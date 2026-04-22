import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from jinja2 import Template
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_MEM0_INIT = _REPO_ROOT / "mem0" / "__init__.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mem0
from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import get_fact_retrieval_messages, parse_messages, process_telemetry_filters, remove_code_blocks
from mem0.utils.openai_retry import call_with_rate_limit_retry
from openai import BadRequestError, OpenAI
from prompts import ANSWER_PROMPT
from tqdm import tqdm

from mem0 import Memory
from src.memzero.progress import Mem0BenchmarkProgress
from src.runtime_config import apply_runtime_env, get_answer_model, get_embedding_settings
from src.token_accounting import (
    build_run_summary,
    bundle_artifact_dir,
    copy_snapshot,
    generate_reporting,
    make_phase_tokens,
    make_token_breakdown,
    prepare_artifact_dir,
    sum_phase_totals,
    write_json,
    write_summary_markdown,
    zero_phase_tokens,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional env loader
    def load_dotenv():
        return False

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for lightweight smoke runs
    tiktoken = None

load_dotenv()

logger = logging.getLogger(__name__)

if Path(mem0.__file__).resolve() != _LOCAL_MEM0_INIT.resolve():
    raise ImportError(
        "mem0_local benchmark must import the repo-local mem0 package. "
        f"Expected {_LOCAL_MEM0_INIT}, got {mem0.__file__}."
    )


def _zero_usage_summary():
    return {
        "internal_llm_prompt_tokens": 0,
        "internal_llm_completion_tokens": 0,
        "internal_llm_total_tokens": 0,
        "internal_llm_calls": 0,
    }


def _sanitize_openai_text(value):
    """Remove control characters that can break provider-side JSON parsing."""
    text = str(value or "")
    text = text.replace("\x00", "")
    text = text.encode("utf-8", "replace").decode("utf-8")
    return re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)


def _sanitize_openai_messages(messages):
    sanitized = []
    for message in messages:
        sanitized.append(
            {
                "role": message.get("role", "user"),
                "content": _sanitize_openai_text(message.get("content", "")),
            }
        )
    return sanitized


class SafeMemory(Memory):
    @staticmethod
    def _normalize_text(value):
        return " ".join(str(value or "").strip().lower().split())

    def _resolve_existing_memory_id(self, resp, temp_uuid_mapping, old_text_to_uuid):
        raw_id = str(resp.get("id", ""))
        if raw_id in temp_uuid_mapping:
            return temp_uuid_mapping[raw_id]

        old_memory = self._normalize_text(resp.get("old_memory"))
        if old_memory and old_memory in old_text_to_uuid:
            return old_text_to_uuid[old_memory]

        action_text = self._normalize_text(resp.get("text"))
        if action_text and action_text in old_text_to_uuid:
            return old_text_to_uuid[action_text]

        return None

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            return super()._add_to_vector_store(messages, metadata, filters, infer)

        sanitized_messages = _sanitize_openai_messages(messages)
        parsed_messages = parse_messages(sanitized_messages)
        custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)

        if custom_fact_extraction_prompt:
            system_prompt = custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        fact_messages = [
            {"role": "system", "content": _sanitize_openai_text(system_prompt)},
            {"role": "user", "content": _sanitize_openai_text(user_prompt)},
        ]
        try:
            response = self.llm.generate_response(
                messages=fact_messages,
                response_format={"type": "json_object"},
            )
        except BadRequestError:
            logger.exception("BadRequest during fact extraction; retrying with sanitized payload.")
            response = self.llm.generate_response(
                messages=_sanitize_openai_messages(fact_messages),
                response_format={"type": "json_object"},
            )

        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        retrieved_old_memory = []
        new_message_embeddings = {}
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                top_k=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())

        temp_uuid_mapping = {}
        old_text_to_uuid = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            old_text_to_uuid[self._normalize_text(item["text"])] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, custom_update_memory_prompt
            )

            try:
                update_messages = [{"role": "user", "content": _sanitize_openai_text(function_calling_prompt)}]
                response = self.llm.generate_response(
                    messages=update_messages,
                    response_format={"type": "json_object"},
                )
            except BadRequestError:
                logger.exception("BadRequest during memory update planning; retrying with sanitized payload.")
                response = self.llm.generate_response(
                    messages=_sanitize_openai_messages(update_messages),
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                response = remove_code_blocks(response)
                new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        for resp in new_memories_with_actions.get("memory", []):
            try:
                action_text = resp.get("text")
                if not action_text:
                    continue

                event_type = resp.get("event")
                resolved_id = self._resolve_existing_memory_id(resp, temp_uuid_mapping, old_text_to_uuid)

                if event_type == "ADD":
                    memory_id = self._create_memory(
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(metadata),
                    )
                    returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                elif event_type == "UPDATE":
                    if resolved_id is None:
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": action_text,
                                "event": "ADD",
                                "fallback_from": "UPDATE",
                            }
                        )
                        continue

                    self._update_memory(
                        memory_id=resolved_id,
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(metadata),
                    )
                    returned_memories.append(
                        {
                            "id": resolved_id,
                            "memory": action_text,
                            "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                        }
                    )
                elif event_type == "DELETE":
                    if resolved_id is None:
                        continue
                    self._delete_memory(memory_id=resolved_id)
                    returned_memories.append(
                        {
                            "id": resolved_id,
                            "memory": action_text,
                            "event": event_type,
                        }
                    )
                elif event_type == "NONE":
                    continue
            except Exception as e:
                logger.error(f"Error processing memory action: {resp}, Error: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories


def _normalize_mem0_usage(usage_snapshot):
    usage_snapshot = usage_snapshot or {}
    return {
        "internal_llm_prompt_tokens": int(usage_snapshot.get("prompt_tokens", 0) or 0),
        "internal_llm_completion_tokens": int(usage_snapshot.get("completion_tokens", 0) or 0),
        "internal_llm_total_tokens": int(usage_snapshot.get("total_tokens", 0) or 0),
        "internal_llm_calls": int(usage_snapshot.get("calls", 0) or 0),
        "scopes": usage_snapshot.get("scopes", {}) or {},
    }


def _conversation_usage_path(output_root, conversation_idx):
    return Path(output_root) / str(conversation_idx) / "conversation_usage.json"


def _speaker_store_path(output_root, conversation_idx, speaker_label):
    return Path(output_root) / str(conversation_idx) / speaker_label


def _write_conversation_usage_summary(output_root, conversation_idx, speaker_usage):
    usage_path = _conversation_usage_path(output_root, conversation_idx)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    total_usage = _zero_usage_summary()
    for usage in speaker_usage.values():
        total_usage["internal_llm_prompt_tokens"] += int(usage.get("internal_llm_prompt_tokens", 0) or 0)
        total_usage["internal_llm_completion_tokens"] += int(usage.get("internal_llm_completion_tokens", 0) or 0)
        total_usage["internal_llm_total_tokens"] += int(usage.get("internal_llm_total_tokens", 0) or 0)
        total_usage["internal_llm_calls"] += int(usage.get("internal_llm_calls", 0) or 0)
    usage_path.write_text(
        json.dumps(
            {
                **total_usage,
                "speakers": speaker_usage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_conversation_usage_summary(output_root, conversation_idx):
    usage_path = _conversation_usage_path(output_root, conversation_idx)
    if not usage_path.exists():
        return _zero_usage_summary()
    return json.loads(usage_path.read_text(encoding="utf-8"))


def _parse_session_datetime(value):
    try:
        return datetime.strptime(str(value), "%I:%M %p on %d %B, %Y")
    except (TypeError, ValueError):
        return None


class MemoryLocalAdd:
    def __init__(
        self,
        data_path=None,
        config_path="config/mem0_local_config.json",
        output_root="run/mem0_local_v1",
        runtime_config=None,
        log_progress=False,
        progress_mode="detailed",
        resume=False,
    ):
        self.data_path = data_path
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.runtime_config = runtime_config or {}
        self.progress = Mem0BenchmarkProgress("mem0_local:add", enabled=log_progress, mode=progress_mode)
        self.resume = resume
        apply_runtime_env(self.runtime_config)
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def _load_base_config(self):
        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Mem0 OSS interprets `custom_fact_extraction_prompt` as a full prompt replacement,
        # unlike the managed service's project-level custom instructions. Keeping it here
        # causes empty `facts` on simple benchmark inputs, so we rely on the native OSS prompt.
        cfg.pop("custom_fact_extraction_prompt", None)
        return cfg

    def _speaker_runtime(self, conversation_idx, speaker_label):
        cfg = deepcopy(self._load_base_config())
        embedding = get_embedding_settings(self.runtime_config)
        mem0_cfg = self.runtime_config.get("mem0_local", {})
        speaker_root = self.output_root / str(conversation_idx) / speaker_label
        speaker_root.mkdir(parents=True, exist_ok=True)
        cfg["vector_store"]["provider"] = mem0_cfg.get("vector_store_provider", cfg["vector_store"]["provider"])
        cfg["vector_store"]["config"]["path"] = speaker_root.as_posix()
        cfg["vector_store"]["config"]["collection_name"] = f"mem0_local_{conversation_idx}_{speaker_label}"
        cfg["llm"]["provider"] = mem0_cfg.get("llm_provider", cfg["llm"]["provider"])
        cfg["llm"]["config"]["model"] = mem0_cfg.get("llm_model", get_answer_model(self.runtime_config))
        cfg["embedder"]["provider"] = embedding["provider"]
        cfg["embedder"]["config"]["model"] = embedding["model"]
        cfg["embedder"]["config"]["embedding_dims"] = embedding["embedding_dims"]
        memory = SafeMemory.from_config(cfg)
        memory.reset()
        return memory

    def _build_speaker_messages(self, chats, speaker_a, speaker_b):
        messages = []
        messages_reverse = []
        for chat in chats:
            chat_text = _sanitize_openai_text(chat["text"])
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat_text}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat_text}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat_text}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat_text}"})
            else:
                raise ValueError(f"Unknown speaker: {chat['speaker']}")
        return messages, messages_reverse

    def _iter_sessions(self, conversation):
        sessions = []
        for key, value in conversation.items():
            if key in {"speaker_a", "speaker_b"} or key.endswith("_date_time") or "timestamp" in key:
                continue
            date_time_key = f"{key}_date_time"
            timestamp = conversation.get(date_time_key)
            sessions.append(
                (
                    _parse_session_datetime(timestamp) or datetime.max,
                    key,
                    timestamp,
                    value,
                )
            )
        for _, key, timestamp, chats in sorted(sessions, key=lambda item: (item[0], item[1])):
            yield key, timestamp, chats

    def _conversation_sessions(self, conversation):
        return list(self._iter_sessions(conversation))

    def _conversation_add_completed(self, conversation_idx):
        usage_path = _conversation_usage_path(self.output_root, conversation_idx)
        speaker_a_store = _speaker_store_path(self.output_root, conversation_idx, "speaker_a")
        speaker_b_store = _speaker_store_path(self.output_root, conversation_idx, "speaker_b")
        return usage_path.exists() and speaker_a_store.exists() and speaker_b_store.exists()

    def _resume_start_index(self):
        if not self.resume:
            return 0
        for idx, item in enumerate(self.data or []):
            if not self._conversation_add_completed(idx):
                return idx
        return len(self.data or [])

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"
        sessions = self._conversation_sessions(conversation)

        self.progress.start_conversation(
            idx,
            conversation_id=str(idx),
            total_units=len(sessions),
            unit_label="session",
            note="starting local memory ingestion",
        )

        memory_a = self._speaker_runtime(idx, "speaker_a")
        memory_b = self._speaker_runtime(idx, "speaker_b")

        for session_idx, (session_key, timestamp, chats) in enumerate(sessions, start=1):
            self.progress.log_step(
                phase="ingest",
                step="session_start",
                current_unit=session_idx,
                note=f"preparing session={session_key}",
            )
            messages, messages_reverse = self._build_speaker_messages(chats, speaker_a, speaker_b)
            self.progress.log_step(
                phase="ingest",
                step="adding_speaker_a",
                current_unit=session_idx,
                note=f"speaker={speaker_a_user_id}",
            )
            memory_a.add(messages, user_id=speaker_a_user_id, metadata={"timestamp": timestamp})
            self.progress.log_step(
                phase="ingest",
                step="adding_speaker_b",
                current_unit=session_idx,
                note=f"speaker={speaker_b_user_id}",
            )
            memory_b.add(messages_reverse, user_id=speaker_b_user_id, metadata={"timestamp": timestamp})
            self.progress.mark_unit_complete(
                phase="ingest",
                step="session_complete",
                current_unit=session_idx,
                note=f"completed session={session_key}",
            )

        _write_conversation_usage_summary(
            self.output_root,
            idx,
            {
                "speaker_a": _normalize_mem0_usage(memory_a.get_llm_usage()),
                "speaker_b": _normalize_mem0_usage(memory_b.get_llm_usage()),
            },
        )
        self.progress.finish_conversation(note="conversation usage summary written")

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        self.output_root.mkdir(parents=True, exist_ok=True)
        total_sessions = sum(len(self._conversation_sessions(item["conversation"])) for item in self.data)
        self.progress.start_run(
            total_conversations=len(self.data),
            total_units=total_sessions,
            unit_label="session",
            note="starting mem0_local add benchmark",
        )
        start_idx = self._resume_start_index()
        for idx, item in tqdm(
            enumerate(self.data),
            total=len(self.data),
            desc="Processing conversations",
            disable=self.progress.tqdm_disabled,
        ):
            conversation_sessions = self._conversation_sessions(item["conversation"])
            if self.resume and idx < start_idx:
                self.progress.mark_conversation_complete(
                    conversation_index=idx,
                    conversation_id=str(idx),
                    unit_count=len(conversation_sessions),
                    note="skipping completed local add conversation",
                )
                continue
            self.process_conversation(item, idx)
        self.progress.finish_run(note="mem0_local add benchmark completed")


class MemoryLocalSearch:
    def __init__(
        self,
        output_path="results/mem0_local_results.json",
        config_path="config/mem0_local_config.json",
        output_root="run/mem0_local_v1",
        top_k=30,
        runtime_config=None,
        log_progress=False,
        progress_mode="detailed",
        resume=False,
    ):
        self.output_path = Path(output_path)
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.top_k = top_k
        self.runtime_config = runtime_config or {}
        self.progress = Mem0BenchmarkProgress("mem0_local:search", enabled=log_progress, mode=progress_mode)
        self.resume = resume
        apply_runtime_env(self.runtime_config)
        self.results = defaultdict(list)
        self.conversation_summaries = []
        openai_kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **openai_kwargs)
        self.answer_template = Template(ANSWER_PROMPT)
        self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
        self.artifact_info = prepare_artifact_dir(self.output_path.parent, "mem0_local_token_accounting")
        self.dataset_path = None

    def _load_base_config(self):
        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.pop("custom_fact_extraction_prompt", None)
        return cfg

    def _speaker_runtime(self, conversation_idx, speaker_label):
        cfg = deepcopy(self._load_base_config())
        embedding = get_embedding_settings(self.runtime_config)
        mem0_cfg = self.runtime_config.get("mem0_local", {})
        speaker_root = self.output_root / str(conversation_idx) / speaker_label
        cfg["vector_store"]["provider"] = mem0_cfg.get("vector_store_provider", cfg["vector_store"]["provider"])
        cfg["vector_store"]["config"]["path"] = speaker_root.as_posix()
        cfg["vector_store"]["config"]["collection_name"] = f"mem0_local_{conversation_idx}_{speaker_label}"
        cfg["llm"]["provider"] = mem0_cfg.get("llm_provider", cfg["llm"]["provider"])
        cfg["llm"]["config"]["model"] = mem0_cfg.get("llm_model", get_answer_model(self.runtime_config))
        cfg["embedder"]["provider"] = embedding["provider"]
        cfg["embedder"]["config"]["model"] = embedding["model"]
        cfg["embedder"]["config"]["embedding_dims"] = embedding["embedding_dims"]
        return SafeMemory.from_config(cfg)

    def _count_tokens(self, text):
        if not text:
            return 0
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def _validate_conversation_runtime(self, conversation_idx):
        required_labels = ("speaker_a", "speaker_b")
        missing_paths = []
        usage_path = _conversation_usage_path(self.output_root, conversation_idx)
        for label in required_labels:
            store_path = _speaker_store_path(self.output_root, conversation_idx, label)
            if not store_path.exists():
                missing_paths.append(store_path.as_posix())
        missing_usage = [] if usage_path.exists() else [usage_path.as_posix()]
        if missing_paths or missing_usage:
            problems = []
            if missing_paths:
                problems.append(f"missing speaker stores: {', '.join(missing_paths)}")
            if missing_usage:
                problems.append(f"missing usage summaries: {', '.join(missing_usage)}")
            raise FileNotFoundError(
                "mem0_local search requires a completed add run for the same output_root; "
                + "; ".join(problems)
            )

    def search_memory(self, memory, user_id, query):
        start = time.time()
        memories = memory.search(query, filters={"user_id": user_id}, top_k=self.top_k)
        end = time.time()
        results = memories.get("results", memories)
        semantic_memories = [
            {
                "memory": item.get("memory", ""),
                "timestamp": (item.get("metadata") or {}).get("timestamp"),
                "score": round(float(item.get("score", 0.0)), 2) if item.get("score") is not None else 0.0,
            }
            for item in results
        ]
        return semantic_memories, end - start

    def answer_question(
        self,
        memory_a,
        memory_b,
        speaker_a_user_id,
        speaker_b_user_id,
        question,
        *,
        question_index=None,
    ):
        self.progress.log_step(
            phase="search",
            step="retrieving_memories",
            current_unit=question_index,
            note="retrieving local memories for both speakers",
        )
        speaker_1_memories, speaker_1_memory_time = self.search_memory(memory_a, speaker_a_user_id, question)
        speaker_2_memories, speaker_2_memory_time = self.search_memory(memory_b, speaker_b_user_id, question)
        total_search_time = speaker_1_memory_time + speaker_2_memory_time

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        answer_prompt = self.answer_template.render(
            speaker_1_user_id=speaker_a_user_id.split("_")[0],
            speaker_2_user_id=speaker_b_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            question=_sanitize_openai_text(question),
        )

        self.progress.log_step(
            phase="answer",
            step="generating_answer",
            current_unit=question_index,
            note="calling answer model",
        )
        start = time.time()
        response = call_with_rate_limit_retry(
            self.openai_client.chat.completions.create,
            request_name="mem0_local search answer generation",
            model=get_answer_model(self.runtime_config),
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        end = time.time()
        usage = getattr(response, "usage", None)

        return {
            "response": response.choices[0].message.content.strip(),
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "search_time": total_search_time,
            "response_time": end - start,
            "prompt_tokens_total": self._count_tokens(answer_prompt),
            "context_tokens": self._count_tokens("\n".join(search_1_memory + search_2_memory)),
            "answer_usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else None,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else None,
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else None,
                "observation_mode": "provider_reported" if usage is not None else "estimated_local",
            },
        }

    def _question_token_bundle(self, answer_bundle):
        answer_usage = answer_bundle["answer_usage"]
        if answer_usage["observation_mode"] == "provider_reported":
            answer_phase = make_phase_tokens(
                answer_usage["prompt_tokens"],
                answer_usage["completion_tokens"],
                answer_usage["total_tokens"],
                "provider_reported",
            )
        else:
            estimated_prompt_tokens = int(answer_bundle["prompt_tokens_total"] or 0)
            answer_phase = make_phase_tokens(
                estimated_prompt_tokens,
                None,
                estimated_prompt_tokens,
                "estimated_local",
            )
        token_breakdown = make_token_breakdown(
            ingest=zero_phase_tokens(),
            retrieval=zero_phase_tokens(),
            answer=answer_phase,
            maintenance=zero_phase_tokens(),
        )
        return {
            "total_tokens": sum_phase_totals(token_breakdown),
            "token_breakdown": token_breakdown,
        }

    def _conversation_summary(
        self,
        conversation_idx,
        question_results,
        internal_prompt_tokens,
        internal_completion_tokens,
        internal_total_tokens,
    ):
        answer_prompt_values = [item["token_breakdown"]["answer"]["prompt_tokens"] for item in question_results]
        answer_completion_values = [item["token_breakdown"]["answer"]["completion_tokens"] for item in question_results]
        answer_total_values = [item["token_breakdown"]["answer"]["total_tokens"] for item in question_results]
        answer_phase = make_phase_tokens(
            sum(value or 0 for value in answer_prompt_values),
            None if any(value is None for value in answer_completion_values) else sum(value or 0 for value in answer_completion_values),
            sum(value or 0 for value in answer_total_values),
            "provider_reported" if not any(value is None for value in answer_completion_values) else "estimated_local",
        )
        token_breakdown = make_token_breakdown(
            ingest=make_phase_tokens(
                internal_prompt_tokens,
                internal_completion_tokens,
                internal_total_tokens,
                "provider_reported",
            ),
            retrieval=zero_phase_tokens(),
            answer=answer_phase,
            maintenance=zero_phase_tokens(),
        )
        return {
            "conversation_id": str(conversation_idx),
            "question_count": len(question_results),
            "total_tokens": sum_phase_totals(token_breakdown),
            "legacy_total_pipeline_tokens": sum(
                int(item.get("total_pipeline_tokens_amortized", 0) or 0) for item in question_results
            ),
            "token_breakdown": token_breakdown,
        }

    def _write_token_accounting_artifacts(self):
        timestamp = self.artifact_info["timestamp"]
        run_summary = build_run_summary(
            technique="mem0_local",
            dataset_path=self.dataset_path or self.output_path,
            result_output_path=self.output_path,
            conversation_summaries=self.conversation_summaries,
        )
        artifact_dir = self.artifact_info["artifact_dir"]
        write_json(artifact_dir / f"mem0_local_token_accounting_{timestamp}.json", run_summary)
        generate_reporting(self.artifact_info, "mem0_local", run_summary)
        snapshots = [
            copy_snapshot(self.output_path, artifact_dir, "mem0_local_results.json"),
            copy_snapshot(self.config_path, artifact_dir, "config_snapshot.json"),
            copy_snapshot(self.runtime_config.get("_config_path"), artifact_dir, "runtime_config_snapshot.toml"),
        ]
        write_summary_markdown(
            artifact_dir / "summary.md",
            technique="mem0_local",
            run_summary=run_summary,
            snapshots=[snapshot for snapshot in snapshots if snapshot],
            notes=[
                "mem0_local maintenance is currently folded into ingest because the OSS runtime does not expose it separately.",
            ],
        )
        bundle_artifact_dir(artifact_dir, self.artifact_info["zip_path"])

    def _load_existing_results(self):
        if not self.resume or not self.output_path.exists():
            return
        with self.output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.results = defaultdict(list, {str(key): value for key, value in payload.items()})

    def _conversation_search_completed(self, conversation_idx, expected_question_count):
        return len(self.results.get(str(conversation_idx), [])) >= expected_question_count

    def _resume_start_index(self, data):
        if not self.resume:
            return 0
        for idx, item in enumerate(data):
            if not self._conversation_search_completed(idx, len(item["qa"])):
                return idx
        return len(data)

    def _truncate_results_for_resume(self, start_idx):
        if not self.resume:
            return
        keys_to_delete = [key for key in list(self.results.keys()) if int(key) >= start_idx]
        for key in keys_to_delete:
            del self.results[key]

    def process_data_file(self, file_path):
        self.dataset_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_results()
        start_idx = self._resume_start_index(data)
        self._truncate_results_for_resume(start_idx)
        total_questions = sum(len(item["qa"]) for item in data)
        self.progress.start_run(
            total_conversations=len(data),
            total_units=total_questions,
            unit_label="question",
            note="starting mem0_local search benchmark",
        )

        for idx, item in tqdm(
            enumerate(data),
            total=len(data),
            desc="Processing conversations",
            disable=self.progress.tqdm_disabled,
        ):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            self.progress.start_conversation(
                idx,
                conversation_id=str(idx),
                total_units=len(qa),
                unit_label="question",
                note="starting local search conversation",
            )

            self._validate_conversation_runtime(idx)
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            memory_a = self._speaker_runtime(idx, "speaker_a")
            memory_b = self._speaker_runtime(idx, "speaker_b")
            usage_summary = _read_conversation_usage_summary(self.output_root, idx)
            internal_prompt_tokens = usage_summary["internal_llm_prompt_tokens"]
            internal_completion_tokens = usage_summary["internal_llm_completion_tokens"]
            internal_total_tokens = usage_summary["internal_llm_total_tokens"]
            internal_calls = usage_summary["internal_llm_calls"]
            amortized_internal_total = internal_total_tokens / len(qa) if qa else 0.0
            amortized_internal_prompt = internal_prompt_tokens / len(qa) if qa else 0.0
            amortized_internal_completion = internal_completion_tokens / len(qa) if qa else 0.0
            if self.resume and idx < start_idx:
                question_results = list(self.results.get(str(idx), []))
                self.conversation_summaries.append(
                    self._conversation_summary(
                        idx,
                        question_results,
                        internal_prompt_tokens,
                        internal_completion_tokens,
                        internal_total_tokens,
                    )
                )
                self.progress.mark_conversation_complete(
                    conversation_index=idx,
                    conversation_id=str(idx),
                    unit_count=len(qa),
                    note="skipping completed local search conversation",
                )
                continue
            question_results = []
            self.results[str(idx)] = []
            if self.resume and idx == start_idx and start_idx < len(data):
                self.progress.log_step(
                    phase="setup",
                    step="resume_conversation",
                    current_unit=1 if qa else None,
                    note="restarting first incomplete conversation from scratch",
                )

            for question_index, question_item in enumerate(
                tqdm(
                    qa,
                    total=len(qa),
                    desc=f"Processing questions for conversation {idx}",
                    leave=False,
                    disable=self.progress.tqdm_disabled,
                ),
                start=1,
            ):
                answer_bundle = self.answer_question(
                    memory_a,
                    memory_b,
                    speaker_a_user_id,
                    speaker_b_user_id,
                    question_item.get("question", ""),
                    question_index=question_index,
                )
                question_token_bundle = self._question_token_bundle(answer_bundle)
                result = {
                    "question": question_item.get("question", ""),
                    "answer": question_item.get("answer", ""),
                    "category": question_item.get("category", -1),
                    "evidence": question_item.get("evidence", []),
                    "response": answer_bundle["response"],
                    "adversarial_answer": question_item.get("adversarial_answer", ""),
                    "speaker_1_memories": answer_bundle["speaker_1_memories"],
                    "speaker_2_memories": answer_bundle["speaker_2_memories"],
                    "num_speaker_1_memories": len(answer_bundle["speaker_1_memories"]),
                    "num_speaker_2_memories": len(answer_bundle["speaker_2_memories"]),
                    "speaker_1_memory_time": answer_bundle["speaker_1_memory_time"],
                    "speaker_2_memory_time": answer_bundle["speaker_2_memory_time"],
                    "search_time": answer_bundle["search_time"],
                    "response_time": answer_bundle["response_time"],
                    "context_tokens": answer_bundle["context_tokens"],
                    "prompt_tokens_total": answer_bundle["prompt_tokens_total"],
                    "internal_llm_prompt_tokens_conversation": internal_prompt_tokens,
                    "internal_llm_completion_tokens_conversation": internal_completion_tokens,
                    "internal_llm_total_tokens_conversation": internal_total_tokens,
                    "internal_llm_calls_conversation": internal_calls,
                    "internal_llm_prompt_tokens_amortized": amortized_internal_prompt,
                    "internal_llm_completion_tokens_amortized": amortized_internal_completion,
                    "internal_llm_total_tokens_amortized": amortized_internal_total,
                    "total_pipeline_tokens_amortized": answer_bundle["prompt_tokens_total"] + amortized_internal_total,
                    "total_tokens": question_token_bundle["total_tokens"],
                    "token_breakdown": question_token_bundle["token_breakdown"],
                    "storage_backend": "chroma",
                }
                self.results[str(idx)].append(result)
                question_results.append(result)
                self.progress.log_step(
                    phase="persist",
                    step="writing_partial_results",
                    current_unit=question_index,
                    note="writing incremental local search results",
                )
                with self.output_path.open("w", encoding="utf-8") as f:
                    json.dump(self.results, f, indent=4)
                self.progress.mark_unit_complete(
                    phase="persist",
                    step="question_complete",
                    current_unit=question_index,
                    note="stored local question result",
                )

            self.conversation_summaries.append(
                self._conversation_summary(
                    idx,
                    question_results,
                    internal_prompt_tokens,
                    internal_completion_tokens,
                    internal_total_tokens,
                )
            )
            self.progress.finish_conversation(note="local search conversation completed")

        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        self._write_token_accounting_artifacts()
        self.progress.finish_run(note="mem0_local search benchmark completed")
