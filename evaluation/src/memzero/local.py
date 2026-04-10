import json
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from jinja2 import Template
from openai import OpenAI
from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import get_fact_retrieval_messages, parse_messages, process_telemetry_filters, remove_code_blocks
from prompts import ANSWER_PROMPT
from tqdm import tqdm

from mem0 import Memory
from src.runtime_config import apply_runtime_env, get_answer_model, get_embedding_settings

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


class LLMUsageTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def callback(self, _llm, response, _params):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        self.calls += 1

    def summary(self):
        return {
            "internal_llm_prompt_tokens": self.prompt_tokens,
            "internal_llm_completion_tokens": self.completion_tokens,
            "internal_llm_total_tokens": self.total_tokens,
            "internal_llm_calls": self.calls,
        }


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

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
                limit=5,
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
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )

            try:
                response = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
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


def _speaker_usage_path(output_root, conversation_idx, speaker_label):
    return Path(output_root) / str(conversation_idx) / speaker_label / "llm_usage.json"


def _write_usage_summary(output_root, conversation_idx, speaker_label, usage_summary):
    usage_path = _speaker_usage_path(output_root, conversation_idx, speaker_label)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(json.dumps(usage_summary, indent=2), encoding="utf-8")


def _read_usage_summary(output_root, conversation_idx, speaker_label):
    usage_path = _speaker_usage_path(output_root, conversation_idx, speaker_label)
    if not usage_path.exists():
        return {
            "internal_llm_prompt_tokens": 0,
            "internal_llm_completion_tokens": 0,
            "internal_llm_total_tokens": 0,
            "internal_llm_calls": 0,
        }
    return json.loads(usage_path.read_text(encoding="utf-8"))


class MemoryLocalAdd:
    def __init__(
        self,
        data_path=None,
        config_path="config/mem0_local_config.json",
        output_root="run/mem0_local_v1",
        runtime_config=None,
    ):
        self.data_path = data_path
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.runtime_config = runtime_config or {}
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
        usage_tracker = LLMUsageTracker()
        speaker_root = self.output_root / str(conversation_idx) / speaker_label
        speaker_root.mkdir(parents=True, exist_ok=True)
        cfg["vector_store"]["provider"] = mem0_cfg.get("vector_store_provider", cfg["vector_store"]["provider"])
        cfg["vector_store"]["config"]["path"] = speaker_root.as_posix()
        cfg["vector_store"]["config"]["collection_name"] = f"mem0_local_{conversation_idx}_{speaker_label}"
        cfg["llm"]["provider"] = mem0_cfg.get("llm_provider", cfg["llm"]["provider"])
        cfg["llm"]["config"]["model"] = mem0_cfg.get("llm_model", get_answer_model(self.runtime_config))
        cfg["llm"]["config"]["response_callback"] = usage_tracker.callback
        cfg["embedder"]["provider"] = embedding["provider"]
        cfg["embedder"]["config"]["model"] = embedding["model"]
        cfg["embedder"]["config"]["embedding_dims"] = embedding["embedding_dims"]
        memory = SafeMemory.from_config(cfg)
        memory.reset()
        memory._usage_tracker = usage_tracker
        return memory

    def _build_speaker_messages(self, chats, speaker_a, speaker_b):
        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
            else:
                raise ValueError(f"Unknown speaker: {chat['speaker']}")
        return messages, messages_reverse

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        memory_a = self._speaker_runtime(idx, "speaker_a")
        memory_b = self._speaker_runtime(idx, "speaker_b")

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]
            messages, messages_reverse = self._build_speaker_messages(chats, speaker_a, speaker_b)
            memory_a.add(messages, user_id=speaker_a_user_id, metadata={"timestamp": timestamp})
            memory_b.add(messages_reverse, user_id=speaker_b_user_id, metadata={"timestamp": timestamp})

        _write_usage_summary(self.output_root, idx, "speaker_a", memory_a._usage_tracker.summary())
        _write_usage_summary(self.output_root, idx, "speaker_b", memory_b._usage_tracker.summary())

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        self.output_root.mkdir(parents=True, exist_ok=True)
        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Processing conversations"):
            self.process_conversation(item, idx)


class MemoryLocalSearch:
    def __init__(
        self,
        output_path="results/mem0_local_results.json",
        config_path="config/mem0_local_config.json",
        output_root="run/mem0_local_v1",
        top_k=30,
        runtime_config=None,
    ):
        self.output_path = Path(output_path)
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.top_k = top_k
        self.runtime_config = runtime_config or {}
        apply_runtime_env(self.runtime_config)
        self.results = defaultdict(list)
        openai_kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **openai_kwargs)
        self.answer_template = Template(ANSWER_PROMPT)
        self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None

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

    def search_memory(self, memory, user_id, query):
        start = time.time()
        memories = memory.search(query, user_id=user_id, limit=self.top_k)
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

    def answer_question(self, memory_a, memory_b, speaker_a_user_id, speaker_b_user_id, question):
        speaker_1_memories, speaker_1_memory_time = self.search_memory(memory_a, speaker_a_user_id, question)
        speaker_2_memories, speaker_2_memory_time = self.search_memory(memory_b, speaker_b_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        answer_prompt = self.answer_template.render(
            speaker_1_user_id=speaker_a_user_id.split("_")[0],
            speaker_2_user_id=speaker_b_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            question=question,
        )

        start = time.time()
        response = self.openai_client.chat.completions.create(
            model=get_answer_model(self.runtime_config),
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        end = time.time()

        return {
            "response": response.choices[0].message.content.strip(),
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "response_time": end - start,
            "prompt_tokens_total": self._count_tokens(answer_prompt),
            "context_tokens": self._count_tokens("\n".join(search_1_memory + search_2_memory)),
        }

    def process_data_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            memory_a = self._speaker_runtime(idx, "speaker_a")
            memory_b = self._speaker_runtime(idx, "speaker_b")
            usage_a = _read_usage_summary(self.output_root, idx, "speaker_a")
            usage_b = _read_usage_summary(self.output_root, idx, "speaker_b")
            internal_prompt_tokens = usage_a["internal_llm_prompt_tokens"] + usage_b["internal_llm_prompt_tokens"]
            internal_completion_tokens = usage_a["internal_llm_completion_tokens"] + usage_b["internal_llm_completion_tokens"]
            internal_total_tokens = usage_a["internal_llm_total_tokens"] + usage_b["internal_llm_total_tokens"]
            internal_calls = usage_a["internal_llm_calls"] + usage_b["internal_llm_calls"]
            amortized_internal_total = internal_total_tokens / len(qa) if qa else 0.0
            amortized_internal_prompt = internal_prompt_tokens / len(qa) if qa else 0.0
            amortized_internal_completion = internal_completion_tokens / len(qa) if qa else 0.0

            for question_item in tqdm(qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False):
                answer_bundle = self.answer_question(
                    memory_a,
                    memory_b,
                    speaker_a_user_id,
                    speaker_b_user_id,
                    question_item.get("question", ""),
                )
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
                    "storage_backend": "chroma",
                }
                self.results[idx].append(result)
                with self.output_path.open("w", encoding="utf-8") as f:
                    json.dump(self.results, f, indent=4)

        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
