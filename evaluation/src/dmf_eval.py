import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from jinja2 import Template
from openai import BadRequestError
from openai import OpenAI
from tqdm import tqdm

from prompts import ANSWER_PROMPT_DMF
from src.runtime_config import apply_runtime_env, get_answer_model

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


class DMFManager:
    def __init__(
        self,
        dataset_path="dataset/locomo10_rag.json",
        output_path="results/dmf_results.json",
        dmf_repo_path="/Users/mat/workspace/personal/paper/agentic-summarization",
        dmf_config_path="config/dmf_benchmark_settings.toml",
        run_root="run/dmf",
        runtime_config=None,
        model=None,
        max_conversations=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.dmf_repo_path = Path(dmf_repo_path).resolve()
        self.dmf_config_path = Path(dmf_config_path).resolve()
        self.runtime_config = runtime_config or {}
        apply_runtime_env(self.runtime_config)
        self.model = model or get_answer_model(self.runtime_config)
        self.max_conversations = max_conversations
        self.results = defaultdict(list)
        self.answer_template = Template(ANSWER_PROMPT_DMF)
        self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
        openai_kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **openai_kwargs)
        self.run_root = Path(run_root).resolve()
        self._ensure_dmf_import_path()
        self._load_dmf_components()

    def _ensure_dmf_import_path(self):
        dmf_src = self.dmf_repo_path / "src"
        if not dmf_src.exists():
            raise FileNotFoundError(f"DMF src path not found: {dmf_src}")
        dmf_src_str = str(dmf_src)
        if dmf_src_str not in sys.path:
            sys.path.insert(0, dmf_src_str)

    def _load_dmf_components(self):
        from dmf.analysis.scoring_engine import ScoringEngine
        from dmf.memory.api import Memory
        from dmf.memory.temporal_memory import TemporalMemory
        from dmf.models.analysis import InteractionProvenance
        from dmf.runtime.pipeline import InteractionPipeline
        from dmf.utils.config_loader import load_dmf_config

        self.ScoringEngine = ScoringEngine
        self.Memory = Memory
        self.TemporalMemory = TemporalMemory
        self.InteractionPipeline = InteractionPipeline
        self.InteractionProvenance = InteractionProvenance
        self.load_dmf_config = load_dmf_config

    def _load_dataset(self):
        with self.dataset_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _slugify(self, value):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return slug or "conversation"

    def _conversation_runtime_paths(self, conversation_id):
        slug = self._slugify(conversation_id)
        conversation_root = self.run_root / slug
        conversation_root.mkdir(parents=True, exist_ok=True)
        chroma_path = conversation_root / "chroma"
        storage_path = conversation_root / "ltm_archive.jsonl"
        runtime_config_path = conversation_root / "dmf_settings.runtime.toml"
        return {
            "root": conversation_root,
            "chroma_path": chroma_path,
            "storage_path": storage_path,
            "runtime_config_path": runtime_config_path,
            "collection_name": f"dmf_{slug}",
        }

    def _materialize_runtime_config(self, conversation_id):
        runtime_paths = self._conversation_runtime_paths(conversation_id)
        config_text = self.dmf_config_path.read_text(encoding="utf-8")

        replacements = {
            "storage_path": runtime_paths["storage_path"].as_posix(),
            "chroma_path": runtime_paths["chroma_path"].as_posix(),
            "collection_name": runtime_paths["collection_name"],
        }
        for key, value in replacements.items():
            config_text = re.sub(
                rf'^{key}\s*=\s*".*"$',
                f'{key} = "{value}"',
                config_text,
                flags=re.MULTILINE,
            )

        runtime_paths["runtime_config_path"].write_text(config_text, encoding="utf-8")
        return runtime_paths["runtime_config_path"]

    def _build_runtime(self, conversation_id):
        config_path = self._materialize_runtime_config(conversation_id)
        cfg = self.load_dmf_config(config_path)
        pipeline = self.InteractionPipeline.from_dmf_config(cfg, analyze_system_prompt=False)
        scoring = self.ScoringEngine.from_dmf_config(cfg)
        temporal_memory = self.TemporalMemory.from_dmf_config(
            cfg,
            nlp_engine=pipeline._nlp_engine,
        )
        memory_api = self.Memory(
            temporal_memory=temporal_memory,
            embedding_engine=pipeline._embedding_engine,
        )
        return {
            "config_path": config_path,
            "cfg": cfg,
            "pipeline": pipeline,
            "scoring": scoring,
            "temporal_memory": temporal_memory,
            "memory_api": memory_api,
        }

    def _format_turn(self, turn):
        timestamp = str(turn.get("timestamp", "")).strip()
        speaker = str(turn.get("speaker", "unknown")).strip()
        text = str(turn.get("text", "")).strip()
        return f"{timestamp} | {speaker}: {text}".strip()

    def _ingest_turn(self, runtime, turn_text, step):
        provenance = self.InteractionProvenance(
            role="user",
            source_turn=step,
            derived_from_model=False,
        )
        report, vector = runtime["pipeline"].analyze_interaction_with_vector(
            turn_text,
            is_system=False,
            provenance=provenance,
        )
        if vector is None:
            return
        runtime["scoring"].calculate_score(report, text=turn_text)
        runtime["temporal_memory"].add_interaction(turn_text, report, vector)

    def _count_tokens(self, text):
        if not text:
            return 0
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def _build_context(self, runtime, question):
        start = time.time()
        context = runtime["memory_api"].get_memory(query_text=question, include_ltm=True)
        end = time.time()
        return context, end - start

    def _sanitize_text_for_api(self, text):
        if not text:
            return ""
        sanitized = str(text)
        sanitized = "".join(ch if (ord(ch) >= 32 or ch in "\n\r\t") else " " for ch in sanitized)
        sanitized = sanitized.encode("utf-8", errors="replace").decode("utf-8")
        return sanitized

    def _answer_question(self, question, context):
        prompt_text = self.answer_template.render(context=context, question=question)
        prompt_text = self._sanitize_text_for_api(prompt_text)
        start = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt_text}],
                temperature=0.0,
            )
        except BadRequestError:
            debug_dir = self.output_path.parent / "dmf_prompt_failures"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"failed_prompt_{int(time.time() * 1000)}.txt"
            debug_path.write_text(prompt_text, encoding="utf-8", errors="replace")
            raise
        end = time.time()
        return response.choices[0].message.content.strip(), end - start, prompt_text

    def _diagnostics_payload(self, runtime):
        return runtime["temporal_memory"].get_recall_diagnostics()

    def process_conversation(self, key, value):
        runtime = self._build_runtime(key)
        for step, turn in enumerate(value["conversation"]):
            turn_text = self._format_turn(turn)
            self._ingest_turn(runtime, turn_text, step)

        conversation_results = []
        for item in value["question"]:
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            category = item.get("category", "")
            context, search_time = self._build_context(runtime, question)
            diagnostics = self._diagnostics_payload(runtime)
            response, response_time, prompt_text = self._answer_question(question, context)

            conversation_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "response": response,
                    "context": context,
                    "search_time": search_time,
                    "response_time": response_time,
                    "context_tokens": self._count_tokens(context),
                    "prompt_tokens_total": self._count_tokens(prompt_text),
                    "internal_llm_prompt_tokens_conversation": 0,
                    "internal_llm_completion_tokens_conversation": 0,
                    "internal_llm_total_tokens_conversation": 0,
                    "internal_llm_calls_conversation": 0,
                    "internal_llm_prompt_tokens_amortized": 0.0,
                    "internal_llm_completion_tokens_amortized": 0.0,
                    "internal_llm_total_tokens_amortized": 0.0,
                    "total_pipeline_tokens_amortized": self._count_tokens(prompt_text),
                    "recalled_count": len(diagnostics.get("final_candidates", [])),
                    "dmf_recall_diagnostics": diagnostics,
                }
            )

        return conversation_results

    def process_all_conversations(self):
        data = self._load_dataset()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)

        items = list(data.items())
        if self.max_conversations is not None:
            items = items[: self.max_conversations]

        for key, value in tqdm(items, desc="Processing conversations"):
            self.results[key].extend(self.process_conversation(key, value))
            with self.output_path.open("w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4)

        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
