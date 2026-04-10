import argparse
import os
from pathlib import Path

from src.runtime_config import DEFAULT_RUNTIME_CONFIG_PATH, load_runtime_config
from src.utils import METHODS, TECHNIQUES

DEFAULT_DMF_REPO_PATH = "/Users/mat/workspace/personal/paper/agentic-summarization"
DEFAULT_DMF_CONFIG_PATH = "config/dmf_benchmark_settings.toml"
DEFAULT_MEM0_LOCAL_CONFIG_PATH = "config/mem0_local_config.json"
DEFAULT_DMF_RUN_ROOT = "run/dmf"
DEFAULT_MEM0_LOCAL_OUTPUT_ROOT = "run/mem0_local_v1"


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def _default_dmf_run_root(locomo_rag_json_path, dmf_config_path):
    dataset_stem = Path(locomo_rag_json_path).stem
    config_stem = Path(dmf_config_path).stem
    return str(Path(DEFAULT_DMF_RUN_ROOT) / dataset_stem / config_stem)


def _default_mem0_local_output_root(locomo_json_path):
    dataset_stem = Path(locomo_json_path).stem
    return str(Path("run/mem0_local") / dataset_stem)


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument(
        "--locomo_json_path",
        type=str,
        default="dataset/locomo10.json",
        help="Path to the LOCOMO session-oriented dataset",
    )
    parser.add_argument(
        "--locomo_rag_json_path",
        type=str,
        default="dataset/locomo10_rag.json",
        help="Path to the LOCOMO linearized dataset",
    )
    parser.add_argument(
        "--runtime_config_path",
        type=str,
        default=str(DEFAULT_RUNTIME_CONFIG_PATH),
        help="Path to the shared runtime config for models and API keys",
    )
    parser.add_argument(
        "--dmf_repo_path",
        type=str,
        default=DEFAULT_DMF_REPO_PATH,
        help="Path to the DMF repository root",
    )
    parser.add_argument(
        "--dmf_config_path",
        type=str,
        default=DEFAULT_DMF_CONFIG_PATH,
        help="Path to the benchmark-specific DMF config",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Optional limit for the number of conversations to process",
    )
    parser.add_argument(
        "--dmf_run_root",
        type=str,
        default=DEFAULT_DMF_RUN_ROOT,
        help="Root directory for DMF benchmark runtime state",
    )
    parser.add_argument(
        "--mem0_local_config_path",
        type=str,
        default=DEFAULT_MEM0_LOCAL_CONFIG_PATH,
        help="Path to the local Mem0 benchmark config",
    )
    parser.add_argument(
        "--mem0_local_output_root",
        type=str,
        default=DEFAULT_MEM0_LOCAL_OUTPUT_ROOT,
        help="Root directory for local Mem0 vector stores",
    )

    args = parser.parse_args()
    runtime_config = load_runtime_config(args.runtime_config_path)
    dmf_cfg = runtime_config.get("dmf", {})
    mem0_local_cfg = runtime_config.get("mem0_local", {})

    dmf_repo_path = args.dmf_repo_path
    if dmf_repo_path == DEFAULT_DMF_REPO_PATH:
        dmf_repo_path = dmf_cfg.get("repo_path", dmf_repo_path)

    dmf_config_path = args.dmf_config_path
    if dmf_config_path == DEFAULT_DMF_CONFIG_PATH:
        dmf_config_path = dmf_cfg.get("config_path", dmf_config_path)

    mem0_local_config_path = args.mem0_local_config_path
    if mem0_local_config_path == DEFAULT_MEM0_LOCAL_CONFIG_PATH:
        mem0_local_config_path = mem0_local_cfg.get("config_path", mem0_local_config_path)

    dmf_run_root = args.dmf_run_root
    if dmf_run_root == DEFAULT_DMF_RUN_ROOT:
        dmf_run_root = _default_dmf_run_root(args.locomo_rag_json_path, dmf_config_path)

    mem0_local_output_root = args.mem0_local_output_root
    if mem0_local_output_root == DEFAULT_MEM0_LOCAL_OUTPUT_ROOT:
        mem0_local_output_root = _default_mem0_local_output_root(args.locomo_json_path)

    # Add your experiment logic here
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")

    if args.technique_type == "mem0":
        from src.memzero.add import MemoryADD
        from src.memzero.search import MemorySearch

        if args.method == "add":
            memory_manager = MemoryADD(data_path=args.locomo_json_path, is_graph=args.is_graph)
            memory_manager.process_all_conversations()
        elif args.method == "search":
            output_file_path = os.path.join(
                args.output_folder,
                f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}_graph_{args.is_graph}.json",
            )
            memory_searcher = MemorySearch(output_file_path, args.top_k, args.filter_memories, args.is_graph)
            memory_searcher.process_data_file(args.locomo_json_path)
    elif args.technique_type == "mem0_local":
        from src.memzero.local import MemoryLocalAdd, MemoryLocalSearch

        if args.method == "add":
            memory_manager = MemoryLocalAdd(
                data_path=args.locomo_json_path,
                config_path=mem0_local_config_path,
                output_root=mem0_local_output_root,
                runtime_config=runtime_config,
            )
            memory_manager.process_all_conversations()
        elif args.method == "search":
            output_file_path = os.path.join(args.output_folder, "mem0_local_results.json")
            memory_searcher = MemoryLocalSearch(
                output_path=output_file_path,
                config_path=mem0_local_config_path,
                output_root=mem0_local_output_root,
                runtime_config=runtime_config,
                top_k=args.top_k,
            )
            memory_searcher.process_data_file(args.locomo_json_path)
    elif args.technique_type == "rag":
        from src.rag import RAGManager

        output_file_path = os.path.join(args.output_folder, f"rag_results_{args.chunk_size}_k{args.num_chunks}.json")
        rag_manager = RAGManager(data_path=args.locomo_rag_json_path, chunk_size=args.chunk_size, k=args.num_chunks)
        rag_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "langmem":
        from src.langmem import LangMemManager

        output_file_path = os.path.join(args.output_folder, "langmem_results.json")
        langmem_manager = LangMemManager(dataset_path=args.locomo_rag_json_path)
        langmem_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "zep":
        from src.zep.add import ZepAdd
        from src.zep.search import ZepSearch

        if args.method == "add":
            zep_manager = ZepAdd(data_path=args.locomo_json_path)
            zep_manager.process_all_conversations("1")
        elif args.method == "search":
            output_file_path = os.path.join(args.output_folder, "zep_search_results.json")
            zep_manager = ZepSearch()
            zep_manager.process_data_file(args.locomo_json_path, "1", output_file_path)
    elif args.technique_type == "openai":
        from src.openai.predict import OpenAIPredict

        output_file_path = os.path.join(args.output_folder, "openai_results.json")
        openai_manager = OpenAIPredict()
        openai_manager.process_data_file(args.locomo_json_path, output_file_path)
    elif args.technique_type == "dmf":
        from src.dmf_eval import DMFManager

        output_file_path = os.path.join(args.output_folder, "dmf_results.json")
        dmf_manager = DMFManager(
            dataset_path=args.locomo_rag_json_path,
            output_path=output_file_path,
            dmf_repo_path=dmf_repo_path,
            dmf_config_path=dmf_config_path,
            run_root=dmf_run_root,
            runtime_config=runtime_config,
            max_conversations=args.max_conversations,
        )
        dmf_manager.process_all_conversations()
    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()
