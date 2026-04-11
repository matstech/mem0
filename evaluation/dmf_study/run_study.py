import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmf_study.analysis import build_study_summary, build_variant_tables, save_variant_tables
from dmf_study.plots import plot_study_summary, plot_variant_conversations
from dmf_study.utils import (
    ensure_dir,
    make_variant_name,
    materialize_config,
    python_executable,
    read_json,
    run_command,
    timestamp_slug,
    write_json,
    zip_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an automatic DMF study")
    parser.add_argument("--dataset-path", type=str, default="dataset/locomo10_rag.json")
    parser.add_argument("--runtime-config-path", type=str, default="config/benchmark_runtime.toml")
    parser.add_argument("--base-config-path", type=str, default="config/dmf_benchmark_settings.toml")
    parser.add_argument("--output-root", type=str, default="results/dmf_study")
    parser.add_argument("--runtime-root", type=str, default="run/dmf_study")
    parser.add_argument("--phase", choices=["budget", "active_context", "ltm"], default="budget")
    parser.add_argument("--active-step", choices=["window", "pruning"], default="window")
    parser.add_argument("--base-budget", type=int, default=2048)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--skip-evals", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def build_variants(args):
    default = {
        "window_size": 10,
        "pruning_frequency_x": 5,
        "recall_limit": 5,
        "distance_threshold": 0.7,
    }
    variants = []
    if args.phase == "budget":
        for budget in [1024, 1536, 2048, 3072, 4096]:
            params = dict(default)
            params["token_budget"] = budget
            variants.append(params)
    elif args.phase == "active_context":
        if args.active_step == "window":
            for window_size in [5, 10, 15]:
                params = dict(default)
                params["token_budget"] = args.base_budget
                params["window_size"] = window_size
                variants.append(params)
        else:
            for pruning_frequency in [3, 5, 10]:
                params = dict(default)
                params["token_budget"] = args.base_budget
                params["pruning_frequency_x"] = pruning_frequency
                variants.append(params)
    else:
        for recall_limit in [3, 5, 8]:
            for distance_threshold in [0.6, 0.7, 0.8]:
                params = dict(default)
                params["token_budget"] = args.base_budget
                params["recall_limit"] = recall_limit
                params["distance_threshold"] = distance_threshold
                variants.append(params)
    return variants


def variant_paths(output_root, runtime_root, variant_name, timestamp):
    artifact_dir = Path(output_root) / f"{variant_name}_{timestamp}"
    tables_dir = artifact_dir / "tables"
    figures_dir = artifact_dir / "figures"
    runtime_dir = Path(runtime_root) / f"{variant_name}_{timestamp}"
    return {
        "artifact_dir": artifact_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "runtime_dir": runtime_dir,
        "results_path": artifact_dir / "dmf_results.json",
        "metrics_path": artifact_dir / "dmf_eval_metrics.json",
        "summary_path": artifact_dir / "summary.md",
        "config_path": artifact_dir / "config_snapshot.toml",
        "manifest_path": artifact_dir / "run_manifest.json",
        "zip_path": Path(output_root) / f"{variant_name}_{timestamp}.zip",
    }


def run_variant(args, params, timestamp):
    variant_name = make_variant_name(params)
    paths = variant_paths(args.output_root, args.runtime_root, variant_name, timestamp)
    ensure_dir(paths["artifact_dir"])
    ensure_dir(paths["tables_dir"])
    ensure_dir(paths["figures_dir"])
    ensure_dir(paths["runtime_dir"])

    materialize_config(args.base_config_path, paths["config_path"], params)

    manifest = {
        "variant_name": variant_name,
        "timestamp": timestamp,
        "dataset_path": args.dataset_path,
        "config_snapshot_path": str(paths["config_path"]),
        "results_path": str(paths["results_path"]),
        "metrics_path": str(paths["metrics_path"]),
        "tables_dir": str(paths["tables_dir"]),
        "figures_dir": str(paths["figures_dir"]),
        "summary_path": str(paths["summary_path"]),
        "zip_path": str(paths["zip_path"]),
        "params": params,
        "status": "planned" if args.dry_run else "running",
    }
    write_json(paths["manifest_path"], manifest)

    if args.dry_run:
        manifest["status"] = "dry_run"
        write_json(paths["manifest_path"], manifest)
        return {"variant_name": variant_name, "paths": paths, "summary_df": None}

    cmd = [
        python_executable(),
        "run_experiments.py",
        "--technique_type",
        "dmf",
        "--output_folder",
        str(paths["artifact_dir"]),
        "--locomo_rag_json_path",
        args.dataset_path,
        "--runtime_config_path",
        args.runtime_config_path,
        "--dmf_config_path",
        str(paths["config_path"]),
        "--dmf_run_root",
        str(paths["runtime_dir"]),
    ]
    if args.max_conversations is not None:
        cmd.extend(["--max_conversations", str(args.max_conversations)])
    run_command(cmd, Path.cwd())

    if not args.skip_evals:
        eval_cmd = [
            python_executable(),
            "evals.py",
            "--input_file",
            str(paths["results_path"]),
            "--output_file",
            str(paths["metrics_path"]),
        ]
        run_command(eval_cmd, Path.cwd())

    summary_df, conversation_df, per_category_df, diagnostics_df = build_variant_tables(
        paths["results_path"],
        paths["metrics_path"] if paths["metrics_path"].exists() else None,
        variant_name,
        params,
        paths["artifact_dir"],
    )
    save_variant_tables(summary_df, conversation_df, per_category_df, diagnostics_df, paths["tables_dir"])
    plot_variant_conversations(conversation_df, paths["figures_dir"], variant_name)

    summary_lines = [
        f"# {variant_name}",
        "",
        f"- dataset: `{args.dataset_path}`",
        f"- token_budget: `{params['token_budget']}`",
        f"- window_size: `{params['window_size']}`",
        f"- pruning_frequency_x: `{params['pruning_frequency_x']}`",
        f"- recall_limit: `{params['recall_limit']}`",
        f"- distance_threshold: `{params['distance_threshold']}`",
    ]
    if not summary_df.empty:
        row = summary_df.iloc[0]
        for column in ("bleu_score", "f1_score", "llm_score", "context_tokens", "prompt_tokens_total", "response_time", "search_time", "recalled_count", "total_tokens"):
            if column in row and not pd_isna(row[column]):
                summary_lines.append(f"- {column}: `{row[column]}`")
    paths["summary_path"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest["status"] = "completed"
    write_json(paths["manifest_path"], manifest)
    zip_directory(paths["artifact_dir"], paths["zip_path"])
    return {"variant_name": variant_name, "paths": paths, "summary_df": summary_df}


def pd_isna(value):
    return value is None or (hasattr(value, "__class__") and value.__class__.__name__ == "NAType") or value != value


def write_aggregate_summary(output_root, summary_frames, timestamp):
    summary_df = build_study_summary(summary_frames)
    summary_dir = ensure_dir(Path(output_root) / f"summary_{timestamp}")
    tables_dir = ensure_dir(summary_dir / "tables")
    figures_dir = ensure_dir(summary_dir / "figures")
    if not summary_df.empty:
        summary_df.to_csv(summary_dir / "summary_table.csv", index=False)
        summary_df.to_csv(tables_dir / "per_variant_table.csv", index=False)
        plot_study_summary(summary_df, figures_dir)

    summary_lines = [
        "# DMF Study Summary",
        "",
        f"- variants: `{len(summary_df)}`",
    ]
    if not summary_df.empty:
        best_quality = summary_df.sort_values(by="llm_score", ascending=False, na_position="last").head(1)
        if not best_quality.empty:
            summary_lines.append(f"- top_variant_by_llm: `{best_quality.iloc[0]['variant']}`")
    (summary_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    zip_directory(summary_dir, Path(output_root) / f"summary_{timestamp}.zip")


def main():
    args = parse_args()
    timestamp = timestamp_slug()
    variants = build_variants(args)
    summary_frames = []
    for params in variants:
        variant_result = run_variant(args, params, timestamp)
        if variant_result["summary_df"] is not None:
            summary_frames.append(variant_result["summary_df"])
    write_aggregate_summary(args.output_root, summary_frames, timestamp)


if __name__ == "__main__":
    main()
