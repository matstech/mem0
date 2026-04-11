import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional at runtime
    plt = None


PHASES = ("ingest", "retrieval", "answer", "maintenance")


def make_phase_tokens(prompt_tokens, completion_tokens, total_tokens, observation_mode):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "observation_mode": observation_mode,
    }


def zero_phase_tokens():
    return make_phase_tokens(0, 0, 0, "known_zero")


def unobserved_phase_tokens():
    return make_phase_tokens(None, None, None, "unobserved")


def make_token_breakdown(ingest, retrieval, answer, maintenance):
    return {
        "ingest": ingest,
        "retrieval": retrieval,
        "answer": answer,
        "maintenance": maintenance,
    }


def sum_phase_totals(token_breakdown):
    totals = []
    for phase_name in PHASES:
        phase_total = token_breakdown[phase_name]["total_tokens"]
        if phase_total is None:
            return None
        totals.append(int(phase_total))
    return int(sum(totals))


def aggregate_run_token_breakdown(conversation_summaries):
    run_breakdown = {}
    for phase_name in PHASES:
        prompt_values = []
        completion_values = []
        total_values = []
        modes = []
        for summary in conversation_summaries:
            phase = summary["token_breakdown"][phase_name]
            prompt_values.append(phase["prompt_tokens"])
            completion_values.append(phase["completion_tokens"])
            total_values.append(phase["total_tokens"])
            modes.append(phase["observation_mode"])
        run_breakdown[phase_name] = make_phase_tokens(
            _sum_nullable_values(prompt_values),
            _sum_nullable_values(completion_values),
            _sum_nullable_values(total_values),
            _aggregate_modes(modes),
        )
    return run_breakdown


def build_run_summary(technique, dataset_path, result_output_path, conversation_summaries):
    run_breakdown = aggregate_run_token_breakdown(conversation_summaries)
    return {
        "technique": technique,
        "dataset_path": str(dataset_path),
        "result_output_path": str(result_output_path),
        "generated_at": utc_timestamp(),
        "run_total_tokens": sum_phase_totals(run_breakdown),
        "run_token_breakdown": run_breakdown,
        "conversation_summaries": conversation_summaries,
    }


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prepare_artifact_dir(base_dir, artifact_name, timestamp=None):
    timestamp = timestamp or utc_timestamp()
    artifact_dir = Path(base_dir) / f"{artifact_name}_{timestamp}"
    tables_dir = artifact_dir / "tables"
    figures_dir = artifact_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "timestamp": timestamp,
        "artifact_dir": artifact_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "zip_path": artifact_dir.with_suffix(".zip"),
    }


def copy_snapshot(source_path, destination_dir, filename=None):
    if not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    destination = Path(destination_dir) / (filename or source.name)
    shutil.copy2(source, destination)
    return destination.as_posix()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path, technique, run_summary, snapshots, notes=None):
    artifact_dir = Path(path).parent
    token_json_files = sorted(artifact_dir.glob("*token_accounting_*.json"))
    token_json_name = token_json_files[0].name if token_json_files else "not_found"
    lines = [
        f"# {technique} Token Accounting Summary",
        "",
        f"- generated_at: `{run_summary['generated_at']}`",
        f"- dataset_path: `{run_summary['dataset_path']}`",
        f"- result_output_path: `{run_summary['result_output_path']}`",
        f"- run_total_tokens: `{run_summary['run_total_tokens']}`",
        f"- conversations: `{len(run_summary['conversation_summaries'])}`",
        "",
        "## Config Snapshots",
        "",
    ]
    if snapshots:
        for snapshot in snapshots:
            lines.append(f"- `{Path(snapshot).name}`")
    else:
        lines.append("- none")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    if plt is None:
        lines.extend(["", "## Plotting", "", "- matplotlib not available; plots were skipped."])
    lines.extend(
        [
            "",
            "## Artifact Contents",
            "",
            f"- token_accounting_json: `{token_json_name}`",
            "- tables/: generated csv outputs",
            "- figures/: generated plots",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_conversation_rows(conversation_summaries):
    rows = []
    for summary in conversation_summaries:
        row = {
            "conversation_id": summary["conversation_id"],
            "question_count": summary["question_count"],
            "total_tokens": summary["total_tokens"],
            "legacy_total_pipeline_tokens": summary["legacy_total_pipeline_tokens"],
        }
        for phase_name in PHASES:
            phase = summary["token_breakdown"][phase_name]
            row[f"{phase_name}_prompt_tokens"] = phase["prompt_tokens"]
            row[f"{phase_name}_completion_tokens"] = phase["completion_tokens"]
            row[f"{phase_name}_total_tokens"] = phase["total_tokens"]
            row[f"{phase_name}_observation_mode"] = phase["observation_mode"]
        rows.append(row)
    return rows


def build_summary_rows(run_summary):
    row = {
        "technique": run_summary["technique"],
        "dataset_path": run_summary["dataset_path"],
        "generated_at": run_summary["generated_at"],
        "run_total_tokens": run_summary["run_total_tokens"],
        "conversation_count": len(run_summary["conversation_summaries"]),
    }
    for phase_name in PHASES:
        phase = run_summary["run_token_breakdown"][phase_name]
        row[f"{phase_name}_prompt_tokens"] = phase["prompt_tokens"]
        row[f"{phase_name}_completion_tokens"] = phase["completion_tokens"]
        row[f"{phase_name}_total_tokens"] = phase["total_tokens"]
        row[f"{phase_name}_observation_mode"] = phase["observation_mode"]
    return [row]


def generate_reporting(artifact_info, technique, run_summary):
    validate_run_summary(run_summary)
    conversation_rows = build_conversation_rows(run_summary["conversation_summaries"])
    summary_rows = build_summary_rows(run_summary)
    write_csv(
        artifact_info["tables_dir"] / "conversation_table.csv",
        fieldnames=list(conversation_rows[0].keys()) if conversation_rows else ["conversation_id"],
        rows=conversation_rows,
    )
    write_csv(
        artifact_info["tables_dir"] / "summary_table.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )
    if plt is None:
        return
    _plot_total_tokens_by_conversation(artifact_info["figures_dir"] / "total_tokens_by_conversation.png", conversation_rows)
    _plot_phase_breakdown_by_conversation(
        artifact_info["figures_dir"] / "phase_breakdown_by_conversation.png",
        conversation_rows,
    )
    _plot_legacy_comparison(
        artifact_info["figures_dir"] / "legacy_vs_total_tokens.png",
        conversation_rows,
        technique,
    )
    _plot_run_breakdown(artifact_info["figures_dir"] / "run_breakdown.png", run_summary)


def bundle_artifact_dir(artifact_dir, zip_path):
    artifact_dir = Path(artifact_dir)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in artifact_dir.rglob("*"):
            archive.write(file_path, file_path.relative_to(artifact_dir.parent))
    return zip_path.as_posix()


def validate_run_summary(run_summary):
    for summary in run_summary["conversation_summaries"]:
        _validate_token_breakdown(summary["token_breakdown"])
        expected_total = sum_phase_totals(summary["token_breakdown"])
        if summary["total_tokens"] != expected_total:
            raise ValueError(
                f"Conversation total_tokens mismatch for {summary['conversation_id']}: "
                f"{summary['total_tokens']} != {expected_total}"
            )
    _validate_token_breakdown(run_summary["run_token_breakdown"])
    expected_run_total = sum_phase_totals(run_summary["run_token_breakdown"])
    if run_summary["run_total_tokens"] != expected_run_total:
        raise ValueError(
            f"Run total_tokens mismatch: {run_summary['run_total_tokens']} != {expected_run_total}"
        )


def _sum_nullable_values(values):
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values))


def _aggregate_modes(modes):
    unique_modes = sorted(set(modes))
    if not unique_modes:
        return "unobserved"
    if len(unique_modes) == 1:
        return unique_modes[0]
    return "mixed"


def _validate_token_breakdown(token_breakdown):
    for phase_name in PHASES:
        phase = token_breakdown[phase_name]
        if phase["observation_mode"] == "unobserved":
            if any(phase[key] is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError(f"Phase {phase_name} is unobserved but has numeric values")
        if phase["observation_mode"] == "known_zero":
            if any(int(phase[key] or 0) != 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError(f"Phase {phase_name} is known_zero but has non-zero values")


def _plot_total_tokens_by_conversation(path, conversation_rows):
    if not conversation_rows:
        return
    labels = [row["conversation_id"] for row in conversation_rows]
    values = [row["total_tokens"] or 0 for row in conversation_rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_title("Total Tokens By Conversation")
    ax.set_xlabel("Conversation")
    ax.set_ylabel("Total Tokens")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_phase_breakdown_by_conversation(path, conversation_rows):
    if not conversation_rows:
        return
    labels = [row["conversation_id"] for row in conversation_rows]
    ingest = [row["ingest_total_tokens"] or 0 for row in conversation_rows]
    retrieval = [row["retrieval_total_tokens"] or 0 for row in conversation_rows]
    answer = [row["answer_total_tokens"] or 0 for row in conversation_rows]
    maintenance = [row["maintenance_total_tokens"] or 0 for row in conversation_rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, ingest, label="ingest")
    ax.bar(labels, retrieval, bottom=ingest, label="retrieval")
    retrieval_bottom = [ingest[i] + retrieval[i] for i in range(len(labels))]
    ax.bar(labels, answer, bottom=retrieval_bottom, label="answer")
    answer_bottom = [retrieval_bottom[i] + answer[i] for i in range(len(labels))]
    ax.bar(labels, maintenance, bottom=answer_bottom, label="maintenance")
    ax.set_title("Token Breakdown By Conversation")
    ax.set_xlabel("Conversation")
    ax.set_ylabel("Tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_legacy_comparison(path, conversation_rows, technique):
    if not conversation_rows:
        return
    labels = [row["conversation_id"] for row in conversation_rows]
    total_tokens = [row["total_tokens"] or 0 for row in conversation_rows]
    legacy_tokens = [row["legacy_total_pipeline_tokens"] or 0 for row in conversation_rows]
    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([x - 0.2 for x in x_positions], total_tokens, width=0.4, label="total_tokens")
    ax.bar([x + 0.2 for x in x_positions], legacy_tokens, width=0.4, label="legacy_total_pipeline_tokens")
    ax.set_title(f"Legacy Comparison For {technique}")
    ax.set_xlabel("Conversation")
    ax.set_ylabel("Tokens")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_run_breakdown(path, run_summary):
    labels = []
    totals = []
    for phase_name in PHASES:
        phase_total = run_summary["run_token_breakdown"][phase_name]["total_tokens"]
        if phase_total is None:
            continue
        labels.append(phase_name)
        totals.append(phase_total)
    if not totals:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, totals)
    ax.set_title(f"{run_summary['technique']} Run Token Breakdown")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Tokens")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
