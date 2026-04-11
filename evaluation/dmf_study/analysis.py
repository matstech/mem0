from pathlib import Path

import pandas as pd

from dmf_study.utils import read_json


RESULT_COLUMNS = [
    "conversation_id",
    "question",
    "category",
    "context_tokens",
    "prompt_tokens_total",
    "response_time",
    "search_time",
    "recalled_count",
    "raw_candidate_count",
    "ranked_candidate_count",
    "final_candidate_count",
    "suppressed_candidate_count",
    "total_tokens",
]

METRIC_COLUMNS = [
    "conversation_id",
    "question",
    "category",
    "bleu_score",
    "f1_score",
    "llm_score",
]


def _flatten_results(path):
    rows = []
    data = read_json(path)
    for conversation_id, items in data.items():
        for item in items:
            row = dict(item)
            row["conversation_id"] = conversation_id
            diagnostics = row.get("dmf_recall_diagnostics") or {}
            row["raw_candidate_count"] = len(diagnostics.get("raw_candidates", []))
            row["ranked_candidate_count"] = len(diagnostics.get("ranked_candidates", []))
            row["final_candidate_count"] = len(diagnostics.get("final_candidates", []))
            row["suppressed_candidate_count"] = len(diagnostics.get("suppressed", []))
            rows.append(row)
    df = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[RESULT_COLUMNS]


def _flatten_eval_metrics(path):
    rows = []
    data = read_json(path)
    for conversation_id, items in data.items():
        for item in items:
            row = dict(item)
            row["conversation_id"] = conversation_id
            rows.append(row)
    df = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[METRIC_COLUMNS]


def _safe_mean(df, column):
    if column not in df.columns or df.empty:
        return None
    return round(float(df[column].mean()), 4)


def build_variant_tables(results_path, metrics_path, variant_name, params, artifact_dir):
    results_df = _flatten_results(results_path)
    metrics_df = _flatten_eval_metrics(metrics_path) if metrics_path and Path(metrics_path).exists() else pd.DataFrame()
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=METRIC_COLUMNS)

    if results_df.empty:
        conversation_df = pd.DataFrame(
            columns=[
                "conversation_id",
                "context_tokens",
                "prompt_tokens_total",
                "response_time",
                "search_time",
                "recalled_count",
                "raw_candidate_count",
                "ranked_candidate_count",
                "final_candidate_count",
                "suppressed_candidate_count",
                "question_count",
            ]
        )
        diagnostics_df = pd.DataFrame(
            columns=[
                "conversation_id",
                "question",
                "category",
                "raw_candidate_count",
                "ranked_candidate_count",
                "final_candidate_count",
                "suppressed_candidate_count",
            ]
        )
    else:
        conversation_df = results_df.groupby("conversation_id").agg(
            context_tokens=("context_tokens", "mean"),
            prompt_tokens_total=("prompt_tokens_total", "mean"),
            response_time=("response_time", "mean"),
            search_time=("search_time", "mean"),
            recalled_count=("recalled_count", "mean"),
            raw_candidate_count=("raw_candidate_count", "mean"),
            ranked_candidate_count=("ranked_candidate_count", "mean"),
            final_candidate_count=("final_candidate_count", "mean"),
            suppressed_candidate_count=("suppressed_candidate_count", "mean"),
            question_count=("question", "count"),
        ).reset_index()

        diagnostics_df = results_df[
            [
                "conversation_id",
                "question",
                "category",
                "raw_candidate_count",
                "ranked_candidate_count",
                "final_candidate_count",
                "suppressed_candidate_count",
            ]
        ].copy()

    if not metrics_df.empty:
        per_category_df = metrics_df.groupby("category").agg(
            bleu_score=("bleu_score", "mean"),
            f1_score=("f1_score", "mean"),
            llm_score=("llm_score", "mean"),
            count=("question", "count"),
        ).reset_index()
    else:
        per_category_df = pd.DataFrame(columns=["category", "bleu_score", "f1_score", "llm_score", "count"])

    summary_row = {
        "variant": variant_name,
        "budget": params["token_budget"],
        "window_size": params["window_size"],
        "pruning_frequency_x": params["pruning_frequency_x"],
        "recall_limit": params["recall_limit"],
        "distance_threshold": params["distance_threshold"],
        "artifact_dir": str(artifact_dir),
        "context_tokens": _safe_mean(results_df, "context_tokens"),
        "prompt_tokens_total": _safe_mean(results_df, "prompt_tokens_total"),
        "response_time": _safe_mean(results_df, "response_time"),
        "search_time": _safe_mean(results_df, "search_time"),
        "recalled_count": _safe_mean(results_df, "recalled_count"),
        "raw_candidate_count": _safe_mean(results_df, "raw_candidate_count"),
        "final_candidate_count": _safe_mean(results_df, "final_candidate_count"),
        "bleu_score": _safe_mean(metrics_df, "bleu_score"),
        "f1_score": _safe_mean(metrics_df, "f1_score"),
        "llm_score": _safe_mean(metrics_df, "llm_score"),
        "total_tokens": _safe_mean(results_df, "total_tokens"),
    }
    summary_df = pd.DataFrame([summary_row])
    return summary_df, conversation_df, per_category_df, diagnostics_df


def save_variant_tables(summary_df, conversation_df, per_category_df, diagnostics_df, tables_dir):
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables_dir / "summary_table.csv", index=False)
    conversation_df.to_csv(tables_dir / "per_conversation_table.csv", index=False)
    per_category_df.to_csv(tables_dir / "per_category_table.csv", index=False)
    diagnostics_df.to_csv(tables_dir / "diagnostics_table.csv", index=False)


def build_study_summary(summary_frames):
    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)
