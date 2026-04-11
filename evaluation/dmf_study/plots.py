import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _quality_column(df):
    for column in ("llm_score", "f1_score", "bleu_score"):
        if column in df.columns and df[column].notna().any():
            return column
    return None


def plot_variant_conversations(conversation_df, figures_dir, variant_name):
    if conversation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(conversation_df["conversation_id"].astype(str), conversation_df["context_tokens"])
    ax.set_title(f"{variant_name}: Context Tokens Per Conversation")
    ax.set_xlabel("Conversation")
    ax.set_ylabel("Context tokens")
    _save(fig, Path(figures_dir) / "context_tokens_per_conversation.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(conversation_df["conversation_id"].astype(str), conversation_df["response_time"])
    ax.set_title(f"{variant_name}: Response Time Per Conversation")
    ax.set_xlabel("Conversation")
    ax.set_ylabel("Seconds")
    _save(fig, Path(figures_dir) / "response_time_per_conversation.png")


def plot_study_summary(summary_df, figures_dir):
    if summary_df.empty:
        return

    quality_col = _quality_column(summary_df)

    for y_column, filename, title, ylabel in (
        ("context_tokens", "budget_vs_context_tokens.png", "Budget vs Context Tokens", "Context tokens"),
        ("prompt_tokens_total", "budget_vs_prompt_tokens.png", "Budget vs Prompt Tokens", "Prompt tokens"),
        ("response_time", "budget_vs_response_time.png", "Budget vs Response Time", "Seconds"),
    ):
        if y_column not in summary_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(summary_df["budget"], summary_df[y_column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Budget")
        ax.set_ylabel(ylabel)
        _save(fig, Path(figures_dir) / filename)

    if quality_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(summary_df["budget"], summary_df[quality_col], marker="o")
        ax.set_title(f"Budget vs {quality_col}")
        ax.set_xlabel("Budget")
        ax.set_ylabel(quality_col)
        _save(fig, Path(figures_dir) / "budget_vs_quality.png")

        if "recalled_count" in summary_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(summary_df["recalled_count"], summary_df[quality_col])
            for _, row in summary_df.iterrows():
                ax.annotate(str(row["budget"]), (row["recalled_count"], row[quality_col]))
            ax.set_title(f"Recalled Count vs {quality_col}")
            ax.set_xlabel("Recalled count")
            ax.set_ylabel(quality_col)
            _save(fig, Path(figures_dir) / "recalled_count_vs_quality.png")

        token_col = "total_tokens" if "total_tokens" in summary_df.columns and summary_df["total_tokens"].notna().any() else "prompt_tokens_total"
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(summary_df[token_col], summary_df[quality_col], c=summary_df["response_time"], cmap="viridis")
        for _, row in summary_df.iterrows():
            ax.annotate(str(row["budget"]), (row[token_col], row[quality_col]))
        ax.set_title("Pareto Quality vs Tokens")
        ax.set_xlabel(token_col)
        ax.set_ylabel(quality_col)
        fig.colorbar(scatter, ax=ax, label="response_time")
        _save(fig, Path(figures_dir) / "pareto_quality_vs_tokens.png")
