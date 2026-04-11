import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def generate_all_plots(overall_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _plot_quality_comparison(overall_df, output_dir / "quality_comparison.png")
    _plot_efficiency_comparison(overall_df, output_dir / "efficiency_comparison.png")
    _plot_token_comparison(overall_df, output_dir / "token_comparison.png")
    _plot_quality_vs_tokens(overall_df, output_dir / "quality_vs_tokens.png")
    _plot_quality_vs_latency(overall_df, output_dir / "quality_vs_latency.png")

def _plot_quality_comparison(df, path):
    if df.empty: return
    metrics = ['BLEU', 'F1', 'LLM Judge']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isnull().all()]
    if not available_metrics: return
    
    ax = df.plot(x='baseline', y=available_metrics, kind='bar', figsize=(10, 6))
    ax.set_title("Quality Comparison Across Baselines")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_efficiency_comparison(df, path):
    if df.empty: return
    metrics = ['search_time', 'response_time']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isnull().all()]
    if not available_metrics: return
    
    ax = df.plot(x='baseline', y=available_metrics, kind='bar', figsize=(10, 6))
    ax.set_title("Efficiency Comparison (Latency)")
    ax.set_ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_token_comparison(df, path):
    if df.empty: return
    metrics = ['context_tokens', 'prompt_tokens_total']
    available_metrics = [m for m in metrics if m in df.columns and not df[m].isnull().all()]
    if not available_metrics: return
    
    ax = df.plot(x='baseline', y=available_metrics, kind='bar', figsize=(10, 6))
    ax.set_title("Token Usage Comparison")
    ax.set_ylabel("Tokens")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_quality_vs_tokens(df, path):
    if df.empty or 'LLM Judge' not in df or 'prompt_tokens_total' not in df: return
    
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.scatter(row['prompt_tokens_total'], row['LLM Judge'], label=row['baseline'])
        plt.annotate(row['baseline'], (row['prompt_tokens_total'], row['LLM Judge']))
    
    plt.title("Quality (LLM Judge) vs. Token Usage")
    plt.xlabel("Prompt Tokens Total")
    plt.ylabel("LLM Judge Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_quality_vs_latency(df, path):
    if df.empty or 'LLM Judge' not in df or 'response_time' not in df: return
    
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.scatter(row['response_time'], row['LLM Judge'], label=row['baseline'])
        plt.annotate(row['baseline'], (row['response_time'], row['LLM Judge']))
    
    plt.title("Quality (LLM Judge) vs. Latency")
    plt.xlabel("Response Time (s)")
    plt.ylabel("LLM Judge Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
