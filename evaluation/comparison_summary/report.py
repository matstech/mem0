import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd

def generate_pdf_report(overall_df, cat_df, details_df, output_path, figures_dir, timestamp, comparison_name):
    figures_dir = Path(figures_dir)
    with PdfPages(output_path) as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.9, "Comparison Summary Report", fontsize=24, ha='center', weight='bold')
        ax.text(0.5, 0.85, f"Name: {comparison_name}", fontsize=14, ha='center')
        ax.text(0.5, 0.82, f"Generated at: {timestamp}", fontsize=14, ha='center')
        
        ax.text(0.1, 0.7, "Included Baselines:", fontsize=16, weight='bold')
        y = 0.65
        for baseline in overall_df['baseline']:
            ax.text(0.15, y, f"- {baseline}", fontsize=12)
            y -= 0.03
            
        ax.text(0.1, y-0.05, "Canonical vs. Legacy Tokens Note:", fontsize=14, weight='bold')
        y -= 0.1
        note_text = (
            "This report distinguishes between canonical tokens (when available via token accounting)\n"
            "and legacy tokens (total_pipeline_tokens_amortized). Canonical metrics are prioritized\n"
            "for comparison when present. Mixed comparisons should be interpreted with caution."
        )
        ax.text(0.1, y, note_text, fontsize=10)
        
        pdf.savefig(fig)
        plt.close()
        
        # Overall Comparison Table
        fig, ax = plt.subplots(figsize=(11, 8.5)) # Landscape for tables
        ax.axis('off')
        ax.set_title("Overall Comparison Table", fontsize=16, weight='bold', pad=20)
        
        table_data = overall_df.round(4).fillna("N/A")
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
        
        pdf.savefig(fig)
        plt.close()
        
        # Plots
        plot_files = [
            "quality_comparison.png",
            "efficiency_comparison.png",
            "token_comparison.png",
            "quality_vs_tokens.png",
            "quality_vs_latency.png"
        ]
        
        for plot_file in plot_files:
            plot_path = figures_dir / plot_file
            if plot_path.exists():
                img = plt.imread(plot_path)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(plot_file.replace('.png', '').replace('_', ' ').title(), fontsize=16, weight='bold')
                pdf.savefig(fig)
                plt.close()
        
        # Analysis/Observations Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.1, 0.9, "Analysis and Observations", fontsize=18, weight='bold')
        
        observations = [
            "1. Quality Analysis: Compare LLM Judge and F1 scores across baselines.",
            "2. Efficiency Analysis: Observe search and response times.",
            "3. Token Usage: Analyze how different techniques impact token consumption.",
            "4. Trade-offs: Quality vs. Cost (tokens) and Quality vs. Latency.",
            "",
            "Detailed observations can be added based on specific dataset characteristics."
        ]
        
        y = 0.85
        for obs in observations:
            ax.text(0.1, y, obs, fontsize=11)
            y -= 0.04
            
        ax.text(0.1, y-0.05, "Methodology Note:", fontsize=14, weight='bold')
        y -= 0.1
        methodology = (
            "Results are aggregated by taking the mean of metrics across all questions in the dataset.\n"
            "Categories are preserved when provided in the input files."
        )
        ax.text(0.1, y, methodology, fontsize=10)

        pdf.savefig(fig)
        plt.close()
