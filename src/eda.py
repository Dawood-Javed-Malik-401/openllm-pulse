import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "outputs"
REPORT_PATH = os.path.join(OUTPUT_DIR, "eda_summary_report.md")


def generate_markdown_report(
    df: pd.DataFrame,
    task_summary: pd.DataFrame,
    quant_counts: pd.Series,
    hw_counts: pd.Series,
    viral_df: pd.DataFrame,
) -> None:
    """Writes an automated Markdown EDA summary report to outputs/eda_summary_report.md (Option E)."""
    known_params = int(df["params_billions"].notna().sum())
    viral_count = int(df["is_viral_outlier"].sum())

    lines = [
        "# OpenLLM Pulse — Automated Exploratory Data Analysis Report",
        "",
        "**Author:** Dawood Javed Malik (`2025-AI-401`)  ",
        "**Course:** Programming for AI — Project 1: ETL + EDA Pipeline",
        "",
        "---",
        "",
        "## 1. Executive Dataset Overview",
        f"- **Total Unique Models Analyzed:** {len(df)}",
        f"- **Total Features Engineered:** {df.shape[1]}",
        f"- **Models with Extracted Parameter Counts:** {known_params} / {len(df)}",
        f"- **Statistical Viral Outliers (IQR Method):** {viral_count} models",
        f"- **Mean Composite Pulse Score:** {df['pulse_score'].mean():.2f} / 100",
        "",
        "## 2. Multi-Category Task Distribution (Option A)",
        "| Task Category (`pipeline_tag`) | Model Count | Total Downloads | Median Likes | Mean Pulse Score |",
        "|---|---:|---:|---:|---:|",
    ]

    for task, row in task_summary.iterrows():
        lines.append(
            f"| `{task}` | {int(row['model_count'])} | {int(row['total_downloads']):,} |"
            f" {row['median_likes']:.1f} | {row['mean_pulse_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Quantization & Weight Formats (Option C)",
            "| Format | Model Count | Share (%) |",
            "|---|---:|---:|",
        ]
    )
    for fmt, cnt in quant_counts.items():
        pct = (cnt / len(df)) * 100
        lines.append(f"| {fmt} | {int(cnt)} | {pct:.1f}% |")

    lines.extend(
        [
            "",
            "## 4. Hardware VRAM Deployment Tiers (Option F)",
            "| Hardware Tier (FP16 VRAM) | Model Count | Share (%) |",
            "|---|---:|---:|",
        ]
    )
    for tier, cnt in hw_counts.items():
        pct = (cnt / len(df)) * 100
        lines.append(f"| {tier} | {int(cnt)} | {pct:.1f}% |")

    lines.extend(
        [
            "",
            "## 5. Top 10 Models by Composite Pulse Score (Option G)",
            "| Rank | Model ID | Task | Params (B) | FP16 VRAM (GB) | Downloads | Likes | Pulse Score |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    top_pulse = df.sort_values(by="pulse_score", ascending=False).head(10)
    for idx, (_, row) in enumerate(top_pulse.iterrows(), start=1):
        params_str = (
            f"{row['params_billions']:.2f}B"
            if pd.notna(row["params_billions"])
            else "N/A"
        )
        vram_str = (
            f"{row['vram_fp16_gb']:.1f} GB"
            if pd.notna(row["vram_fp16_gb"])
            else "N/A"
        )
        lines.append(
            f"| {idx} | `{row['model_id']}` | `{row['pipeline_tag']}` | {params_str} |"
            f" {vram_str} | {int(row['downloads']):,} | {int(row['likes']):,} |"
            f" **{row['pulse_score']:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## 6. Top 10 Viral Outliers (Downloads > Q3 + 1.5 * IQR)",
            "| Model ID | Task | Downloads | Daily Velocity | Likes | Pulse Score |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in viral_df.head(10).iterrows():
        lines.append(
            f"| `{row['model_id']}` | `{row['pipeline_tag']}` | {int(row['downloads']):,} |"
            f" {row['daily_download_velocity']:,.1f}/day | {int(row['likes']):,} |"
            f" {row['pulse_score']:.2f} |"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"[EDA] Automated Markdown summary report saved to '{REPORT_PATH}'.")


def run_eda(df: pd.DataFrame) -> None:
    """Conducts structured Exploratory Data Analysis (EDA), prints insights, and saves 7 plots + MD report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Normalize SQLite integer booleans back to Python bool
    df["is_viral_outlier"] = df["is_viral_outlier"].astype(bool)

    print("\n" + "=" * 65)
    print("       EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 65)

    # 1. Dataset Overview & Missing Value Check
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    known_params = df["params_billions"].notna().sum()
    viral_total = int(df["is_viral_outlier"].sum())
    print(f"    Models with Extracted Parameter Size: {known_params} / {len(df)}")
    print(f"    Viral Outlier Models (IQR Method):    {viral_total} models")

    # 2. Descriptive Statistics for Numerical Columns
    print("\n[2] Summary Statistics (Numerical Features):")
    num_cols = [
        "downloads",
        "likes",
        "params_billions",
        "vram_fp16_gb",
        "daily_download_velocity",
        "pulse_score",
    ]
    print(df[num_cols].describe().round(2).to_string())

    # 3. Multi-Task Pipeline Category Breakdown (Option A)
    print("\n[3] Multi-Category Task Breakdown (Option A):")
    task_summary = (
        df.groupby("pipeline_tag")
        .agg(
            model_count=("model_id", "count"),
            total_downloads=("downloads", "sum"),
            median_likes=("likes", "median"),
            mean_pulse_score=("pulse_score", "mean"),
        )
        .sort_values(by="total_downloads", ascending=False)
        .round(2)
    )
    print(task_summary.to_string())

    # 4. Architecture Dominance Breakdown
    print("\n[4] Top 10 Model Architectures & Performance Metrics:")
    arch_summary = (
        df.groupby("architecture")
        .agg(
            model_count=("model_id", "count"),
            total_downloads=("downloads", "sum"),
            avg_likes=("likes", "mean"),
            avg_pulse=("pulse_score", "mean"),
        )
        .sort_values(by="model_count", ascending=False)
        .round(1)
    )
    print(arch_summary.head(10).to_string())

    # 5. Quantization & Hardware Tier Breakdown (Options C & F)
    print("\n[5] Weight Quantization Formats (Option C):")
    quant_counts = df["quantization_format"].value_counts()
    print(quant_counts.to_string())

    print("\n[6] Hardware VRAM Deployment Tiers (Option F):")
    hw_counts = df["hardware_tier"].value_counts()
    print(hw_counts.to_string())

    # 6. Top 10 Pulse Score Leaderboard (Option G)
    print("\n[7] Top 10 Open-Source Models by Composite Pulse Score (Option G):")
    leaderboard_cols = [
        "model_id",
        "pipeline_tag",
        "params_billions",
        "vram_fp16_gb",
        "downloads",
        "likes",
        "pulse_score",
    ]
    print(
        df.sort_values(by="pulse_score", ascending=False)[leaderboard_cols]
        .head(10)
        .to_string(index=False)
    )

    # Plot 1: Top Model Architectures by Count
    plt.figure(figsize=(10, 6))
    arch_counts = df["architecture"].value_counts().head(10)
    sns.barplot(
        x=arch_counts.values,
        y=arch_counts.index,
        hue=arch_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 Open-Source Model Architectures in Ecosystem", fontsize=14)
    plt.xlabel("Number of Models", fontsize=12)
    plt.ylabel("Architecture Family", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_architecture_distribution.png"), dpi=300)
    plt.close()

    # Plot 2: Model Scaling (Size Tiers) Distribution
    plt.figure(figsize=(9, 5))
    tier_order = [
        "Small (<=3B)",
        "Medium (3B-10B)",
        "Large (10B-35B)",
        "XL (>35B)",
        "Unspecified",
    ]
    existing_tiers = [t for t in tier_order if t in df["size_tier"].values]
    sns.countplot(
        data=df,
        x="size_tier",
        order=existing_tiers,
        hue="size_tier",
        palette="magma",
        legend=False,
    )
    plt.title("Distribution of LLM Parameter Size Tiers", fontsize=14)
    plt.xlabel("Parameter Scaling Tier", fontsize=12)
    plt.ylabel("Model Count", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_size_tier_distribution.png"), dpi=300)
    plt.close()

    # Plot 3: Top Open-Source Licenses
    plt.figure(figsize=(10, 6))
    top_licenses = df["license"].value_counts().head(8)
    sns.barplot(
        x=top_licenses.values,
        y=top_licenses.index,
        hue=top_licenses.index,
        palette="crest",
        legend=False,
    )
    plt.title("Top 8 Open-Source Licenses Across Top Models", fontsize=14)
    plt.xlabel("Number of Models", fontsize=12)
    plt.ylabel("License Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_license_breakdown.png"), dpi=300)
    plt.close()

    # Plot 4: Downloads vs. Likes Correlation (Log-Scale Scatter Plot)
    plt.figure(figsize=(10, 6))
    valid_engagement = df[(df["downloads"] > 0) & (df["likes"] > 0)].copy()
    if not valid_engagement.empty:
        sns.scatterplot(
            data=valid_engagement,
            x="downloads",
            y="likes",
            hue="pipeline_tag",
            style="is_viral_outlier",
            alpha=0.85,
            s=80,
            palette="deep",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title(
            "Community Engagement: Downloads vs. Likes by Task (Log Scale)",
            fontsize=14,
        )
        plt.xlabel("Downloads (Last 30 Days - Log Scale)", fontsize=12)
        plt.ylabel("Community Likes (Log Scale)", fontsize=12)
        plt.legend( bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "4_downloads_vs_likes_scatter.png"), dpi=300
        )
    plt.close()

    # Plot 5: Numerical Feature Correlation Heatmap (Option E)
    plt.figure(figsize=(9, 7))
    corr_cols = [
        "params_billions",
        "vram_fp16_gb",
        "downloads",
        "likes",
        "daily_download_velocity",
        "age_days",
        "pulse_score",
    ]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
    )
    plt.title("Pearson Correlation Heatmap of Key Model Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Plot 6: Monthly Model Release Timeline (Option E)
    plt.figure(figsize=(11, 5))
    valid_months = df[df["created_month"] != "Unknown"]
    if not valid_months.empty:
        monthly_counts = valid_months["created_month"].value_counts().sort_index()
        # Display latest 24 active months for visual clarity
        monthly_counts = monthly_counts.tail(24)
        sns.lineplot(
            x=monthly_counts.index,
            y=monthly_counts.values,
            marker="o",
            linewidth=2.5,
            color="#2563eb",
        )
        plt.title("Release Velocity: Models Created per Month (Recent 24 Months)", fontsize=14)
        plt.xlabel("Release Month (YYYY-MM)", fontsize=12)
        plt.ylabel("Number of Top Models Released", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "6_release_timeline.png"), dpi=300)
    plt.close()

    # Plot 7: Top 15 Models by Composite Pulse Score (Option G)
    plt.figure(figsize=(11, 6))
    top15 = df.sort_values(by="pulse_score", ascending=False).head(15)
    sns.barplot(
        data=top15,
        x="pulse_score",
        y="model_id",
        hue="pipeline_tag",
        dodge=False,
        palette="rocket",
    )
    plt.title("Top 15 Open-Source Models by Composite Pulse Score (0-100)", fontsize=14)
    plt.xlabel("Composite Pulse Score", fontsize=12)
    plt.ylabel("Model Repository ID", fontsize=12)
    plt.xlim(0, 105)
    plt.legend(title="Task Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_top_pulse_leaderboard.png"), dpi=300)
    plt.close()

    # Generate Automated Markdown Report (Option E)
    viral_df = df[df["is_viral_outlier"]].sort_values(
        by="downloads", ascending=False
    )
    generate_markdown_report(df, task_summary, quant_counts, hw_counts, viral_df)

    print("\n[EDA] Saved 7 high-resolution charts and 1 Markdown report to 'outputs/'.")
    print("=" * 65 + "\n")