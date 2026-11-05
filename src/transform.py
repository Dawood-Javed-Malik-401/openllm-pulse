from datetime import datetime
import re
from typing import List, Optional
import numpy as np
import pandas as pd


def parse_license(tags: List[str]) -> str:
    """Extracts the license name from the Hugging Face tags list."""
    if not isinstance(tags, list):
        return "Unknown"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split("license:")[1].strip().lower()
    return "Unknown"


def parse_architecture(model_id: str, tags: List[str]) -> str:
    """Identifies the primary LLM family/architecture from tags or model ID."""
    known_families = [
        "llama",
        "mistral",
        "mixtral",
        "qwen",
        "gemma",
        "phi",
        "deepseek",
        "falcon",
        "yi",
        "gpt2",
        "bert",
        "mpnet",
        "minilm",
        "t5",
        "flan",
        "blip",
        "llava",
        "clip",
        "opt",
        "bloom",
        "starcoder",
    ]
    combined_text = (
        model_id + " " + " ".join(tags if isinstance(tags, list) else [])
    ).lower()

    for family in known_families:
        if family in combined_text:
            return family.capitalize()
    return "Other"


def parse_parameters_billions(model_id: str, tags: List[str]) -> Optional[float]:
    """Extracts parameter count in Billions (e.g., '7B' -> 7.0, '500M' -> 0.5)."""
    search_text = f"{model_id} " + " ".join(
        tags if isinstance(tags, list) else []
    )

    moe_match = re.search(
        r"\b(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]\b", search_text
    )
    if moe_match:
        experts = float(moe_match.group(1))
        size = float(moe_match.group(2))
        return round(experts * size, 2)

    b_match = re.search(
        r"(?:^|[-_/:\s])(\d+(?:\.\d+)?)\s*[bB](?:$|[-_/:\s])", search_text
    )
    if b_match:
        val = float(b_match.group(1))
        if 0.05 <= val <= 2000:
            return val

    m_match = re.search(
        r"(?:^|[-_/:\s])(\d+(?:\.\d+)?)\s*[mM](?:$|[-_/:\s])", search_text
    )
    if m_match:
        val = float(m_match.group(1))
        if 10 <= val <= 9999:
            return round(val / 1000.0, 3)

    return np.nan


def assign_size_tier(params: float) -> str:
    """Categorizes model parameter counts into clean scaling tiers."""
    if pd.isna(params) or params <= 0:
        return "Unspecified"
    if params <= 3.0:
        return "Small (<=3B)"
    if params <= 10.0:
        return "Medium (3B-10B)"
    if params <= 35.0:
        return "Large (10B-35B)"
    return "XL (>35B)"


def parse_quantization(model_id: str, tags: List[str]) -> str:
    """Identifies weight quantization/storage format from model ID or tags (Option C)."""
    combined = (
        model_id + " " + " ".join(tags if isinstance(tags, list) else [])
    ).lower()

    if "gguf" in combined or "ggml" in combined:
        return "GGUF"
    if "awq" in combined:
        return "AWQ"
    if "gptq" in combined:
        return "GPTQ"
    if "exl2" in combined:
        return "EXL2"
    if "fp8" in combined or "int8" in combined or "8bit" in combined:
        return "FP8/INT8"
    if "int4" in combined or "4bit" in combined or "bnb" in combined:
        return "INT4"
    if "safetensors" in combined:
        return "Safetensors (FP16/BF16)"
    return "Standard (PyTorch/Bin)"


def assign_hardware_tier(vram_fp16: float) -> str:
    """Assigns hardware deployment tier based on estimated FP16 VRAM in GB (Option F)."""
    if pd.isna(vram_fp16) or vram_fp16 <= 0:
        return "Unknown Hardware Tier"
    if vram_fp16 <= 8.0:
        return "Edge / Laptop (<=8GB VRAM)"
    if vram_fp16 <= 24.0:
        return "Consumer GPU (12-24GB VRAM)"
    if vram_fp16 <= 48.0:
        return "Workstation (48GB VRAM)"
    return "Datacenter (80GB+ VRAM)"


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalizes a numeric Pandas Series to the [0.0, 1.0] scale safely."""
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)


def transform_models_data(validated_records: List[dict]) -> pd.DataFrame:
    """Transforms validated Pydantic records into an enriched Pandas DataFrame."""
    print("[TRANSFORM] Converting validated records to Pandas DataFrame...")
    df = pd.DataFrame(validated_records)

    if df.empty:
        raise ValueError("DataFrame is empty. No records to transform.")

    # 1. Deduplicate based on unique model ID
    df = df.drop_duplicates(subset=["id"]).copy()

    # 2. Rename columns for clean database storage
    df = df.rename(columns={"id": "model_id", "createdAt": "created_at"})

    # 3. Extract Author and Model Name from 'author/model_name'
    split_id = df["model_id"].str.split("/", n=1, expand=True)
    if split_id.shape[1] > 1:
        df["author"] = split_id[0].fillna("Community_User")
        df["model_name"] = split_id[1].fillna(df["model_id"])
    else:
        df["author"] = "Community_User"
        df["model_name"] = df["model_id"]

    # 4. Feature Engineering: License, Architecture, Parameter Counts, and Size Tier
    df["license"] = df["tags"].apply(parse_license)
    df["architecture"] = [
        parse_architecture(mid, t)
        for mid, t in zip(df["model_id"], df["tags"])
    ]
    df["params_billions"] = [
        parse_parameters_billions(mid, t)
        for mid, t in zip(df["model_id"], df["tags"])
    ]
    df["size_tier"] = df["params_billions"].apply(assign_size_tier)

    # 5. Option C: Quantization Format Detection
    df["quantization_format"] = [
        parse_quantization(mid, t)
        for mid, t in zip(df["model_id"], df["tags"])
    ]

    # 6. Compute Engagement Metric: Likes per 1,000 Downloads
    df["likes_per_1k_downloads"] = np.where(
        df["downloads"] > 0,
        np.round((df["likes"] / df["downloads"]) * 1000, 4),
        0.0,
    )

    # 7. Format Dates & Option C: Model Age (Days) and Daily Download Velocity
    parsed_dates = pd.to_datetime(
        df["created_at"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    now_naive = datetime.utcnow()
    calculated_age = (now_naive - parsed_dates).dt.days
    median_age = int(calculated_age.median()) if calculated_age.notna().any() else 365
    df["age_days"] = calculated_age.fillna(median_age).clip(lower=1).astype(int)
    df["daily_download_velocity"] = np.round(df["downloads"] / df["age_days"], 2)

    df["created_month"] = parsed_dates.dt.strftime("%Y-%m").fillna("Unknown")
    df["created_at"] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(
        "Unknown"
    )

    # 8. Option C: Statistical IQR Viral Outlier Detection on Downloads
    q1 = df["downloads"].quantile(0.25)
    q3 = df["downloads"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)
    df["is_viral_outlier"] = df["downloads"] > upper_fence

    # 9. Option F: Hardware VRAM Estimation (FP16 & INT4) and Hardware Tiering
    df["vram_fp16_gb"] = np.round(df["params_billions"] * 2.2, 2)
    df["vram_int4_gb"] = np.round(df["params_billions"] * 0.65, 2)
    df["hardware_tier"] = df["vram_fp16_gb"].apply(assign_hardware_tier)

    # 10. Convert Tag Lists to Comma-Separated Strings for SQLite/CSV
    df["tag_count"] = df["tags"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["tags_str"] = df["tags"].apply(
        lambda x: ", ".join(x[:10]) if isinstance(x, list) else ""
    )

    # 11. Option G: Composite Pulse Score (0-100)
    # 40% Log Downloads + 30% Log Likes + 20% Log Velocity + 10% Tag Completeness
    norm_dl = min_max_normalize(np.log1p(df["downloads"]))
    norm_likes = min_max_normalize(np.log1p(df["likes"]))
    norm_vel = min_max_normalize(np.log1p(df["daily_download_velocity"]))
    norm_tags = min_max_normalize(df["tag_count"].astype(float))

    df["pulse_score"] = np.round(
        (0.40 * norm_dl + 0.30 * norm_likes + 0.20 * norm_vel + 0.10 * norm_tags)
        * 100,
        2,
    )

    # 12. Select and order final structured columns
    final_columns = [
        "model_id",
        "author",
        "model_name",
        "pipeline_tag",
        "architecture",
        "params_billions",
        "size_tier",
        "quantization_format",
        "vram_fp16_gb",
        "vram_int4_gb",
        "hardware_tier",
        "license",
        "library_name",
        "downloads",
        "likes",
        "likes_per_1k_downloads",
        "age_days",
        "daily_download_velocity",
        "is_viral_outlier",
        "pulse_score",
        "tag_count",
        "created_at",
        "created_month",
        "tags_str",
    ]
    df = (
        df[final_columns]
        .sort_values(by="pulse_score", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"[TRANSFORM] Transformation complete. Final shape: {df.shape[0]} rows,"
        f" {df.shape[1]} columns."
    )
    return df