import json
import os
from typing import List, Optional
import requests

API_URL = "https://huggingface.co/api/models"
RAW_DATA_PATH = os.path.join("data", "raw_models.json")

DEFAULT_TASK_CATEGORIES = [
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "sentence-similarity",
]


def fetch_models_from_api(
    limit_per_category: int = 100,
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> list:
    """Extracts open-source LLM metadata across multiple task categories from the Hugging Face API."""
    if categories is None:
        categories = DEFAULT_TASK_CATEGORIES

    if limit is not None:
        limit_per_category = max(1, limit // len(categories))

    all_raw_data = []
    total_expected = limit_per_category * len(categories)
    print(
        f"[EXTRACT] Requesting top {limit_per_category} models across"
        f" {len(categories)} categories (~{total_expected} total)..."
    )

    for category in categories:
        print(f"  -> Fetching category: '{category}' (limit={limit_per_category})...")
        params = {
            "filter": category,
            "sort": "downloads",
            "direction": "-1",
            "limit": limit_per_category,
            "full": "true",
        }

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        category_data = response.json()

        for item in category_data:
            if isinstance(item, dict) and not item.get("pipeline_tag"):
                item["pipeline_tag"] = category

        all_raw_data.extend(category_data)
        print(f"     Retrieved {len(category_data)} models for '{category}'.")

    os.makedirs("data", exist_ok=True)
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as file:
        json.dump(all_raw_data, file, indent=2)

    print(
        f"[EXTRACT] Successfully extracted {len(all_raw_data)} total records"
        f" across {len(categories)} categories and saved to '{RAW_DATA_PATH}'."
    )
    return all_raw_data


if __name__ == "__main__":
    fetch_models_from_api(limit_per_category=100)