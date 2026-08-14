"""
router.py — the "head" file. Its only job: given domain(s) detected by
Ollama/Gemma, load and merge the corresponding category folder(s)' data.

Imported by streamlit_app.py -- not run standalone.
"""

import os

import numpy as np
import pandas as pd

CATEGORY_DIR = "categories"
VALID_CATEGORIES = ["cs", "math", "physics", "econ", "q_bio"]


def load_category_data(category: str) -> dict:
    """Loads one category's metadata + both embedding arrays from disk."""
    base = os.path.join(CATEGORY_DIR, category)
    return {
        "metadata": pd.read_parquet(os.path.join(base, "metadata.parquet")),
        "abstract_embeds": np.load(os.path.join(base, "abstract_embeddings.npy")),
        "goal_embeds": np.load(os.path.join(base, "goal_embeddings.npy")),
        "paper_ids": np.load(os.path.join(base, "paper_ids.npy"), allow_pickle=True),
    }


def load_and_merge(domains: list[str]):
    """
    Opens the category folder(s) matching Ollama's detected domain(s) and
    concatenates them into one pool, so a cross-domain query (e.g. cs + q_bio)
    is ranked fairly across both, not merged after separate top-10s.
    """
    valid_domains = [d for d in domains if d in VALID_CATEGORIES]
    if not valid_domains:
        return None

    all_data = [load_category_data(d) for d in valid_domains]

    return {
        "abstract_embeds": np.vstack([d["abstract_embeds"] for d in all_data]),
        "goal_embeds": np.vstack([d["goal_embeds"] for d in all_data]),
        "paper_ids": np.concatenate([d["paper_ids"] for d in all_data]),
        "metadata": pd.concat([d["metadata"] for d in all_data], ignore_index=True),
    }