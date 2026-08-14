"""
search_engine.py — the actual two-stage cosine similarity + fusion logic.
Both stages now use SPECTER2 embeddings (abstract-level and research-goal-level),
just encoding different text into the same embedding space per stage.

Imported by streamlit_app.py -- not run standalone.
"""

import numpy as np

TOP_K_STAGE1 = 10


def run_search(query_abstract_vec: "np.ndarray", query_goal_vec: "np.ndarray", pool: dict):
    """
    pool: output of router.load_and_merge() -- contains abstract_embeds,
    goal_embeds, paper_ids, metadata, all aligned by row index.

    Returns a DataFrame of the top 10 results, ranked by fused score.
    """
    # Stage 1: abstract-level similarity across the full pool
    abstract_sims = pool["abstract_embeds"] @ query_abstract_vec
    top10_idx = np.argsort(-abstract_sims)[:TOP_K_STAGE1]
    top10_paper_ids = pool["paper_ids"][top10_idx]
    top10_abstract_scores = abstract_sims[top10_idx] * 100

    # Stage 2: research-goal similarity, computed ONLY on the 10 survivors
    # reuse top10_idx directly -- abstract_embeds / goal_embeds / paper_ids
    # share the same row order, so no separate paper_id lookup is needed
    goal_subset = pool["goal_embeds"][top10_idx]
    goal_sims = goal_subset @ query_goal_vec
    top10_goal_scores = goal_sims * 100

    # Fuse and rank
    final_scores = top10_abstract_scores + top10_goal_scores
    rank_order = np.argsort(-final_scores)

    ranked_paper_ids = top10_paper_ids[rank_order]
    ranked_final = final_scores[rank_order]
    ranked_abstract = top10_abstract_scores[rank_order]
    ranked_goal = top10_goal_scores[rank_order]

    meta_indexed = pool["metadata"].set_index("paperid")
    results_df = meta_indexed.loc[ranked_paper_ids].reset_index()
    results_df["final_score"] = ranked_final
    results_df["abstract_score"] = ranked_abstract
    results_df["goal_score"] = ranked_goal

    return results_df