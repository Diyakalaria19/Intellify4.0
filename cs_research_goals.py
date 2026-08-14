"""
Phase A, Step 2 — Generate 'research_goal' column for each category's papers.

This script processes one category at a time, saves progress every N rows,
and can resume from a checkpoint if it crashes or you stop it.

Requires:
    pip install ollama pandas pyarrow tqdm
"""

import json
import os
import time

import pandas as pd
import ollama
from tqdm import tqdm


CHECKPOINT_EVERY = 500
MAX_RETRIES = 3


def get_research_goal(abstract: str) -> str:
    """
    Calls the local Ollama Gemma model once for a single abstract.
    Returns the extracted research goal string.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model="gemma3:4b",

                messages=[
                    {
                        "role": "user",
                        "content": f"""
You are a research assistant.

Given the following paper abstract, extract its core research goal
as a single concise sentence of maximum 25 words.

Focus on WHAT the paper is trying to achieve or solve,
not the method or results.

Respond ONLY with valid JSON in this exact format:

{{"research_goal": "..."}}

Abstract:
{abstract}
"""
                    }
                ],

                format="json",

                options={
                    "temperature": 0.2
                }
            )

            content = response["message"]["content"]

            parsed = json.loads(content)

            return parsed.get("research_goal", "").strip()

        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"  [parse error, attempt {attempt+1}/{MAX_RETRIES}]: {e}"
            )
            time.sleep(1)

        except Exception as e:
            print(
                f"  [Ollama error, attempt {attempt+1}/{MAX_RETRIES}]: {e}"
            )
            time.sleep(2 ** attempt)

    return "EXTRACTION_FAILED"


def process_category(category: str, input_dir: str, checkpoint_dir: str):
    """
    Reads categories/{category}/cs_papers.json,
    adds a research_goal column,
    checkpoints progress along the way,
    and writes the final Parquet file.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{category}_checkpoint.parquet"
    )

    input_path = os.path.join(
        input_dir,
        "cs_papers.json"
    )

    output_path = os.path.join(
        input_dir,
        "cs_research_goals.parquet"
    )


    # --- Resume from checkpoint if one exists ---
    # Otherwise start fresh from the JSON file.

    if os.path.exists(checkpoint_path):

        df = pd.read_parquet(checkpoint_path)

        print(
            f"[{category}] Resuming from checkpoint: "
            f"{df['research_goal'].notna().sum()} / {len(df)} done"
        )

    else:

        df = pd.read_json(input_path)

        df["research_goal"] = pd.NA

        print(
            f"[{category}] Starting fresh: {len(df)} papers"
        )


    # --- Process only rows that haven't successfully been processed ---

    pending_mask = (
        df["research_goal"].isna()
        | (df["research_goal"] == "EXTRACTION_FAILED")
    )

    pending_indices = df[pending_mask].index


    for count, idx in enumerate(
        tqdm(pending_indices, desc=f"{category}"),
        start=1
    ):

        abstract = df.at[idx, "abstract"]

        df.at[idx, "research_goal"] = get_research_goal(abstract)


        # --- Save checkpoint every 500 processed rows ---

        if count % CHECKPOINT_EVERY == 0:

            df.to_parquet(checkpoint_path)

            print(
                f"\n[{category}] Checkpoint saved "
                f"after {count} processed rows."
            )


    # --- Final save ---

    df.to_parquet(checkpoint_path)

    df.to_parquet(output_path)

    print(
        f"[{category}] Done. Saved to {output_path}"
    )


    failed = (
        df["research_goal"] == "EXTRACTION_FAILED"
    ).sum()

    if failed:

        print(
            f"[{category}] WARNING: {failed} rows failed extraction "
            f"and can be retried by running the script again."
        )


if __name__ == "__main__":

    process_category(
        "cs",
        input_dir=r"F:\Diya\CS",
        checkpoint_dir=r"F:\Diya\CS"
    )