"""
Phase A, Step 2 — Generate 'research_goal' column using batch processing.

This version processes multiple abstracts in a single Ollama call,
which is significantly faster than making one call per paper.

Requires:
    pip install ollama pandas pyarrow tqdm

Model:
    gemma3:4b
"""

import json
import os
import time

import pandas as pd
import ollama
from tqdm import tqdm


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BATCH_SIZE = 10
CHECKPOINT_EVERY = 500
MAX_RETRIES = 3


# ---------------------------------------------------------
# PROCESS ONE BATCH OF ABSTRACTS
# ---------------------------------------------------------

def get_research_goals_batch(abstracts):
    """
    Sends a batch of abstracts to Gemma 3 4B and returns
    one research goal for each abstract.

    abstracts:
        List of abstract strings.

    returns:
        List of research goal strings in the same order
        as the input abstracts.
    """

    for attempt in range(MAX_RETRIES):

        try:

            # Build the numbered list of abstracts
            abstract_text = ""

            for i, abstract in enumerate(abstracts):
                abstract_text += (
                    f"\n\n--- ABSTRACT {i + 1} ---\n"
                    f"{abstract}"
                )


            response = ollama.chat(

                model="gemma3:4b",

                messages=[
                    {
                        "role": "user",

                        "content": f"""
You are a research assistant.

You will receive {len(abstracts)} research paper abstracts.

For EACH abstract, extract its core research goal as ONE concise
sentence of maximum 25 words.

Focus on WHAT the paper is trying to achieve or solve,
not the method or results.

You MUST return exactly {len(abstracts)} research goals.

The order MUST remain exactly the same as the abstracts.

Respond ONLY with valid JSON in this exact format:

{{
    "research_goals": [
        "research goal for abstract 1",
        "research goal for abstract 2",
        "research goal for abstract 3"
    ]
}}

{abstract_text}
"""
                    }
                ],

                format="json",

                options={
                    "temperature": 0.2
                }
            )


            # -------------------------------------------------
            # Extract model response
            # -------------------------------------------------

            content = response["message"]["content"]

            parsed = json.loads(content)

            research_goals = parsed.get("research_goals", [])


            # -------------------------------------------------
            # Validate the number of returned goals
            # -------------------------------------------------

            if len(research_goals) != len(abstracts):

                raise ValueError(
                    f"Expected {len(abstracts)} research goals, "
                    f"but received {len(research_goals)}"
                )


            # Make sure every result is a string
            research_goals = [
                str(goal).strip()
                for goal in research_goals
            ]


            return research_goals


        # -----------------------------------------------------
        # JSON / structure error
        # -----------------------------------------------------

        except (json.JSONDecodeError, KeyError, ValueError) as e:

            print(
                f"  [Batch parse error, "
                f"attempt {attempt + 1}/{MAX_RETRIES}]: {e}"
            )

            time.sleep(1)


        # -----------------------------------------------------
        # Ollama / other error
        # -----------------------------------------------------

        except Exception as e:

            print(
                f"  [Ollama error, "
                f"attempt {attempt + 1}/{MAX_RETRIES}]: {e}"
            )

            time.sleep(2 ** attempt)


    # ---------------------------------------------------------
    # If all attempts fail
    # ---------------------------------------------------------

    return ["EXTRACTION_FAILED"] * len(abstracts)


# ---------------------------------------------------------
# PROCESS CATEGORY
# ---------------------------------------------------------

def process_category(
    category: str,
    input_dir: str,
    checkpoint_dir: str
):

    """
    Reads the category JSON file, generates research goals
    in batches, checkpoints progress, and saves the final
    Parquet file.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)


    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{category}_checkpoint.parquet"
    )


    input_path = os.path.join(
        input_dir,
        f"{category}_papers.json"
    )


    output_path = os.path.join(
        input_dir,
        f"{category}_research_goals.parquet"
    )


    # ---------------------------------------------------------
    # LOAD DATA / RESUME
    # ---------------------------------------------------------

    if os.path.exists(checkpoint_path):

        df = pd.read_parquet(checkpoint_path)

        print(
            f"[{category}] Resuming from checkpoint: "
            f"{df['research_goal'].notna().sum()} / "
            f"{len(df)} done"
        )

    else:

        df = pd.read_json(input_path)

        df["research_goal"] = pd.NA

        print(
            f"[{category}] Starting fresh: "
            f"{len(df)} papers"
        )


    # ---------------------------------------------------------
    # FIND PAPERS THAT STILL NEED PROCESSING
    # ---------------------------------------------------------

    pending_mask = (
        df["research_goal"].isna()
        |
        (df["research_goal"] == "EXTRACTION_FAILED")
    )

    pending_indices = df[pending_mask].index.tolist()


    print(
        f"[{category}] Papers remaining: "
        f"{len(pending_indices)}"
    )


    # ---------------------------------------------------------
    # PROCESS IN BATCHES
    # ---------------------------------------------------------

    processed_since_checkpoint = 0


    for start in tqdm(
        range(0, len(pending_indices), BATCH_SIZE),
        desc=f"{category}"
    ):

        # Get indices for this batch
        batch_indices = pending_indices[
            start:start + BATCH_SIZE
        ]


        # Get abstracts for those indices
        batch_abstracts = [
            df.at[idx, "abstract"]
            for idx in batch_indices
        ]


        # -----------------------------------------------------
        # SEND ENTIRE BATCH TO GEMMA
        # -----------------------------------------------------

        research_goals = get_research_goals_batch(
            batch_abstracts
        )


        # -----------------------------------------------------
        # PUT RESULTS BACK INTO DATAFRAME
        # -----------------------------------------------------

        for idx, goal in zip(
            batch_indices,
            research_goals
        ):

            df.at[idx, "research_goal"] = goal


        processed_since_checkpoint += len(batch_indices)


        # -----------------------------------------------------
        # CHECKPOINT EVERY 500 PAPERS
        # -----------------------------------------------------

        if processed_since_checkpoint >= CHECKPOINT_EVERY:

            df.to_parquet(checkpoint_path)

            print(
                f"\n[{category}] Checkpoint saved. "
                f"Processed {start + len(batch_indices)} "
                f"papers in this run."
            )

            processed_since_checkpoint = 0


    # ---------------------------------------------------------
    # FINAL SAVE
    # ---------------------------------------------------------

    df.to_parquet(checkpoint_path)

    df.to_parquet(output_path)


    print(
        f"[{category}] Done. "
        f"Saved to {output_path}"
    )


    # ---------------------------------------------------------
    # REPORT FAILURES
    # ---------------------------------------------------------

    failed = (
        df["research_goal"] == "EXTRACTION_FAILED"
    ).sum()


    if failed:

        print(
            f"[{category}] WARNING: "
            f"{failed} rows failed extraction "
            f"and can be retried."
        )


# ---------------------------------------------------------
# RUN CATEGORY
# ---------------------------------------------------------

if __name__ == "__main__":

    process_category(
        "eess",
        input_dir=r"F:\Diya\EESS",
        checkpoint_dir=r"F:\Diya\EESS"
    )