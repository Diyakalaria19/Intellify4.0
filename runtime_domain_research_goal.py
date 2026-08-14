"""
ollama_client.py — runtime domain + research goal extraction.

This does NOT replace category_research_goals.py. That script runs offline,
once, per category, and only extracts research goals (domain is already known
since you're running it inside that category's folder/script).

This file runs at RUNTIME, once per user query, and does something the
precompute script never had to do: figure out WHICH domain(s) the query
belongs to, in addition to extracting its research goal -- because an
incoming user query hasn't been assigned to any category yet.

The research-goal extraction instructions below are copied verbatim from
category_research_goals.py's prompt, so the STYLE of goal extraction matches
between precomputed dataset goals and runtime query goals. Only the domain
classification instruction is new.

Requires: pip install ollama
Setup: ollama pull gemma3:4b   (must match model name used in precompute)
"""

import json
import time

import ollama

MODEL_NAME = "gemma3:4b"  # matches category_research_goals.py exactly
MAX_RETRIES = 3

VALID_DOMAINS = {"cs", "math", "physics", "econ", "q_bio"}


def extract_domain_and_goal(idea_description: str) -> dict:
    """
    Returns: {"domains": ["cs", ...], "research_goal": "..."}
    """
    prompt = f"""
You are a research assistant.

TASK 1 — Domain classification:
Identify which of these domains this research idea belongs to (one or more):
cs, math, physics, econ, q_bio.

TASK 2 — Research goal extraction:
Given the following idea description, extract its core research goal
as a single concise sentence of maximum 25 words.

Focus on WHAT the idea is trying to achieve or solve,
not the method or results.

Respond ONLY with valid JSON in this exact format:

{{"domains": ["..."], "research_goal": "..."}}

Idea description:
{idea_description}
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.2},
            )

            content = response["message"]["content"]
            parsed = json.loads(content)

            parsed["domains"] = [d for d in parsed.get("domains", []) if d in VALID_DOMAINS]
            parsed["research_goal"] = parsed.get("research_goal", "").strip()

            return parsed

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [parse error, attempt {attempt+1}/{MAX_RETRIES}]: {e}")
            time.sleep(1)

        except Exception as e:
            print(f"  [Ollama error, attempt {attempt+1}/{MAX_RETRIES}]: {e}")
            time.sleep(2 ** attempt)

    # match the same failure convention as category_research_goals.py
    return {"domains": [], "research_goal": "EXTRACTION_FAILED"}


if __name__ == "__main__":
    test_input = "A lightweight neural architecture for real-time protein folding prediction on edge devices."
    result = extract_domain_and_goal(test_input)
    print(result)
