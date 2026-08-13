"""
STEP 1: Build 5 category JSON files from the arXiv snapshot.

Eligibility rule: a paper is included only if ALL its categories fall
within your 5 target domains (math, eess, cs, q-bio, econ). Cross-domain
papers (e.g. cs + math) are included in BOTH relevant output files.

Quota: math, eess, cs, q-bio -> 10,000 papers each (random sample if more
are eligible). econ -> ALL eligible papers (falls short of 10k).

Each output record has exactly these 5 fields:
    paper_id, title, abstract, author, category

Output: math_papers.json, eess_papers.json, cs_papers.json,
        q-bio_papers.json, econ_papers.json
"""

import json
import random

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
RANDOM_SEED = 42

target_categories = {"math", "eess", "cs", "q-bio","q-fin"}
QUOTAS = {
    "math": 10000,
    "eess": 10000,
    "cs": 10000,
    "q-bio": 10000,
    "q-fin" : 10000,
}

buckets = {cat: [] for cat in target_categories}
total_scanned = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)
        total_scanned += 1

        cats = paper.get("categories", "").split()
        if not cats:
            continue

        top_levels = {c.split(".")[0] for c in cats}
        if not top_levels.issubset(target_categories):
            continue  # excluded - touches a domain outside your 5

        record = {
            "paper_id": paper.get("id"),
            "title": paper.get("title", "").strip(),
            "abstract": paper.get("abstract", "").strip(),
            "author": paper.get("authors", "").strip(),
            "category": paper.get("categories", "").strip(),
        }

        for domain in top_levels:
            buckets[domain].append(record)

print(f"Total papers scanned: {total_scanned}\n")

random.seed(RANDOM_SEED)

for domain, papers in buckets.items():
    quota = QUOTAS[domain]
    if quota is not None and len(papers) > quota:
        selected = random.sample(papers, quota)
    else:
        selected = papers

    out_path = f"{domain}_papers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print(f"{domain:6s}: {len(papers)} eligible -> saved {len(selected)} to {out_path}")

print("\nDone.")