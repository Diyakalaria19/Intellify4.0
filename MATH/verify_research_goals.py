import pandas as pd
df = pd.read_parquet(r"F:\Diya\MATH\math_research_goals.parquet")
print(df["research_goal"].notna().sum(), "/", len(df))


# import pandas as pd
# df = pd.read_parquet(r"F:\Diya\BIO\q-bio_research_goals.parquet")
# failed = df[df["research_goal"] == "EXTRACTION_FAILED"]
# print(f"{len(failed)} papers failed extraction")
# print(failed.index.tolist())