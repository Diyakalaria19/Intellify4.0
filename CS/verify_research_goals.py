import pandas as pd

df = pd.read_parquet(r"F:\Diya\CS\cs_checkpoint.parquet")

print(df[["abstract", "research_goal"]].head(10))

# import pandas as pd
# df = pd.read_parquet(r"F:\Diya\MATH\math_research_goals.parquet")
# print(df["research_goal"].notna().sum(), "/", len(df))

