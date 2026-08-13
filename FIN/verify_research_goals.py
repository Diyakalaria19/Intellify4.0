# import pandas as pd
# df = pd.read_parquet(r"F:\Diya\FIN\q-fin_research_goals.parquet")
# failed = df[df["research_goal"] == "EXTRACTION_FAILED"]
# print(f"{len(failed)} papers failed extraction")
# print(failed.index.tolist())

import pandas as pd
df = pd.read_parquet(r"F:\Diya\FIN\q-fin_research_goals.parquet")
print(df["research_goal"].notna().sum(), "/", len(df))

