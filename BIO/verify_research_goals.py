import pandas as pd
df = pd.read_parquet(r"C:\Users\student\Desktop\BIO\q-bio_checkpoint.parquet")
failed = df[df["research_goal"] == "EXTRACTION_FAILED"]
print(f"{len(failed)} papers failed extraction")
print(failed.index.tolist())