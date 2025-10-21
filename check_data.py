import pandas as pd
df = pd.read_csv("data/data.csv")
for col in df.columns:
    # show rows where column is non-numeric (ignoring empty)
    bad = df[~df[col].astype(str).str.strip().replace('.','',regex=False).str.isnumeric()]
    if not bad.empty:
        print(f"\n-- Non-numeric or suspicious values in column: {col} --")
        print(bad.head(10))
