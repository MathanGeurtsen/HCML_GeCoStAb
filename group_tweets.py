import pandas as pd

df = pd.read_csv("archive/data.csv")
df.dropna(inplace=True)

df = df.groupby('Handle').agg(lambda x: " ".join(list(set(x.tolist()))))

print(df.head())
print(len(df))

df.to_csv("archive/grouped_data.csv")