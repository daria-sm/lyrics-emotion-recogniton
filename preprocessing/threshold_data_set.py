import pandas as pd

df = pd.read_json("../data/processed/limit-all.json", lines=True)

# Extract valence and arousal values from the matching_docs dictionary
df["valence"] = df["matching_docs"].apply(lambda x: x.get("Valence"))
df["arousal"] = df["matching_docs"].apply(lambda x: x.get("Arousal"))

# Convert valence and arousal columns to numeric (float)
df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")
threshold = 0.3
# Filter the dataset to create a subset where both valence and arousal are greater than 0.6
subset = df[
    (df["valence"] > 0.5 + threshold) & (df["arousal"] > 0.5 + threshold) | (df["valence"] > 0.5 + threshold) &
    (df["arousal"] < 0.5 - threshold) | (df["valence"] < 0.5 - threshold) & (df["arousal"] > 0.5 + threshold) | (
            df["valence"] < 0.5 - threshold) & (df["arousal"] < 0.5 - threshold)]

with open("../data/processed/limit-threshold-03.json", 'w') as file:
    for _, row in subset.iterrows():
        json_row = row.to_json()
        file.write(json_row + '\n')
