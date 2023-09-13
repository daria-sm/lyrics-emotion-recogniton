import pandas as pd

df = pd.read_json("../data/processed/limit-all.json", lines=True)


# Function to update 'Q_song'
def update_q_song(record):
    if record['Q_song'] == 'Q2':
        record['Q_song'] = 'Q1'
    if record['Q_song'] == 'Q4':
        record['Q_song'] = 'Q3'
    return record


df['matching_docs'] = df['matching_docs'].apply(update_q_song)

with open("../data/processed/limit-all-binary-arousal.json", 'w') as file:
    for _, row in df.iterrows():
        json_row = row.to_json()
        file.write(json_row + '\n')
