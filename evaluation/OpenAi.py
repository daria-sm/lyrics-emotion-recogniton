import openai
import pandas as pd

from evaluation.metrics import print_pandas_accuracy
from preprocessing.extract_lyrics_words import beautify

openai.api_key = "api-key"  # Replace with your actual API key

prompt = "We divide the Russel's' circumflex model in Q1 positive valence and positive arousal Q2 negative valence positive arousal Q3 negative valence and negative arousal Q4 positive valence negative arousal"

df = pd.read_json("../data/processed/limit-100.json", lines=True)
result_df = pd.DataFrame()

for index, row in df.iterrows():
    lyrics = beautify(row["lyrics"])
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}, {"role": "system",
                                                          "content": "Which string  'Q1' 'Q2' 'Q3' 'Q4' represents the quadrant for the lyrics '{0}'? "
                                                                     "\n you must only answer with the string nothing else."
                                                          .format(lyrics)}],
        max_tokens=2
    )
    result_q = response.choices[0].message.content
    if result_q in ["Q1", "Q2", "Q3", "Q4"]:
        result_df['lastfm_id'] = df['lastfm_id']
        result_df['Actual'] = df['matching_docs'].apply(lambda x: x['Q_song'])
        result_df.loc[index, 'Predicted'] = result_q

print(result_df.head(10))
print_pandas_accuracy(result_df)
