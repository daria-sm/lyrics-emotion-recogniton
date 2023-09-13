# import modules & set up logging
import pandas
from gensim.models.word2vec import Word2Vec

from preprocessing.extract_lyrics_words import extract_lyrics_sentences_tokenized

sentences = []  # lyrics of the songs

songs_path = "../data/processed/limit-70-per-cent.json"
df = pandas.read_json(songs_path, lines=True)
# a clean df to store results
result_df = pandas.DataFrame()
not_analyzed_songs = 0
for index, row in df.iterrows():
    sentences.extend(extract_lyrics_sentences_tokenized(row["lyrics"]))
# train word2vec on the all sentences from lyrics
# parameter min count removes words that appear not frequently in the corpus
# workers can be used to train the model
# corpus will be sentences
model = Word2Vec(sentences, min_count=1, workers=4)  # default = 1 worker = no parallelization
model.build_vocab(sentences)  # can be a non-repeatable, 1-pass generator
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
# Get the KeyedVectors
word_vectors = model.wv

# Save KeyedVectors to a file
word_vectors.save_word2vec_format("../data/models/word2vec_vectors_70.bin", binary=True)

print("Done")
