from gensim.models import KeyedVectors

# Load the pre-trained Google Word2Vec model
model = KeyedVectors.load_word2vec_format('../data/models/GoogleNews-vectors-negative300.bin.gz', binary=True)
