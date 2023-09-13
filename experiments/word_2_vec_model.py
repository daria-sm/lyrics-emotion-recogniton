# Define the two sets of words

from gensim.models import KeyedVectors
from evaluation.Word2Vec import Word2Vec, Word2VecBySentence, Word2VecSentenceToTag, Word2VecBinaryValence, \
    Word2VecBinaryArousal
from evaluation.Word2VecBOW import Word2VecBOW
from preprocessing.extract_lyrics_words import beautify, extract_adjectives

model = KeyedVectors.load_word2vec_format("../data/models/word2vec_vectors_70.bin", binary=True)
print("All words")
onw_model = Word2Vec(model, "../data/processed/limit-30-per-cent.json")
onw_model.execute()
onw_model.print_evaluation()
print(onw_model.result_df.head())
print("Adjectives")
# # Adjectives
google_model_experiment = Word2Vec(model, "../data/processed/limit-30-per-cent.json", extract_adjectives)
google_model_experiment.execute()
google_model_experiment.print_evaluation()
print("Sentence")
by_Sentence_experiment = Word2VecBySentence(model, "../data/processed/limit-30-per-cent.json")
by_Sentence_experiment.execute()
by_Sentence_experiment.print_evaluation()
print(by_Sentence_experiment.result_df.head())
print("Sentence to tag")
own_model_sentence = Word2VecSentenceToTag(model, "../data/processed/limit-30-per-cent.json")
own_model_sentence.execute()
own_model_sentence.print_evaluation()
print(own_model_sentence.result_df.head())

print("Binary Valence")
by_Sentence_experiment = Word2VecBinaryValence(model, "../data/processed/limit-all-binary-valence.json")
by_Sentence_experiment.execute()
by_Sentence_experiment.print_evaluation()
#
print("Binary Arousal")
by_Sentence_experiment = Word2VecBinaryArousal(model, "../data/processed/limit-all-binary-arousal.json")
by_Sentence_experiment.execute()
by_Sentence_experiment.print_evaluation()
#
print("Word2VecBOW")
google_model_experiment_word2_vec_bof = Word2VecBOW(model, "../data/processed/limit-30-per-cent.json", beautify)
google_model_experiment_word2_vec_bof.execute()
google_model_experiment_word2_vec_bof.print_evaluation()