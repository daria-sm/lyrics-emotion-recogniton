from evaluation.Word2Vec import Word2Vec, Word2VecBySentence, Word2VecBinaryValence, Word2VecBinaryArousal, \
    Word2VecSentenceToTag
from evaluation.Word2VecBOW import Word2VecBOW
from preprocessing.extract_lyrics_words import extract_adjectives, beautify
from training.word_2_vec_google_model import model

# #
print("All words")
google_model_experiment = Word2Vec(model, "../data/processed/limit-10.json")
google_model_experiment.execute()
google_model_experiment.print_evaluation()

# print("Adjectives")
# # # Adjectives
# google_model_experiment = Word2Vec(model, "../data/processed/limit-10.json", extract_adjectives)
# google_model_experiment.execute()
# google_model_experiment.print_evaluation()
# #
# #
# print("BySentence")
# by_Sentence_experiment = Word2VecBySentence(model, "../data/processed/limit-10.json")
# by_Sentence_experiment.execute()
# by_Sentence_experiment.print_evaluation()
# #
# print("Sentence to tag")
# google_model_experiment = Word2VecSentenceToTag(model, "../data/processed/limit-10.json")
# google_model_experiment.execute()
# google_model_experiment.print_evaluation()
# #
# print("Binary Valence")
# by_Sentence_experiment = Word2VecBinaryValence(model, "../data/processed/limit-all-binary-valence.json")
# by_Sentence_experiment.execute()
# by_Sentence_experiment.print_evaluation()
# #
# print("Binary Arousal")
# by_Sentence_experiment = Word2VecBinaryArousal(model, "../data/processed/limit-all-binary-arousal.json")
# by_Sentence_experiment.execute()
# by_Sentence_experiment.print_evaluation()
# #
# print("Word2VecBOW")
# google_model_experiment_word2_vec_bof = Word2VecBOW(model, "../data/processed/limit-10.json", beautify)
# google_model_experiment_word2_vec_bof.execute()
# google_model_experiment_word2_vec_bof.print_evaluation()
