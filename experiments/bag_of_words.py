from evaluation.BagOfWords import BagOfWords, BagOfWordsBinaryValence, BagOfWordsBinaryArousal
from preprocessing.extract_lyrics_words import extract_adjectives, extract_adverbs

# Extract all the words
bag_of_words_experiment = BagOfWords("../data/processed/limit-10.json")
bag_of_words_experiment.execute()
bag_of_words_experiment.print_evaluation()

# Extract only adjectives
bag_of_words_experiment = BagOfWords("../data/processed/limit-10.json", extract_adjectives)
bag_of_words_experiment.execute()
bag_of_words_experiment.print_evaluation()

# Extract only adverbs
bag_of_words_experiment = BagOfWords("../data/processed/limit-10.json", extract_adverbs)
bag_of_words_experiment.execute()
bag_of_words_experiment.print_evaluation()
# explain with some examples why adverbs are not good for tokenized lyrics


# Binary bag of words on valence
bag_of_words_experiment = BagOfWordsBinaryValence("../data/processed/limit-all-binary-valence.json")
bag_of_words_experiment.execute()
bag_of_words_experiment.print_evaluation()


# Binary bag of words on arousal
bag_of_words_experiment = BagOfWordsBinaryArousal("../data/processed/limit-all-binary-arousal.json")
bag_of_words_experiment.execute()
bag_of_words_experiment.print_evaluation()
