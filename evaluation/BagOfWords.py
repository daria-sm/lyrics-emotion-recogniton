import pandas as pd

from evaluation.metrics import print_pandas_accuracy, print_pandas_precision, print_pandas_recall, \
    print_pandas_f1_score, print_pandas_precision_non_binary, print_pandas_recall_non_binary, \
    print_pandas_f1_score_non_binary
from preprocessing.extract_lyrics_words import extract_words_from_lexicon


class BagOfWords:
    def __init__(self, data_path, preprocessing_method=extract_words_from_lexicon):
        self.result_df = pd.DataFrame()
        self.df = pd.read_json(data_path, lines=True)
        self.word_scores = {}
        self.prepare_tag_vectors()
        self.preprocessing_method = preprocessing_method

    def prepare_tag_vectors(self):
        # Read the CSV file
        word_scores_df = pd.read_csv("../data/processed/NRC-VAD-Lexicon.csv", sep="\t", header=None)

        # Create a dictionary of word scores (word: (valence, arousal))
        for index, row in word_scores_df.iterrows():
            word = row[0]  # Word is in the first column
            valence = row[1]  # Valence is in the second column
            arousal = row[2]  # Arousal is in the third column
            self.word_scores[word] = (valence, arousal)

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        # Bag of Words representation (word counts)
        vocabulary = list(self.word_scores.keys())
        word_counts = [words.count(word) for word in self.word_scores]

        # Calculate aggregated valence and arousal for the song
        total_valence = 0
        total_arousal = 0
        num_words = sum(word_counts)
        word_count_dict = {}
        if num_words == 0:
            print("No features found")
            return None

        for word, count in zip(vocabulary, word_counts):
            if word in self.word_scores and count != 0:
                valence, arousal = self.word_scores[word]
                total_valence += valence * count
                total_arousal += arousal * count
                word_count_dict[word] = count

        # Normalize aggregated scores
        normalized_valence = total_valence / num_words
        normalized_arousal = total_arousal / num_words

        # Classify emotion based on the normalized scores
        if normalized_valence > 0.5 and normalized_arousal > 0.5:
            emotion = "Q1"
        elif normalized_valence > 0.5 and normalized_arousal <= 0.5:
            emotion = "Q4"
        elif normalized_valence <= 0.5 and normalized_arousal > 0.5:
            emotion = "Q2"
        else:
            emotion = "Q3"
        return emotion

    def execute(self):
        for index, row in self.df.iterrows():
            result_q = self.compare_vectors(self.preprocessing_method(row["lyrics"]))
            if result_q:
                self.result_df['lastfm_id'] = self.df['lastfm_id']
                self.result_df['Actual'] = self.df['matching_docs'].apply(lambda x: x['Q_song'])
                self.result_df.loc[index, 'Predicted'] = result_q

    def print_evaluation(self):
        self.result_df.dropna(inplace=True)
        print_pandas_accuracy(self.result_df)
        print_pandas_precision_non_binary(self.result_df)
        print_pandas_recall_non_binary(self.result_df)
        print_pandas_f1_score_non_binary(self.result_df)
        print(self.result_df)


class BagOfWordsBinaryValence(BagOfWords):

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        # Bag of Words representation (word counts)
        vocabulary = list(self.word_scores.keys())
        word_counts = [words.count(word) for word in self.word_scores]

        # Calculate aggregated valence and arousal for the song
        total_valence = 0
        num_words = sum(word_counts)
        word_count_dict = {}
        if num_words == 0:
            print("No features found")
            return None

        for word, count in zip(vocabulary, word_counts):
            if word in self.word_scores and count != 0:
                valence, arousal = self.word_scores[word]
                total_valence += valence * count
                word_count_dict[word] = count

        # Normalize aggregated scores
        normalized_valence = total_valence / num_words

        # Classify emotion based on the normalized scores
        if normalized_valence >= 0.5:
            emotion = "Q1"
        else:
            emotion = "Q2"
        return emotion

    def print_evaluation(self):
        self.result_df.dropna(inplace=True)
        print_pandas_accuracy(self.result_df)
        print_pandas_precision(self.result_df)
        print_pandas_recall(self.result_df)
        print_pandas_f1_score(self.result_df)
        print(self.result_df.head())


class BagOfWordsBinaryArousal(BagOfWordsBinaryValence):

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        # Bag of Words representation (word counts)
        vocabulary = list(self.word_scores.keys())
        word_counts = [words.count(word) for word in self.word_scores]

        # Calculate aggregated valence and arousal for the song
        total_arousal = 0
        num_words = sum(word_counts)
        word_count_dict = {}
        if num_words == 0:
            print("No features found")
            return None

        for word, count in zip(vocabulary, word_counts):
            if word in self.word_scores and count != 0:
                valence, arousal = self.word_scores[word]
                total_arousal += arousal * count
                word_count_dict[word] = count

        # Normalize aggregated scores
        normalized_arousal = total_arousal / num_words

        # Classify emotion based on the normalized scores
        if normalized_arousal >= 0.5:
            emotion = "Q1"
        else:
            emotion = "Q3"
        return emotion
