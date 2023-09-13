from abc import ABC

import numpy as np
import pandas as pd
import spacy

from evaluation.metrics import print_pandas_accuracy, print_pandas_precision_non_binary, print_pandas_recall_non_binary, \
    print_pandas_f1_score_non_binary
from scipy.spatial.distance import cosine

from preprocessing.extract_lyrics_words import extract_words, extract_lyrics_sentences, extract_adjectives

from collections import Counter

from preprocessing.extract_lyrics_words import extract_lyrics_sentences_tokenized


class Word2Vec(ABC):
    def __init__(self, model, data_path, preprocessing_method=extract_words):
        self.model = model
        self.result_df = pd.DataFrame()
        self.df = pd.read_json(data_path, lines=True)
        self.prepare_tag_vectors()
        self.preprocessing_method = preprocessing_method

    def prepare_tag_vectors(self):
        df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')
        filtered_df_q1 = df[df['Quadrant'] == "Q1"]
        filtered_df_q2 = df[df['Quadrant'] == "Q2"]
        filtered_df_q3 = df[df['Quadrant'] == "Q3"]
        filtered_df_q4 = df[df['Quadrant'] == "Q4"]

        set2 = set(filtered_df_q1['emotion_tag'][0:5])
        set3 = set(filtered_df_q2['emotion_tag'][0:5])
        set4 = set(filtered_df_q3['emotion_tag'][0:5])
        set5 = set(filtered_df_q4['emotion_tag'][0:5])

        # Calculate the average vector for q1
        set2_vectors = [self.model[word] for word in set2 if word in self.model]
        self.set2_avg_vector = np.mean(set2_vectors, axis=0)

        # Calculate the average vector for q2
        set3_vectors = [self.model[word] for word in set3 if word in self.model]
        self.set3_avg_vector = np.mean(set3_vectors, axis=0)

        # Calculate the average vector for q3
        set4_vectors = [self.model[word] for word in set4 if word in self.model]
        self.set4_avg_vector = np.mean(set4_vectors, axis=0)

        # Calculate the average vector for q4
        set5_vectors = [self.model[word] for word in set5 if word in self.model]
        self.set5_avg_vector = np.mean(set5_vectors, axis=0)

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        set1_vectors = [self.model[word] for word in words if word in self.model]
        if not set1_vectors:
            return None
        set1_avg_vector = np.mean(set1_vectors, axis=0)
        maximum = 0
        result_q = ""
        if set1_avg_vector.ndim == 1:
            # Calculate the cosine similarity distance Q1
            cosine_distance = 1 - cosine(set1_avg_vector, self.set2_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q1"
                maximum = cosine_distance

            # Calculate the cosine similarity distance Q2
            cosine_distance = 1 - cosine(set1_avg_vector, self.set3_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q2"

            # Calculate the cosine similarity distance Q3
            cosine_distance = 1 - cosine(set1_avg_vector, self.set4_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q3"

            # Calculate the cosine similarity distance Q4
            cosine_distance = 1 - cosine(set1_avg_vector, self.set5_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q4"
        return result_q

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


class Word2VecBySentence(Word2Vec):
    """Compares each sentence of the song with the tags from each quadrant, polling the most matching quadrant
    the quadrant with the most sentences attached will be considered the song quadrant"""

    def execute(self):
        for index, row in self.df.iterrows():
            sentences = extract_lyrics_sentences(row["lyrics"])
            result_q_list = []
            for sentence in sentences:
                if len(sentence) > 0:
                    if result_q := self.compare_vectors(sentence):
                        result_q_list.append(result_q)
            self.result_df['lastfm_id'] = self.df['lastfm_id']
            self.result_df['Actual'] = self.df['matching_docs'].apply(lambda x: x['Q_song'])
            # Extract most repeated quadrant
            quadrant_counter = Counter(result_q_list)
            if result_q_list:
                self.result_df.loc[index, 'Predicted'] = quadrant_counter.most_common(1)[0][0]
            else:
                self.result_df.loc[index, 'Predicted'] = None


class Word2VecSentenceToTag(Word2VecBySentence):
    """Compares every sentence to the 10 most common tags in each quadrant
    an average vector is created for every tag quadrant, being the biggest vector the most probable"""

    def prepare_tag_vectors(self):
        df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')
        filtered_df_q1 = df[df['Quadrant'] == "Q1"]
        filtered_df_q2 = df[df['Quadrant'] == "Q2"]
        filtered_df_q3 = df[df['Quadrant'] == "Q3"]
        filtered_df_q4 = df[df['Quadrant'] == "Q4"]

        set2 = set(filtered_df_q1['emotion_tag'][0:5])
        set3 = set(filtered_df_q2['emotion_tag'][0:5])
        set4 = set(filtered_df_q3['emotion_tag'][0:5])
        set5 = set(filtered_df_q4['emotion_tag'][0:5])

        # Calculate the average vector for q1
        self.q1_vector = [self.model[word] for word in set2 if word in self.model]

        # Calculate the average vector for q2
        self.q2_vector = [self.model[word] for word in set3 if word in self.model]

        # Calculate the average vector for q3
        self.q3_vector = [self.model[word] for word in set4 if word in self.model]

        # Calculate the average vector for q4
        self.q4_vector = [self.model[word] for word in set5 if word in self.model]

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        set1_vectors = [self.model[word] for word in words if word in self.model]
        if not set1_vectors:
            return None
        set1_avg_vector = np.mean(set1_vectors, axis=0)
        maximum = 0
        result_q = ''
        similarity_q1 = [1 - cosine(set1_avg_vector, tag) for tag in self.q1_vector]
        similarity_q2 = [1 - cosine(set1_avg_vector, tag) for tag in self.q2_vector]
        similarity_q3 = [1 - cosine(set1_avg_vector, tag) for tag in self.q3_vector]
        similarity_q4 = [1 - cosine(set1_avg_vector, tag) for tag in self.q4_vector]
        if sum(similarity_q1) > maximum:
            maximum = sum(similarity_q1)
            result_q = "Q1"
        if sum(similarity_q2) > maximum:
            maximum = sum(similarity_q2)
            result_q = "Q2"
        if sum(similarity_q3) > maximum:
            maximum = sum(similarity_q1)
            result_q = "Q3"
        if sum(similarity_q4) > maximum:
            result_q = "Q4"
        return result_q


class Word2VecSpacy(Word2VecSentenceToTag):

    def __init__(self, data_path):
        nlp = spacy.load('en_core_web_lg')
        super().__init__(nlp, data_path)

    def prepare_tag_vectors(self):
        df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')
        filtered_df_q1 = df[df['Quadrant'] == "Q1"]
        filtered_df_q2 = df[df['Quadrant'] == "Q2"]
        filtered_df_q3 = df[df['Quadrant'] == "Q3"]
        filtered_df_q4 = df[df['Quadrant'] == "Q4"]

        set2 = set(filtered_df_q1['emotion_tag'][0:5])
        set3 = set(filtered_df_q2['emotion_tag'][0:5])
        set4 = set(filtered_df_q3['emotion_tag'][0:5])
        set5 = set(filtered_df_q4['emotion_tag'][0:5])

        # Calculate the average vector for q1
        self.q1_vector = [self.model(word) for word in set2]

        # Calculate the average vector for q2
        self.q2_vector = [self.model(word) for word in set3]

        # Calculate the average vector for q3
        self.q3_vector = [self.model(word) for word in set4]

        # Calculate the average vector for q4
        self.q4_vector = [self.model(word) for word in set5]

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        set1_avg_vector = self.model(words)
        if not set1_avg_vector:
            return None
        maximum = 0
        result_q = ''
        similarity_q1 = [set1_avg_vector.similarity(tag) for tag in self.q1_vector]
        similarity_q2 = [set1_avg_vector.similarity(tag) for tag in self.q2_vector]
        similarity_q3 = [set1_avg_vector.similarity(tag) for tag in self.q3_vector]
        similarity_q4 = [set1_avg_vector.similarity(tag) for tag in self.q4_vector]

        if sum(similarity_q1) > maximum:
            maximum = sum(similarity_q1)
            result_q = "Q1"
        if sum(similarity_q2) > maximum:
            maximum = sum(similarity_q2)
            result_q = "Q2"
        if sum(similarity_q3) > maximum:
            maximum = sum(similarity_q1)
            result_q = "Q3"
        if sum(similarity_q4) > maximum:
            result_q = "Q4"
        return result_q


class Word2VecBinaryValence(Word2Vec):

    def prepare_tag_vectors(self):
        df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')
        filtered_df_q1 = df[df['Quadrant'] == "Q1"]
        filtered_df_q2 = df[df['Quadrant'] == "Q2"]
        filtered_df_q3 = df[df['Quadrant'] == "Q3"]
        filtered_df_q4 = df[df['Quadrant'] == "Q4"]

        set_q1_q4 = set(filtered_df_q1['emotion_tag'][0:5]).union(set(filtered_df_q4['emotion_tag'][0:5]))
        set_q2_q3 = set(filtered_df_q2['emotion_tag'][0:5]).union(set(filtered_df_q3['emotion_tag'][0:5]))

        # Calculate the average vector for q1
        set_q1_q4_vectors = [self.model[word] for word in set_q1_q4 if word in self.model]
        self.set_q1_q4_vector = np.mean(set_q1_q4_vectors, axis=0)

        # Calculate the average vector for q2
        set_q2_q3_vectors = [self.model[word] for word in set_q2_q3 if word in self.model]
        self.set_q2_q3_avg_vector = np.mean(set_q2_q3_vectors, axis=0)

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        set1_vectors = [self.model[word] for word in words if word in self.model]
        if not set1_vectors:
            return None
        set1_avg_vector = np.mean(set1_vectors, axis=0)
        maximum = 0
        result_q = ""
        if set1_avg_vector.ndim == 1:
            # Calculate the cosine similarity distance Q1
            cosine_distance = 1 - cosine(set1_avg_vector, self.set_q1_q4_vector)
            if cosine_distance > maximum:
                result_q = "Q1"
                maximum = cosine_distance

            # Calculate the cosine similarity distance Q2
            cosine_distance = 1 - cosine(set1_avg_vector, self.set_q2_q3_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q2"
        return result_q


class Word2VecBinaryArousal(Word2Vec):

    def prepare_tag_vectors(self):
        df = pd.read_csv('../data/processed/tags_vad_q_sorted.csv')
        filtered_df_q1 = df[df['Quadrant'] == "Q1"]
        filtered_df_q2 = df[df['Quadrant'] == "Q2"]
        filtered_df_q3 = df[df['Quadrant'] == "Q3"]
        filtered_df_q4 = df[df['Quadrant'] == "Q4"]

        set_q1_q2 = set(filtered_df_q1['emotion_tag'][0:5]).union(set(filtered_df_q2['emotion_tag'][0:5]))
        set_q3_q4 = set(filtered_df_q3['emotion_tag'][0:5]).union(set(filtered_df_q4['emotion_tag'][0:5]))

        # Calculate the average vector for q1
        set_q1_q4_vectors = [self.model[word] for word in set_q1_q2 if word in self.model]
        self.set_q1_q2_vector = np.mean(set_q1_q4_vectors, axis=0)

        # Calculate the average vector for q2
        set_q2_q3_vectors = [self.model[word] for word in set_q3_q4 if word in self.model]
        self.set_q3_q4_avg_vector = np.mean(set_q2_q3_vectors, axis=0)

    def compare_vectors(self, words):
        """Returns the mox approximate quadrant to the set of words"""
        set1_vectors = [self.model[word] for word in words if word in self.model]
        if not set1_vectors:
            return None
        set1_avg_vector = np.mean(set1_vectors, axis=0)
        maximum = 0
        result_q = ""
        if set1_avg_vector.ndim == 1:
            # Calculate the cosine similarity distance Q1
            cosine_distance = 1 - cosine(set1_avg_vector, self.set_q1_q2_vector)
            if cosine_distance > maximum:
                result_q = "Q1"
                maximum = cosine_distance

            # Calculate the cosine similarity distance Q2
            cosine_distance = 1 - cosine(set1_avg_vector, self.set_q3_q4_avg_vector)
            if cosine_distance > maximum:
                result_q = "Q3"
        return result_q
