import pandas as pd
import spacy
from scipy.spatial.distance import cosine

from evaluation.Word2Vec import Word2Vec

from collections import Counter

nlp = spacy.load("en_core_web_sm")
allowed_pos_tags = {"ADJ", "NOUN", "VERB"}


class Word2VecBOW(Word2Vec):
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
        # Bag of Words representation (word counts)
        doc = nlp(words)

        # Initialize a Counter to count word frequencies
        word_counts = Counter()

        # Iterate through tokens and count words remove stop words
        for token in doc:
            if not token.is_punct and not token.is_space and not token.is_stop and token.pos_ in allowed_pos_tags:
                word_counts[token.text.lower()] += 1

        most_common_words = [word for word, _ in word_counts.most_common(5) if word in self.model]
        most_common_word_vector = {word: self.model[word] for word in most_common_words}
        similarity_dict = {}
        quadrants = []
        for word in most_common_words:
            similarity_dict["Q1"] = sum([1 - cosine(most_common_word_vector[word], tag) for tag in self.q1_vector])
            similarity_dict["Q2"] = sum([1 - cosine(most_common_word_vector[word], tag) for tag in self.q2_vector])
            similarity_dict["Q3"] = sum([1 - cosine(most_common_word_vector[word], tag) for tag in self.q3_vector])
            similarity_dict["Q4"] = sum([1 - cosine(most_common_word_vector[word], tag) for tag in self.q4_vector])
            quadrants.append(max(similarity_dict, key=similarity_dict.get))

        quadrants_counter = Counter(quadrants)

        # Find the most common string
        if quadrants_counter.most_common(1):
            return quadrants_counter.most_common(1)[0][0]
        print(quadrants)
        return None
