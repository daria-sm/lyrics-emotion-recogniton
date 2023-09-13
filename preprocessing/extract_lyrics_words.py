import re

import nltk
import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import string
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_lg")


def beautify(lyrics):
    lyrics = lyrics.lower()
    lyrics = lyrics.replace("<br>", " ")

    lyrics = BeautifulSoup(lyrics, 'html.parser').get_text()
    return lyrics


def extract_adjectives(lyrics, pos_tags=None):
    if pos_tags is None:
        pos_tags = ["ADJ"]
    lyrics = lyrics.lower()
    # Remove stopwords
    sentences = extract_lyrics_sentences(lyrics)
    relevant_words = []
    # Input sentence
    for sentence in sentences:
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ in pos_tags:
                relevant_words.append(token.text.lower())
    return relevant_words


def extract_adverbs(lyrics):
    return extract_adjectives(lyrics, ["ADV"])


def extract_words(lyrics):
    lyrics = beautify(lyrics)
    tokens = word_tokenize(lyrics)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]
    cleaned_words = [re.sub(r'[^A-Za-z]', '', word) for word in filtered_words]
    return [word for word in cleaned_words if word != ""]


# read lexicon_nrc_vad
lexicon_path = "../data/processed/NRC-VAD-Lexicon.csv"
columns = ["word", "Valence", "Arousal", "Dominance"]
nrc_vad_df = pandas.read_csv(lexicon_path, names=columns, sep='\t')


def extract_words_from_lexicon(lyrics):
    lyrics = beautify(lyrics)
    tokens = word_tokenize(lyrics)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]
    lexicon_words = [word for word in filtered_words if nrc_vad_df['word'].isin([word]).any()]
    return lexicon_words


def extract_lyrics_sentences(lyrics: str):
    lyrics = lyrics.replace("<br>", "\n")
    lyrics = BeautifulSoup(lyrics, 'html.parser').get_text()
    sentences = [sentence.strip() for sentence in re.split(r'[\r\n]+', lyrics) if sentence.strip()]
    return sentences


def extract_lyrics_sentences_tokenized(lyrics: str):
    sentences = extract_lyrics_sentences(lyrics)
    # Tokenize each sentence into words, excluding punctuation
    tokenized_sentences = [nltk.word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation))) for
                           sentence in sentences]
    return remove_stop_words(tokenized_sentences)


def remove_stop_words(tokenized_sentences):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in tokenized_sentences:
        filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences
