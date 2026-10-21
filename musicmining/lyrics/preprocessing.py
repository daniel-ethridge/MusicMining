"""Functionality for retrieving lyrics from the Genius API and processing them"""

import re
from typing import Optional

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _convert_pos(pos: str):
    if pos.startswith("N"):
        return "n"
    if pos.startswith("V"):
        return "v"
    if pos.startswith("J"):
        return "a"
    if pos.startswith("R"):
        "r"
    if pos.startswith("J"):
        "a"

    return "n"


def _contraction_to_words(contraction: str):
    contraction = contraction.lower()

    contract_dict = {"i'm": "i am", "you're": "you are", "we're": "we are", "i'll": "i will", "they're": "they are"}

    if contraction in contract_dict:
        words = contract_dict[contraction]
        return words

    return contraction


def to_count_matrix(lyrics: list[str], **kwargs) -> pd.DataFrame:
    """
    Create a CountVectorizer matrix.

    Parameters
    ----------
    lyrics : list[str]
        A list of song lyrics. Each song's lyrics should be one string in the list.
    **kwargs
        Keyword arguments passed to sklearn.feature_extraction.text.CountVectorizer

    Returns
    -------
    pd.DataFrame
        A dataframe of word counts for each song's lyrics.
    """
    count_vec = CountVectorizer(**kwargs)
    out = count_vec.fit_transform(lyrics)
    return pd.DataFrame(out.toarray(), columns=count_vec.get_feature_names_out())


def to_tfidf_matrix(lyrics: list[str], **kwargs) -> pd.DataFrame:
    """
    Create a TfidfVectorizer matrix.

    Parameters
    ----------
    lyrics : list[str]
        A list of song lyrics. Each song's lyrics should be one string in the list.
    **kwargs
        Keyword arguments passed to sklearn.feature_extraction.text.TfidfVectorizer

    Returns
    -------
    pd.DataFrame
        A dataframe of normalized frequency values for each song's lyrics.
    """
    tfidf_vec = TfidfVectorizer(**kwargs)
    out = tfidf_vec.fit_transform(lyrics)
    return pd.DataFrame(out.toarray(), columns=tfidf_vec.get_feature_names_out())


def clean_genius_lyrics(lyrics: str, stopwords: Optional[list[str]] = None) -> str:
    """
    Clean the raw lyrics returned from the Genius API.

    Removes bracketed words (e.g. [Intro], [Chorus]), numbers, non-words/spaces (i.e. punctuation), extra white space
    capital letters, and stopwords (if provided). Also converts select contractions to actual words and lemmatizes each
    word.

    Parameters
    ----------
    lyrics : str
        The raw lyrics returned from the Genius API
    stopwords : Optional[list[str]], optional, default = None
        Meaningless words (e.g. a, but, or, the, etc.) to remove.

    Returns
    -------
    str
        The processed lyrics
    """
    # remove words in brackets, numbers, and non-words/spaces except apostrophes
    lyrics = re.sub(r"\[(.*?)\]|([0-9])|(\n)|[^\w\s\']", "", lyrics)

    # remove added white space
    lyrics = re.sub(r"\s{2,}", " ", lyrics)

    # remove capitalization
    lyrics = lyrics.lower()

    # Remove stopwords if applicable
    lyrics = lyrics.split()
    if stopwords is not None:
        lyrics = [word for word in lyrics if word not in stopwords]

    # replace contractions with words
    lyrics = [_contraction_to_words(word) for word in lyrics]

    # Get rid of multiword list entries
    lyrics = " ".join(lyrics)
    lyrics = lyrics.split()

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lyrics_w_pos = nltk.pos_tag(lyrics)
    lyrics = [lemmatizer.lemmatize(tagged_word[0], _convert_pos(tagged_word[1])) for tagged_word in lyrics_w_pos]

    # remove apostrophes left in earlier
    lyrics = " ".join(lyrics)
    lyrics = re.sub(r"(\')", "", lyrics)

    return lyrics
