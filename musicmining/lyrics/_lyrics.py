import re
from typing import Optional

import nltk
import pandas as pd
from lyricsgenius import Genius
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
    count_vec = CountVectorizer(**kwargs)
    out = count_vec.fit_transform(lyrics)
    return pd.DataFrame(out.toarray(), columns=count_vec.get_feature_names_out())


def to_tfidf_matrix(lyrics: list[str], **kwargs) -> pd.DataFrame:
    tfidf_vec = TfidfVectorizer(**kwargs)
    out = tfidf_vec.fit_transform(lyrics)
    return pd.DataFrame(out.toarray(), columns=tfidf_vec.get_feature_names_out())


def get_lyrics_from_genius(access_token: str, song: str, artist: str) -> str | None:
    genius = Genius(access_token)
    song_obj = genius.search_song(song, artist)

    if song_obj is None:
        return None

    return song_obj.lyrics


def clean_genius_lyrics(lyrics: str, stopwords: Optional[list[str]] = None):
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
