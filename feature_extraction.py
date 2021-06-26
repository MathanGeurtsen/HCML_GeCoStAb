import nltk
from nltk import (tokenize, ngrams)

from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from numpy.core.defchararray import index
import numpy as np
import pandas as pd

import re
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from typing import List, Tuple

from auxiliary import dict_sort

"""
TODO: 
* Remove RTs
* What about Emoji's?
* https://medium.com/verifa/feature-generation-from-tweets-9af0565ad6e6
"""

Tweet = str
STEMMER = PorterStemmer()
TOKENIZER = RegexpTokenizer(r"\w+")
STOP_WORDS = stopwords.words("english")

"""Some helper fuctinos"""


def CleanTweet(tweet: Tweet) -> Tweet:
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", "", tweet)  # remove URLs
    tweet = re.sub("@[^\s]+", "", tweet)  # remove usernames
    tweet = re.sub(r"#([^\s]+)", r"\1", tweet)  # remove the # in #hashtag
    tweet = TOKENIZER.tokenize(
        tweet
    )  # remove repeated characters (helloooooooo into hello)
    tweet = [
        STEMMER.stem(word) for word in tweet if (word not in STOP_WORDS)
    ]  # stemming (jumping -> jump)
    tweet = " ".join(tweet)
    return tweet


def compute_ngram_frequencies(tweet: Tweet) -> dict:
    n_gram_frequencies: dict = {n + 1: defaultdict(lambda: 0) for n in range(1)}
    words = tweet.split(" ")

    for n_gram_size in n_gram_frequencies:
        for word_index in range(len(words) - n_gram_size + 1):
            n_gram = tuple(words[word_index : word_index + n_gram_size])
            n_gram_frequencies[n_gram_size][n_gram] = (
                n_gram_frequencies[n_gram_size][n_gram] + 1
            )

    return n_gram_frequencies


def extract_features(data: pd.DataFrame, max_features: int = 150, seed:int = 1) -> Tuple:
    vectorizer = CountVectorizer(max_features=max_features, min_df=5, max_df=0.7)

    X = vectorizer.fit_transform(data.CleanTweet).toarray()

    voc_sorted = dict_sort(vectorizer.vocabulary_)
    
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    Y = data.BinaryParty.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X_train = pd.DataFrame(X_train, columns=voc_sorted.keys())

    return (X_train, X_test, y_train, y_test, vectorizer)


def extract_features_csv(file_name: str, max_features: int = 150) -> Tuple:
    """ facade over extract_features
    """

    data = pd.read_csv(file_name)
    data.dropna(inplace=True)
    return extract_features(data, max_features=max_features)

def group_tweets(target_dir:str, source_file_name:str, grouped_file_name:str) -> None:
    """ takes in the extracted tweets and groups by user. 
    """

    df = pd.read_csv(target_dir + source_file_name)
    df.dropna(inplace=True)
    df = df.groupby('Handle').agg(lambda x: " ".join(list(set(x.tolist()))))
    df.to_csv(target_dir + grouped_file_name)

def sanitize_data(target_dir:str, source_file: str, sanitized_file:str) -> None:
    """ cleans tweets using a standard tokenizer and adds binary labels to the dataset. 
    """

    data = pd.read_csv(target_dir + source_file )    
    data.dropna(inplace=True)

    # Add clean tweets, binary party to data
    clean_tweets = []
    for i in range(len(data)):
        tweet = data.iloc[i]["Tweet"]
        cleanTweet = CleanTweet(tweet)
        clean_tweets.append(cleanTweet)

    data["CleanTweet"] = clean_tweets
    data["BinaryParty"] = (data["Party"] == "Democrat").astype(int)


    data.to_csv(target_dir + sanitized_file)

if __name__ == "__main__":

    """Data to sanitize"""
    path = "archive/grouped_data.csv"

    """Read original data"""
    data = pd.read_csv(path)
    data.dropna(inplace=True)

    """Add clean tweets, binary party to data"""
    clean_tweets = []
    for i in range(len(data)):
        tweet = data.iloc[i]["Tweet"]
        cleanTweet = CleanTweet(tweet)
        clean_tweets.append(cleanTweet)

    data["CleanTweet"] = clean_tweets
    data["BinaryParty"] = (data["Party"] == "Democrat").astype(int)

    """Save new data"""
    data.to_csv(path)
