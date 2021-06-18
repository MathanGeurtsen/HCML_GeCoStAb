from nltk import tokenize
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

import re
from collections import defaultdict
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from typing import List, Tuple

"""
TODO: 
* Remove RTs
* What about Emoji's?
* https://medium.com/verifa/feature-generation-from-tweets-9af0565ad6e6
"""

Tweet = str
STEMMER = PorterStemmer()
TOKENIZER = RegexpTokenizer(r'\w+')
STOP_WORDS = stopwords.words("english")

"""Some helper fuctinos"""
def CleanTweet(tweet: Tweet) -> Tweet:
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = TOKENIZER.tokenize(tweet) # remove repeated characters (helloooooooo into hello)
    tweet = [ STEMMER.stem(word) for word in tweet if (word not in STOP_WORDS) ] # stemming (jumping -> jump)
    tweet = ' '.join(tweet)
    return tweet

def compute_ngram_frequencies(tweet: Tweet) -> dict:
    n_gram_frequencies: dict = { n + 1: defaultdict(lambda: 0) for n in range(1) }
    words = tweet.split(" ")

    for n_gram_size in n_gram_frequencies:
        for word_index in range(len(words) - n_gram_size + 1):
            n_gram = tuple(words[word_index:word_index + n_gram_size])
            n_gram_frequencies[n_gram_size][n_gram] = n_gram_frequencies[n_gram_size][n_gram] + 1
    
    return n_gram_frequencies

def extract_features(data: pd.DataFrame, max_features: int = 2000) -> Tuple:
    vectorizer = CountVectorizer(max_features=max_features, min_df=5, max_df=0.7)

    X = vectorizer.fit_transform(data.CleanTweet).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    Y = data.BinaryParty.to_numpy()

    return train_test_split(X, Y, test_size=0.2)

def extract_features_csv(file_name: str, max_features:int =150) -> Tuple:
    data = pd.read_csv(file_name)
    data.dropna(inplace=True)
    return extract_features(data, max_features=max_features)


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

    data['CleanTweet'] = clean_tweets
    data['BinaryParty'] = (data['Party'] == "Democrat").astype(int)

    """Save new data"""
    data.to_csv(path)
