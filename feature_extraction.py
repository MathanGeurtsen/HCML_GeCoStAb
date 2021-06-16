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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

"""
TODO: 
* Remove RTs
* What about Emoji's?
"""

Tweet = str

#https://medium.com/verifa/feature-generation-from-tweets-9af0565ad6e6

data = pd.read_csv("archive/NoRTdata.csv")

data.dropna(inplace=True)
data = data.sample(n=80000)

STEMMER = PorterStemmer()
TOKENIZER = RegexpTokenizer(r'\w+')
STOP_WORDS = stopwords.words("english")

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

clean_data: list = []
columns = ['party', 'tweet']

#1 = Democrat, 0 = Republican
data['BinaryParty'] = (data['Party'] == "Democrat").astype(int)
print(data.BinaryParty)


clean_tweets = []
for i in range(len(data)):
    tweet = data.iloc[i]["Tweet"]

    cleanTweet = CleanTweet(tweet)

    clean_tweets.append(cleanTweet)

data['CleanTweet'] = clean_tweets



vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(data.CleanTweet).toarray()

tfidfconverter = TfidfTransformer()

X = tfidfconverter.fit_transform(X).toarray()
Y = data.BinaryParty.to_numpy()

print(len(X), len(Y))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print(y_predict)
print("Accuracy = " + str(accuracy_score(y_true=y_test, y_pred=y_predict)))

