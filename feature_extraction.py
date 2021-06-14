import numpy as np
from numpy.core.defchararray import index
import pandas as pd

#https://medium.com/verifa/feature-generation-from-tweets-9af0565ad6e6

data = pd.read_csv("archive\ExtractedTweets.csv")
print(data.head())

data.dropna(inplace=True)
data.to_csv("archive\ExtractedTweets.txt", header=None, index=None, sep='\t')