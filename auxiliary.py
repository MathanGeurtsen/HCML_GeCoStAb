import numpy as np
import pandas as pd

import graphviz

from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

from typing import Dict

def dict_sort(in_dict: Dict) -> Dict:
    """ convenience function to sort a dictionary by value. adapted from stackoverflow.com/a/613218
    """
    return {k: v for k, v in sorted(in_dict.items(), key=lambda item: item[1])}

def dict_swap(in_dict: Dict) -> Dict:
    """ convenience function to swap keys and values in a dictionary, assumes both are unique.
    """
    return {x[1]:x[0] for x in in_dict.items()}

def plot_tree(clf: DecisionTreeClassifier, vec: CountVectorizer) -> None:
    """ Plots the decision tree with graphviz, output is a pdf file. """

    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=dict_swap(vec.vocabulary_),
        class_names=["rep", "dem"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    print("visualized tree to iris.pdf") 

def print_metrics(y_test: np.array, y_predict: np.array) -> None:
    """ calculates and ouputs standard model evaluation metrics.
    """

    print("Accuracy = " + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
    print("Precision = " + str(precision_score(y_true=y_test, y_pred=y_predict)))
    print("Recall = " + str(recall_score(y_true=y_test, y_pred=y_predict)), end="\n\n")

    
