#!/usr/bin/env python3
from feature_extraction import extract_features_csv
from sklearn import tree

from auxiliary import (print_metrics, plot_tree)

import sys

def train_model(data_file: str, max_features=200) -> None:
    """ Train a decision tree model based on a given datafile. 
    """

    print("training tree model... ", end="")
    sys.stdout.flush()
    X_train, X_test, y_train, y_test, vec = extract_features_csv(
        data_file, max_features=200
    )

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    print("Done")

    print_metrics(y_test, y_predict)
    plot_tree(clf, vec)
    return clf, X_train, X_test, y_train, y_test, vec, y_predict


if __name__ == "__main__":
    _ = train_model("archive/grouped_data.csv")
