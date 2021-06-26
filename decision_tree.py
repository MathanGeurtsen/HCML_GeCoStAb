#!/usr/bin/env python3
from feature_extraction import extract_features_csv
from sklearn import tree

from auxiliary import (print_metrics, plot_tree)

def train_model(data_file, max_features=200):
    print("training tree model... ", end="")
    X_train, X_test, y_train, y_test, voc = extract_features_csv(
        data_file, max_features=200
    )

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    print("Done")

    print_metrics(y_test, y_predict)
    plot_tree(clf, voc)
    return clf, X_train, X_test, y_train, y_test, voc, y_predict


if __name__ == "__main__":
    _ = train_model("archive/grouped_data.csv")
