#!/usr/bin/env python

from feature_extraction import extract_features_csv
from sklearn.neural_network import MLPClassifier

from auxiliary import print_metrics

from typing import Tuple

def train_model(data_file: str, max_features: int=200, max_iter: int=10000) -> Tuple:
    print("training dnn model... ", end="")
    X_train, X_test, y_train, y_test, voc = extract_features_csv(data_file, max_features=max_features)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=max_iter)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    print("Done")

    print_metrics(y_test, y_predict)

    return clf, X_train, X_test, y_train, y_test, voc, y_predict

if __name__ == "__main__":
    _ = train_model("archive/grouped_data.csv")

