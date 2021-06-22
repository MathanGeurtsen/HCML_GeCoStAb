#!/usr/bin/env python

from feature_extraction import extract_features_csv
from sklearn.neural_network import MLPClassifier

from auxiliary import print_metrics

def run_model():
    X_train, X_test, y_train, y_test, voc = extract_features_csv("archive/grouped_data.csv", max_features=200)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=10000)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    print_metrics(y_test, y_predict)
    return clf, X_train, X_test, y_train, y_test, voc, y_predict

if __name__ == "__main__":
    _ = run_model()

