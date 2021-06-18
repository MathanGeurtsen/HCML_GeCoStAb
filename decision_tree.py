#!/usr/bin/env python3
from feature_extraction import extract_features_csv
from sklearn import tree

from auxiliary import print_metrics, plot_tree

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, voc = extract_features_csv(
        "archive/grouped_data.csv", max_features=200
    )

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print_metrics(y_test, y_predict)
    plot_tree(clf, voc)
