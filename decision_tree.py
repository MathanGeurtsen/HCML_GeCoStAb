import pandas as pd
from feature_extraction import extract_features_csv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree

from auxiliary import print_metrics

if __name__ == "__main__": 
    X_train, X_test, y_train, y_test = extract_features_csv("archive/data.csv", max_features=200)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print_metrics(y_test, y_predict)
