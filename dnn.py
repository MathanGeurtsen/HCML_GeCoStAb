import pandas as pd
from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("archive/data.csv")
data.dropna(inplace=True)
x_train, x_test, y_train, y_test = extract_features(data)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=100)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

print(y_predict)
print("Accuracy = " + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
print("Precision = " + str(precision_score(y_true=y_test, y_pred=y_predict)))
print("Recall = " + str(recall_score(y_true=y_test, y_pred=y_predict)))