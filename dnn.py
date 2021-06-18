import pandas as pd
from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel
from interpret import show

data = pd.read_csv("archive/grouped_data.csv")
data.dropna(inplace=True)
x_train, x_test, y_train, y_test, _ = extract_features(data)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=10000)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

print(y_predict)
print("Accuracy = "  + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
print("Precision = " + str(precision_score(y_true=y_test, y_pred=y_predict)))
print("Recall = "    + str(recall_score(y_true=y_test, y_pred=y_predict)))



lime1 = LimeTabular(predict_fn=clf.predict_proba, data=x_train, random_state=1)
shap1 = ShapKernel(predict_fn=clf.predict_proba, data=x_train)

lime_local1_1 = lime1.explain_local(list(x_test[:1]), list(y_test[:1]), name='LIME_MLP1_1')
show(lime_local1_1)