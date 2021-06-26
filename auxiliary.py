from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import graphviz
from sklearn import tree


def plot_tree(clf, voc):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=list(
            map(lambda x: x[0], sorted(voc.items(), key=lambda x: x[1]))
        ),
        class_names=["rep", "dem"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    print("visualized tree to iris.pdf") 
    

def print_metrics(y_test: np.array, y_predict: np.array) -> None:
    print("Accuracy = " + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
    print("Precision = " + str(precision_score(y_true=y_test, y_pred=y_predict)))
    print("Recall = " + str(recall_score(y_true=y_test, y_pred=y_predict)), end="\n\n")

def classify_user():
    raise NotImplementedError
