from feature_extraction import (group_tweets, sanitize_data)

import numpy as np
import pandas as pd

import graphviz

from sklearn import tree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score



def plot_tree(clf, vec):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=list(
            map(lambda x: x[0], sorted(vec.vocabulary_.items(), key=lambda x: x[1]))
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

def tree_classify_explain(tree_model, X):
    node_indicator = tree_model.decision_path(X)
    leaf_id = clf.apply(X_test)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample {id}:\n'.format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

            print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
          "{inequality} {threshold})".format(
              node=node_id,
              sample=sample_id,
              feature=feature[node_id],
              value=X_test[sample_id, feature[node_id]],
              inequality=threshold_sign,
              threshold=threshold[node_id]))

def classify_explain_user(vec, tree_model, dnn_model, X_train, user_file):
    grouped_file   = "user_grouped.csv"
    sanitized_file = "user_sanitized.csv"

    user_file_path = user_file.split("/")
    if len(user_file_path) > 1:
        base_dir = user_file_path[0]
        user_file = "/".join(user_file_path[1:])
    else:
        base_dir = "./"

    group_tweets(base_dir, user_file, grouped_file)
    sanitize_data(base_dir, grouped_file, sanitized_file)

    data = pd.read_csv(base_dir + sanitized_file)
    data.dropna(inplace=True)

    X = vec.transform(data.CleanTweet).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    X = pd.DataFrame(X, columns=vec.vocabulary_.keys())
    tree_model_prediction = tree_model.predict(X)
    dnn_model_prediction  = dnn_model.predict(X)
    # TODO: run exlanation over tree prediction (see https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)

    # TODO: run explanation over dnn prediction with LIME and SHAP
    
