from feature_extraction import (group_tweets, sanitize_data)
from dnn import train_model
from auxiliary import (dict_sort, dict_swap)

import pandas as pd
from pandas.core.frame import (DataFrame, Series)

from interpret.blackbox import (LimeTabular, ShapKernel)
from interpret import show

from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer

import sys

def lime_local_explanation(clf: MLPClassifier, X_train: DataFrame, datapoint: DataFrame, label: Series) -> None:
    """ creates a local post hoc LIME explanation for a black box model, outputs a link to a hosted explanation (NB: the python shell must still be running for the server to still be up!) 
    """

    print("creating LIME explanation of dnn model... ", end="")
    sys.stdout.flush()
    lime_tab = LimeTabular(predict_fn=clf.predict_proba, data=X_train, random_state=1)
    lime_local = lime_tab.explain_local(datapoint, label, name='LIME local explanation')
    print("Done")
    show(lime_local)

def shap_local_explanation(clf: MLPClassifier, X_train: DataFrame, datapoint: DataFrame, label: Series) -> None:
    """ creates a local post hoc SHAP explanation for a black box model, outputs a link to a hosted explanation (NB: the python shell must still be running for the server to still be up!) 
    """

    print("creating SHAP explanation of dnn model... ", end="")
    sys.stdout.flush()
    shap_kern = ShapKernel(predict_fn=clf.predict_proba, data=X_train)
    shap_local = shap_kern.explain_local(datapoint, label, name='SHAP local explanation')
    print("Done")
    show(shap_local)

def tree_classify_explain(tree_model: DecisionTreeClassifier, X: DataFrame, vec: CountVectorizer) -> None:
    """ generates a local explanation of a decision tree by way of decision path. adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html 
    """

    # these objects are required for tree interface
    sample_id = 0 
    node_indicator = tree_model.decision_path(X)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    leaf_id = tree_model.apply(X)
    thresholds = tree_model.tree_.threshold
    features = tree_model.tree_.feature

    # tree interface uses feature numbers internally which we need to convert to feature names
    voc_swapped = dict_swap(vec.vocabulary_)
    feature_name = lambda x: voc_swapped[features[node_id]]

    print('Rules used for prediction in Tree model:')
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            break
        sign = "<=" if  (X[feature_name(node_id)][0] <= thresholds[node_id]) else ">"
        print(f"Decision {node_id} : \"{feature_name(node_id)}\" = {X[feature_name(node_id)][0]} {sign} {round(thresholds[node_id], 4)})")
    proba = tree_model.predict_proba(X)[0]
    pred = "Democrat" if tree_model.predict(X)[0] else "Republican"
    print(f"Tree model prediction: user is a {pred} with {max(proba)*100}% certainty")


def classify_explain_user(vec: CountVectorizer, tree_model: DecisionTreeClassifier, dnn_model: MLPClassifier, X_train: DataFrame, user_file: str) -> None:
    """ classifies a user based on both the tree model and the neural network model, then used various methods for generating explanations
    """

    grouped_file   = "user_grouped.csv"
    sanitized_file = "user_sanitized.csv"

    user_file_path = user_file.split("/")
    if len(user_file_path) > 1:
        base_dir = user_file_path[0]
        user_file = "/".join(user_file_path[1:])
    else:
        base_dir = "./"

    print(f"\nClassifying and explaining classifications for user file: \"{user_file}\"...\n")

    group_tweets(base_dir, user_file, grouped_file)
    sanitize_data(base_dir, grouped_file, sanitized_file)

    data = pd.read_csv(base_dir + sanitized_file)
    data.dropna(inplace=True)

    X = vec.transform(data.CleanTweet).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    X = pd.DataFrame(X, columns=dict_sort(vec.vocabulary_))

    tree_classify_explain(tree_model, X, vec)

    pred = "Democrat" if dnn_model.predict(X)[0] else "Republican"
    print(f"\ndnn prediction: {pred}")
    
    lime_local_explanation(dnn_model, X_train, X, data.BinaryParty)
    shap_local_explanation(dnn_model, X_train, X, data.BinaryParty)



if __name__ == "__main__":
    clf, X_train, X_test, y_train, y_test, vec, y_predict = train_model("archive/grouped_data.csv")

    lime_local_explanation(clf, X_train, X_test[:1], y_test[:1])
    shap_local_explanation(clf, X_train, X_test[:1], y_test[:1])
