#!/usr/bin/env python

import sys
import dnn
import decision_tree

from post_proc_explanation import (lime_local_explanation, shap_local_explanation, classify_explain_user)
from feature_extraction import (group_tweets, sanitize_data)

from time import sleep

UNPROCESSED_DATA_FILE = "ExtractedTweets.csv"
GROUPED_DATA_FILE     = "grouped_data.csv"
SANITISED_DATA_FILE   = "grouped_data_sanitised.csv"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help" "-?"]:
        print(
            "Usage: HCML_GeCoStAb "
            + "[<data_directory>] [<user_file>] [<preprocess_data>]",
            end="\n\n",
        )
        print(
            "Classifies partisan tweets. All arguments optional. Trains an explainable model and a black box model on data in <data-directory> (defaults to \"archive\", then explains a local classification on <user-file> (defaults to \"example-user.csv\". performs preprocessing if <preprocess_data> is given and set to True. (defaults to True)"
        )
        sys.exit()

    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
        if data_directory[len(data_directory) -1] == "\\":
            data_directory = data_directory[:-1]
            data_directory = data_directory + "/"
    else:
        data_directory = "./archive/"
            
    if len(sys.argv) > 2:
        user_file = sys.argv[2]
    else:
        user_file = "example-user.csv"

    preprocess_data = False if ( len(sys.argv) == 4 and sys.argv[3].lower() not in ["y", "yes", "true", "1"] ) else True

    if preprocess_data:
        print("preprocessing data... ", end="")
        sys.stdout.flush()
        group_tweets(data_directory, UNPROCESSED_DATA_FILE, GROUPED_DATA_FILE)
        sanitize_data(data_directory, GROUPED_DATA_FILE, SANITISED_DATA_FILE)
        print("Done")

    data_file = data_directory + SANITISED_DATA_FILE

    dnn_tuple = dnn.train_model(data_file)
    dnn_model, X_train, X_test, y_train, y_test, voc, y_predict = dnn_tuple

    tree_tuple = decision_tree.train_model(data_file)
    tree_model = tree_tuple[0]

    vec = tree_tuple[5]
    X_train = tree_tuple[1]
    
    classify_explain_user(vec, tree_model, dnn_model, X_train, user_file)
