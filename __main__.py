#!/usr/bin/env python

import sys
import dnn
import decision_tree

from post_proc_explanation import (lime_local_explanation, shap_local_explanation)
from feature_extraction import (group_tweets, sanitize_data)

from auxiliary import classify_user

from time import sleep

UNPROCESSED_DATA_FILE = "ExtractedTweets.csv"
GROUPED_DATA_FILE     = "grouped_data.csv"
SANITISED_DATA_FILE   = "grouped_data_sanitised.csv"

if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print(
            "Usage: HCML_GeCoStAb "
            + "<data_directory> <user_file> [<preprocess_data>]",
            end="\n\n",
        )
        print(
            "Classifies partisan tweets. Trains an explainable model and a black box model on data in <data-directory>, then explains a local classification on <user-file>. Assumed data is already preprocessed, can performs preprocessing if <preprocess_data> is given and set to True."
        )
        sys.exit()

    # data_file = "grouped_data.csv"
    # user_file = "example-user.csv"

    data_directory = sys.argv[1]
    if data_directory[len(data_directory) -1] == "\\":
        data_directory = data_directory[:-1]
    data_directory = data_directory + "/"
        
    user_file = sys.argv[2]

    preprocess_data = True if ( len(sys.argv) == 4 and sys.argv[3].lower() in ["y", "yes", "true", "1"] ) else False

    if preprocess_data:
        print("preprocessing data... ", end="")
        group_tweets(data_directory, UNPROCESSED_DATA_FILE, GROUPED_DATA_FILE)
        sleep(1)
        sanitize_data(data_directory, GROUPED_DATA_FILE, SANITISED_DATA_FILE)
        sleep(1)
        print("Done")

    data_file = data_directory + SANITISED_DATA_FILE

    ###########

    dnn_tuple = dnn.train_model(data_file)
    dnn_model, X_train, X_test, y_train, y_test, voc, y_predict = dnn_tuple

    tree_tuple = decision_tree.train_model(data_file)
    tree_model = tree_tuple[0]
    
    classify_user()

    # lime_local_explanation(clf, X_train, X_test[:1], y_test[:1])
    # shap_local_explanation(clf, X_train, X_test[:1], y_test[:1])


    
    

    
    

    
