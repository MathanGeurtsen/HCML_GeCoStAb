from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
def print_metrics(y_test:np.array , y_predict: np.array) -> None:
    print("Accuracy = " + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
    print("Precision = " + str(precision_score(y_true=y_test, y_pred=y_predict)))
    print("Recall = " + str(recall_score(y_true=y_test, y_pred=y_predict)))


