import pandas as pd
from interpret.blackbox import (LimeTabular, ShapKernel)
from interpret import show



from dnn import train_model

def lime_local_explanation(clf, X_train, datapoint, label):
    print("creating LIME explanation...")
    lime_tab = LimeTabular(predict_fn=clf.predict_proba, data=X_train, random_state=1)
    lime_local = lime_tab.explain_local(datapoint, label, name='LIME local explanation')
    show(lime_local)

def shap_local_explanation(clf, X_train, datapoint, label):
    print("creating SHAP explanation...")
    shap_kern = ShapKernel(predict_fn=clf.predict_proba, data=X_train)
    shap_local = shap_kern.explain_local(datapoint, label, name='SHAP local explanation')
    show(shap_local)


if __name__ == "__main__":
    clf, X_train, X_test, y_train, y_test, vec, y_predict = train_model("archive/grouped_data.csv")

    lime_local_explanation(clf, X_train, X_test[:1], y_test[:1])
    shap_local_explanation(clf, X_train, X_test[:1], y_test[:1])
