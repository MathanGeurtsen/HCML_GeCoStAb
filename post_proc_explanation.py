from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel
from interpret import show


from dnn import run_model

def lime_local_explanation(clf, X_train, datapoint, label):
    lime_tab = LimeTabular(predict_fn=clf.predict_proba, data=X_train, random_state=1)
    lime_local = lime_tab.explain_local(datapoint, label, name='LIME local explanation')
    show(lime_local)

def shap_local_explanation(clf, X_train, datapoint, label):
    shap_kern = ShapKernel(predict_fn=clf.predict_proba, data=X_train)
    shap_local = shap_kern.explain_local(datapoint, label, name='SHAP local explanation')
    show(shap_local)


if __name__ == "__main__":
    print("training model...")
    clf, X_train, X_test, y_train, y_test, voc, y_predict = run_model()

    print("creating local explanations...")
    lime_local_explanation(clf, X_train, X_test[:1], y_test[:1])
    shap_local_explanation(clf, X_train, X_test[:1], y_test[:1])
######################
# FROM ASSIGNMENT \/ #
######################


# # Lime explanations for the single hidden layer NN model
# lime_mlp1 = LimeTabular(predict_fn=mlp_1L.predict, data=X_train_class, random_state=1)

# lime_local_mlp1 = lime_mlp1.explain_local(X_dev_class.loc[randomList, :], y_dev_class.loc[randomList, :], name='LIME_mlp1')

# show(lime_local_mlp1)


# # The SHAP explanations for the single hidden layer NN model
# shap_mlp1 = ShapKernel(predict_fn=mlp_1L.predict, data=background_val, feature_names=X_train_class.columns)
# shap_local_mlp1 = shap_mlp1.explain_local(X_dev_class.loc[randomList, :], y_dev_class.loc[randomList, :], name='SHAP_mlp1')
# show(shap_local_mlp1)
