import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def load_preprocessed_data(filepath):
    X_train, X_test, y_train, y_test = joblib.load(filepath)
    return X_train, X_test, y_train, y_test

def load_trained_model(filepath):
    model = joblib.load(filepath)
    return model

def classificationReport(y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names = ["class 0", "class 1"]))

def confusionMatrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels = ["class 0", "class 1"])
    disp.plot()
    plt.savefig(title + ".png")
    plt.close()

def rocAucCurve(y_test, y_prob, title):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = "ROC Curve (AUC = {:.2f})".format(roc_auc))
    plt.plot([0,1], [0, 1], linestyle = "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig(title + ".png")
    plt.close()

if __name__ == "__main__":
    # load the preprocessed data 
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")

    # evaluate logistic regression 
    logreg = load_trained_model("model_logreg.pkl")
    # prediction
    y_pred_logreg = logreg.predict(X_test)
    y_pred_prob_logreg = logreg.predict_proba(X_test)[:, 1]
    # classification report
    classificationReport(y_test, y_pred_logreg)
    # confusion matrix 
    confusionMatrix(y_test, y_pred_logreg, "Confusion Matrix for Logistic Regression")
    # roc-auc curve
    rocAucCurve(y_test, y_pred_prob_logreg, "ROC-AUC Curve for Logistic Regression")

    # evaluate decision tree classifier 
    dtc = load_trained_model("model_dtc.pkl")
    # prediction
    y_pred_dtc = dtc.predict(X_test)
    y_pred_prob_dtc = dtc.predict_proba(X_test)[:, 1]
    # classification report
    classificationReport(y_test, y_pred_dtc)
    # confusion matrix 
    confusionMatrix(y_test, y_pred_dtc, "Confusion Matrix for Decision Tree Classifier")
    # roc-auc curve
    rocAucCurve(y_test, y_pred_prob_dtc, "ROC-AUC Curve for Decision Tree Classifier")

    # evaluate random forest classifier 
    rfc = load_trained_model("model_rfc.pkl")
    # prediction
    y_pred_rfc = rfc.predict(X_test)
    y_pred_prob_rfc = rfc.predict_proba(X_test)[:, 1]
    # classification report
    classificationReport(y_test, y_pred_rfc)
    # confusion matrix 
    confusionMatrix(y_test, y_pred_rfc, "Confusion Matrix for Random Forest Classifier")
    # roc-auc curve
    rocAucCurve(y_test, y_pred_prob_rfc, "ROC-AUC Curve for Random Forest Classifier")
