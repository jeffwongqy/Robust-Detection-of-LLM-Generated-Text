import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib

def load_preprocessed_data(filepath):
    X_train, X_test, y_train, y_test = joblib.load(filepath)
    return X_train, X_test, y_train, y_test

def model_training(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    model = GridSearchCV(model, 
                         param_grid = param_grid, 
                         cv = cv, 
                         scoring = "accuracy", 
                         refit = "accuracy", 
                         return_train_score = True, 
                         verbose = 3, 
                         n_jobs = 1)
    model.fit(X, y)

    print(model.best_params_)
    print(model.best_score_)
    print()
    return model

def save_model(model, filepath):
    return joblib.dump(model, filepath)



if __name__ == "__main__":
    # load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")

    # for logistic regression 
    logreg_param = { 'C': [0.01, 0.1, 1, 10], 
                    'max_iter': [100, 200, 300]}
    model_logreg = model_training(LogisticRegression(random_state = 42), X_train, y_train, logreg_param)

    # for decision trees classifier
    dtc_param = {'max_depth':[3, 4, 5], 
                 'min_samples_leaf': [3, 4, 5], 
                 'min_samples_split': [3, 4, 5]}
    model_dtc = model_training(DecisionTreeClassifier(random_state = 42), X_train, y_train, dtc_param)

    # for random forest classifier
    rfc_param = {'n_estimators': [20, 30, 40, 50], 
                 'max_depth':[3, 4, 5], 
                 'min_samples_leaf': [3, 4, 5], 
                 'min_samples_split': [3, 4, 5]}
    model_rfc = model_training(RandomForestClassifier(random_state = 42), X_train, y_train, rfc_param)

    # save models
    save_model(model_logreg, "model_logreg.pkl")
    save_model(model_dtc, "model_dtc.pkl")
    save_model(model_rfc, "model_rfc.pkl")