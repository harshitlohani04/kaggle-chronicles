# Model Training and evaluation
from pipeline import X, y, ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rmse import rmse
import numpy as np
import pickle as pkl
import optuna
from sklearn.metrics import roc_auc_score


# Hyper-Parameter tuning using Optuna
def objective(trial, X, y, testx, testy):
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth", 2, 30)
    batch_size = 100000
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, X.shape[0])
        X_batch = X[start:end]
        y_batch = y[start:end]
        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, n_jobs = -1)
        model.fit(X_batch, y_batch)
    pred_val = model.predict(testx)
    score = roc_auc_score(testy, pred_val)

    return score


def fittingModel(x, y, model):
    batch_size = 100000
    n_batches = int(np.ceil(x.shape[0] / batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, x.shape[0])
        X_batch = x[start:end]
        y_batch = y[start:end]
        model.fit(X_batch, y_batch)
    return model


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

ct.fit(X_train)
Xnew = ct.transform(X_train)
newXval = ct.transform(X_val)

study = optuna.create_study(direction = "maximize")
study.optimize(lambda trial : objective(trial, Xnew, y_train, newXval, y_val), n_trials = 30)

# Retrieving the the best parameters
best_params = study.best_trial.params
print(best_params)

model = RandomForestClassifier(n_estimators = best_params["n_estimators"], max_depth = best_params["max_depth"], n_jobs = -1, random_state = 42)
finalmodel = fittingModel(Xnew, y_train, model)
y_pred = finalmodel.predict(newXval)
print(y_pred)

with open("bcics/Code/bestModel.pkl", "wb") as f:
    pkl.dump(finalmodel, f)

