# Model Training and evaluation
from pipeline import X, y, ct
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
import optuna
from sklearn.metrics import roc_auc_score
import xgboost as xgb

PARAMS = {
    "max_depth": [2, 32],
    "eta": [0.01, 0.1],
    "subsample": [0.5, 1],
    "colsample_bytree": [0.5, 1],
    "lambda": [0.1, 10],
    "alpha": [0.1, 10]
}


# Hyper-Parameter tuning using Optuna
def objective(trial, X, y, testx, testy):
    max_depth = trial.suggest_int("max_depth", PARAMS["max_depth"][0], PARAMS["max_depth"][1])
    eta = trial.suggest_float("eta", PARAMS["eta"][0], PARAMS["eta"][1])
    subsample = trial.suggest_float("subsample", PARAMS["subsample"][0], PARAMS["subsample"][1])
    colsample_bytree = trial.suggest_float("colsample_bytree", PARAMS["colsample_bytree"][0], PARAMS["colsample_bytree"][1])
    lambda1 = trial.suggest_float("lambda", PARAMS["lambda"][0], PARAMS["lambda"][1])
    alpha = trial.suggest_float("alpha", PARAMS["alpha"][0], PARAMS["alpha"][1])

    current_params = {
        "max_depth": max_depth,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "binary:logistic",
        "lambda": lambda1,
        "alpha": alpha
    }

    batch_size = 100000

    rounds = trial.suggest_int("rounds", 50, 200)

    n_batches = int(np.ceil(X.shape[0] / batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, X.shape[0])
        y_batch = y[start:end]
        X_batch = xgb.DMatrix(X[start:end], label = y_batch)
        model = xgb.train(current_params, X_batch, rounds)
    pred_val = model.predict(xgb.DMatrix(testx, label = testy))
    score = roc_auc_score(testy, pred_val)

    return score


def fittingModel(x, y, params, rounds):
    batch_size = 100000
    n_batches = int(np.ceil(x.shape[0] / batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, x.shape[0])
        y_batch = y[start:end]
        X_batch = xgb.DMatrix(x[start:end], label = y_batch)
        model = xgb.train(params, X_batch, rounds)
    return model


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

ct.fit(X_train)
Xnew = ct.transform(X_train)
newXval = ct.transform(X_val)

study = optuna.create_study(direction = "maximize")
study.optimize(lambda trial : objective(trial, Xnew, y_train, newXval, y_val), n_trials = 40)

# Retrieving the the best parameters
best_params = study.best_trial.params
print(best_params)

bestParams = {
    "max_depth" : best_params["max_depth"],
    "eta" : best_params["eta"],
    "subsample" : best_params["subsample"],
    "colsample_bytree" : best_params["colsample_bytree"],
    "objective" : "binary:logistic",
    "lambda": best_params["lambda"],
    "alpha": best_params["alpha"]
}
rounds = best_params["rounds"]

finalmodel = fittingModel(Xnew, y_train, bestParams, rounds)
y_pred = finalmodel.predict(xgb.DMatrix(newXval, label = y_val))
print(y_pred)

with open("bcics/Code/bestModel.pkl", "wb") as f:
    pkl.dump(finalmodel, f)

