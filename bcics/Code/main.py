# Model Training and evaluation
from pipeline import X, y, ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rmse import rmse
import numpy as np


def findBestModel(x, testx, y, testy, params):
    minimum = 100
    batch_size = 100000  # Adjust based on memory capacity
    n_batches = int(np.ceil(X_train.shape[0] / batch_size))
    for estimator, depth in params:
        print("hello")
        classifier = RandomForestClassifier(n_estimators = estimator, max_depth = depth, n_jobs = -1)
        classifier.fit(x, y)
        y_pred = classifier.predict(testx)
        error = rmse(y_pred, testy)
        if error<minimum:
            minimum = error
            best_model = classifier
    return best_model, minimum


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

ct.fit(X_train)
Xnew = ct.transform(X_train)
newXval = ct.transform(X_val)

param = [(100, 5), (200, 5), (500, 5)]
model, minError = findBestModel(Xnew, newXval, y_train, y_val, param)
print(minError)

