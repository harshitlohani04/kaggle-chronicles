from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))