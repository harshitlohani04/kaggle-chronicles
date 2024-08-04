import numpy as np
import pandas as pd
import os

rd = "D:/kaggle-chronicles/bpopm/"

train_data = pd.read_csv(os.path.join(rd, "Data/playground-series-s4e8/train.csv"))

print(train_data.describe())
print(train_data.isnull().sum())
