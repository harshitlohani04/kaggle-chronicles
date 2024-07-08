import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import set_config
from custom_transformers import custom_LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

rd = "D:/My Competitions/Binary Classification of Insurance Cross Selling/Data/train"

train_data = pd.read_csv(os.path.join(rd, "train.csv"))

# Sorting the columns into numeric and object data
objCols = train_data.select_dtypes(include = object)
intCols = train_data.select_dtypes(include = np.number)

# Visualizing the data
print(train_data.describe())

# Dropping the unneccessary columns
train_data = train_data.drop(columns=["id", "Driving_License"], axis=1)

objSteps = [("label encoder", custom_LabelEncoder())] # For the obj columns
intSteps = [("scaler", StandardScaler())] # For the integer columns

set_config(display="diagram")

pipe1 = Pipeline(steps=objSteps)
pipe2 = Pipeline(steps=intSteps)



