import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from custom_transformers import custom_LabelEncoder, ColumnDropper
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

rd = "D:/My Competitions/Binary Classification of Insurance Cross Selling/Data"

train_data = pd.read_csv(os.path.join(rd, "train/train.csv"))

# Sorting the columns into numeric and object data
objCols = train_data.select_dtypes(include = object)
intCols = train_data.select_dtypes(include = np.number)

objSteps = [("label encoder", custom_LabelEncoder())] # For the obj columns
intSteps = [("scaler", StandardScaler())] # For the integer columns

pipe1 = Pipeline(steps=objSteps)
pipe2 = Pipeline(steps=intSteps)

ct = ColumnTransformer(transformers=[("obj cols", pipe1, objCols), ("int cols", pipe2, intCols)])

finalPipeline = make_pipeline(ColumnDropper, ct)

X = train_data.drop(columns=["Response"], axis = 1)
y = train_data["Response"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)



