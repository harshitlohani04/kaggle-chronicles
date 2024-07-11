import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from custom_transformers import custom_LabelEncoder, ColumnDropper
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

rd = "D:/My Competitions/Binary Classification of Insurance Cross Selling/Data"

train_data = pd.read_csv(os.path.join(rd, "train/train.csv"))

cd = ColumnDropper()
cd.fit(train_data)
train_data = cd.transform(train_data)

X = train_data.drop(columns=["Response"], axis = 1)
y = train_data["Response"]

# Sorting the columns into numeric and object data
objCols = X.select_dtypes(include = object).columns
intCols = X.select_dtypes(include = np.number).columns

objSteps = [("label encoder", custom_LabelEncoder())] # For the obj columns
intSteps = [("scaler", StandardScaler())] # For the integer columns

pipe1 = Pipeline(steps=objSteps)
pipe2 = Pipeline(steps=intSteps)

ct = ColumnTransformer(transformers=[("obj cols", pipe1, objCols), ("int cols", pipe2, intCols)])


