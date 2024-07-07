import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

rd = "D:/My Competitions/Binary Classification of Insurance Cross Selling/Data/train"

train_data = pd.read_csv(os.path.join(rd, "train.csv"))

# Sorting the columns into numeric and object data
objCols = train_data.select_dtypes(include = object)
intCols = train_data.select_dtypes(include = np.number)

# Visualizing the data
print(train_data.describe())

# Dropping the unneccessary columns
train_data = train_data.drop(columns=["id", "Driving_License"], axis=1)

# Modifying the vehicle-age column

