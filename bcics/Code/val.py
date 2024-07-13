import os
import pandas as pd
from pipeline import rd, ct
from custom_transformers import ColumnDropper
import pickle

with open("bcics/Code/bestModel.pkl", "rb") as f:
    model = pickle.load(f)

testData = pd.read_csv(os.path.join(rd, "test/test.csv"))
cd = ColumnDropper()
cd.fit(testData)
train_data = cd.transform(testData)

testData = ct.fit_transform(testData)

predictions = model.predict(testData)
print(predictions)
