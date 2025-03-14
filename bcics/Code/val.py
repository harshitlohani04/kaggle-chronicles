import os
import pandas as pd
from pipeline import rd, ct
from custom_transformers import ColumnDropper
import pickle
import numpy as np
import xgboost as xgb

with open("bcics/Code/bestModel.pkl", "rb") as f:
    model = pickle.load(f)

testData = pd.read_csv(os.path.join(rd, "test/test.csv"))
ids = testData.id.to_numpy(dtype = np.int64)
print(ids)

cd = ColumnDropper()
cd.fit(testData)
test_data = cd.transform(testData)

test_data = ct.fit_transform(test_data)

predictions = np.round(model.predict(xgb.DMatrix(test_data)), 2)
# predictions = np.array([1 if pred>=0.1 else 0 for pred in predictions])

submission = np.concatenate([ids.reshape(-1, 1), predictions.reshape(-1, 1)], axis = 1)
submission = pd.DataFrame(submission, columns = ["id", "Response"])
submission.to_csv("bcics/Code/submission.csv", index = False)
