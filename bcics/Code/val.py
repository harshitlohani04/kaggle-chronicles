import os
import pandas as pd
from pipeline import finalPipeline, rd

testData = pd.read_csv(os.path.join(rd, "test/test.csv"))
