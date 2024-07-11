# Model Training and evaluation
from pipeline import X, y, ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, roc_curve

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)

ct.fit(X_train)
Xnew = ct.transform(X_train)

