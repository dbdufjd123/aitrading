import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib, os

class QuoteDirectionModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, subsample=0.7, colsample_bytree=0.7,
            eval_metric="mlogloss", random_state=42
        )
        self.scaler = None

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def save(self, dirpath="models", name="xgb_quote"):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, os.path.join(dirpath, f"{name}.pkl"))
        joblib.dump(self.scaler, os.path.join(dirpath, f"{name}_scaler.pkl"))

    @classmethod
    def load(cls, dirpath="models", name="xgb_quote"):
        m = cls()
        m.model = joblib.load(os.path.join(dirpath, f"{name}.pkl"))
        m.scaler = joblib.load(os.path.join(dirpath, f"{name}_scaler.pkl"))
        return m