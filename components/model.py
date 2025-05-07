from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from config.logger import logger

class MLModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])

    def predict(self, df: pd.DataFrame):
        X = df.drop(columns=[df.columns[0]])
        y = df[df.columns[0]]
        self.pipeline.fit(X, y)
        df['prediction'] = self.pipeline.predict(X)
        score = round(self.pipeline.score(X, y), 3)
        logger.info(f"Model score (R_sqr) = {score}")
        return (df[['prediction']], score)
