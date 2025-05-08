from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from config.logger import logger

class MLModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])

    def predict(self, df: pd.DataFrame):

        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        df_final = pd.DataFrame({'predictions': predictions})
        score = round(self.pipeline.score(X_test, y_test), 3)
        logger.info(f"Model score (R_sqr) = {score}")
        return (df_final, score)
