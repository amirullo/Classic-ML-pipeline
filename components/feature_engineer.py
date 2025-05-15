from abc import ABC, abstractmethod
import pandas as pd


class FeatureEngineer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LagFeatureEngineer(FeatureEngineer):
    def transform(self, df):
        df['lag_1'] = df.iloc[:, 0].shift(1)
        return df

class FillNAFeatures(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill()
        df.fillna(0, inplace=True)
        return df

