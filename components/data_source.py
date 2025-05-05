import pandas as pd

from config.logger import logger

class CSVDataSource:
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
