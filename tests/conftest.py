import pytest
import pandas as pd

@pytest.fixture
def mock_loaded_data():
    path = './sp500_closing_prices.csv'
    df = pd.read_csv(path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


