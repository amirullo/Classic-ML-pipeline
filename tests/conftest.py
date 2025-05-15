# import pytest
# import pandas as pd

# @pytest.fixture
# def mock_loaded_data():
#     path = './sp500_closing_prices.csv'
#     df = pd.read_csv(path, index_col='Date')
#     df.index = pd.to_datetime(df.index)
#     df = df.sort_index()
#     return df


import os
import pytest
import pandas as pd

@pytest.fixture
def mock_loaded_data():
    # Gets the path of the current test file and appends the CSV filename
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'sp500_closing_prices.csv')
    
    df = pd.read_csv(path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df
