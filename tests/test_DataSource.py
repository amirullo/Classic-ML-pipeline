import pytest
import pandas as pd
from unittest.mock import patch
from components.data_source import YahooFinDataSource
import numpy as np


@pytest.fixture
def data_source():
    return YahooFinDataSource()


def test_get_tickers(data_source):
    mock_table = pd.DataFrame({'Symbol': [f'TICK{i}' for i in range(15)]})
    with patch('pandas.read_html', return_value=[mock_table]):
        tickers = data_source.get_tickers()
        assert isinstance(tickers, list)
        assert '^GSPC' in tickers
        assert len(tickers) == 11  # 10 + 1 (GSPC)


@patch('components.data_source.yf.download')
def test_download_data(mock_download, data_source):
    mock_download.return_value = pd.DataFrame({
        'Close': [100, 101, 102]
    }, index=pd.date_range("2023-01-01", periods=3))

    tickers = ['TICK1', 'TICK2']
    closing_prices = data_source.download_data(tickers)

    assert isinstance(closing_prices, dict)
    assert set(closing_prices.keys()) == set(tickers)
    for df in closing_prices.values():
        assert isinstance(df, pd.Series)


def test_tidy_up_data(data_source, mock_loaded_data):
    closing_prices = {
        ('TICK1', 'TICK1'): pd.Series([100, 101], index=pd.date_range("2023-01-01", periods=2)),
        ('^GSPC', '^GSPC'): pd.Series([4767, 4767], index=pd.date_range("2023-01-01", periods=2))
    }
    df = data_source.tidy_up_data(closing_prices)
    assert isinstance(df, pd.DataFrame)
    assert 'target' in df.columns
    assert 'TICK1' in df.columns
    assert np.isclose(df['target'].mean(), mock_loaded_data['^GSPC'].mean(), rtol=.005)
