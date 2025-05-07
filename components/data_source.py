import pandas as pd
import os
import time
import yfinance as yf
import datetime as dt
from abc import ABC, abstractmethod
from config.logger import logger

class BaseDataSource(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

class CSVDataSource(BaseDataSource):
    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

class YahooFinDataSource(BaseDataSource):
    def load_data(self) -> pd.DataFrame:
        df = self.download_data()
        # df = pd.DataFrame()
        # for fname in os.listdir(path):
        #     input_path = os.path.join(path, fname)
        #     if fname.endswith('.csv') and os.path.exists(input_path):
        #         df = self.read_data_from_file(input_path)
        # if df.shape[0] == 0:
        #     df = self.download_data()
        #     df.to_csv(os.path.join(path, 'sp500_closing_prices.csv'))
        return df

    def download_data(self) -> pd.DataFrame:
        logger.info(f"Getting data from YahooFinance")

        # Get list of SP500 tickers from wiki
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        # Set periods to download
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=30 * 12 * 5)
        closing_prices = {}

        tickers = tickers[:10]

        # Add SP500 index to tickers
        tickers.append('^GSPC')

        for ticker in tickers:
            try:
                # Replace dots with hyphens for Yahoo Finance format
                ticker_yf = ticker.replace('.', '-')
                data = yf.download(ticker_yf,
                                   start=start_date, end=end_date,
                                   progress=False,
                                   # threads=False,
                                   period="1mo",
                                   # auto_adjust=False
                                   )
                if not data.empty:
                    closing_prices[ticker] = data['Close']

                # Sleep briefly to avoid hitting rate limits
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")

        df = pd.concat(closing_prices, axis=1)
        df.columns = [x for x, y in df.columns]
        df = df.sort_index()
        df.rename(columns={'^GSPC': 'target'}, inplace=True)
        logger.info(f"Data downloaded from YahooFinance")

        return df

    def read_data_from_file(self, path: str) -> pd.DataFrame:
        logger.info(f"Reading data from file")
        df = pd.read_csv(path, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df