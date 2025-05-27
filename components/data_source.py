import pandas as pd
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
        logger.info("Getting data from YahooFinance")
        tickers = self.get_tickers()
        closing_prices = self.download_data(tickers)
        df = self.tidy_up_data(closing_prices)
        return df

    def get_tickers(self):
        # Get list of SP500 tickers from wiki
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        tickers = tickers[:10]

        # Add SP500 index to tickers
        tickers.append('^GSPC')
        return tickers

    def download_data(self, tickers) -> pd.DataFrame:

        # Set periods to download
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=30 * 12 * 5)
        closing_prices = {}

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

        logger.info("Data downloaded from YahooFinance")

        return closing_prices

    def tidy_up_data(self, closing_prices):
        df = pd.concat(closing_prices, axis=1)
        df.columns = [x for x, y in df.columns]
        df = df.sort_index()
        df.rename(columns={'^GSPC': 'target'}, inplace=True)
        return df

    def read_data_from_file(self, path: str) -> pd.DataFrame:
        logger.info("Reading data from file")
        df = pd.read_csv(path, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df