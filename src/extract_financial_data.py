from typing import Union

import pandas as pd
import yfinance as yf


class FinancialDataExtractor:

    def __init__(self, symbol: str, start: str, end: str, interval: str = "1d"):
        """
        Initializes the TechnicalAnalysisIndicators class with the given
        parameters.

        Args:
            symbol (str): The ticker symbol that we are interested in
            start (str): start date for the period to be tested
            end (str): end date for the period to be tested
            interval (str, optional): frequency of the data fetched. Defaults to "1d".
        """
        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")

        if start > end:
            raise ValueError("Start date should be before end date.")

        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.data = self.get_data()

    def get_data(self) -> Union[pd.DataFrame, None]:
        """
        Fetches the historical data for the ticker symbol from Yahoo
        Finance using the yfinance library.

        Raises:
            ValueError: if the dataframe is empty

        Returns:
            Union[pd.DataFrame, None]: either the dataframe with the historical data
            or None if an error occurred
        """
        try:
            ticker_data = yf.download(
                self.symbol, start=self.start, end=self.end, interval=self.interval
            ).reset_index()
            # reset index to make Date a column
            if ticker_data.empty:
                raise ValueError("Invalid ticker symbol.")
            return ticker_data

        except Exception as e:
            print(f"Error getting data: {e}")
            return None

    def fill_missing_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills the missing dates in the dataframe.

        Args:
            data (pd.DataFrame): the dataframe to fill the missing dates for

        Returns:
            pd.DataFrame: the dataframe with the missing dates filled
        """
        data_copy = data.copy()
        all_dates = pd.date_range(start=self.start, end=self.end)
        df_all_dates = pd.DataFrame(all_dates, columns=["Date"])

        data_copy = df_all_dates.merge(data, on="Date", how="left")
        return data_copy

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the returns for the given dataframe.

        Args:
            data (pd.DataFrame): the dataframe to calculate the returns for

        Returns:
            pd.DataFrame: the dataframe with the returns
        """
        data_copy = data.copy()
        data_copy["Close"] = data_copy["Close"].ffill()
        data_copy["Return"] = data_copy["Close"] / data_copy["Close"].shift(1) - 1
        return data_copy

    def extraction_flow(self) -> pd.DataFrame:
        """
        Combines the methods in the class to extract the financial data.

        Returns:
            pd.DataFrame: pricing data stored
            in a pandas dataframe
        """
        data = self.fill_missing_dates(self.data)
        data = self.calculate_returns(data)
        return data
