from typing import Union, Dict
import pandas as pd
import datetime as dt
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import volume_weighted_average_price
from langchain_community.tools import tool


@tool("get_stock_prices")
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicators for a given ticker."""
    try:
        # Download stock data for the last 6 months (use daily interval to get the latest data)
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24),  # 6 months ago
            end=dt.datetime.now(),
            interval='1d'  # Use daily data instead of weekly
        )

        # Check if data is empty
        if data.empty:
            return f"No data found for ticker '{ticker}'."

        # Flatten the MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Reset index and convert 'Date' to string
        data.reset_index(inplace=True)
        if 'Date' in data.columns:
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        # Check for required columns
        required_columns = ['High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return f"Required columns {required_columns} are missing. Available columns: {data.columns.tolist()}"

        # Drop rows with missing values in required columns
        data.dropna(subset=required_columns, inplace=True)

        # Initialize indicators dictionary
        indicators = {}

        # RSI Indicator
        try:
            rsi_series = RSIIndicator(data['Close'], window=14).rsi()
            rsi_values = list(enumerate(rsi_series.dropna()))
            indicators["RSI"] = {data['Date'].iloc[i]: round(value, 2) for i, value in rsi_values[-12:]}
        except Exception as e:
            indicators["RSI"] = f"Error calculating RSI: {str(e)}"

        # Stochastic Oscillator
        try:
            sto_series = StochasticOscillator(
                data['High'], data['Low'], data['Close'], window=14).stoch()
            sto_values = list(enumerate(sto_series.dropna()))
            indicators["Stochastic_Oscillator"] = {data['Date'].iloc[i]: round(value, 2) for i, value in
                                                   sto_values[-12:]}
        except Exception as e:
            indicators["Stochastic_Oscillator"] = f"Error calculating Stochastic Oscillator: {str(e)}"

        # MACD
        try:
            macd = MACD(data['Close'])

            # MACD Line
            macd_series = macd.macd()
            macd_values = list(enumerate(macd_series.dropna()))
            indicators["MACD"] = {data['Date'].iloc[i]: round(value, 2) for i, value in macd_values[-12:]}

            # MACD Signal Line
            macd_signal_series = macd.macd_signal()
            macd_signal_values = list(enumerate(macd_signal_series.dropna()))
            indicators["MACD_Signal"] = {data['Date'].iloc[i]: round(value, 2) for i, value in macd_signal_values[-12:]}
        except Exception as e:
            indicators["MACD"] = f"Error calculating MACD: {str(e)}"
            indicators["MACD_Signal"] = f"Error calculating MACD Signal: {str(e)}"

        # VWAP
        try:
            if 'Volume' in data.columns:
                vwap_series = volume_weighted_average_price(
                    high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']
                )
                vwap_values = list(enumerate(vwap_series.dropna()))
                indicators["vwap"] = {data['Date'].iloc[i]: round(value, 2) for i, value in vwap_values[-12:]}
            else:
                indicators["vwap"] = "Volume data not available."
        except Exception as e:
            indicators["vwap"] = f"Error calculating VWAP: {str(e)}"

        # Return stock prices and indicators
        return {
            'stock_price': data.to_dict(orient='records'),
            'indicators': indicators
        }

    except Exception as e:
        return f"Error fetching price data: {str(e)}"
