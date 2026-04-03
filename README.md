# Gold Price Predictor (Time-Series Forecasting)

## Project Overview
This project is an end-to-end time-series forecasting pipeline built with PyTorch. It predicts future gold prices (localized to per gram) by analyzing historical market trends using a Long Short-Term Memory (LSTM) architecture. 

Rather than relying on static datasets, the system features an automated data pipeline that fetches over 50 years of live historical data directly from Yahoo Finance, processing it dynamically for future price prediction.

## Key Features
* **Automated Pipeline:** Dynamically fetches and processes over 13,500 historical market records (1972-Present) via `yfinance`.
* **PyTorch Architecture:** Custom `nn.Module` featuring an LSTM layer (64 hidden units) coupled with a linear output layer optimized via MSE Loss and Adam optimizer.
* **Autoregressive Forecasting:** Uses a 30-day sliding window approach. The CLI tool allows users to input any future date, and the model predicts the price step-by-step up to that exact day by feeding its own predictions back into the sequence.
* **Localized Context:** Automatically converts standard Troy Ounce market prices into Grams for practical, real-world utility.

## Tech Stack
* **Deep Learning:** Python, PyTorch (`nn.LSTM`)
* **Data Processing:** Scikit-Learn (`MinMaxScaler`), NumPy, Pandas
* **Data Extraction:** Yahoo Finance API (`yfinance`), JSON

## How to Run
1. Install dependencies: `pip install torch yfinance scikit-learn numpy pandas`
2. Fetch the latest dataset (Generates `real_db.json`): `python data_scrapper.py`
3. Train the model and launch the interactive CLI: `python main.py`
