# Gold Price Predictor (Per Gram) 📈

A time-series forecasting model built with PyTorch to predict future gold prices per gram. This project automates the retrieval of real-world financial data, processes it using a sliding window approach, and utilizes a Long Short-Term Memory (LSTM) neural network to forecast future values based on historical trends.

## Features
* **Automated Data Pipeline:** Fetches up-to-date historical gold data directly from Yahoo Finance (`GC=F`).
* **Unit Conversion:** Automatically converts the standard Troy Ounce market price into Grams for localized, practical forecasting.
* **Custom PyTorch Architecture:** Implements a custom `nn.Module` with an LSTM layer and a linear output layer for sequence prediction.
* **Interactive Forecasting:** Includes a CLI tool that allows users to input any future date and calculates the predicted gold price per gram, step-by-step, up to that specific day.

## Technologies Used
* **Deep Learning:** Python, PyTorch (`nn.LSTM`, `optim.Adam`, `MSELoss`)
* **Data Processing & Scaling:** Scikit-Learn (`MinMaxScaler`), NumPy
* **Data Extraction:** `yfinance`, `json`

## How to Run

1. Make sure you have the required dependencies installed:

        pip install torch yfinance scikit-learn numpy pandas

2. **Important: Fetch the data first!** Run the data scrapper to download the latest gold prices and generate the local `real_db.json` database:

        python data_scrapper.py

3. Run the main script to train the model and start the prediction interface:

        python main.py

4. When prompted, enter a future date (in `YYYY-MM-DD` format) to see the step-by-step price prediction per gram.