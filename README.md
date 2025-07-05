# Stock-Price-Prediction-using-LSTM
A Streamlit web app that predicts Indian NSE stock prices using LSTM neural networks and technical indicators like RSI, SMA, and EMA. Includes interactive charts, 52-week metrics, and CSV export functionality.

## 🎥 Demo

👉 Live App: https://stock-price-prediction-using-lstm-nnrhip88vm5qm6mwnzwtlx.streamlit.app/

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- yfinance
- Matplotlib, Plotly
- Streamlit

## ✨ Features

- LSTM-based stock price forecasting (7 to 15 days)
- Technical indicators: SMA, EMA, RSI
- 52-week high and low statistics
- Actual vs Predicted price graph
- Top 20 NSE stocks dropdown
- CSV export of prediction
- SMA fallback prediction if model fails

## ⚙️ How It Works

1. Fetch historical stock data using yfinance.
2. Engineer features like SMA, EMA, RSI, etc.
3. Scale and reshape data for the LSTM model.
4. Train an LSTM model on historical trends.
5. Generate future price predictions.
6. Display results and charts on Streamlit dashboard.

## 🧩 Installation

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
