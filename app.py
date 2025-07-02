import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import pandas_ta as ta

import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Indian Stock Price Prediction")

# Top 50 NSE stock symbols with readable names
stock_dict = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ASIANPAINT.NS": "Asian Paints",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ITC.NS": "ITC",
    "HCLTECH.NS": "HCL Technologies",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "WIPRO.NS": "Wipro",
    "NTPC.NS": "NTPC",
    "TECHM.NS": "Tech Mahindra"
}

# Sidebar for inputs
st.sidebar.header("📊 Configuration")

# Stock selection
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    options=list(stock_dict.keys()),
    format_func=lambda x: f"{stock_dict[x]} ({x})"
)

# Simplified parameters
prediction_days = st.sidebar.selectbox("Prediction Period", [1, 5, 7, 15], index=2)
data_period = st.sidebar.selectbox("Historical Data", ["1y", "2y", "3y", "5y"], index=1)
sequence_length = st.sidebar.slider("LSTM Sequence Length (days)", 30, 100, 60)

# Submit button
predict_button = st.sidebar.button("🔮 Predict Stock Price", type="primary")

@st.cache_data(ttl=3600)
def load_data(symbol, period):
    """Load stock data with caching"""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_rsi(prices, window=14):
    """Calculate RSI manually"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series(index=prices.index, dtype=float)

def calculate_technical_indicators(data):
    """Calculate basic technical indicators"""
    try:
        close_prices = data['Close']
        
        # Create indicators dictionary
        indicators = {}
        
        # Moving averages
        indicators['SMA_20'] = close_prices.rolling(window=20).mean()
        indicators['EMA_20'] = close_prices.ewm(span=20).mean()
        
        # RSI
        try:
            indicators['RSI'] = ta.rsi(close_prices, length=14)
        except:
            indicators['RSI'] = calculate_rsi(close_prices)
        
        # Convert to DataFrame
        indicators_df = pd.DataFrame(indicators, index=close_prices.index)
        return indicators_df
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return pd.DataFrame(index=data.index)

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        targets.append(data[i])
    
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def predict_simple_moving_average(close_prices, prediction_days):
    """Simple moving average prediction as fallback"""
    try:
        last_prices = close_prices.tail(30).values
        trend = np.polyfit(range(len(last_prices)), last_prices, 1)[0]
        
        predictions = []
        last_price = close_prices.iloc[-1]
        
        for i in range(prediction_days):
            next_price = last_price + (trend * (i + 1))
            predictions.append(next_price)
        
        return np.array(predictions), {'Method': 'Simple Moving Average'}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.array([close_prices.iloc[-1]] * prediction_days), {'Method': 'Last Price'}

def predict_lstm(close_prices, seq_length, prediction_days):
    """LSTM prediction function"""
    try:
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(scaled_data, seq_length)
        
        if len(X) < 100:  # Not enough data for LSTM
            return predict_simple_moving_average(close_prices, prediction_days)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        model = build_lstm_model(seq_length)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        with st.spinner("Training LSTM model..."):
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0,
                
            )
        
        # Make predictions
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        
        for _ in range(prediction_days):
            pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Calculate metrics
        test_predictions = model.predict(X_test, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test)
        
        mse = mean_squared_error(y_test_actual, test_predictions)
        mae = mean_absolute_error(y_test_actual, test_predictions)
        rmse = np.sqrt(mse)
        
        return predictions.flatten(), {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'History': history}
        
    except Exception as e:
        st.error(f"LSTM prediction failed: {e}")
        return predict_simple_moving_average(close_prices, prediction_days)

# Main app
if selected_stock:
    # Load data
    with st.spinner("Loading stock data..."):
        data = load_data(selected_stock, data_period)
    
    if data.empty:
        st.error("❌ No data found. Please check your selection and try again.")
    else:
        # Display basic info
        st.success(f"✅ Loaded {len(data)} days of data for {stock_dict[selected_stock]}")
        
        # Extract close prices as Series
        close_prices = data['Close']
        
        # Current price info and 52-week high/low
        try:
            current_price = float(close_prices.iloc[-1])
            prev_price = float(close_prices.iloc[-2])
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            # Calculate 52-week high/low
            high_52w = float(data['High'].rolling(window=252).max().iloc[-1])
            low_52w = float(data['Low'].rolling(window=252).min().iloc[-1])
            
            # Distance from 52W high/low
            dist_from_high = ((current_price - high_52w) / high_52w) * 100
            dist_from_low = ((current_price - low_52w) / low_52w) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("Daily Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
            with col3:
                st.metric("52W High", f"₹{high_52w:.2f}", f"{dist_from_high:.1f}%")
            with col4:
                st.metric("52W Low", f"₹{low_52w:.2f}", f"{dist_from_low:.1f}%")
                
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(data)
        
        # Price chart with SMA and EMA
        st.subheader("📈 Stock Price Chart")
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=data.index, y=close_prices,
            name="Close Price", line=dict(color='blue', width=2)
        ))
        
        # Add moving averages
        if not indicators.empty:
            if 'SMA_20' in indicators.columns:
                fig.add_trace(go.Scatter(
                    x=indicators.index, y=indicators['SMA_20'],
                    name="SMA 20", line=dict(color='orange', dash='dash')
                ))
            
            if 'EMA_20' in indicators.columns:
                fig.add_trace(go.Scatter(
                    x=indicators.index, y=indicators['EMA_20'],
                    name="EMA 20", line=dict(color='green', dash='dash')
                ))
        
        fig.update_layout(
            title=f"{stock_dict[selected_stock]} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators - RSI and Volume
        if not indicators.empty:
            st.subheader("📊 Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                if 'RSI' in indicators.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=indicators.index, y=indicators['RSI'],
                        name="RSI", line=dict(color='purple')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI")
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # Volume Chart
                if 'Volume' in data.columns:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=data.index, y=data['Volume'],
                        name="Volume", marker_color='lightblue'
                    ))
                    fig_vol.update_layout(title="Trading Volume", yaxis_title="Volume")
                    st.plotly_chart(fig_vol, use_container_width=True)
        
        # Predictions section
        if predict_button:
            st.subheader("🔮 Price Predictions")
            
            with st.spinner("Generating predictions..."):
                predictions, metrics = predict_lstm(close_prices, sequence_length, prediction_days)
            
            # Display predictions
            future_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=prediction_days
            )
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions
            })
            
            # Prediction chart
            fig_pred = go.Figure()
            
            # Historical data (last 30 days)
            recent_data = data.tail(30)
            fig_pred.add_trace(go.Scatter(
                x=recent_data.index, y=recent_data['Close'],
                name="Historical", line=dict(color='blue')
            ))
            
            # Predictions
            fig_pred.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Predicted Price'],
                name="Predictions", line=dict(color='red', dash='dash'),
                marker=dict(size=8)
            ))
            
            fig_pred.update_layout(
                title=f"{prediction_days}-Day Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction table
            forecast_df['Change'] = forecast_df['Predicted Price'].diff()
            forecast_df['Change %'] = forecast_df['Predicted Price'].pct_change() * 100
            
            # Display table
            st.subheader("📋 Prediction Details")
            display_df = forecast_df.copy()
            display_df['Predicted Price'] = display_df['Predicted Price'].round(2)
            display_df['Change'] = display_df['Change'].round(2)
            display_df['Change %'] = display_df['Change %'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Model performance
            if 'MSE' in metrics:
                st.subheader("📈 Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{metrics['MSE']:.2f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
            
            # Download predictions
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{selected_stock}_predictions.csv",
                mime="text/csv"
            )



# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This app predicts Indian stock prices using:
- **LSTM Neural Networks** for predictions
- **Technical Indicators** (RSI, SMA, EMA)
- **Volume Analysis**
- **52-Week High/Low tracking**

**Features:**
- SMA & EMA on main chart
- RSI indicator chart
- Volume analysis chart
- 7-day price predictions

**Disclaimer:** This is for educational purposes only. 
Not financial advice.
""")

# Add some basic styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ⚠️ Educational Disclaimer (main page)
st.warning("⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice!")