import yfinance as yf
import pandas as pd
import plotly.express as px
import difflib
import streamlit as st
import io

# Predefined list of stock ticker symbols
ticker_list = [
    {"name": "Adani Green Energy Ltd.", "symbol": "ADANIGREEN.NS"},
    {"name": "Reliance Industries", "symbol": "RELIANCE.NS"},
    {"name": "Tata Consultancy Services", "symbol": "TCS.NS"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK.NS"},
    {"name": "Infosys", "symbol": "INFY.NS"},
    {"name": "Wipro", "symbol": "WIPRO.NS"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK.NS"},
    {"name": "Bajaj Finance", "symbol": "BAJFINANCE.NS"},
    {"name": "Hindustan Unilever", "symbol": "HINDUNILVR.NS"}
]

# Function to fetch stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"⚠️ No data found for {stock_symbol}. Check the stock symbol or date range.")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data for {stock_symbol}: {e}")
        return None

# Function to display stock suggestions based on user input
def suggest_stocks(keyword):
    company_names = [ticker["name"] for ticker in ticker_list]
    suggestions = difflib.get_close_matches(keyword, company_names, n=5, cutoff=0.3)
    return suggestions

# Function to plot stock graph using Plotly
def plot_stock_graph(data, stock_symbol):
    fig = px.line(data, x=data.index, y="Close", title=f"Stock Price of {stock_symbol}")
    st.plotly_chart(fig)

# Function to save data as CSV
def save_to_csv(data, stock_symbol, start_date, end_date):
    filename = f"{stock_symbol}_{start_date}_to_{end_date}.csv"
    return filename, data.to_csv(index=True).encode('utf-8')

# Streamlit UI
def main():
    st.title("Stock Data Downloader & Visualizer")

    # User enters keyword to search for stock
    keyword = st.text_input("Type a company name or keyword (e.g., Adani, Tata): ").strip()
    
    if keyword:
        # Get suggestions based on the input keyword
        suggestions = suggest_stocks(keyword)
        
        if suggestions:
            st.write("🔍 Did you mean one of these?")
            selected_company = st.selectbox("Select a company", suggestions)
            stock_symbol = next(ticker['symbol'] for ticker in ticker_list if ticker['name'] == selected_company)
            st.write(f"You selected: {selected_company} ({stock_symbol})")
        else:
            st.warning("⚠️ No suggestions found. Please try a different keyword.")
            return
        
        # Date inputs for the stock data range
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        if start_date and end_date:
            st.write(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
            data = fetch_stock_data(stock_symbol, start_date, end_date)

            if data is not None:
                # Show stock graph
                plot_stock_graph(data, stock_symbol)

                # Save data as CSV and provide download option
                csv_filename, csv_data = save_to_csv(data, stock_symbol, start_date, end_date)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()








