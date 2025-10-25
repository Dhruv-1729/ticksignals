import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import sqlalchemy 
from datetime import datetime, timedelta
import re
import os
from scipy import stats

# --- App Configuration ---
st.set_page_config(page_title="TickSignals", layout="wide")

# --- Initialize Neon Connection ---
# This automatically and securely reads from your st.secrets
# [connections.postgresql]
try:
    conn = st.connection("postgresql", type="sql")
except Exception as e:
    st.error(f"Failed to connect to the database. Please check secrets: {e}")
    # Stop the app if DB connection fails
    st.stop()


# --- Helper Functions (MODIFIED FOR NEON) ---

@st.cache_data(ttl=3600)
def get_stock_data_with_caching(ticker):
    """
    Fetches stock data, using the persistent Neon (PostgreSQL) database for caching.
    """
    sanitized_ticker = re.sub(r'[^a-zA-Z0-9_]', '', ticker).lower() # PostgreSQL prefers lowercase tables
    fetch_log = []
    
    # Check if table exists and get the last date
    try:
        # Use conn.run() for schema/metadata queries
        existing_tables = conn.run(f"SELECT table_name FROM information_schema.tables WHERE table_name = '{sanitized_ticker}';")
        
        if existing_tables:
            # Use conn.query() for pandas DataFrames
            last_date_df = conn.query(f'SELECT MAX("Date") FROM "{sanitized_ticker}"')
            last_date_str = last_date_df.iloc[0, 0]
            start_date = (pd.to_datetime(last_date_str) + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = "2010-01-01"
            
    except Exception as e:
        fetch_log.append(f"DB read error for {ticker}: {e}. Defaulting to full history.")
        start_date = "2010-01-01"

    # Download new data if needed
    if start_date <= datetime.now().strftime('%Y-%m-%d'):
        fetch_log.append(f"Fetching new data for {ticker} from {start_date}...")
        new_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
            
            # Ensure column names are compatible with SQL
            new_data.rename(columns=str.capitalize, inplace=True)
            df_to_write = new_data.reset_index()
            # Rename for consistency, ensure 'Date' is lowercase for some SQL variants if needed
            df_to_write.rename(columns={'index': 'Date'}, inplace=True)
            
            # Write new data to the Neon database
            try:
                # Use conn.write() to write a DataFrame to a table
                # We use session.execute to use df.to_sql for more control
                with conn.session() as s:
                    df_to_write.to_sql(sanitized_ticker, 
                                     con=s.bind, 
                                     if_exists='append', 
                                     index=False,
                                     method='multi')
                fetch_log.append(f"Successfully cached {len(df_to_write)} new rows for {ticker}.")
            except Exception as e:
                fetch_log.append(f"Error writing to Neon DB for {ticker}: {e}")
    else:
        fetch_log.append(f"Data for {ticker} is up to date.")
    
    # Read all data for the ticker from the Neon DB
    try:
        data = conn.query(f'SELECT * FROM "{sanitized_ticker}"')
        date_col = next((col for col in data.columns if col.lower() == 'date'), 'Date')
        
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)
        # Handle potential duplicates from overlapping fetches
        data = data[~data.index.duplicated(keep='last')]
    except Exception as e:
        st.error(f"Error reading {ticker} data from Neon: {e}")
        data = pd.DataFrame()
    
    return data, fetch_log

def store_signals_in_db(ticker, signals_df):
    """Stores newly generated signals in the Neon database."""
    
    # Create the all_signals table if it doesn't exist
    try:
        with conn.session() as s:
            s.execute(sqlalchemy.text('''
                CREATE TABLE IF NOT EXISTS all_signals (
                    "Date" TEXT, 
                    "Ticker" TEXT, 
                    "Signal" TEXT, 
                    "Price" REAL, 
                    "Confidence_Pct" INTEGER,
                    PRIMARY KEY ("Date", "Ticker")
                );
            '''))
            s.commit()
    except Exception as e:
        st.warning(f"Could not create all_signals table: {e}")

    # Prepare data for insertion
    df_to_insert = signals_df.copy()
    df_to_insert['Signal'] = df_to_insert['Signal'].apply(lambda x: "Buy" if x == 2 else "Sell")
    df_to_insert['Confidence_Pct'] = df_to_insert['Confidence_%'].astype(int)
    df_to_insert['Ticker'] = ticker
    df_to_insert['Price'] = df_to_insert['Close']
    df_to_insert['Date'] = df_to_insert.index.strftime('%Y-%m-%d')
    
    df_to_insert = df_to_insert[['Date', 'Ticker', 'Signal', 'Price', 'Confidence_Pct']]

    # Use a SQL "INSERT ... ON CONFLICT DO UPDATE" to perform an "upsert"
    # This avoids errors if we try to insert a duplicate key (Date, Ticker)
    insert_sql = """
    INSERT INTO all_signals ("Date", "Ticker", "Signal", "Price", "Confidence_Pct")
    VALUES (:Date, :Ticker, :Signal, :Price, :Confidence_Pct)
    ON CONFLICT ("Date", "Ticker") DO UPDATE SET
        "Signal" = EXCLUDED."Signal",
        "Price" = EXCLUDED."Price",
        "Confidence_Pct" = EXCLUDED."Confidence_Pct";
    """
    
    try:
        with conn.session() as s:
            for record in df_to_insert.to_dict('records'):
                s.execute(sqlalchemy.text(insert_sql), record)
            s.commit()
    except Exception as e:
        st.error(f"Error storing signals in Neon DB: {e}")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(data, period=14):
    """Calculate Average True Range for volatility"""
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def is_trending_market(data, lookback=50):
    """Determine if market is trending or choppy"""
    # ADX-like calculation
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    plus_dm = data['High'].diff()
    minus_dm = -data['Low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr_smooth = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_smooth)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    return adx > 20  # Trending if ADX > 20

def generate_enhanced_signals(data):
    """
    Enhanced signal generation with earlier entries and better exits.
    Uses multiple confirmations and momentum filters.
    """
    data['Signal'] = 0
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Volume
    if 'Volume' in data.columns:
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    else:
        data['Volume_Ratio'] = 1
    
    # Price momentum
    data['Price_ROC'] = data['Close'].pct_change(10) * 100
    
    # Trend detection
    data['Is_Trending'] = is_trending_market(data)
    
    # BUY CONDITIONS - Multiple confirmations needed
    buy_conditions = (
        (data['SMA20'] > data['SMA50']) & 
        (data['SMA20'].shift(1) <= data['SMA50'].shift(1)) &
        (data['SMA50'] >= data['SMA200'] * 0.98) &  # Within 2% of SMA200
        (data['RSI'] < 70) &
        (data['RSI'] > 30) &
        (data['MACD_Hist'] > data['MACD_Hist'].shift(1)) &
        (data['Price_ROC'] > -5) &
        (data['Volume_Ratio'] > 0.8) &
        (data['Is_Trending'] == True)
    )
    
    # SELL CONDITIONS
    sell_conditions = (
        (data['SMA20'] < data['SMA50']) & 
        (data['SMA20'].shift(1) >= data['SMA50'].shift(1)) &
        (
            ((data['RSI'] > 70) & (data['MACD_Hist'] < 0)) |
            ((data['Price_ROC'] < -8) & (data['MACD_Hist'] < data['MACD_Hist'].shift(1)))
        )
    )
    
    data.loc[buy_conditions, 'Signal'] = 2
    data.loc[sell_conditions, 'Signal'] = -2
    
    return data

def calculate_confidence_score(forecast_data, signal_type):
    """
    Calculate confidence score (0-100%) based on technical indicators.
    """
    score = 0
    max_score = 0
    
    # Days to crossover scoring (closer = better, but not too close)
    max_score += 20
    days = abs(forecast_data['days_to_crossover'])
    if 3 <= days <= 8: score += 20
    elif 2 <= days <= 10: score += 15
    elif days <= 12: score += 10
    
    # RSI scoring
    max_score += 15
    rsi = forecast_data['rsi']
    if signal_type == 'BUY':
        if 30 <= rsi <= 50: score += 15
        elif 50 < rsi <= 60: score += 10
        elif 25 <= rsi < 30 or 60 < rsi <= 65: score += 7
    else:  # SELL
        if 50 <= rsi <= 70: score += 15
        elif 40 <= rsi < 50: score += 10
        elif 70 < rsi <= 75 or 35 <= rsi < 40: score += 7
    
    # MACD alignment scoring
    max_score += 20
    macd_hist = forecast_data['macd_histogram']
    macd_slope = forecast_data['macd_slope']
    if signal_type == 'BUY':
        if macd_hist > -0.05 and macd_slope > 0.01: score += 20
        elif macd_hist > -0.1 and macd_slope > 0: score += 12
        elif macd_hist > -0.2: score += 6
    else:  # SELL
        if macd_hist < 0.05 and macd_slope < -0.01: score += 20
        elif macd_hist < 0.1 and macd_slope < 0: score += 12
        elif macd_hist < 0.2: score += 6
    
    # Volume trend scoring
    max_score += 15
    vol_trend = forecast_data['volume_trend']
    if vol_trend > 15: score += 15
    elif vol_trend > 5: score += 10
    elif vol_trend > 0: score += 5
    elif vol_trend > -10: score += 3
    
    # Convergence rate scoring
    max_score += 15
    conv_rate = forecast_data['convergence_rate']
    if conv_rate > 0.03: score += 15
    elif conv_rate > 0.02: score += 10
    elif conv_rate > 0.01: score += 7
    elif conv_rate > 0.005: score += 4
    
    # Price momentum scoring
    max_score += 15
    price_roc = forecast_data['price_roc']
    if signal_type == 'BUY':
        if -3 <= price_roc <= 2: score += 15
        elif -5 <= price_roc <= 5: score += 10
        elif -8 <= price_roc <= 8: score += 5
    else:  # SELL
        if -2 <= price_roc <= 3: score += 15
        elif -5 <= price_roc <= 5: score += 10
        elif -8 <= price_roc <= 8: score += 5
    
    confidence_pct = int((score / max_score) * 100)
    return min(100, max(0, confidence_pct)) 

def calculate_forecast_metrics(data):
    """Advanced forecasting for predicting buy AND sell signals with confidence scoring."""
    if len(data) < 200:
        return None
    
    lookback = 20
    recent_data = data.tail(lookback).copy()
    
    sma50_velocity = (recent_data['SMA50'].iloc[-1] - recent_data['SMA50'].iloc[0]) / lookback
    sma200_velocity = (recent_data['SMA200'].iloc[-1] - recent_data['SMA200'].iloc[0]) / lookback
    current_gap = recent_data['SMA50'].iloc[-1] - recent_data['SMA200'].iloc[-1]
    relative_velocity = sma50_velocity - sma200_velocity
    
    if relative_velocity != 0:
        days_to_crossover = -current_gap / relative_velocity
    else:
        days_to_crossover = float('inf')
    
    gap_5_days_ago = data['SMA50'].iloc[-5] - data['SMA200'].iloc[-5] if len(data) >= 5 else current_gap
    convergence_rate = abs(gap_5_days_ago - current_gap) / 5 if gap_5_days_ago != current_gap else 0
    
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal_line
    macd_slope = (macd.iloc[-1] - macd.iloc[-5]) / 5 if len(macd) >= 5 else 0
    
    volume_trend = 0
    if 'Volume' in data.columns:
        recent_vol_avg = recent_data['Volume'].tail(5).mean()
        historical_vol_avg = data['Volume'].tail(50).mean()
        volume_trend = (recent_vol_avg / historical_vol_avg - 1) * 100 if historical_vol_avg > 0 else 0
    
    rsi_value = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
    price_roc = ((data['Close'].iloc[-1] / data['Close'].iloc[-20]) - 1) * 100 if len(data) >= 20 else 0
    current_state = "above" if current_gap > 0 else "below"
    
    forecast = {
        'ticker': None, 'current_gap': current_gap, 'gap_percentage': (abs(current_gap) / data['Close'].iloc[-1]) * 100,
        'sma50_velocity': sma50_velocity, 'sma200_velocity': sma200_velocity, 'relative_velocity': relative_velocity,
        'convergence_rate': convergence_rate, 'days_to_crossover': days_to_crossover, 'macd_value': macd.iloc[-1],
        'macd_signal': signal_line.iloc[-1], 'macd_histogram': macd_histogram.iloc[-1], 'macd_slope': macd_slope,
        'price_roc': price_roc, 'volume_trend': volume_trend, 'rsi': rsi_value, 'current_price': data['Close'].iloc[-1],
        'sma50': data['SMA50'].iloc[-1], 'sma200': data['SMA200'].iloc[-1], 'current_state': current_state
    }
    
    if current_state == "below" and relative_velocity > 0 and 2 <= days_to_crossover <= 12:
        if (25 <= rsi_value <= 65 and macd_histogram.iloc[-1] > -0.2 and 
            convergence_rate > 0.008 and price_roc > -10):
            confidence = calculate_confidence_score(forecast, 'BUY')
            if confidence >= 40:
                forecast['signal'] = 'STRONG_BUY_FORECAST' if confidence >= 70 else 'BUY_FORECAST'
                forecast['confidence'] = confidence
            else:
                forecast['signal'] = 'NEUTRAL'
                forecast['confidence'] = 0
        else:
            forecast['signal'] = 'NEUTRAL'
            forecast['confidence'] = 0
            
    elif current_state == "above" and relative_velocity < 0 and 2 <= days_to_crossover <= 12:
        if (25 <= rsi_value <= 75 and macd_histogram.iloc[-1] < 0.2 and 
            convergence_rate > 0.008 and price_roc < 10):
            confidence = calculate_confidence_score(forecast, 'SELL')
            if confidence >= 40:
                forecast['signal'] = 'STRONG_SELL_FORECAST' if confidence >= 70 else 'SELL_FORECAST'
                forecast['confidence'] = confidence
            else:
                forecast['signal'] = 'NEUTRAL'
                forecast['confidence'] = 0
        else:
            forecast['signal'] = 'NEUTRAL'
            forecast['confidence'] = 0
    else:
        forecast['signal'] = 'NEUTRAL'
        forecast['confidence'] = 0
    
    return forecast

def calculate_signal_confidence(data, signal_index):
    """
    Enhanced confidence calculation based on signal quality.
    """
    score = 0
    max_score = 0
    
    signal_type = data['Signal'].iloc[signal_index]
    rsi = data['RSI'].iloc[signal_index]
    current_price = data['Close'].iloc[signal_index]
    sma20 = data['SMA20'].iloc[signal_index]
    sma50 = data['SMA50'].iloc[signal_index]
    macd_hist = data['MACD_Hist'].iloc[signal_index]
    price_roc = data['Price_ROC'].iloc[signal_index]
    volume_ratio = data['Volume_Ratio'].iloc[signal_index] if 'Volume_Ratio' in data.columns else 1
    
    # Gap size at crossover (20 points) - tighter is better
    max_score += 20
    gap_pct = abs((sma20 - sma50) / current_price) * 100
    if gap_pct < 0.3: score += 20
    elif gap_pct < 0.8: score += 15
    elif gap_pct < 1.5: score += 10
    elif gap_pct < 2.5: score += 5
    
    # RSI positioning (25 points)
    max_score += 25
    if signal_type == 2:  # BUY
        if 35 <= rsi <= 55: score += 25
        elif 30 <= rsi <= 60: score += 18
        elif 25 <= rsi <= 65: score += 10
    else:  # SELL
        if 45 <= rsi <= 65: score += 25
        elif 40 <= rsi <= 70: score += 18
        elif 35 <= rsi <= 75: score += 10
    
    # MACD alignment (20 points)
    max_score += 20
    if signal_type == 2:  # BUY
        if macd_hist > 0: score += 20
        elif macd_hist > -0.3: score += 12
        elif macd_hist > -0.8: score += 6
    else:  # SELL
        if macd_hist < 0: score += 20
        elif macd_hist < 0.3: score += 12
        elif macd_hist < 0.8: score += 6
    
    # Volume strength (15 points)
    max_score += 15
    if volume_ratio > 1.5: score += 15
    elif volume_ratio > 1.2: score += 12
    elif volume_ratio > 0.9: score += 8
    elif volume_ratio > 0.7: score += 4
    
    # Momentum alignment (20 points)
    max_score += 20
    if signal_type == 2:  # BUY
        if price_roc > 2: score += 20
        elif price_roc > 0: score += 15
        elif price_roc > -3: score += 8
    else:  # SELL
        if price_roc < -2: score += 20
        elif price_roc < 0: score += 15
        elif price_roc < 3: score += 8
    
    confidence_pct = int((score / max_score) * 100)
    return confidence_pct

# --- New Streamlit Helper Functions ---

def format_large_number(num):
    """Formats large numbers into M, B, T strings."""
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    if num >= 1e6:
        return f"${num/1e6:.2f}M"
    return f"${num:,.2f}"

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    """Fetches key stock info for the UI."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Use fastInfo for more reliable last price, fallback to regularMarketPrice
        price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        
        return {
            "Price": f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A",
            "Market Cap": format_large_number(info.get('marketCap')),
            "Volume": f"{info.get('volume', 'N/A'):,}" if isinstance(info.get('volume'), int) else "N/A",
            "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A"
        }
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return {
            "Price": "N/A", "Market Cap": "N/A", "Volume": "N/A", "P/E Ratio": "N/A"
        }

@st.cache_resource
def load_csv_data(file_name, from_db=False):
    """
    Loads data. If from_db=True, loads from Neon DB.
    Otherwise, attempts to load from a local CSV (for uploaded files).
    """
    if from_db:
        try:
            # Note: PostgreSQL table names are case-sensitive if quoted.
            # Assuming we standardize on lowercase for these tables.
            db_table_name = file_name.replace('.csv', '').lower()
            if db_table_name == "latest_signals":
                # Special query for latest_signals
                query = """
                SELECT t1."Date", t1."Ticker", t1."Signal", t1."Price", t1."Confidence_Pct" as "Confidence_%"
                FROM all_signals t1
                INNER JOIN (
                    SELECT "Ticker", MAX("Date") as "MaxDate"
                    FROM all_signals
                    GROUP BY "Ticker"
                ) t2 ON t1."Ticker" = t2."Ticker" AND t1."Date" = t2."MaxDate"
                ORDER BY t1."Confidence_Pct" DESC, t1."Date" DESC;
                """
                df = conn.query(query)
            else:
                # Standard table query
                df = conn.query(f'SELECT * FROM "{db_table_name}"')
                
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.warning(f"Could not load data from database table '{db_table_name}': {e}")
            return None
    else:
        # Original CSV loading logic (for user uploads)
        try:
            df = pd.read_csv(file_name)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except FileNotFoundError:
            return None
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            return None

# --- Modified Core Functions for Streamlit ---

def generate_chart_for_ticker_mod(ticker, show_forecast=False):
    """
    Generates chart, performance, and forecast strings for Streamlit.
    MODIFIED: Returns fig, perf_summary, forecast_summary. Does NOT call fig.show().
    Uses Neon DB.
    """
    data, fetch_log = get_stock_data_with_caching(ticker)
    if data.empty:
        st.error(f"ERROR: No data found for ticker '{ticker}'")
        return None, "", ""

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = generate_enhanced_signals(data)
    
    is_invested = False
    cleaned_signals = []
    for index, row in data.iterrows():
        signal = row['Signal']
        if not is_invested and signal == 2:
            cleaned_signals.append(signal)
            is_invested = True
        elif is_invested and signal == -2:
            cleaned_signals.append(signal)
            is_invested = False
        else:
            cleaned_signals.append(0)
    data['Signal'] = cleaned_signals
    
    data['Forecast_Buy'] = False
    data['Forecast_Sell'] = False
    forecast_accuracy = {'buy': {'correct': 0, 'total': 0}, 'sell': {'correct': 0, 'total': 0}}
    
    for i in range(200, len(data) - 15):
        historical_data = data.iloc[:i].copy()
        forecast = calculate_forecast_metrics(historical_data)
        
        if forecast and forecast['confidence'] >= 50:
            if forecast['signal'] in ['STRONG_BUY_FORECAST', 'BUY_FORECAST']:
                data.loc[data.index[i], 'Forecast_Buy'] = True
                future_window = data.iloc[i:min(i+16, len(data))]
                if (future_window['Signal'] == 2).any():
                    forecast_accuracy['buy']['correct'] += 1
                forecast_accuracy['buy']['total'] += 1
                
            elif forecast['signal'] in ['STRONG_SELL_FORECAST', 'SELL_FORECAST']:
                data.loc[data.index[i], 'Forecast_Sell'] = True
                future_window = data.iloc[i:min(i+16, len(data))]
                if (future_window['Signal'] == -2).any():
                    forecast_accuracy['sell']['correct'] += 1
                forecast_accuracy['sell']['total'] += 1

    # Performance simulation
    initial_capital = 10000.00
    cash, shares, cost_basis, transaction_log, portfolio_values = initial_capital, 0, 0, [], []
    first_price_bh = data['Close'].dropna().iloc[0]
    data['Buy and Hold'] = (initial_capital / first_price_bh) * data['Close']
    first_valid_index = data.dropna(subset=['SMA50', 'SMA200']).index.get_loc(data.dropna(subset=['SMA50', 'SMA200']).index[0])

    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        if i == first_valid_index and shares == 0 and (data['SMA50'].iloc[i] > data['SMA200'].iloc[i]):
            signal_type, signal_code = "Initial Buy", 2
        else:
            signal_type, signal_code = None, data['Signal'].iloc[i]

        if signal_code == -2 and shares > 0:
            sale_value = shares * current_price
            gain_loss = sale_value - cost_basis
            transaction_log.append({"Date": data.index[i], "Signal": "Sell", "Price": current_price, 
                                   "Amount": sale_value, "Profit": gain_loss})
            cash, shares, cost_basis = sale_value, 0, 0
        elif signal_code == 2 and cash > 0:
            purchase_value = cash
            transaction_log.append({"Date": data.index[i], "Signal": signal_type if signal_type else "Buy", 
                                   "Price": current_price, "Amount": purchase_value, "Profit": 0})
            shares, cost_basis, cash = cash / current_price, purchase_value, 0
        
        portfolio_values.append(cash + (shares * current_price))
    
    data['Strategy Value'] = portfolio_values
    forecast_data = calculate_forecast_metrics(data)
    
    # Prophet forecast
    df_prophet = data.reset_index().rename(columns={'index': 'ds', 'Date': 'ds', 'Close': 'y'})
    forecast_daily = None
    if len(df_prophet) > 1:
        try:
            model = Prophet(daily_seasonality=True).fit(df_prophet)
            future_daily = model.make_future_dataframe(periods=365*2)
            forecast_daily = model.predict(future_daily)
        except Exception as e:
            st.warning(f"Could not generate Prophet forecast: {e}")

    # --- Capture Output as Strings ---
    forecast_summary = "No active forecast."
    if show_forecast and forecast_data and forecast_data['signal'] != 'NEUTRAL':
        forecast_summary = (
            f"Signal: {forecast_data['signal']} (Confidence: {forecast_data['confidence']}%)\n"
            f"Current Price: ${forecast_data['current_price']:.2f}\n"
            f"SMA50: ${forecast_data['sma50']:.2f} | SMA200: ${forecast_data['sma200']:.2f}\n"
            f"RSI: {forecast_data['rsi']:.1f}\n"
            f"Gap: ${forecast_data['current_gap']:.2f} ({forecast_data['gap_percentage']:.2f}%)\n"
            f"Days to Crossover: {forecast_data['days_to_crossover']:.1f}\n"
            f"MACD: {forecast_data['macd_value']:.2f} | Signal: {forecast_data['macd_signal']:.2f}\n"
            f"Volume Trend: {forecast_data['volume_trend']:.1f}%"
        )
        
        if forecast_accuracy['buy']['total'] > 0 or forecast_accuracy['sell']['total'] > 0:
            forecast_summary += "\n\nHISTORICAL FORECAST ACCURACY:\n"
            if forecast_accuracy['buy']['total'] > 0:
                buy_acc = (forecast_accuracy['buy']['correct'] / forecast_accuracy['buy']['total']) * 100
                forecast_summary += f"Buy: {forecast_accuracy['buy']['correct']}/{forecast_accuracy['buy']['total']} ({buy_acc:.1f}%)\n"
            if forecast_accuracy['sell']['total'] > 0:
                sell_acc = (forecast_accuracy['sell']['correct'] / forecast_accuracy['sell']['total']) * 100
                forecast_summary += f"Sell: {forecast_accuracy['sell']['correct']}/{forecast_accuracy['sell']['total']} ({sell_acc:.1f}%)"

    final_strategy_value = data['Strategy Value'].iloc[-1]
    final_bh_value = data['Buy and Hold'].iloc[-1]
    strategy_return = ((final_strategy_value - initial_capital) / initial_capital) * 100
    bh_return = ((final_bh_value - initial_capital) / initial_capital) * 100
    
    performance_summary = (
        f"Strategy Final Value: ${final_strategy_value:,.2f} ({strategy_return:+.2f}%)\n"
        f"Buy & Hold Final Value: ${final_bh_value:,.2f} ({bh_return:+.2f}%)\n"
        f"Outperformance: {strategy_return - bh_return:+.2f}%\n"
        f"Total Transactions: {len(transaction_log)}"
    )
    # --- End Capture ---

    # Create 3-row chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.50, 0.25, 0.25],
                        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
                        subplot_titles=(f'{ticker} Price Chart', 'Portfolio Value ($10,000 Initial Investment)', 'Recent Transactions'))

    # Row 1: Price chart
    if forecast_daily is not None:
        fig.add_trace(go.Scatter(x=forecast_daily['ds'], y=forecast_daily['yhat_upper'], mode='lines', 
                                line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty', 
                                name='Uncertainty'), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast_daily['ds'], y=forecast_daily['yhat_lower'], mode='lines', 
                                line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty', 
                                showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast_daily['ds'], y=forecast_daily['yhat'], mode='lines', 
                                line=dict(color='rgb(31, 119, 180)', width=2), name='Forecast'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='black', width=2), 
                            name=f'{ticker} Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', 
                            line=dict(color='orange', width=1.5, dash='dash'), name='50-Day SMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', 
                            line=dict(color='purple', width=1.5, dash='dash'), name='200-Day SMA'), row=1, col=1)

    buy_signals = data[data['Signal'] == 2].copy()
    sell_signals = data[data['Signal'] == -2].copy()
    forecast_buy_signals = data[data['Forecast_Buy'] == True].copy()
    forecast_sell_signals = data[data['Forecast_Sell'] == True].copy()
    
    if len(forecast_buy_signals) > 0:
        fig.add_trace(go.Scatter(x=forecast_buy_signals.index, y=forecast_buy_signals['Close'], mode='markers', 
                                marker=dict(symbol='diamond', color='lightgreen', size=10, 
                                           line=dict(color='darkgreen', width=1.5)), 
                                name='Buy Forecast', legendrank=3), row=1, col=1)
    
    if len(forecast_sell_signals) > 0:
        fig.add_trace(go.Scatter(x=forecast_sell_signals.index, y=forecast_sell_signals['Close'], mode='markers',
                                marker=dict(symbol='diamond', color='lightcoral', size=10, 
                                           line=dict(color='darkred', width=1.5)),
                                name='Sell Forecast', legendrank=4), row=1, col=1)
    
    # Current forecast star
    if show_forecast and forecast_data and forecast_data['signal'] != 'NEUTRAL' and forecast_data['confidence'] >= 50:
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1]
        
        if 'BUY' in forecast_data['signal']:
            marker_color = 'gold' if forecast_data['confidence'] >= 70 else 'yellow'
            marker_size = 20 if forecast_data['confidence'] >= 70 else 16
            arrow_color = 'darkorange'
        else:
            marker_color = 'orange' if forecast_data['confidence'] >= 70 else 'lightsalmon'
            marker_size = 20 if forecast_data['confidence'] >= 70 else 16
            arrow_color = 'darkred'
        
        fig.add_trace(go.Scatter(x=[current_date], y=[current_price], mode='markers',
                                marker=dict(symbol='star', color=marker_color, size=marker_size, 
                                           line=dict(color=arrow_color, width=2)),
                                name='Current Forecast', legendrank=0,
                                hovertemplate=f"<b>ACTIVE FORECAST</b><br>{forecast_data['signal']}<br>Confidence: {forecast_data['confidence']}%<br>Est. {forecast_data['days_to_crossover']:.1f} days<extra></extra>"),
                     row=1, col=1)
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                            marker=dict(symbol='triangle-up', color='green', size=13), 
                            name='Buy Signal', legendrank=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                            marker=dict(symbol='triangle-down', color='red', size=13), 
                            name='Sell Signal', legendrank=2), row=1, col=1)

    # Row 2: Portfolio performance
    fig.add_trace(go.Scatter(x=data.index, y=data['Buy and Hold'], mode='lines',
                            line=dict(color='gray', width=2), name='Buy & Hold', legendrank=5), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Strategy Value'], mode='lines',
                            line=dict(color='blue', width=2.5), name='MA Strategy', legendrank=6), row=2, col=1)
    fig.add_hline(y=initial_capital, line=dict(color='red', dash='dash', width=1), 
                  annotation_text=f"Initial ${initial_capital:,.0f}", row=2, col=1)

    # Row 3: Transaction table
    if transaction_log:
        log_df = pd.DataFrame(transaction_log).tail(20)
        fig.add_trace(go.Table(
            header=dict(values=['Date', 'Signal', 'Price', 'Amount', 'Profit'], 
                       font=dict(size=11), align="left", fill_color='lightgray'),
            cells=dict(values=[log_df['Date'].dt.strftime('%Y-%m-%d'), log_df['Signal'], 
                              log_df['Price'].map('${:,.2f}'.format), 
                              log_df['Amount'].map('${:,.2f}'.format), 
                              log_df['Profit'].apply(lambda x: f"${x:,.2f}" if x != 0 else "-")], 
                      font=dict(size=10), align="left", height=20)
        ), row=3, col=1)
    
    chart_title = f'{ticker} Analysis: Enhanced MA Strategy | Strategy: ${final_strategy_value:,.0f} vs Buy&Hold: ${final_bh_value:,.0f}'
    fig.update_layout(
        title=dict(text=chart_title, x=0.5, xanchor='center', font=dict(size=13)),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, 
                    bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='gray', borderwidth=1, font=dict(size=9)),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        yaxis1_title="Price (USD)",
        yaxis2_title="Portfolio Value (USD)",
        dragmode='zoom',
        height=1000
    )
    fig.update_xaxes(rangeselector=dict(buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=3, label="3m", step="month", stepstepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=2, label="2y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all", label="All")
    ]), bgcolor="rgba(255, 255, 255, 0.8)", activecolor="lightblue",
       x=0.01, y=1.04, xanchor='left', yanchor='top', font=dict(size=9)), row=1, col=1)
    
    return fig, performance_summary, forecast_summary


def process_all_tickers_mod(ticker_list, logger_callback):
    """
    Mass processing mode.
    MODIFIED: Uses Neon DB and returns DataFrame.
    """
    logger_callback(f"Processing {len(ticker_list)} tickers...")
    
    for i, ticker in enumerate(ticker_list, 1):
        logger_callback(f"\n[{i}/{len(ticker_list)}] Processing: {ticker}")
        data, _ = get_stock_data_with_caching(ticker)
        if data.empty:
            logger_callback(f"Skipping {ticker}, no data found.")
            continue
        
        data = generate_enhanced_signals(data)
        
        is_invested = False
        cleaned_signals = []
        for index, row in data.iterrows():
            signal = row['Signal']
            if not is_invested and signal == 2:
                cleaned_signals.append(signal)
                is_invested = True
            elif is_invested and signal == -2:
                cleaned_signals.append(signal)
                is_invested = False
            else:
                cleaned_signals.append(0)
        data['Signal'] = cleaned_signals
        
        signals_to_store = data[data['Signal'] != 0].copy()
        if not signals_to_store.empty:
            for idx in signals_to_store.index:
                try:
                    signal_position = data.index.get_loc(idx)
                    if isinstance(signal_position, slice):
                        signal_position = signal_position.start
                    confidence = calculate_signal_confidence(data, signal_position)
                    signals_to_store.loc[idx, 'Confidence_%'] = confidence
                except Exception as e:
                    signals_to_store.loc[idx, 'Confidence_%'] = 0
            
            store_signals_in_db(ticker, signals_to_store)
            logger_callback(f"Stored {len(signals_to_store)} signals for {ticker}")
    
    # Export latest signals from Neon DB
    logger_callback("\n--- Exporting Latest Signals ---")
    latest_signals_df = load_csv_data("latest_signals.csv", from_db=True)
    
    if latest_signals_df is not None and not latest_signals_df.empty:
        # Save a local copy for download
        output_csv = 'latest_signals.csv'
        latest_signals_df.to_csv(output_csv, index=False)
        logger_callback(f"Exported latest signals to '{output_csv}' for download.")
    else:
        logger_callback("No latest signals found in database.")
        latest_signals_df = pd.DataFrame()
    
    logger_callback("\nMass processing complete!")
    return latest_signals_df


def forecast_mode_mod(ticker_list, logger_callback):
    """
    Forecast mode.
    MODIFIED: Uses Neon DB and returns DataFrame.
    """
    logger_callback(f"Analyzing {len(ticker_list)} tickers for forecasts...\n")
    
    forecasts = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(ticker_list, 1):
        status_text = f"[{i}/{len(ticker_list)}] Analyzing {ticker}..."
        logger_callback(status_text)
        progress_bar.progress(i / len(ticker_list))
        
        data, _ = get_stock_data_with_caching(ticker)
        if data.empty or len(data) < 200:
            logger_callback(f"[{i}/{len(ticker_list)}] Analyzing {ticker}... Insufficient data")
            continue
        
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = calculate_rsi(data['Close'], period=14)
        
        forecast = calculate_forecast_metrics(data)
        if forecast and forecast['signal'] != 'NEUTRAL' and forecast['confidence'] >= 40:
            forecast['ticker'] = ticker
            forecasts.append(forecast)
            logger_callback(f"[{i}/{len(ticker_list)}] Analyzing {ticker}... {forecast['signal']} ({forecast['confidence']}%)")
        else:
            logger_callback(f"[{i}/{len(ticker_list)}] Analyzing {ticker}... Neutral")
    
    progress_bar.empty()
    
    if forecasts:
        forecast_df = pd.DataFrame(forecasts)
        
        # Sort by priority
        priority_order = {'STRONG_BUY_FORECAST': 1, 'STRONG_SELL_FORECAST': 2, 'BUY_FORECAST': 3, 'SELL_FORECAST': 4}
        forecast_df['priority'] = forecast_df['signal'].map(priority_order)
        forecast_df = forecast_df.sort_values(['priority', 'confidence'], ascending=[True, False])
        forecast_df['date'] = datetime.now().strftime('%Y-%m-%d')
        
        output_df = forecast_df[['ticker', 'date', 'signal', 'confidence', 'days_to_crossover', 'current_price', 
                                 'gap_percentage', 'rsi', 'macd_histogram', 'price_roc', 'volume_trend', 
                                 'convergence_rate']].copy()
        output_df.columns = ['Ticker', 'Date', 'Forecast_Signal', 'Confidence_%', 'Days_To_Crossover', 'Current_Price', 
                            'Gap_%', 'RSI', 'MACD_Histogram', 'Price_ROC_%', 'Volume_Trend_%', 'Convergence_Rate']
        
        output_df['Date'] = pd.to_datetime(output_df['Date']).dt.strftime('%Y-%m-%d')
        # Rounding for display
        output_df['Days_To_Crossover'] = output_df['Days_To_Crossover'].round(1)
        output_df['Current_Price'] = output_df['Current_Price'].round(2)
        
        # Save to Neon DB
        try:
            with conn.session() as s:
                output_df.to_sql("forecast_signals", 
                                 con=s.bind, 
                                 if_exists='replace', # Replace old forecasts
                                 index=False,
                                 method='multi')
            logger_callback(f"Forecast complete! {len(output_df)} opportunities found and saved to DB.")
        except Exception as e:
            logger_callback(f"Error saving forecasts to Neon DB: {e}")

        # Save a local copy for download
        output_file = 'forecast_signals.csv'
        output_df.to_csv(output_file, index=False)
        
        return output_df
    else:
        logger_callback("\nNo forecast signals found.")
        return pd.DataFrame()


# --- Streamlit UI (Modified to load from DB) ---

st.sidebar.title("TickSignals")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Analyzer", "Forecast Signals", "Trade Signals"]
)

if app_mode == "Analyzer":
    st.title("Signal Analyzer")
    
    single_tab, mass_tab, forecast_tab = st.tabs(["Single Ticker", "Mass Run", "Forecast Run"])

    with single_tab:
        st.header("Single Ticker Analysis")
        ticker_input = st.text_input("Enter Ticker Symbol", "SPY").upper()
        
        if st.button("Run Analysis"):
            if not ticker_input:
                st.warning("Please enter a ticker symbol.")
            else:
                with st.spinner(f"Analyzing {ticker_input}..."):
                    
                    # Display Stock Info Metrics
                    stock_info = get_stock_info(ticker_input)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Last Price", stock_info["Price"])
                    col2.metric("Market Cap", stock_info["Market Cap"])
                    col3.metric("Volume", stock_info["Volume"])
                    col4.metric("P/E Ratio", stock_info["P/E Ratio"])
                    
                    # Generate and Display Chart
                    fig, perf_summary, forecast_summary = generate_chart_for_ticker_mod(ticker_input, show_forecast=True)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col_perf, col_forecast = st.columns(2)
                        with col_perf:
                            st.subheader("Performance Summary")
                            st.text(perf_summary)
                        with col_forecast:
                            st.subheader("Forecast Summary")
                            st.text(forecast_summary)
                    else:
                        st.error("Could not generate chart for this ticker.")

    with mass_tab:
        st.header("Mass Run - All Tickers")
        st.write("Process a list of tickers to generate and store trade signals in the persistent database.")
        
        # Option to upload a new CSV
        uploaded_file = st.file_uploader("Upload Ticker CSV (Optional, uses 'vanguard.csv' if not provided)", type="csv")
        
        if st.button("Run Mass Signal Analysis"):
            ticker_list = []
            if uploaded_file is not None:
                try:
                    # Use the uploaded file's in-memory representation
                    ticker_df = pd.read_csv(uploaded_file)
                    ticker_list = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list)} tickers from uploaded file.")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
            else:
                try:
                    ticker_df = pd.read_csv('vanguard.csv') # Default file
                    ticker_list = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list)} tickers from default 'vanguard.csv'.")
                except FileNotFoundError:
                    st.error("Default 'vanguard.csv' not found. Please upload a ticker file.")
            
            if ticker_list:
                st.subheader("Processing Log")
                log_area = st.container(height=300)
                
                def mass_logger(message):
                    log_area.info(message)

                with st.spinner("Processing all tickers... This may take a long time."):
                    latest_signals_df = process_all_tickers_mod(ticker_list, mass_logger)
                
                st.success("Mass Run Complete!")
                st.dataframe(latest_signals_df)
                
                if not latest_signals_df.empty:
                    st.download_button(
                        label="Download latest_signals.csv",
                        data=latest_signals_df.to_csv(index=False).encode('utf-8'),
                        file_name="latest_signals.csv",
                        mime="text/csv"
                    )

    with forecast_tab:
        st.header("Predictive Forecast Run")
        st.write("Analyze all tickers for potential *upcoming* buy/sell signals. Results are saved to the persistent database.")
        
        # Option to upload a new CSV
        uploaded_file_forecast = st.file_uploader("Upload Ticker CSV (Optional, uses 'vanguard.csv' if not provided)", type="csv", key="forecast_uploader")

        if st.button("Run Forecast Analysis"):
            ticker_list_forecast = []
            if uploaded_file_forecast is not None:
                try:
                    ticker_df = pd.read_csv(uploaded_file_forecast)
                    ticker_list_forecast = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list_forecast)} tickers from uploaded file.")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
            else:
                try:
                    ticker_df = pd.read_csv('vanguard.csv') # Default file
                    ticker_list_forecast = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list_forecast)} tickers from default 'vanguard.csv'.")
                except FileNotFoundError:
                    st.error("Default 'vanguard.csv' not found. Please upload a ticker file.")

            if ticker_list_forecast:
                st.subheader("Processing Log")
                log_area_forecast = st.container(height=300)

                def forecast_logger(message):
                    log_area_forecast.info(message)
                
                with st.spinner("Running forecast analysis... This may take a long time."):
                    forecast_df = forecast_mode_mod(ticker_list_forecast, forecast_logger)
                
                st.success("Forecast Run Complete!")
                st.dataframe(forecast_df)

                if not forecast_df.empty:
                    st.download_button(
                        label="Download forecast_signals.csv",
                        data=forecast_df.to_csv(index=False).encode('utf-8'),
                        file_name="forecast_signals.csv",
                        mime="text/csv"
                    )

elif app_mode == "Forecast Signals":
    st.title("Forecast Signals")
    st.info("This table shows predictive signals from the persistent database, sorted by date (newest first).")
    st.write("Run a 'Forecast Run' from the 'Analyzer' tab to generate or update this data.")

    # Load from the 'forecast_signals' table in Neon DB
    forecast_data = load_csv_data("forecast_signals.csv", from_db=True)
    
    if forecast_data is not None:
        if not forecast_data.empty:
            if 'Date' in forecast_data.columns:
                st.dataframe(forecast_data.sort_values(by='Date', ascending=False), use_container_width=True)
            else:
                st.warning("Forecast data is missing the 'Date' column for sorting.")
                st.dataframe(forecast_data, use_container_width=True)
        else:
            st.warning("No forecast data found. Please run a 'Forecast Run' from the 'Analyzer' tab.")
    else:
        st.warning("No 'forecast_signals' table found in database. Please run a 'Forecast Run' from the 'Analyzer' tab to generate it.")

elif app_mode == "Trade Signals":
    st.title("Trade Signals")
    st.info("This table shows the latest confirmed trade signal for each ticker from the persistent database, sorted by date (newest first).")
    st.write("Run a 'Mass Run' from the 'Analyzer' tab to generate or update this data.")
    
    # Load from the 'all_signals' table in Neon DB
    signal_data = load_csv_data("latest_signals.csv", from_db=True)
    
    if signal_data is not None:
        if not signal_data.empty:
            if 'Date' in signal_data.columns:
                st.dataframe(signal_data.sort_values(by='Date', ascending=False), use_container_width=True)
            else:
                st.warning("Signal data is missing the 'Date' column for sorting.")
                st.dataframe(signal_data, use_container_width=True)
        else:
            st.warning("No signal data found. Please run a 'Mass Run' from the 'Analyzer' tab.")
    else:
        st.warning("No 'all_signals' table found in database. Please run a 'Mass Run' from the 'Analyzer' tab to generate it.")
