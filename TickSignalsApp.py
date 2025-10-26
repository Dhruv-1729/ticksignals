import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import re
from scipy import stats
import hashlib

# --- App Configuration ---
st.set_page_config(page_title="TickSignals", layout="wide")

# --- Initialize Session State ---
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'page_visits' not in st.session_state:
    st.session_state.page_visits = 0
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]

# Increment page visits
st.session_state.page_visits += 1

# --- Database Connection ---
def get_db_connection():
    """Creates a SQLAlchemy engine for PostgreSQL connection"""
    try:
        db_url = st.secrets["connections"]["postgresql"]["url"]
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# --- Analytics Functions ---
def log_ticker_search(ticker):
    """Log ticker search to database"""
    engine = get_db_connection()
    if engine is None:
        return
    
    try:
        with engine.connect() as conn:
            # Create analytics table if it doesn't exist
            create_table = text("""
                CREATE TABLE IF NOT EXISTS analytics_searches (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            """)
            conn.execute(create_table)
            conn.commit()
            
            # Insert search log
            insert_query = text("""
                INSERT INTO analytics_searches (ticker, session_id)
                VALUES (:ticker, :session_id)
            """)
            conn.execute(insert_query, {
                "ticker": ticker,
                "session_id": st.session_state.session_id
            })
            conn.commit()
    except Exception as e:
        pass  # Silently fail to not disrupt user experience

def log_page_visit():
    """Log page visit to database"""
    engine = get_db_connection()
    if engine is None:
        return
    
    try:
        with engine.connect() as conn:
            # Create visits table if it doesn't exist
            create_table = text("""
                CREATE TABLE IF NOT EXISTS analytics_visits (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    page_count INTEGER
                )
            """)
            conn.execute(create_table)
            conn.commit()
            
            # Insert visit log
            insert_query = text("""
                INSERT INTO analytics_visits (session_id, page_count)
                VALUES (:session_id, :page_count)
            """)
            conn.execute(insert_query, {
                "session_id": st.session_state.session_id,
                "page_count": st.session_state.page_visits
            })
            conn.commit()
    except Exception as e:
        pass

def get_analytics_data():
    """Retrieve analytics data from database"""
    engine = get_db_connection()
    if engine is None:
        return None
    
    analytics = {}
    
    try:
        with engine.connect() as conn:
            # Total visits
            visits_query = text("SELECT COUNT(*) FROM analytics_visits")
            total_visits = conn.execute(visits_query).scalar() or 0
            analytics['total_visits'] = total_visits
            
            # Unique sessions
            sessions_query = text("SELECT COUNT(DISTINCT session_id) FROM analytics_visits")
            unique_sessions = conn.execute(sessions_query).scalar() or 0
            analytics['unique_sessions'] = unique_sessions
            
            # Current active users (last 5 minutes)
            active_query = text("""
                SELECT COUNT(DISTINCT session_id) 
                FROM analytics_visits 
                WHERE timestamp > NOW() - INTERVAL '5 minutes'
            """)
            active_users = conn.execute(active_query).scalar() or 0
            analytics['active_users'] = active_users
            
            # Most searched tickers
            popular_query = text("""
                SELECT ticker, COUNT(*) as search_count 
                FROM analytics_searches 
                GROUP BY ticker 
                ORDER BY search_count DESC 
                LIMIT 10
            """)
            popular_tickers = pd.read_sql_query(popular_query, conn)
            analytics['popular_tickers'] = popular_tickers
            
            # Recent searches
            recent_query = text("""
                SELECT ticker, timestamp, session_id 
                FROM analytics_searches 
                ORDER BY timestamp DESC 
                LIMIT 20
            """)
            recent_searches = pd.read_sql_query(recent_query, conn)
            analytics['recent_searches'] = recent_searches
            
            # Total searches
            total_searches_query = text("SELECT COUNT(*) FROM analytics_searches")
            total_searches = conn.execute(total_searches_query).scalar() or 0
            analytics['total_searches'] = total_searches
            
            # Visits over time (last 7 days)
            visits_time_query = text("""
                SELECT DATE(timestamp) as date, COUNT(*) as visits
                FROM analytics_visits
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            visits_over_time = pd.read_sql_query(visits_time_query, conn)
            analytics['visits_over_time'] = visits_over_time
            
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")
        return None
    
    return analytics

# Log this page visit
log_page_visit()

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def get_stock_data_with_caching(ticker):
    """Fetches stock data using PostgreSQL database for caching."""
    engine = get_db_connection()
    if engine is None:
        return pd.DataFrame(), ["Database connection failed"]
    
    sanitized_ticker = re.sub(r'[^a-zA-Z0-9_]', '', ticker).lower()
    table_name = f"stock_{sanitized_ticker}"
    
    fetch_log = []
    
    try:
        with engine.connect() as conn:
            check_table = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """)
            table_exists = conn.execute(check_table, {"table_name": table_name}).scalar()
            
            if table_exists:
                last_date_query = text(f'SELECT MAX("Date") FROM {table_name}')
                last_date_str = conn.execute(last_date_query).scalar()
                
                if last_date_str:
                    start_date = (pd.to_datetime(last_date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start_date = "2010-01-01"
            else:
                start_date = "2010-01-01"
                create_table = text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        "Date" DATE PRIMARY KEY,
                        "Open" FLOAT,
                        "High" FLOAT,
                        "Low" FLOAT,
                        "Close" FLOAT,
                        "Volume" BIGINT
                    )
                """)
                conn.execute(create_table)
                conn.commit()
    except Exception as e:
        fetch_log.append(f"Error checking existing data: {e}")
        start_date = "2010-01-01"
    
    if start_date <= datetime.now().strftime('%Y-%m-%d'):
        fetch_log.append(f"Fetching new data for {ticker} from {start_date}...")
        new_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        
        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
            new_data.rename(columns=str.capitalize, inplace=True)
            df_to_write = new_data.reset_index()
            df_to_write.rename(columns={'index': 'Date'}, inplace=True)
            
            try:
                df_to_write.to_sql(table_name, engine, if_exists='append', index=False)
                fetch_log.append(f"Added {len(df_to_write)} new records")
            except Exception as e:
                fetch_log.append(f"Error writing data: {e}")
    else:
        fetch_log.append(f"Data for {ticker} is up to date.")
    
    try:
        with engine.connect() as conn:
            query = text(f'SELECT * FROM {table_name} ORDER BY "Date"')
            data = pd.read_sql_query(query, conn)
            
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
            else:
                return pd.DataFrame(), fetch_log
                
    except Exception as e:
        fetch_log.append(f"Error reading data: {e}")
        data = pd.DataFrame()
    
    return data, fetch_log

def store_signals_in_db(ticker, signals_df):
    """Stores newly generated signals in the PostgreSQL database."""
    engine = get_db_connection()
    if engine is None:
        return
    
    try:
        with engine.connect() as conn:
            create_table = text("""
                CREATE TABLE IF NOT EXISTS all_signals (
                    "Date" DATE,
                    "Ticker" TEXT,
                    "Signal" TEXT,
                    "Price" FLOAT,
                    "Confidence_Pct" INTEGER,
                    PRIMARY KEY ("Date", "Ticker")
                )
            """)
            conn.execute(create_table)
            conn.commit()
            
            for date, row in signals_df.iterrows():
                signal_type = "Buy" if row['Signal'] == 2 else "Sell"
                confidence = int(row.get('Confidence_%', 0))
                
                insert_query = text("""
                    INSERT INTO all_signals ("Date", "Ticker", "Signal", "Price", "Confidence_Pct")
                    VALUES (:date, :ticker, :signal, :price, :confidence)
                    ON CONFLICT ("Date", "Ticker") 
                    DO UPDATE SET "Signal" = :signal, "Price" = :price, "Confidence_Pct" = :confidence
                """)
                
                conn.execute(insert_query, {
                    "date": date.strftime('%Y-%m-%d'),
                    "ticker": ticker,
                    "signal": signal_type,
                    "price": float(row['Close']),
                    "confidence": confidence
                })
            
            conn.commit()
            
    except Exception as e:
        st.error(f"Error storing signals: {e}")

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
    
    return adx > 20

def generate_enhanced_signals(data):
    """Enhanced signal generation with earlier entries and better exits."""
    data['Signal'] = 0
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    if 'Volume' in data.columns:
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    else:
        data['Volume_Ratio'] = 1
    
    data['Price_ROC'] = data['Close'].pct_change(10) * 100
    data['Is_Trending'] = is_trending_market(data)
    
    buy_conditions = (
        (data['SMA20'] > data['SMA50']) & 
        (data['SMA20'].shift(1) <= data['SMA50'].shift(1)) &
        (data['SMA50'] >= data['SMA200'] * 0.98) &
        (data['RSI'] < 70) &
        (data['RSI'] > 30) &
        (data['MACD_Hist'] > data['MACD_Hist'].shift(1)) &
        (data['Price_ROC'] > -5) &
        (data['Volume_Ratio'] > 0.8) &
        (data['Is_Trending'] == True)
    )
    
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
    """Calculate confidence score (0-100%) based on technical indicators."""
    score = 0
    max_score = 0
    
    max_score += 20
    days = abs(forecast_data['days_to_crossover'])
    if 3 <= days <= 8: score += 20
    elif 2 <= days <= 10: score += 15
    elif days <= 12: score += 10
    
    max_score += 15
    rsi = forecast_data['rsi']
    if signal_type == 'BUY':
        if 30 <= rsi <= 50: score += 15
        elif 50 < rsi <= 60: score += 10
        elif 25 <= rsi < 30 or 60 < rsi <= 65: score += 7
    else:
        if 50 <= rsi <= 70: score += 15
        elif 40 <= rsi < 50: score += 10
        elif 70 < rsi <= 75 or 35 <= rsi < 40: score += 7
    
    max_score += 20
    macd_hist = forecast_data['macd_histogram']
    macd_slope = forecast_data['macd_slope']
    if signal_type == 'BUY':
        if macd_hist > -0.05 and macd_slope > 0.01: score += 20
        elif macd_hist > -0.1 and macd_slope > 0: score += 12
        elif macd_hist > -0.2: score += 6
    else:
        if macd_hist < 0.05 and macd_slope < -0.01: score += 20
        elif macd_hist < 0.1 and macd_slope < 0: score += 12
        elif macd_hist < 0.2: score += 6
    
    max_score += 15
    vol_trend = forecast_data['volume_trend']
    if vol_trend > 15: score += 15
    elif vol_trend > 5: score += 10
    elif vol_trend > 0: score += 5
    elif vol_trend > -10: score += 3
    
    max_score += 15
    conv_rate = forecast_data['convergence_rate']
    if conv_rate > 0.03: score += 15
    elif conv_rate > 0.02: score += 10
    elif conv_rate > 0.01: score += 7
    elif conv_rate > 0.005: score += 4
    
    max_score += 15
    price_roc = forecast_data['price_roc']
    if signal_type == 'BUY':
        if -3 <= price_roc <= 2: score += 15
        elif -5 <= price_roc <= 5: score += 10
        elif -8 <= price_roc <= 8: score += 5
    else:
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
    """Enhanced confidence calculation based on signal quality."""
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
    
    max_score += 20
    gap_pct = abs((sma20 - sma50) / current_price) * 100
    if gap_pct < 0.3: score += 20
    elif gap_pct < 0.8: score += 15
    elif gap_pct < 1.5: score += 10
    elif gap_pct < 2.5: score += 5
    
    max_score += 25
    if signal_type == 2:
        if 35 <= rsi <= 55: score += 25
        elif 30 <= rsi <= 60: score += 18
        elif 25 <= rsi <= 65: score += 10
    else:
        if 45 <= rsi <= 65: score += 25
        elif 40 <= rsi <= 70: score += 18
        elif 35 <= rsi <= 75: score += 10
    
    max_score += 20
    if signal_type == 2:
        if macd_hist > 0: score += 20
        elif macd_hist > -0.3: score += 12
        elif macd_hist > -0.8: score += 6
    else:
        if macd_hist < 0: score += 20
        elif macd_hist < 0.3: score += 12
        elif macd_hist < 0.8: score += 6
    
    max_score += 15
    if volume_ratio > 1.5: score += 15
    elif volume_ratio > 1.2: score += 12
    elif volume_ratio > 0.9: score += 8
    elif volume_ratio > 0.7: score += 4
    
    max_score += 20
    if signal_type == 2:
        if price_roc > 2: score += 20
        elif price_roc > 0: score += 15
        elif price_roc > -3: score += 8
    else:
        if price_roc < -2: score += 20
        elif price_roc < 0: score += 15
        elif price_roc < 3: score += 8
    
    confidence_pct = int((score / max_score) * 100)
    return confidence_pct

# --- Streamlit Helper Functions ---

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
def load_csv_data(file_name):
    """Loads a CSV file, handling FileNotFoundError."""
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
    """Generates chart, performance, and forecast strings for Streamlit."""
    data, fetch_log = get_stock_data_with_caching(ticker)
    if data.empty:
        st.error(f"ERROR: No data found for ticker '{ticker}'")
        return None, "", "", pd.DataFrame()

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

    # Build signals table
    signals_table_data = []
    
    # Add actual signals
    actual_signals = data[data['Signal'] != 0].copy()
    for idx in actual_signals.index:
        signal_type = "Buy" if actual_signals.loc[idx, 'Signal'] == 2 else "Sell"
        signals_table_data.append({
            'Date': idx.strftime('%Y-%m-%d'),
            'Type': signal_type,
            'Price': f"${actual_signals.loc[idx, 'Close']:.2f}",
            'Signal_Sort': 1 if signal_type == "Buy" else 2
        })
    
    # Add forecast signals
    forecast_buys = data[data['Forecast_Buy'] == True].copy()
    for idx in forecast_buys.index:
        signals_table_data.append({
            'Date': idx.strftime('%Y-%m-%d'),
            'Type': 'Buy (Prediction)',
            'Price': f"${forecast_buys.loc[idx, 'Close']:.2f}",
            'Signal_Sort': 3
        })
    
    forecast_sells = data[data['Forecast_Sell'] == True].copy()
    for idx in forecast_sells.index:
        signals_table_data.append({
            'Date': idx.strftime('%Y-%m-%d'),
            'Type': 'Sell (Prediction)',
            'Price': f"${forecast_sells.loc[idx, 'Close']:.2f}",
            'Signal_Sort': 4
        })
    
    signals_df = pd.DataFrame(signals_table_data)
    if not signals_df.empty:
        signals_df = signals_df.sort_values('Date', ascending=False)
    
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

    performance_summary = ""

    # Create 2-row chart (removed portfolio value row)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.70, 0.30],
                        specs=[[{"type": "scatter"}], [{"type": "table"}]],
                        subplot_titles=(f'{ticker} Price Chart', 'Signals'))

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

    # Row 2: Signals table
    if not signals_df.empty:
        table_df = signals_df[['Date', 'Type', 'Price']].tail(30)
        fig.add_trace(go.Table(
            header=dict(values=['Date', 'Signal Type', 'Price'], 
                       font=dict(size=12), align="left", fill_color='lightgray'),
            cells=dict(values=[table_df['Date'], table_df['Type'], table_df['Price']], 
                      font=dict(size=11), align="left", height=25)
        ), row=2, col=1)
    
    chart_title = f'{ticker} Analysis'
    fig.update_layout(
        title=dict(text=chart_title, x=0.5, xanchor='center', font=dict(size=14)),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, 
                    bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='gray', borderwidth=1, font=dict(size=10)),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        yaxis1_title="Price (USD)",
        dragmode='pan',
        height=900
    )
    
    # Enable mobile-friendly zoom and pan
    fig.update_xaxes(
        fixedrange=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="rgba(255, 255, 255, 0.8)",
            activecolor="lightblue",
            x=0.01, y=1.02, xanchor='left', yanchor='top', font=dict(size=10)
        ),
        row=1, col=1
    )
    fig.update_yaxes(fixedrange=False, row=1, col=1)
    
    return fig, performance_summary, forecast_summary, signals_df


def process_all_tickers_mod(ticker_list, logger_callback):
    """Mass processing mode."""
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
    
    logger_callback("\n--- Exporting Latest Signals ---")
    engine = get_db_connection()
    latest_signals_df = pd.DataFrame()
    
    if engine:
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT t1."Date", t1."Ticker", t1."Signal", t1."Price", t1."Confidence_Pct" as "Confidence_%"
                    FROM all_signals t1
                    INNER JOIN (
                        SELECT "Ticker", MAX("Date") as "MaxDate"
                        FROM all_signals
                        GROUP BY "Ticker"
                    ) t2 ON t1."Ticker" = t2."Ticker" AND t1."Date" = t2."MaxDate"
                    ORDER BY t1."Confidence_Pct" DESC, t1."Date" DESC
                """)
                latest_signals_df = pd.read_sql_query(query, conn)
                
                if not latest_signals_df.empty:
                    output_csv = 'latest_signals.csv'
                    latest_signals_df.to_csv(output_csv, index=False)
                    logger_callback(f"Exported latest signals to '{output_csv}'")
        except Exception as e:
            logger_callback(f"Error exporting signals: {e}")
    
    logger_callback("\nMass processing complete!")
    return latest_signals_df


def forecast_mode_mod(ticker_list, logger_callback):
    """Forecast mode."""
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
        logger_callback("\nProcessing forecasts...")
        
        forecast_df = pd.DataFrame(forecasts)
        
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
        output_df['Days_To_Crossover'] = output_df['Days_To_Crossover'].round(1)
        output_df['Current_Price'] = output_df['Current_Price'].round(2)
        
        output_file = 'forecast_signals.csv'
        output_df.to_csv(output_file, index=False)
        
        logger_callback(f"Forecast complete! {len(output_df)} opportunities found.")
        return output_df
    else:
        logger_callback("\nNo forecast signals found.")
        return pd.DataFrame()


# --- Streamlit UI ---

# Admin Mode Toggle
if 'show_admin_login' not in st.session_state:
    st.session_state.show_admin_login = False

if not st.session_state.admin_mode:
    if st.sidebar.button("🔐"):
        st.session_state.show_admin_login = not st.session_state.show_admin_login
    
    if st.session_state.show_admin_login:
        with st.sidebar:
            admin_password = st.text_input("Enter Password:", type="password", key="admin_password_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if admin_password == "dhruv10":
                        st.session_state.admin_mode = True
                        st.session_state.show_admin_login = False
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error("Incorrect password")
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_admin_login = False
                    st.rerun()
else:
    st.sidebar.success("Admin Mode Active")
    if st.sidebar.button("🔓 Logout"):
        st.session_state.admin_mode = False
        st.rerun()

st.sidebar.title("TickSignals")

# Build tab list
tab_list = ["Ticker Analyzer", "Forecast Signals History", "Trade Signals History"]
if st.session_state.admin_mode:
    tab_list.append("Analytics Dashboard")

app_mode = st.sidebar.radio("Tabs", tab_list)

if app_mode == "Ticker Analyzer":
    st.title("Ticker Analyzer")
    
    single_tab, mass_tab, forecast_tab = st.tabs(["Single Ticker", "Mass Run", "Forecast Run"])

    with single_tab:
        st.header("Single Ticker Analysis")
        ticker_input = st.text_input("Enter Ticker Symbol", "").upper()
        
        if st.button("Run Analysis"):
            if not ticker_input:
                st.warning("Please enter a ticker symbol.")
            else:
                # Log the ticker search
                log_ticker_search(ticker_input)
                
                with st.spinner(f"Analyzing {ticker_input}..."):
                    
                    # Display Stock Info Metrics
                    stock_info = get_stock_info(ticker_input)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Last Price", stock_info["Price"])
                    col2.metric("Market Cap", stock_info["Market Cap"])
                    col3.metric("Volume", stock_info["Volume"])
                    col4.metric("P/E Ratio", stock_info["P/E Ratio"])
                    
                    # Generate and Display Chart
                    fig, perf_summary, forecast_summary, signals_df = generate_chart_for_ticker_mod(ticker_input, show_forecast=True)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'displaylogo': False
                        })
                        
                        if forecast_summary and forecast_summary != "No active forecast.":
                            st.subheader("Forecast Analysis")
                            st.text(forecast_summary)
                    else:
                        st.error("Could not generate chart for this ticker.")

    with mass_tab:
        st.header("Mass Run - All Tickers")
        st.write("Process a list of tickers to generate and store trade signals.")
        
        uploaded_file = st.file_uploader("Upload Ticker CSV (Optional, uses Vanguard ETF 1500 if not provided)", type="csv")
        
        if st.button("Run Mass Signal Analysis"):
            ticker_list = []
            if uploaded_file is not None:
                try:
                    ticker_df = pd.read_csv(uploaded_file)
                    ticker_list = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list)} tickers from uploaded file.")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
            else:
                try:
                    ticker_df = pd.read_csv('vanguard.csv')
                    ticker_list = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list)} tickers from Vanguard ETF 1500.")
                except FileNotFoundError:
                    st.error("Default 'vanguard.csv' not found. Please upload a ticker file.")
            
            if ticker_list:
                st.subheader("Processing Log")
                log_area = st.container(height=300)
                
                def mass_logger(message):
                    log_area.info(message)

                with st.spinner("Processing all tickers"):
                    latest_signals_df = process_all_tickers_mod(ticker_list, mass_logger)
                
                st.success("Mass Run Complete!")
                st.dataframe(latest_signals_df)
                
                st.download_button(
                    label="Download latest_signals.csv",
                    data=latest_signals_df.to_csv(index=False).encode('utf-8'),
                    file_name="latest_signals.csv",
                    mime="text/csv"
                )

    with forecast_tab:
        st.header("Prediction Forecast Run")
        st.write("Analyze all tickers for potential *upcoming* buy/sell signals.")
        
        uploaded_file_forecast = st.file_uploader("Upload Ticker CSV (Optional, uses Vanguard ETF 1500 if not provided)", type="csv", key="forecast_uploader")

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
                    ticker_df = pd.read_csv('vanguard.csv')
                    ticker_list_forecast = ticker_df.iloc[:, 0].tolist()
                    st.info(f"Using {len(ticker_list_forecast)} tickers from default 'vanguard.csv'.")
                except FileNotFoundError:
                    st.error("Default 'vanguard.csv' not found. Please upload a ticker file.")

            if ticker_list_forecast:
                st.subheader("Processing Log")
                log_area_forecast = st.container(height=300)

                def forecast_logger(message):
                    log_area_forecast.info(message)
                
                with st.spinner("Running forecast analysis... This may take a while"):
                    forecast_df = forecast_mode_mod(ticker_list_forecast, forecast_logger)
                
                st.success("Forecast Run Complete!")
                st.dataframe(forecast_df)

                st.download_button(
                    label="Download forecast_signals.csv",
                    data=forecast_df.to_csv(index=False).encode('utf-8'),
                    file_name="forecast_signals.csv",
                    mime="text/csv"
                )

elif app_mode == "Forecast Signals History":
    st.title("Forecast Signals History")

    forecast_data = load_csv_data("forecast_signals.csv")
    
    if forecast_data is not None:
        if not forecast_data.empty:
            if 'Date' in forecast_data.columns:
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
                st.dataframe(forecast_data.sort_values(by='Date', ascending=False), use_container_width=True)
            else:
                st.warning("Forecast file is missing the 'Date' column for sorting.")
                st.dataframe(forecast_data, use_container_width=True)
        else:
            st.warning("No forecast data found. Please run a 'Forecast Run' from the 'Analyzer' tab.")
    else:
        st.warning("No file found. Please run a 'Forecast Run' from the 'Analyzer' tab")

elif app_mode == "Trade Signals History":
    st.title("Trade Signals History")    
    signal_data = load_csv_data("latest_signals.csv")
    
    if signal_data is not None:
        if not signal_data.empty:
            if 'Date' in signal_data.columns:
                signal_data['Date'] = pd.to_datetime(signal_data['Date'])
                st.dataframe(signal_data.sort_values(by='Date', ascending=False), use_container_width=True)
            else:
                st.warning("Signal file is missing the 'Date' column for sorting.")
                st.dataframe(signal_data, use_container_width=True)
        else:
            st.warning("No signal data found. Please run a 'Mass Run' from the 'Analyzer' tab.")
    else:
        st.warning("No 'latest_signals.csv' file found. Please run a 'Mass Run' from the 'Analyzer' tab to generate it.")

elif app_mode == "Analytics Dashboard":
    if not st.session_state.admin_mode:
        st.error("Access Denied: Admin mode required")
    else:
        st.title("Analytics Dashboard")
        
        analytics = get_analytics_data()
        
        if analytics:
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Visits", f"{analytics['total_visits']:,}")
            with col2:
                st.metric("Unique Users", f"{analytics['unique_sessions']:,}")
            with col3:
                st.metric("Active Now", f"{analytics['active_users']}")
            with col4:
                st.metric("Total Searches", f"{analytics['total_searches']:,}")
            
            st.divider()
            
            # Two column layout
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Most Searched Tickers")
                if not analytics['popular_tickers'].empty:
                    st.dataframe(
                        analytics['popular_tickers'].rename(columns={'ticker': 'Ticker', 'search_count': 'Searches'}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No search data yet")
            
            with col_right:
                st.subheader("Recent Searches")
                if not analytics['recent_searches'].empty:
                    recent_df = analytics['recent_searches'][['ticker', 'timestamp']].copy()
                    recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(
                        recent_df.rename(columns={'ticker': 'Ticker', 'timestamp': 'Time'}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No search history yet")
            
            st.divider()
            
            # Visits over time chart
            st.subheader("Visits Over Time (Last 30 Days)")
            if not analytics['visits_over_time'].empty:
                fig_visits = go.Figure()
                fig_visits.add_trace(go.Scatter(
                    x=analytics['visits_over_time']['date'],
                    y=analytics['visits_over_time']['visits'],
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                fig_visits.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Visits",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_visits, use_container_width=True)
            else:
                st.info("Not enough data to display trend")
            
            st.divider()
            
            # Additional Stats
            st.subheader("Detailed Statistics")
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                avg_searches_per_user = analytics['total_searches'] / max(analytics['unique_sessions'], 1)
                st.metric("Avg Searches per User", f"{avg_searches_per_user:.1f}")
            
            with col_stat2:
                avg_visits_per_user = analytics['total_visits'] / max(analytics['unique_sessions'], 1)
                st.metric("Avg Visits per User", f"{avg_visits_per_user:.1f}")
            
        else:
            st.error("Unable to load analytics data")
