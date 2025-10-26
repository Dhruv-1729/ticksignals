import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import sqlite3
from datetime import datetime, timedelta
import re
import os
from scipy import stats

def get_stock_data_with_caching(ticker, db_name='stock_data.db'):
    """Fetches stock data, using a local SQLite database for caching."""
    sanitized_ticker = re.sub(r'[^a-zA-Z0-9_]', '', ticker)
    conn = sqlite3.connect(db_name)
    
    try:
        last_date_str = pd.read_sql(f'SELECT MAX(Date) FROM "{sanitized_ticker}"', conn).iloc[0, 0]
        start_date = (pd.to_datetime(last_date_str) + timedelta(days=1)).strftime('%Y-%m-%d')
    except Exception:
        start_date = "2010-01-01"

    if start_date <= datetime.now().strftime('%Y-%m-%d'):
        print(f"Fetching new data for {ticker} from {start_date}...")
        new_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
            new_data.rename(columns=str.capitalize, inplace=True)
            df_to_write = new_data.reset_index()
            df_to_write.rename(columns={'index': 'Date'}, inplace=True)
            df_to_write.to_sql(sanitized_ticker, conn, if_exists='append', index=False)
    
    try:
        data = pd.read_sql(f'SELECT * FROM "{sanitized_ticker}"', conn)
        date_col = next((col for col in data.columns if col.lower() == 'date'), None)
        if not date_col:
            conn.close()
            return pd.DataFrame()
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)
    except Exception:
        data = pd.DataFrame()
    
    conn.close()
    return data

def store_signals_in_db(ticker, signals_df, db_name='stock_data.db'):
    """Stores newly generated signals in the database with confidence scores."""
    conn = sqlite3.connect(db_name)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS all_signals (
            Date TEXT, Ticker TEXT, Signal TEXT, Price REAL, Confidence_Pct INTEGER,
            PRIMARY KEY (Date, Ticker)
        )
    ''')
    
    for date, row in signals_df.iterrows():
        signal_type = "Buy" if row['Signal'] == 2 else "Sell"
        confidence = int(row.get('Confidence_%', 0))
        conn.execute("INSERT OR REPLACE INTO all_signals VALUES (?, ?, ?, ?, ?)",
                     (date.strftime('%Y-%m-%d'), ticker, signal_type, row['Close'], confidence))
    
    conn.commit()
    conn.close()
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
    Enhanced signal generation with improved sell signal accuracy and profit protection.
    Uses multiple confirmations and momentum filters with better exit strategies.
    """
    data['Signal'] = 0
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    
    # Additional technical indicators for better sell signals
    data['SMA10'] = data['Close'].rolling(window=10).mean()
    data['ATR'] = calculate_atr(data, period=14)
    data['BB_Upper'] = data['SMA20'] + (2 * data['Close'].rolling(20).std())
    data['BB_Lower'] = data['SMA20'] - (2 * data['Close'].rolling(20).std())
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
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
    data['Price_ROC_5'] = data['Close'].pct_change(5) * 100
    
    # Trend detection
    data['Is_Trending'] = is_trending_market(data)
    
    # Calculate profit/loss tracking for better sell decisions
    data['Entry_Price'] = 0.0
    data['Max_Price_Since_Entry'] = 0.0
    data['Current_Gain_Loss_Pct'] = 0.0
    
    # Track entry prices and gains for profit protection
    entry_price = 0
    max_price = 0
    for i in range(len(data)):
        if data['Signal'].iloc[i] == 2:  # Buy signal
            entry_price = data['Close'].iloc[i]
            max_price = entry_price
            data.iloc[i, data.columns.get_loc('Entry_Price')] = entry_price
        elif data['Signal'].iloc[i] == -2:  # Sell signal
            entry_price = 0
            max_price = 0
            data.iloc[i, data.columns.get_loc('Entry_Price')] = 0
        
        if entry_price > 0:
            current_price = data['Close'].iloc[i]
            max_price = max(max_price, current_price)
            gain_loss_pct = ((current_price - entry_price) / entry_price) * 100
            data.iloc[i, data.columns.get_loc('Entry_Price')] = entry_price
            data.iloc[i, data.columns.get_loc('Max_Price_Since_Entry')] = max_price
            data.iloc[i, data.columns.get_loc('Current_Gain_Loss_Pct')] = gain_loss_pct
    
    # BUY CONDITIONS - Multiple confirmations needed (keeping existing good logic)
    buy_conditions = (
        # Early crossover detection - SMA20 crossing above SMA50
        (data['SMA20'] > data['SMA50']) & 
        (data['SMA20'].shift(1) <= data['SMA50'].shift(1)) &
        
        # Long-term trend confirmation - must be above or approaching SMA200
        (data['SMA50'] >= data['SMA200'] * 0.98) &  # Within 2% of SMA200
        
        # RSI not overbought
        (data['RSI'] < 70) &
        (data['RSI'] > 30) &  # Not oversold either (avoid falling knives)
        
        # MACD confirmation - momentum turning positive
        (data['MACD_Hist'] > data['MACD_Hist'].shift(1)) &  # Histogram improving
        
        # Price momentum positive or turning
        (data['Price_ROC'] > -5) &  # Not in freefall
        
        # Volume confirmation
        (data['Volume_Ratio'] > 0.8) &  # Reasonable volume
        
        # Trending market (avoid choppy conditions)
        (data['Is_Trending'] == True)
    )
    
    # Significantly improved SELL CONDITIONS with multiple layers of protection
    sell_conditions = (
        # Primary sell condition: SMA20 crossing below SMA50 with confirmation
        (data['SMA20'] < data['SMA50']) & 
        (data['SMA20'].shift(1) >= data['SMA50'].shift(1)) &
        
        # AND at least one of these confirmations:
        (
            # Confirmation 1: Strong bearish momentum with RSI overbought
            ((data['RSI'] > 75) & (data['MACD_Hist'] < 0) & (data['Price_ROC_5'] < -2)) |
            
            # Confirmation 2: Profit protection - protect gains above 15%
            ((data['Current_Gain_Loss_Pct'] > 15) & (data['RSI'] > 65) & (data['MACD_Hist'] < data['MACD_Hist'].shift(1))) |
            
            # Confirmation 3: Strong bearish momentum with volume
            ((data['Price_ROC'] < -10) & (data['MACD_Hist'] < data['MACD_Hist'].shift(1)) & (data['Volume_Ratio'] > 1.2)) |
            
            # Confirmation 4: Bollinger Band breakdown with momentum
            ((data['BB_Position'] < 0.2) & (data['MACD_Hist'] < 0) & (data['Price_ROC_5'] < -3)) |
            
            # Confirmation 5: Multiple timeframe bearish alignment
            ((data['SMA10'] < data['SMA20']) & (data['RSI'] > 60) & (data['MACD_Hist'] < 0) & (data['Price_ROC'] < -5)) |
            
            # Confirmation 6: Stop loss protection - prevent major losses
            (data['Current_Gain_Loss_Pct'] < -8)
        )
    )
    
    data.loc[buy_conditions, 'Signal'] = 2
    data.loc[sell_conditions, 'Signal'] = -2
    
    return data

def calculate_confidence_score(forecast_data, signal_type):
    """
    Enhanced confidence score calculation with better sell signal accuracy.
    Higher score = stronger signal alignment.
    """
    score = 0
    max_score = 0
    
    # Days to crossover scoring (20 points)
    max_score += 20
    days = abs(forecast_data['days_to_crossover'])
    if 3 <= days <= 8: score += 20
    elif 2 <= days <= 10: score += 15
    elif days <= 12: score += 10
    
    # RSI scoring with improved sell signal logic (20 points)
    max_score += 20
    rsi = forecast_data['rsi']
    if signal_type == 'BUY':
        if 30 <= rsi <= 50: score += 20
        elif 50 < rsi <= 60: score += 15
        elif 25 <= rsi < 30 or 60 < rsi <= 65: score += 10
    else:  # SELL - more aggressive RSI scoring for better exits
        if 70 <= rsi <= 80: score += 20  # Strong overbought
        elif 65 <= rsi <= 85: score += 15  # Good overbought
        elif 60 <= rsi <= 90: score += 10  # Moderate overbought
        elif 55 <= rsi <= 95: score += 5   # Weak overbought
    
    # MACD alignment scoring (20 points)
    max_score += 20
    macd_hist = forecast_data['macd_histogram']
    macd_slope = forecast_data['macd_slope']
    if signal_type == 'BUY':
        if macd_hist > -0.05 and macd_slope > 0.01: score += 20
        elif macd_hist > -0.1 and macd_slope > 0: score += 12
        elif macd_hist > -0.2: score += 6
    else:  # SELL - more sensitive to bearish MACD
        if macd_hist < -0.05 and macd_slope < -0.01: score += 20
        elif macd_hist < 0 and macd_slope < 0: score += 15
        elif macd_hist < 0.05 and macd_slope < 0: score += 10
        elif macd_hist < 0.1: score += 5
    
    # Volume trend scoring (15 points)
    max_score += 15
    vol_trend = forecast_data['volume_trend']
    if signal_type == 'BUY':
        if vol_trend > 15: score += 15
        elif vol_trend > 5: score += 10
        elif vol_trend > 0: score += 5
        elif vol_trend > -10: score += 3
    else:  # SELL - higher volume on sell signals is good
        if vol_trend > 20: score += 15  # Strong selling volume
        elif vol_trend > 10: score += 12
        elif vol_trend > 0: score += 8
        elif vol_trend > -5: score += 5
    
    # Convergence rate scoring (15 points)
    max_score += 15
    conv_rate = forecast_data['convergence_rate']
    if conv_rate > 0.03: score += 15
    elif conv_rate > 0.02: score += 10
    elif conv_rate > 0.01: score += 7
    elif conv_rate > 0.005: score += 4
    
    # Price momentum scoring with enhanced sell logic (20 points)
    max_score += 20
    price_roc = forecast_data['price_roc']
    if signal_type == 'BUY':
        if -3 <= price_roc <= 2: score += 20
        elif -5 <= price_roc <= 5: score += 15
        elif -8 <= price_roc <= 8: score += 10
    else:  # SELL - more aggressive on negative momentum
        if price_roc < -5: score += 20  # Strong negative momentum
        elif price_roc < -2: score += 15  # Good negative momentum
        elif price_roc < 0: score += 10   # Weak negative momentum
        elif price_roc < 3: score += 5    # Neutral momentum
    
    # Additional sell-specific scoring (10 points)
    if signal_type == 'SELL':
        max_score += 10
        # Higher confidence for sell signals with multiple confirmations
        confirmation_count = 0
        if rsi > 70: confirmation_count += 1
        if macd_hist < 0: confirmation_count += 1
        if price_roc < -2: confirmation_count += 1
        if vol_trend > 5: confirmation_count += 1
        
        if confirmation_count >= 3: score += 10
        elif confirmation_count >= 2: score += 7
        elif confirmation_count >= 1: score += 4
    
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
        'ticker': None,
        'current_gap': current_gap,
        'gap_percentage': (abs(current_gap) / data['Close'].iloc[-1]) * 100,
        'sma50_velocity': sma50_velocity,
        'sma200_velocity': sma200_velocity,
        'relative_velocity': relative_velocity,
        'convergence_rate': convergence_rate,
        'days_to_crossover': days_to_crossover,
        'macd_value': macd.iloc[-1],
        'macd_signal': signal_line.iloc[-1],
        'macd_histogram': macd_histogram.iloc[-1],
        'macd_slope': macd_slope,
        'price_roc': price_roc,
        'volume_trend': volume_trend,
        'rsi': rsi_value,
        'current_price': data['Close'].iloc[-1],
        'sma50': data['SMA50'].iloc[-1],
        'sma200': data['SMA200'].iloc[-1],
        'current_state': current_state
    }
    
    # Improved BUY forecast with tighter criteria
    if current_state == "below" and relative_velocity > 0 and 2 <= days_to_crossover <= 12:
        if (25 <= rsi_value <= 65 and macd_histogram.iloc[-1] > -0.2 and 
            convergence_rate > 0.008 and price_roc > -10):
            
            confidence = calculate_confidence_score(forecast, 'BUY')
            
            # Only generate signal if confidence is decent
            if confidence >= 40:
                if confidence >= 70:
                    forecast['signal'] = 'STRONG_BUY_FORECAST'
                else:
                    forecast['signal'] = 'BUY_FORECAST'
                forecast['confidence'] = confidence
            else:
                forecast['signal'] = 'NEUTRAL'
                forecast['confidence'] = 0
        else:
            forecast['signal'] = 'NEUTRAL'
            forecast['confidence'] = 0
            
    # Improved SELL forecast with tighter criteria
    elif current_state == "above" and relative_velocity < 0 and 2 <= days_to_crossover <= 12:
        if (25 <= rsi_value <= 75 and macd_histogram.iloc[-1] < 0.2 and 
            convergence_rate > 0.008 and price_roc < 10):
            
            confidence = calculate_confidence_score(forecast, 'SELL')
            
            # Only generate signal if confidence is decent
            if confidence >= 40:
                if confidence >= 70:
                    forecast['signal'] = 'STRONG_SELL_FORECAST'
                else:
                    forecast['signal'] = 'SELL_FORECAST'
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

def generate_chart_for_ticker(ticker, show_forecast=False):
    """Generates chart for a ticker with all analysis."""
    data = get_stock_data_with_caching(ticker)
    if data.empty:
        print(f"ERROR: No data found for ticker '{ticker}'")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Generate enhanced signals
    data = generate_enhanced_signals(data)
    
    # Alternate signals
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
    
    # Historical forecasts - only high confidence ones
    data['Forecast_Buy'] = False
    data['Forecast_Sell'] = False
    forecast_accuracy = {'buy': {'correct': 0, 'total': 0}, 'sell': {'correct': 0, 'total': 0}}
    
    print(f"Calculating historical forecasts...")
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
    
    print(f"Found {forecast_accuracy['buy']['total']} buy and {forecast_accuracy['sell']['total']} sell forecasts")

    # Performance simulation with $10,000 initial capital
    initial_capital = 10000.00
    cash, shares, cost_basis, transaction_log, portfolio_values = initial_capital, 0, 0, [], []
    
    # Buy and Hold calculation
    first_price_bh = data['Close'].dropna().iloc[0]
    data['Buy and Hold'] = (initial_capital / first_price_bh) * data['Close']
    
    # Find first valid index where both SMAs exist
    first_valid_index = data.dropna(subset=['SMA50', 'SMA200']).index.get_loc(data.dropna(subset=['SMA50', 'SMA200']).index[0])

    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        
        # Initial buy if above SMA200
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
        model = Prophet(daily_seasonality=True).fit(df_prophet)
        future_daily = model.make_future_dataframe(periods=365*2)
        forecast_daily = model.predict(future_daily)

    # Display forecast info
    print(f"\nGenerating interactive plot for {ticker}...")
    if show_forecast and forecast_data and forecast_data['signal'] != 'NEUTRAL':
        print("\n" + "="*60)
        print(f"FORECAST ANALYSIS FOR {ticker}")
        print("="*60)
        print(f"Signal: {forecast_data['signal']} (Confidence: {forecast_data['confidence']}%)")
        print(f"Current Price: ${forecast_data['current_price']:.2f}")
        print(f"SMA50: ${forecast_data['sma50']:.2f} | SMA200: ${forecast_data['sma200']:.2f}")
        print(f"RSI: {forecast_data['rsi']:.1f}")
        print(f"Gap: ${forecast_data['current_gap']:.2f} ({forecast_data['gap_percentage']:.2f}%)")
        print(f"Days to Crossover: {forecast_data['days_to_crossover']:.1f}")
        print(f"MACD: {forecast_data['macd_value']:.2f} | Signal: {forecast_data['macd_signal']:.2f}")
        print(f"Volume Trend: {forecast_data['volume_trend']:.1f}%")
        print("="*60)
        
        if forecast_accuracy['buy']['total'] > 0 or forecast_accuracy['sell']['total'] > 0:
            print(f"\nHISTORICAL FORECAST ACCURACY:")
            print("-"*60)
            if forecast_accuracy['buy']['total'] > 0:
                buy_acc = (forecast_accuracy['buy']['correct'] / forecast_accuracy['buy']['total']) * 100
                print(f"Buy: {forecast_accuracy['buy']['correct']}/{forecast_accuracy['buy']['total']} ({buy_acc:.1f}%)")
            if forecast_accuracy['sell']['total'] > 0:
                sell_acc = (forecast_accuracy['sell']['correct'] / forecast_accuracy['sell']['total']) * 100
                print(f"Sell: {forecast_accuracy['sell']['correct']}/{forecast_accuracy['sell']['total']} ({sell_acc:.1f}%)")
            print("="*60 + "\n")

    # Calculate performance metrics
    final_strategy_value = data['Strategy Value'].iloc[-1]
    final_bh_value = data['Buy and Hold'].iloc[-1]
    strategy_return = ((final_strategy_value - initial_capital) / initial_capital) * 100
    bh_return = ((final_bh_value - initial_capital) / initial_capital) * 100
    
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"  Strategy Final Value: ${final_strategy_value:,.2f} ({strategy_return:+.2f}%)")
    print(f"  Buy & Hold Final Value: ${final_bh_value:,.2f} ({bh_return:+.2f}%)")
    print(f"  Outperformance: {strategy_return - bh_return:+.2f}%")
    print(f"  Total Transactions: {len(transaction_log)}")

    # Create 3-row chart (Price, Portfolio Performance, Transaction Table)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.50, 0.25, 0.25],
                        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
                        subplot_titles=(f'{ticker} Price Chart', 'Portfolio Value ($10,000 Initial Investment)', 'Recent Transactions'))

    # Row 1: Price chart with signals
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
    
    # Historical forecast diamonds
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
        
        fig.add_annotation(x=current_date, y=current_price * 1.05,
                          text=f"<b>{forecast_data['signal']}</b><br>({forecast_data['confidence']}% | {forecast_data['days_to_crossover']:.1f} days)",
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=arrow_color,
                          ax=0, ay=-40, font=dict(size=10, color=arrow_color),
                          bgcolor='rgba(255, 255, 255, 0.9)', bordercolor=arrow_color, borderwidth=2,
                          row=1, col=1)
    
    # Actual signals
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                            marker=dict(symbol='triangle-up', color='green', size=13), 
                            name='Buy Signal', legendrank=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                            marker=dict(symbol='triangle-down', color='red', size=13), 
                            name='Sell Signal', legendrank=2), row=1, col=1)

    # Row 2: Portfolio performance comparison
    fig.add_trace(go.Scatter(x=data.index, y=data['Buy and Hold'], mode='lines',
                            line=dict(color='gray', width=2), name='Buy & Hold', legendrank=5), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Strategy Value'], mode='lines',
                            line=dict(color='blue', width=2.5), name='MA Strategy', legendrank=6), row=2, col=1)
    
    # Add reference line at initial investment
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
    else:
        fig.add_trace(go.Table(
            header=dict(values=['Date', 'Signal', 'Price', 'Amount', 'Profit'], 
                       font=dict(size=11), align="left", fill_color='lightgray'),
            cells=dict(values=[[], [], [], [], []], 
                      font=dict(size=10), align="left")
        ), row=3, col=1)

    # Fetch stats
    try:
        stock_info = yf.Ticker(ticker).info
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        
        market_cap = stock_info.get('marketCap', 'N/A')
        pe_ratio = stock_info.get('trailingPE', 'N/A')
        eps = stock_info.get('trailingEps', 'N/A')
        beta = stock_info.get('beta', 'N/A')
        volume = stock_info.get('volume', 'N/A')
        
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = "N/A"
        
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        eps_str = f"${eps:.2f}" if isinstance(eps, (int, float)) else "N/A"
        beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
        volume_str = f"{volume:,}" if isinstance(volume, int) else "N/A"
        
        stats_text = (f"Price: ${current_price:.2f} | "
                     f"Prev Close: ${prev_close:.2f} | "
                     f"Market Cap: {market_cap_str} | "
                     f"P/E: {pe_str} | "
                     f"EPS: {eps_str} | "
                     f"Beta: {beta_str} | "
                     f"Volume: {volume_str}")
    except Exception as e:
        print(f"Error fetching stock info: {e}")
        stats_text = f"Current Price: ${data['Close'].iloc[-1]:.2f}"

    chart_title = f'{ticker} Analysis: Enhanced MA Strategy | Strategy: ${final_strategy_value:,.0f} vs Buy&Hold: ${final_bh_value:,.0f}'
    if show_forecast and forecast_data and forecast_data['signal'] != 'NEUTRAL':
        chart_title += f" | Forecast: {forecast_data['signal']} ({forecast_data['confidence']}%)"

    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0.5,
            xanchor='center',
            font=dict(size=13)
        ),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, 
                    bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='gray', borderwidth=1, font=dict(size=9)),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        yaxis1_title="Price (USD)",
        yaxis2_title="Portfolio Value (USD)",
        dragmode='zoom',
        height=1000
    )

    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        xanchor='center', yanchor='bottom',
        showarrow=False,
        font=dict(size=10, color='#333'),
        bgcolor='rgba(255, 255, 200, 0.9)',
        bordercolor='#999',
        borderwidth=1,
        borderpad=6
    )
    
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1m", step="month", stepmode="backward"),
                                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                                dict(count=2, label="2y", step="year", stepmode="backward"),
                                                dict(count=5, label="5y", step="year", stepmode="backward"),
                                                dict(step="all", label="All")],
                                       bgcolor="rgba(255, 255, 255, 0.8)", activecolor="lightblue",
                                       x=0.01, y=1.04, xanchor='left', yanchor='top', font=dict(size=9)),
                     row=1, col=1)
    fig.show()

def process_single_ticker(ticker):
    """Processes a single ticker with full analysis and chart."""
    print(f"\n{'='*60}")
    print(f"SINGLE TICKER MODE: {ticker}")
    print('='*60)
    
    data = get_stock_data_with_caching(ticker)
    if data.empty:
        print(f"ERROR: Could not fetch data for {ticker}")
        return
    
    # Generate enhanced signals
    data = generate_enhanced_signals(data)
    
    # Alternate signals (only take opposite signals)
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
        # Calculate confidence
        for idx in signals_to_store.index:
            try:
                signal_position = data.index.get_loc(idx)
                if isinstance(signal_position, slice):
                    signal_position = signal_position.start
                confidence = calculate_signal_confidence(data, signal_position)
                signals_to_store.loc[idx, 'Confidence_%'] = confidence
            except Exception as e:
                print(f"Warning: Could not calculate confidence: {e}")
                signals_to_store.loc[idx, 'Confidence_%'] = 0
        
        store_signals_in_db(ticker, signals_to_store)
    
    generate_chart_for_ticker(ticker, show_forecast=True)

def calculate_signal_confidence(data, signal_index):
    """
    Enhanced confidence calculation with improved sell signal accuracy.
    """
    score = 0
    max_score = 0
    
    signal_type = data['Signal'].iloc[signal_index]
    rsi = data['RSI'].iloc[signal_index]
    current_price = data['Close'].iloc[signal_index]
    sma20 = data['SMA20'].iloc[signal_index]
    sma50 = data['SMA50'].iloc[signal_index]
    sma200 = data['SMA200'].iloc[signal_index]
    macd_hist = data['MACD_Hist'].iloc[signal_index]
    price_roc = data['Price_ROC'].iloc[signal_index]
    
    volume_ratio = data['Volume_Ratio'].iloc[signal_index] if 'Volume_Ratio' in data.columns else 1
    
    # Get additional indicators if available
    price_roc_5 = data['Price_ROC_5'].iloc[signal_index] if 'Price_ROC_5' in data.columns else price_roc
    bb_position = data['BB_Position'].iloc[signal_index] if 'BB_Position' in data.columns else 0.5
    current_gain_loss = data['Current_Gain_Loss_Pct'].iloc[signal_index] if 'Current_Gain_Loss_Pct' in data.columns else 0
    
    # Gap size at crossover (20 points) - tighter is better
    max_score += 20
    gap_pct = abs((sma20 - sma50) / current_price) * 100
    if gap_pct < 0.3: score += 20
    elif gap_pct < 0.8: score += 15
    elif gap_pct < 1.5: score += 10
    elif gap_pct < 2.5: score += 5
    
    # RSI positioning with enhanced sell logic (25 points)
    max_score += 25
    if signal_type == 2:  # BUY
        if 35 <= rsi <= 55: score += 25
        elif 30 <= rsi <= 60: score += 18
        elif 25 <= rsi <= 65: score += 10
    else:  # SELL - more aggressive RSI scoring
        if 75 <= rsi <= 85: score += 25  # Strong overbought
        elif 70 <= rsi <= 90: score += 20  # Good overbought
        elif 65 <= rsi <= 95: score += 15  # Moderate overbought
        elif 60 <= rsi <= 100: score += 10  # Weak overbought
        elif 55 <= rsi <= 100: score += 5   # Very weak overbought
    
    # MACD alignment (20 points)
    max_score += 20
    if signal_type == 2:  # BUY
        if macd_hist > 0: score += 20
        elif macd_hist > -0.3: score += 12
        elif macd_hist > -0.8: score += 6
    else:  # SELL - more sensitive to bearish MACD
        if macd_hist < -0.1: score += 20  # Strong bearish
        elif macd_hist < 0: score += 15   # Bearish
        elif macd_hist < 0.1: score += 10  # Weak bearish
        elif macd_hist < 0.3: score += 5   # Very weak bearish
    
    # Volume strength (15 points)
    max_score += 15
    if signal_type == 2:  # BUY
        if volume_ratio > 1.5: score += 15
        elif volume_ratio > 1.2: score += 12
        elif volume_ratio > 0.9: score += 8
        elif volume_ratio > 0.7: score += 4
    else:  # SELL - higher volume is better for sell signals
        if volume_ratio > 2.0: score += 15  # Very high selling volume
        elif volume_ratio > 1.5: score += 12  # High selling volume
        elif volume_ratio > 1.2: score += 10  # Good selling volume
        elif volume_ratio > 0.9: score += 6   # Moderate volume
        elif volume_ratio > 0.7: score += 3   # Low volume
    
    # Momentum alignment with enhanced sell logic (20 points)
    max_score += 20
    if signal_type == 2:  # BUY
        if price_roc > 2: score += 20
        elif price_roc > 0: score += 15
        elif price_roc > -3: score += 8
    else:  # SELL - more aggressive on negative momentum
        if price_roc < -8: score += 20  # Very strong negative momentum
        elif price_roc < -5: score += 18  # Strong negative momentum
        elif price_roc < -2: score += 15  # Good negative momentum
        elif price_roc < 0: score += 10   # Weak negative momentum
        elif price_roc < 3: score += 5    # Neutral momentum
    
    # Additional sell-specific scoring (10 points)
    if signal_type == -2:  # SELL signals get extra scoring
        max_score += 10
        confirmation_count = 0
        
        # Count confirmations for sell signal
        if rsi > 70: confirmation_count += 1
        if macd_hist < 0: confirmation_count += 1
        if price_roc < -2: confirmation_count += 1
        if price_roc_5 < -1: confirmation_count += 1
        if volume_ratio > 1.2: confirmation_count += 1
        if bb_position < 0.3: confirmation_count += 1  # Near lower Bollinger Band
        if current_gain_loss > 10: confirmation_count += 1  # Profit protection
        
        if confirmation_count >= 4: score += 10  # Very strong confirmation
        elif confirmation_count >= 3: score += 8  # Strong confirmation
        elif confirmation_count >= 2: score += 6  # Moderate confirmation
        elif confirmation_count >= 1: score += 3  # Weak confirmation
    
    confidence_pct = int((score / max_score) * 100)
    return confidence_pct


def process_all_tickers(ticker_list):
    """Mass processing mode with enhanced signals."""
    print("\n" + "="*60)
    print("MASS PROCESSING MODE")
    print("="*60)
    print(f"Processing {len(ticker_list)} tickers...")
    
    for i, ticker in enumerate(ticker_list, 1):
        print(f"\n[{i}/{len(ticker_list)}] Processing: {ticker}")
        data = get_stock_data_with_caching(ticker)
        if data.empty:
            print(f"Skipping {ticker}, no data found.")
            continue
        
        # Generate enhanced signals
        data = generate_enhanced_signals(data)
        
        # Alternate signals
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
            print(f"Stored {len(signals_to_store)} signals for {ticker}")
    
    # Export latest signals
    print("\n--- Exporting Latest Signals ---")
    conn = sqlite3.connect('stock_data.db')
    try:
        query = """
        SELECT t1.Date, t1.Ticker, t1.Signal, t1.Price, t1.Confidence_Pct as 'Confidence_%'
        FROM all_signals t1
        INNER JOIN (
            SELECT Ticker, MAX(Date) as MaxDate
            FROM all_signals
            GROUP BY Ticker
        ) t2 ON t1.Ticker = t2.Ticker AND t1.Date = t2.MaxDate
        ORDER BY t1.Confidence_Pct DESC, t1.Date DESC;
        """
        latest_signals_df = pd.read_sql_query(query, conn)
        
        if not latest_signals_df.empty:
            output_csv = 'latest_signals.csv'
            latest_signals_df.to_csv(output_csv, index=False)
            print(f"Exported latest signals to '{output_csv}'")
            
            print("\n" + "="*60)
            print("LATEST SIGNALS SUMMARY")
            print("="*60)
            if 'Confidence_%' in latest_signals_df.columns:
                avg_conf = latest_signals_df['Confidence_%'].mean()
                high_conf = len(latest_signals_df[latest_signals_df['Confidence_%'] >= 80])
                print(f"Total Signals: {len(latest_signals_df)}")
                print(f"Average Confidence: {avg_conf:.1f}%")
                print(f"High Confidence (≥70%): {high_conf}")
                print("="*60)
    except Exception as e:
        print(f"Error exporting signals: {e}")
    finally:
        conn.close()
    
    print("\nMass processing complete!")


def forecast_mode(ticker_list):
    """Forecast mode: Analyzes all tickers for upcoming buy/sell signals."""
    print("\n" + "="*60)
    print("FORECAST MODE - Predictive Signal Analysis")
    print("="*60)
    print(f"Analyzing {len(ticker_list)} tickers...\n")
    
    forecasts = []
    
    for i, ticker in enumerate(ticker_list, 1):
        print(f"[{i}/{len(ticker_list)}] Analyzing {ticker}...", end=' ')
        
        data = get_stock_data_with_caching(ticker)
        if data.empty or len(data) < 200:
            print("Insufficient data")
            continue
        
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = calculate_rsi(data['Close'], period=14)
        
        forecast = calculate_forecast_metrics(data)
        if forecast and forecast['signal'] != 'NEUTRAL' and forecast['confidence'] >= 40:
            forecast['ticker'] = ticker
            forecasts.append(forecast)
            print(f"{forecast['signal']} ({forecast['confidence']}%)")
        else:
            print("Neutral")
    
    if forecasts:
        # Build 60-day history of forecasts
        all_forecast_history = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        for i, ticker in enumerate(ticker_list, 1):
            print(f"[History {i}/{len(ticker_list)}] {ticker}...", end=' ', flush=True)
            
            data = get_stock_data_with_caching(ticker)
            if data.empty or len(data) < 200:
                print("skip")
                continue
            
            # Filter data to 60-day window
            data_window = data.loc[(data.index >= start_date) & (data.index <= end_date)].copy()
            
            if len(data_window) == 0:
                print("no data")
                continue
            
            # Calculate indicators for each day in the window
            for idx in data_window.index:
                historical_data = data.loc[data.index <= idx].copy()
                
                if len(historical_data) < 200:
                    continue
                
                historical_data['SMA50'] = historical_data['Close'].rolling(window=50).mean()
                historical_data['SMA200'] = historical_data['Close'].rolling(window=200).mean()
                historical_data['RSI'] = calculate_rsi(historical_data['Close'], period=14)
                
                forecast = calculate_forecast_metrics(historical_data)
                if forecast and forecast['signal'] != 'NEUTRAL':
                    # Apply strict confidence filtering
                    if forecast['signal'] in ['STRONG_BUY_FORECAST', 'STRONG_SELL_FORECAST']:
                        if forecast['confidence'] >= 80:  # Strong signals need 80%+
                            forecast['ticker'] = ticker
                            forecast['date'] = idx
                            all_forecast_history.append(forecast)
                    elif forecast['signal'] in ['BUY_FORECAST', 'SELL_FORECAST']:
                        if forecast['confidence'] >= 65:  # Regular signals need 65%+
                            forecast['ticker'] = ticker
                            forecast['date'] = idx
                            all_forecast_history.append(forecast)
            
            print("done")
        
        # Create dataframe from 60-day history
        if all_forecast_history:
            forecast_df = pd.DataFrame(all_forecast_history)
        else:
            forecast_df = pd.DataFrame(forecasts)
        
        # Sort by date (newest first), then by priority
        priority_order = {'STRONG_BUY_FORECAST': 1, 'STRONG_SELL_FORECAST': 2, 'BUY_FORECAST': 3, 'SELL_FORECAST': 4}
        forecast_df['priority'] = forecast_df['signal'].map(priority_order)
        forecast_df = forecast_df.sort_values(['date'], ascending=[False], ignore_index=True) if 'date' in forecast_df.columns else forecast_df.sort_values(['priority', 'confidence'], ascending=[True, False])
        
        output_df = forecast_df[['ticker', 'date', 'signal', 'confidence', 'days_to_crossover', 'current_price', 
                                 'gap_percentage', 'rsi', 'macd_histogram', 'price_roc', 'volume_trend', 
                                 'convergence_rate']].copy() if 'date' in forecast_df.columns else forecast_df[['ticker', 'signal', 'confidence', 'days_to_crossover', 'current_price', 
                                 'gap_percentage', 'rsi', 'macd_histogram', 'price_roc', 'volume_trend', 
                                 'convergence_rate']].copy()
        output_df.columns = ['Ticker', 'Date', 'Forecast_Signal', 'Confidence_%', 'Days_To_Crossover', 'Current_Price', 
                            'Gap_%', 'RSI', 'MACD_Histogram', 'Price_ROC_%', 'Volume_Trend_%', 'Convergence_Rate'] if 'date' in forecast_df.columns else ['Ticker', 'Forecast_Signal', 'Confidence_%', 'Days_To_Crossover', 'Current_Price', 
                            'Gap_%', 'RSI', 'MACD_Histogram', 'Price_ROC_%', 'Volume_Trend_%', 'Convergence_Rate']
        
        # Format date column if it exists
        if 'Date' in output_df.columns:
            output_df['Date'] = pd.to_datetime(output_df['Date']).dt.strftime('%Y-%m-%d')
        
        output_df['Days_To_Crossover'] = output_df['Days_To_Crossover'].round(1)
        output_df['Current_Price'] = output_df['Current_Price'].round(2)
        output_df['Gap_%'] = output_df['Gap_%'].round(2)
        output_df['RSI'] = output_df['RSI'].round(1)
        output_df['MACD_Histogram'] = output_df['MACD_Histogram'].round(3)
        output_df['Price_ROC_%'] = output_df['Price_ROC_%'].round(2)
        output_df['Volume_Trend_%'] = output_df['Volume_Trend_%'].round(1)
        output_df['Convergence_Rate'] = output_df['Convergence_Rate'].round(3)
        
        output_file = 'forecast_signals.csv'
        output_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print(f"FORECAST RESULTS - {len(output_df)} Opportunities Found")
        print("="*60)
        print(output_df.to_string(index=False))
        print("\n" + "="*60)
        print(f"Forecast saved to '{output_file}'")
        print("="*60)
        
        strong_buy = output_df[output_df['Forecast_Signal'] == 'STRONG_BUY_FORECAST']
        buy = output_df[output_df['Forecast_Signal'] == 'BUY_FORECAST']
        strong_sell = output_df[output_df['Forecast_Signal'] == 'STRONG_SELL_FORECAST']
        sell = output_df[output_df['Forecast_Signal'] == 'SELL_FORECAST']
        
        print(f"\nSummary:")
        print(f"  Strong Buy: {len(strong_buy)} | Buy: {len(buy)}")
        print(f"  Strong Sell: {len(strong_sell)} | Sell: {len(sell)}")
        if len(output_df) > 0:
            print(f"  Avg Confidence: {output_df['Confidence_%'].mean():.1f}%")
            print(f"  Avg Days to Crossover: {output_df['Days_To_Crossover'].mean():.1f}")
    else:
        print("\nNo forecast signals found.")

def main_menu():
    """Main menu for program operation modes."""
    print("\n" + "="*60)
    print("STOCK SIGNAL ANALYZER")
    print("MA Crossover Strategy with Predictive Forecasting")
    print("="*60)
    print("\nAvailable Modes:")
    print("  1) MASS RUN - Process all tickers from CSV")
    print("  2) SINGLE RUN - Analyze one ticker with chart")
    print("  3) FORECAST RUN - Generate predictive signals")
    print("  4) EXIT")
    print("="*60)
    
    choice = input("\nSelect mode (1-4): ").strip()
    return choice

if __name__ == "__main__":
    csv_file = 'vanguard.csv'
    # csv_file = 'nasdaq_snp.csv'

    
    ticker_df = pd.read_csv(csv_file)
    ticker_list = ticker_df.iloc[:, 0].tolist()
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            process_all_tickers(ticker_list)
            input("\nPress Enter to return to menu...")
        elif choice == '2':
            ticker = input("\nEnter ticker symbol: ").strip().upper()
            if ticker:
                process_single_ticker(ticker)
            input("\nPress Enter to return to menu...")
        elif choice == '3':
            forecast_mode(ticker_list)
            input("\nPress Enter to return to menu...")
        elif choice == '4':
            print("\nExiting program")
            break
        else:
            print("\nInvalid choice. Please select 1-4.")
