import pandas as pd
import pandas_ta as ta
import numpy as np

def extract_v2_features(df_m5, return_full=False):
    """
    Extracts the advanced 15-feature array for the V2 Trend Model.
    Expected Input: df_m5 (DataFrame containing M5 candles with 'time', 'open', 'high', 'low', 'close')
    Returns: A single row DataFrame containing the latest feature vector if return_full is False.
             Otherwise returns the entire DataFrame with features attached.
    """
    df = df_m5.copy()
    
    # 1. Momentum History
    df['RSI'] = df.ta.rsi(length=14)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['RSI_Lag2'] = df['RSI'].shift(2)
    
    # 2. Trend Context
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['Dist_EMA_50'] = (df['close'] - df['EMA_50']) / df['close']
    df['Dist_EMA_200'] = (df['close'] - df['EMA_200']) / df['close']
    
    # 3. Volatility
    adx_df = df.ta.adx(length=14)
    if adx_df is not None and not adx_df.empty:
        adx_col = [col for col in adx_df.columns if col.startswith('ADX')]
        df['ADX'] = adx_df[adx_col[0]] if adx_col else 0
    else:
        df['ADX'] = 0
    df['ATR'] = df.ta.atr(length=14)
    
    # 4. Time Context
    # Ensure time is datetime format (if not already)
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
        
    df['Hour'] = df['time'].dt.hour
    df['DayOfWeek'] = df['time'].dt.dayofweek
    
    # 5. Velocity
    df['ROC_1'] = df.ta.roc(length=1)
    df['ROC_3'] = df.ta.roc(length=3)
    df['ROC_5'] = df.ta.roc(length=5)
    
    # 6. Candle Shape
    candle_range = df['high'] - df['low'] + 1e-8
    df['Rel_Body'] = abs(df['close'] - df['open']) / candle_range
    df['Rel_Upper_Wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
    df['Rel_Lower_Wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
    
    # Clean NaNs and infs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    if return_full:
        return df
        
    feature_cols = [
        'RSI', 'RSI_Lag1', 'RSI_Lag2',  
        'Dist_EMA_50', 'Dist_EMA_200',  
        'ADX', 'ATR',                   
        'Hour', 'DayOfWeek',            
        'ROC_1', 'ROC_3', 'ROC_5',      
        'Rel_Body', 'Rel_Upper_Wick', 'Rel_Lower_Wick'
    ]
    
    # Return the very last row's features as a 2D DataFrame (expected by sklearn predict_proba)
    X_live = df[feature_cols].iloc[[-1]]
    return X_live
