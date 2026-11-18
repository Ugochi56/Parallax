import pandas as pd
import pandas_ta as ta
import numpy as np

def extract_v3_features(df_m5, df_h1, df_h4, return_full=False):
    """
    Extracts the 21-dimensional multi-timeframe feature vector.
    
    Expected Input:
      - df_m5: M5 candles with 'time', 'open', 'high', 'low', 'close', 'tick_volume'
      - df_h1: H1 candles
      - df_h4: H4 candles
      
    Returns:
      - A single row DataFrame containing the latest feature vector (return_full=False)
      - Or the full merged DataFrame with features (return_full=True)
    """
    # 1. Process M5 Features
    m5 = df_m5.copy()
    m5['RSI'] = m5.ta.rsi(length=14)
    m5['RSI_Lag1'] = m5['RSI'].shift(1)
    m5['RSI_Lag2'] = m5['RSI'].shift(2)
    m5['EMA_50'] = m5.ta.ema(length=50)
    m5['EMA_200'] = m5.ta.ema(length=200)
    m5['Dist_EMA_50'] = (m5['close'] - m5['EMA_50']) / m5['close']
    m5['Dist_EMA_200'] = (m5['close'] - m5['EMA_200']) / m5['close']
    adx_m5 = m5.ta.adx(length=14)
    if adx_m5 is not None and not adx_m5.empty:
        adx_col = [col for col in adx_m5.columns if col.startswith('ADX')]
        m5['ADX'] = adx_m5[adx_col[0]] if adx_col else 0
    else:
        m5['ADX'] = 0
    m5['ATR'] = m5.ta.atr(length=14) / m5['close']  # Normalized as % of price for scale-invariance
    
    if not pd.api.types.is_datetime64_any_dtype(m5['time']):
        m5['time'] = pd.to_datetime(m5['time'])
        
    m5['Hour'] = m5['time'].dt.hour
    m5['DayOfWeek'] = m5['time'].dt.dayofweek
    m5['ROC_1'] = m5.ta.roc(length=1)
    m5['ROC_3'] = m5.ta.roc(length=3)
    m5['ROC_5'] = m5.ta.roc(length=5)
    c_range = m5['high'] - m5['low'] + 1e-8
    m5['Rel_Body'] = abs(m5['close'] - m5['open']) / c_range
    m5['Rel_Upper_Wick'] = (m5['high'] - m5[['open', 'close']].max(axis=1)) / c_range
    m5['Rel_Lower_Wick'] = (m5[['open', 'close']].min(axis=1) - m5['low']) / c_range

    # 2. Process H1 Features
    h1 = df_h1.copy()
    h1['H1_RSI'] = h1.ta.rsi(length=14)
    h1['H1_EMA_50'] = h1.ta.ema(length=50)
    h1['H1_Dist_EMA_50'] = (h1['close'] - h1['H1_EMA_50']) / h1['close']
    adx_h1 = h1.ta.adx(length=14)
    if adx_h1 is not None and not adx_h1.empty:
        adx_col = [col for col in adx_h1.columns if col.startswith('ADX')]
        h1['H1_ADX'] = adx_h1[adx_col[0]] if adx_col else 0
    else:
        h1['H1_ADX'] = 0
    
    if not pd.api.types.is_datetime64_any_dtype(h1['time']):
        h1['time'] = pd.to_datetime(h1['time'])
        
    h1_cols = ['time', 'H1_RSI', 'H1_Dist_EMA_50', 'H1_ADX']
    h1_features = h1[h1_cols].copy()

    # 3. Process H4 Features
    h4 = df_h4.copy()
    h4['H4_RSI'] = h4.ta.rsi(length=14)
    h4['H4_EMA_50'] = h4.ta.ema(length=50)
    h4['H4_Dist_EMA_50'] = (h4['close'] - h4['H4_EMA_50']) / h4['close']
    adx_h4 = h4.ta.adx(length=14)
    if adx_h4 is not None and not adx_h4.empty:
        adx_col = [col for col in adx_h4.columns if col.startswith('ADX')]
        h4['H4_ADX'] = adx_h4[adx_col[0]] if adx_col else 0
    else:
        h4['H4_ADX'] = 0
    
    if not pd.api.types.is_datetime64_any_dtype(h4['time']):
        h4['time'] = pd.to_datetime(h4['time'])
        
    h4_cols = ['time', 'H4_RSI', 'H4_Dist_EMA_50', 'H4_ADX']
    h4_features = h4[h4_cols].copy()

    # 4. Sort for merge_asof and ensure identical datetime64[ns] dtype to avoid pandas merge resolution errors
    def _to_clean_dt(s):
        dt_s = pd.to_datetime(s)
        if getattr(dt_s.dt, 'tz', None) is not None:
            dt_s = dt_s.dt.tz_localize(None)
        return dt_s.astype('datetime64[ns]')

    m5['time'] = _to_clean_dt(m5['time'])
    h1_features['time'] = _to_clean_dt(h1_features['time'])
    h4_features['time'] = _to_clean_dt(h4_features['time'])

    m5 = m5.sort_values('time')
    h1_features = h1_features.sort_values('time')
    h4_features = h4_features.sort_values('time')

    merged = pd.merge_asof(m5, h1_features, on='time', direction='backward')
    merged = pd.merge_asof(merged, h4_features, on='time', direction='backward')

    merged.replace([np.inf, -np.inf], 0, inplace=True)
    merged.fillna(0, inplace=True)

    feature_cols = [
        'RSI', 'RSI_Lag1', 'RSI_Lag2',  
        'Dist_EMA_50', 'Dist_EMA_200',  
        'ADX', 'ATR',                   
        'Hour', 'DayOfWeek',            
        'ROC_1', 'ROC_3', 'ROC_5',      
        'Rel_Body', 'Rel_Upper_Wick', 'Rel_Lower_Wick',
        'H1_RSI', 'H1_Dist_EMA_50', 'H1_ADX',
        'H4_RSI', 'H4_Dist_EMA_50', 'H4_ADX'
    ]

    if return_full:
        return merged, feature_cols

    X_live = merged[feature_cols].iloc[[-1]]
    return X_live
