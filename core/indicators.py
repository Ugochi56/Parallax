import pandas as pd
import numpy as np

def calculate_rsi_simple(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss_safe = loss.fillna(0)
    rs = np.where(loss_safe != 0, gain / loss_safe, 1e10)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)

def calculate_adx_simple(df, period=14):
    if df is None or len(df) < period + 1:
        return pd.Series(dtype=float)
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    
    pdm_s = pd.Series(np.insert(pdm, 0, 0.0)).ewm(alpha=1/period, adjust=False).mean().values
    ndm_s = pd.Series(np.insert(ndm, 0, 0.0)).ewm(alpha=1/period, adjust=False).mean().values
    tr_s = pd.Series(np.insert(tr, 0, 0.0)).ewm(alpha=1/period, adjust=False).mean().values
    
    tr_s_safe = np.where(tr_s > 0, tr_s, 1e-8)
    pdi = 100 * (pdm_s / tr_s_safe)
    ndi = 100 * (ndm_s / tr_s_safe)
    sum_di = pdi + ndi
    dx = 100 * np.abs(pdi - ndi) / np.where(sum_di != 0, sum_di, 1e-8)
    return pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()

def calculate_atr_simple(df, period=14):
    if df is None or len(df) == 0:
        return 0.0
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    if len(h) < 2:
        return float(h[0] - l[0]) if len(h) else 0.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    res = pd.Series(np.insert(tr, 0, h[0] - l[0])).ewm(alpha=1/period, adjust=False).mean()
    if res.empty or pd.isna(res.iloc[-1]):
        return 0.0
    return float(res.iloc[-1])

def calculate_poc(df, num_bins=50):
    """
    Calculates the Point of Control (POC) using Tick Volume.
    Fast vectorized implementation using np.histogram.
    """
    if df is None or len(df) == 0 or 'tick_volume' not in df.columns:
        return 0.0
        
    min_price = float(df['low'].min())
    max_price = float(df['high'].max())
    
    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        return float(min_price) if (min_price is not None and not pd.isna(min_price)) else 0.0
        
    try:
        bins = np.linspace(min_price, max_price, num_bins + 1)
        typical_price = (df['high'].values + df['low'].values + df['close'].values) / 3.0
        vol_vals = df['tick_volume'].values
        counts, bin_edges = np.histogram(typical_price, bins=bins, weights=vol_vals)
        max_idx = int(np.argmax(counts))
        return float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0)
    except Exception as e:
        print(f"[ERROR] POC Calculation failed: {e}")
        return 0.0

def map_market_structure(df, lookback=2):
    """
    Calculates Pivot Highs and Lows (Fractals) to map market structure (HH, HL, LH, LL).
    lookback=2 means a pivot must be higher/lower than the 2 candles before and after it.
    Fast vectorized numpy implementation for high-speed replay.
    """
    if df is None or len(df) < lookback * 2:
        return []
        
    highs = df['high'].values
    lows = df['low'].values
    times = df['time'].values if 'time' in df.columns else range(len(df))
    n = len(highs)
    
    pivots = []
    for i in range(lookback, n - lookback):
        ch = highs[i]
        cl = lows[i]
        is_high = True
        is_low = True
        
        for j in range(1, lookback + 1):
            if highs[i-j] >= ch or highs[i+j] >= ch:
                is_high = False
            if lows[i-j] <= cl or lows[i+j] <= cl:
                is_low = False
                
        if is_high:
            pivots.append({'type': 'HIGH', 'price': float(ch), 'time': times[i], 'idx': i})
        if is_low:
            pivots.append({'type': 'LOW', 'price': float(cl), 'time': times[i], 'idx': i})
            
    # Classify structural shifts
    structure = []
    last_high = None
    last_low = None
    
    for p in pivots:
        if p['type'] == 'HIGH':
            p['struct'] = 'HH' if (last_high is None or p['price'] > last_high['price']) else 'LH'
            last_high = p
            structure.append(p)
        elif p['type'] == 'LOW':
            p['struct'] = 'LL' if (last_low is None or p['price'] < last_low['price']) else 'HL'
            last_low = p
            structure.append(p)
            
    return structure

def map_market_structure_two_tier(df, major_lookback=5, minor_lookback=2):
    """
    Two-Tier Market Structure & Inducement (IDM) Engine:
    - Major Structure (lookback=5): Institutional macro swing pivots (HH, HL, LH, LL).
    - Minor Structure (lookback=2): Internal retail fractals used to identify Inducement (IDM).
    
    Returns:
        major_struct: list of major pivots with structural classifications
        minor_struct: list of minor pivots with structural classifications
        idm_info: dict containing major/minor highs/lows and IDM levels
    """
    if df is None or len(df) < major_lookback * 2:
        return [], [], {
            'major_high': None, 'major_low': None,
            'idm_high': None, 'idm_low': None,
            'major_highs': [], 'major_lows': [],
            'minor_highs': [], 'minor_lows': []
        }
        
    major_struct = map_market_structure(df, lookback=major_lookback)
    minor_struct = map_market_structure(df, lookback=minor_lookback)
    
    major_highs = [p for p in major_struct if p['type'] == 'HIGH']
    major_lows = [p for p in major_struct if p['type'] == 'LOW']
    
    last_major_high = major_highs[-1] if major_highs else None
    last_major_low = major_lows[-1] if major_lows else None
    
    minor_highs = [p for p in minor_struct if p['type'] == 'HIGH']
    minor_lows = [p for p in minor_struct if p['type'] == 'LOW']
    
    idm_high = None
    if last_major_high and minor_highs:
        leg_minor_highs = [p for p in minor_highs if p['idx'] > last_major_high['idx'] and p['price'] < last_major_high['price']]
        if leg_minor_highs:
            idm_high = leg_minor_highs[-1]
        elif len(minor_highs) >= 2:
            idm_high = minor_highs[-1]
            
    idm_low = None
    if last_major_low and minor_lows:
        leg_minor_lows = [p for p in minor_lows if p['idx'] > last_major_low['idx'] and p['price'] > last_major_low['price']]
        if leg_minor_lows:
            idm_low = leg_minor_lows[-1]
        elif len(minor_lows) >= 2:
            idm_low = minor_lows[-1]
            
    idm_info = {
        'major_high': last_major_high,
        'major_low': last_major_low,
        'idm_high': idm_high,
        'idm_low': idm_low,
        'major_highs': major_highs,
        'major_lows': major_lows,
        'minor_highs': minor_highs,
        'minor_lows': minor_lows,
    }
    return major_struct, minor_struct, idm_info

def get_higher_timeframe_target(df_h1, df_m15, current_price, direction):
    """
    Scans higher timeframe data (H1 first, falling back to M15) for the nearest structural pivot.
    Used to resolve wide, technically sound Take Profit targets for lower timeframe entries.
    """
    # 1. Check H1 structure
    if df_h1 is not None and len(df_h1) >= 5:
        try:
            h1_struct = map_market_structure(df_h1, lookback=2)
            if direction == 'BUY':
                h1_target = min([p['price'] for p in h1_struct if p['type'] == 'HIGH' and p['price'] > current_price], default=None)
            else:
                h1_target = max([p['price'] for p in h1_struct if p['type'] == 'LOW' and p['price'] < current_price], default=None)

            if h1_target is not None:
                return h1_target
        except Exception as e:
            print(f"[WARNING] get_higher_timeframe_target H1 check failed: {e}")

    # 2. Fall back to M15 (df_adx) structure
    if df_m15 is not None and len(df_m15) >= 5:
        try:
            m15_struct = map_market_structure(df_m15, lookback=2)
            if direction == 'BUY':
                m15_target = min([p['price'] for p in m15_struct if p['type'] == 'HIGH' and p['price'] > current_price], default=None)
            else:
                m15_target = max([p['price'] for p in m15_struct if p['type'] == 'LOW' and p['price'] < current_price], default=None)
            if m15_target is not None:
                return m15_target
        except Exception as e:
            print(f"[WARNING] get_higher_timeframe_target M15 check failed: {e}")

    return None
