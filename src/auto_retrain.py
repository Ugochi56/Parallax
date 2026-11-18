import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

# Configure UTF-8 encoding for standard output to support printing emojis in Windows cmd/powershell
sys.stdout.reconfigure(encoding='utf-8')

# Must add root path so core and engine are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *
from engine.feature_extractor_v3 import extract_v3_features

HISTORIC_BARS = 200000

def get_ml_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    return df

def compute_triple_barrier_labels(df, lookahead=12, pt_mult=1.0, sl_mult=1.0):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['ATR'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    for i in range(n - lookahead):
        atr = atrs[i]
        # Convert normalized ATR (ratio) to dollar ATR for price barrier calculation
        dollar_atr = (atr * closes[i]) if atr < 1.0 else atr
        if dollar_atr <= 0 or np.isnan(dollar_atr):
            dollar_atr = 1.0
        pt = closes[i] + pt_mult * dollar_atr
        sl = closes[i] - sl_mult * dollar_atr
        
        window_highs = highs[i+1 : i+1+lookahead]
        window_lows = lows[i+1 : i+1+lookahead]
        
        hit_pt = np.where(window_highs >= pt)[0]
        hit_sl = np.where(window_lows <= sl)[0]
        
        first_pt = hit_pt[0] if len(hit_pt) > 0 else len(window_highs)
        first_sl = hit_sl[0] if len(hit_sl) > 0 else len(window_lows)
        
        if first_pt < first_sl:
            labels[i] = 1
        elif first_sl < first_pt:
            labels[i] = 0
        else:
            labels[i] = 1 if closes[i+lookahead] > closes[i] else 0
            
    return labels

def safe_save_model(model, target_path, retries=5, delay=1.0):
    """
    Saves joblib model atomically using a .tmp file and a retry loop to handle Windows file locks (PermissionError).
    """
    import time
    tmp_path = target_path + ".tmp"
    joblib.dump(model, tmp_path)
    for attempt in range(retries):
        try:
            os.replace(tmp_path, target_path)
            return True
        except PermissionError as e:
            if attempt == retries - 1:
                print(f"[ERROR] Could not save model to {target_path} after {retries} attempts: {e}")
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
                raise e
            time.sleep(delay)
    return False

def train_trend_brain(df_raw, feature_cols):
    print("\n🧠 1. Structuring V2 Trend Features...")
    df = df_raw.copy()
    
    # Core Live Labeling: Triple-Barrier Method (Look ahead 12 candles / 1 hour)
    label_lookahead = 12
    labels = compute_triple_barrier_labels(df, lookahead=label_lookahead, pt_mult=1.0, sl_mult=1.0)
    df['Target'] = labels
    
    # Drop lookahead boundary rows from dataset
    df = df.iloc[:-label_lookahead]
    
    X = df[feature_cols]
    y = df['Target']
    
    # Time-Series Split with Purging and Embargoing
    split_idx = int(len(df) * 0.85)
    embargo = 50
    
    # Purge: Training set ends before lookahead window at split boundary
    X_train = X.iloc[:split_idx - label_lookahead]
    y_train = y.iloc[:split_idx - label_lookahead]
    
    # Embargo: Test set starts after embargo period to decay feature correlation
    X_test = X.iloc[split_idx + embargo:]
    y_test = y.iloc[split_idx + embargo:]

    print(f"🤖 Training Deep Forest Array on {len(X_train)} instances...")
    # Regularized Random Forest parameters to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=10, 
        min_samples_leaf=5,
        min_samples_split=10, 
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("🧪 Validating Accuracy against unseen blind data...")
    test_probs = model.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    
    # Only test trades where the AI was confident
    confident_idx = max_probs > AI_TREND_THRESHOLD
    y_filtered = y_test[confident_idx]
    preds = model.predict(X_test)[confident_idx]
    
    if len(y_filtered) > 0:
        acc = accuracy_score(y_filtered, preds)
        print("="*40)
        print(f"✅ Trend Model Accuracy: {acc:.2%}")
        print(f"📊 Trades Setup Passed: {len(y_filtered)} / {len(y_test)}")
        print("="*40)
        
        # Lowered saving threshold to 51% since data leakage is now eliminated
        if acc >= 0.51:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), TREND_MODEL_FILE)
            safe_save_model(model, model_path)
            print(f"💾 Evolution Complete. {TREND_MODEL_FILE} permanently overwritten!")
        else:
            print("⚠️ Accuracy too low (Below 51%). Model Update REJECTED.")
    else:
        print("❌ No confidence setups identified. Skipping build.")


def train_reversal_brain(df_raw, feature_cols):
    print("\n🧠 2. Structuring V2 RSI Reversal Matrix...")
    df = df_raw.copy()
    
    # 1. Filter anomalies
    conditions = [
        df['RSI'] <= 30,
        df['RSI'] >= 70
    ]
    df['Anomaly_Trigger'] = np.select(conditions, [1, -1], default=0)
    anomaly_df = df[df['Anomaly_Trigger'] != 0].copy()
    
    if len(anomaly_df) < 50:
        print("🚫 Not enough RSI extremities found recently. Skipping Reversal model.")
        return

    print("⚖️ Deep Labeling Results (1.0 ATR Snap)...")
    labels = []
    
    # Use reset index to avoid datetime indexing offset errors
    anomaly_df = anomaly_df.reset_index()
    
    for i in range(len(anomaly_df)):
        trigger_type = anomaly_df.loc[i, 'Anomaly_Trigger']
        trigger_price = anomaly_df.loc[i, 'close']
        trigger_atr = anomaly_df.loc[i, 'ATR']
        original_idx = anomaly_df.loc[i, 'index']
        
        # Convert normalized ATR (ratio) to dollar ATR for price barrier calculation
        dollar_atr = (trigger_atr * trigger_price) if trigger_atr < 1.0 else trigger_atr
        if dollar_atr <= 0 or np.isnan(dollar_atr):
            dollar_atr = trigger_price * 0.0035

        # Look ahead 15 candles (1 hour 15 mins) safely using structural index
        max_idx = min(original_idx + 15, len(df))
        future_window = df.loc[original_idx+1 : max_idx]
        
        if trigger_type == 1:  # Oversold (BUY)
            if future_window['high'].max() >= trigger_price + (dollar_atr * 1.0):
                labels.append(1)
            else: labels.append(0)
        else: # Overbought (SELL)
            if future_window['low'].min() <= trigger_price - (dollar_atr * 1.0):
                labels.append(1)
            else: labels.append(0)
            
    anomaly_df['Target'] = labels
    
    X = anomaly_df[feature_cols]
    y = anomaly_df['Target']
    
    # Time-Series Split on Anomaly Indices with Purging & Embargoing
    split_idx = int(len(anomaly_df) * 0.85)
    split_original_idx = anomaly_df.loc[split_idx, 'index']
    
    # Purge: Training set contains anomalies occurring before split_original_idx - 15 (lookahead)
    train_mask = anomaly_df['index'] < split_original_idx - 15
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Embargo: Test set contains anomalies occurring after split_original_idx + 50 (decay time)
    test_mask = anomaly_df['index'] >= split_original_idx + 50
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"🤖 Training Reversal Neural Net on {len(X_train)} squeezes...")
    # Regularized Random Forest parameters to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=8, 
        min_samples_leaf=5, 
        random_state=42, 
        class_weight="balanced", 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    
    confident_idx = max_probs > AI_REVERSAL_THRESHOLD
    y_filtered = y_test[confident_idx]
    preds = model.predict(X_test)[confident_idx]
    
    if len(y_filtered) > 0:
        acc = accuracy_score(y_filtered, preds)
        print("="*40)
        print(f"✅ Reversal Model Accuracy: {acc:.2%}")
        print(f"📊 Trades Setup Passed: {len(y_filtered)} / {len(y_test)}")
        print("="*40)
        
        if acc >= 0.55:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), REVERSAL_MODEL_FILE)
            safe_save_model(model, model_path)
            print(f"💾 Evolution Complete. {REVERSAL_MODEL_FILE} permanently overwritten!")
        else:
            print("⚠️ Accuracy too low (Below 55%). Reversal Model Update REJECTED.")
    else:
        print("❌ No confidence setups identified. Skipping build.")


def train_autonomous_brain():
    print("[NEXUS AUTO-RETRAINER] Booting Master Orchestrator...")
    
    if not mt5.initialize():
        print("[ERROR] MT5 Init failed.")
        return
        
    print(f"[DOWNLOAD] Downloading latest {HISTORIC_BARS} M5 candles from Broker...")
    df_m5 = get_ml_data(SYMBOL, mt5.TIMEFRAME_M5, HISTORIC_BARS)
    
    print("[DOWNLOAD] Downloading corresponding H1 and H4 candles...")
    df_h1 = get_ml_data(SYMBOL, mt5.TIMEFRAME_H1, HISTORIC_BARS // 12 + 100)
    df_h4 = get_ml_data(SYMBOL, mt5.TIMEFRAME_H4, HISTORIC_BARS // 48 + 100)
    
    if df_m5 is None or df_h1 is None or df_h4 is None:
        print("[ERROR] Failed to pull data.")
        mt5.shutdown()
        return

    print("[FEATURE] Extracting 21 Advanced Multi-Timeframe Features...")
    df, feature_cols = extract_v3_features(df_m5, df_h1, df_h4, return_full=True)
    df = df.reset_index(drop=True)

    # 1. Train SMC Trend Filter
    train_trend_brain(df, feature_cols)

    # 2. Train RSI Mean-Reversion Model
    train_reversal_brain(df, feature_cols)

    mt5.shutdown()

if __name__ == "__main__":
    train_autonomous_brain()
