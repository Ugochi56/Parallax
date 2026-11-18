import sys
from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time
from core.state_manager import nexus_state

class VWAPReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("VWAP_REVERSION")
        self.last_signal_time = None
        self.throttle_timer = 0
        self.sd_multiplier = 2.0  # 2.0 Standard Deviations for institutional statistical anomaly stretches

    def get_buy_lock(self, symbol=None):
        if symbol is None: symbol = SYMBOL
        return nexus_state.get(f'vwap_buy_locked_{symbol}', False)

    def set_buy_lock(self, status, symbol=None):
        if symbol is None: symbol = SYMBOL
        nexus_state.set(f'vwap_buy_locked_{symbol}', status)

    def get_sell_lock(self, symbol=None):
        if symbol is None: symbol = SYMBOL
        return nexus_state.get(f'vwap_sell_locked_{symbol}', False)

    def set_sell_lock(self, status, symbol=None):
        if symbol is None: symbol = SYMBOL
        nexus_state.set(f'vwap_sell_locked_{symbol}', status)

    def calculate_vwap_bands(self, df):
        import pandas as pd
        df = df.copy()
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True, drop=False)
            
        try:
            # Anchored VWAP (Daily)
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['tick_volume'])
            df['VWAP'] = vwap
        except Exception as e:
            df['VWAP'] = np.nan

        # Fallback if ta.vwap fails or returns all NaNs
        if df['VWAP'].isna().all():
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            volume = df['tick_volume']
            cum_pv = (typical_price * volume).cumsum()
            cum_v = volume.cumsum()
            df['VWAP'] = cum_pv / np.where(cum_v == 0, 1.0, cum_v)
            
        try:
            # Calculate standard deviation of the difference between close and VWAP
            diff = df['close'] - df['VWAP']
            std_dev = diff.rolling(window=100).std()
            
            df['VWAP_Upper'] = df['VWAP'] + (std_dev * self.sd_multiplier)
            df['VWAP_Lower'] = df['VWAP'] - (std_dev * self.sd_multiplier)
            return df
        except Exception as e:
            return None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get('observe_only', False))
        symbol = kwargs.get('symbol', SYMBOL)
        signal_payload = None
        action_msg = "[VWAP_TRACK]"
        
        df_vwap = self.calculate_vwap_bands(df_m5)
        if df_vwap is None or len(df_vwap) < 100:
            return {'payload': None, 'ui': "[VWAP_INIT]"}
            
        latest = df_vwap.iloc[-1]
        vwap_val = latest.get('VWAP', 0)
        upper_band = latest.get('VWAP_Upper', 0)
        lower_band = latest.get('VWAP_Lower', 0)
        last_time = df_m5.iloc[-1]['time']
        
        if vwap_val == 0 or np.isnan(upper_band) or np.isnan(lower_band):
            return {'payload': None, 'ui': "[VWAP_WAIT]"}
            
        # UI Action Feedback string
        dist_pct = abs((current_price - vwap_val) / vwap_val) * 100
        action_msg = f"[vDiv:{dist_pct:.1f}%]"

        # Calculate ADX trend strength from df_adx (M15)
        adx_val = 0.0
        if df_adx is not None and len(df_adx) >= 30:
            from core.indicators import calculate_adx_simple
            adx_series = calculate_adx_simple(df_adx, ADX_PERIOD)
            if adx_series is not None and not adx_series.empty:
                adx_val = adx_series.iloc[-1]

        # Calculate H1 trend direction (confirmed closed candle)
        trend = 'FLAT'
        if df_h1 is not None and len(df_h1) >= 50:
            h1_idx = -2 if len(df_h1) >= 2 else -1
            ema_50_h1 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[h1_idx]
            close_h1 = df_h1.iloc[h1_idx]['close']
            trend = 'UP' if close_h1 > ema_50_h1 else 'DOWN'

        # Calculate M15 trend direction (confirmed closed candle)
        m15_trend = 'FLAT'
        if df_adx is not None and len(df_adx) >= 50:
            m15_idx = -2 if len(df_adx) >= 2 else -1
            ema_50_m15 = df_adx['close'].ewm(span=50, adjust=False).mean().iloc[m15_idx]
            close_m15 = df_adx.iloc[m15_idx]['close']
            m15_trend = 'UP' if close_m15 > ema_50_m15 else 'DOWN'

        # Dynamic Normalization Reset
        if not observe_only:
            if current_price >= vwap_val and self.get_buy_lock(symbol):
                self.set_buy_lock(False, symbol)
                print("\n[VWAP] [ANOMALY RESOLVED] Asset normalized above VWAP. Buy triggers unlocked.")
            if current_price <= vwap_val and self.get_sell_lock(symbol):
                self.set_sell_lock(False, symbol)
                print("\n[VWAP] [ANOMALY RESOLVED] Asset normalized below VWAP. Sell triggers unlocked.")

        # Signal Logic
        if not observe_only and self.last_signal_time != last_time and (time.time() - self.throttle_timer > 3.0):
            # Scalp risk override: purely mathematical strat
            r_pct = min(current_risk, 0.5)
            sl_padding = atr * SL_ATR_MULTIPLIER

            if current_price >= upper_band and not self.get_sell_lock(symbol):                        
                # Trend Guard: Block counter-trend short if strong uptrend (ADX or EMA alignment)
                is_uptrend = (adx_val > ADX_TREND_START and trend == 'UP') or (trend == 'UP' and m15_trend == 'UP')
                if is_uptrend:
                    action_msg = f"[vDiv:{dist_pct:.1f}%] 🚫 Blocked: Uptrend"
                else:
                    self.throttle_timer = time.time()
                    self.last_signal_time = last_time
                    self.set_sell_lock(True, symbol)
                    from engine.feature_extractor_v3 import extract_v3_features
                    feat_vec = extract_v3_features(df_m5, df_h1, df_h4).values[0].tolist()
                    signal_payload = {
                        'signal': 'SELL', 
                        'sl': current_price + sl_padding, 
                        'tp_price': vwap_val,
                        'risk_override': r_pct, 
                        'confidence': 0.85, 
                        'comment': "VWAP Stretch SELL",
                        'features': feat_vec
                    }
                    print(f"\n[VWAP] [EXTREME VOLUMETRIC STRETCH] (+{self.sd_multiplier} SD). Snapping Back.")
                
            elif current_price <= lower_band and not self.get_buy_lock(symbol):                        
                # Trend Guard: Block counter-trend long if strong downtrend (ADX or EMA alignment)
                is_downtrend = (adx_val > ADX_TREND_START and trend == 'DOWN') or (trend == 'DOWN' and m15_trend == 'DOWN')
                if is_downtrend:
                    action_msg = f"[vDiv:{dist_pct:.1f}%] 🚫 Blocked: Downtrend"
                else:
                    self.throttle_timer = time.time()
                    self.last_signal_time = last_time
                    self.set_buy_lock(True, symbol)
                    from engine.feature_extractor_v3 import extract_v3_features
                    feat_vec = extract_v3_features(df_m5, df_h1, df_h4).values[0].tolist()
                    signal_payload = {
                        'signal': 'BUY', 
                        'sl': current_price - sl_padding, 
                        'tp_price': vwap_val,
                        'risk_override': r_pct, 
                        'confidence': 0.85, 
                        'comment': "VWAP Stretch BUY",
                        'features': feat_vec
                    }
                    print(f"\n[VWAP] [EXTREME VOLUMETRIC DEPRESSION] (-{self.sd_multiplier} SD). Snapping Back.")

        if signal_payload is not None:
            entry = current_price
            sl = signal_payload.get('sl')
            tp = signal_payload.get('tp_price')
            direction = signal_payload.get('signal')
            if sl is not None and tp is not None:
                risk_dist = abs(entry - sl)
                reward_dist = max(0.0, tp - entry) if direction == 'BUY' else max(0.0, entry - tp)
                min_rr = getattr(sys.modules.get('core.config'), 'MIN_DYNAMIC_RR', 1.5)
                if risk_dist <= 0 or (reward_dist / risk_dist) < min_rr:
                    rr_val = (reward_dist / risk_dist) if risk_dist > 0 else 0.0
                    print(f"\n[VWAP] [SIGNAL ABORTED]: R:R {rr_val:.2f} < {min_rr:.1f}")
                    signal_payload = None
                    action_msg = f"{action_msg} 🚫 Low R:R"

        return {'payload': signal_payload, 'ui': action_msg}
