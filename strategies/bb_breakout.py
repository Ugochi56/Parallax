"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import time
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from core.config import ADX_PERIOD, STRATEGY_RR
from core.indicators import calculate_adx_simple, calculate_rsi_simple


class BBBreakoutStrategy(BaseStrategy):
    """Public Reference Adapter: textbook 20-period Bollinger Band volatility breakout."""

    def __init__(self, ai_model=None):
        super().__init__("BB_BREAKOUT")
        self.trend_model = ai_model
        self.bb_length = 20
        self.bb_std = 2.0
        self.last_signal_time = None
        self.throttle_timer = 0

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < self.bb_length:
            return {"payload": None, "ui": "[BB:INIT]"}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        df = df_m5 if ("BBU" in df_m5.columns and "BBL" in df_m5.columns and "BBM" in df_m5.columns) else df_m5.copy()
        if "BBU" not in df.columns:
            sma = df["close"].rolling(self.bb_length).mean()
            std = df["close"].rolling(self.bb_length).std()
            df["BBM"] = sma
            df["BBU"] = sma + self.bb_std * std
            df["BBL"] = sma - self.bb_std * std

        curr_rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else float(calculate_rsi_simple(df["close"], 14).iloc[-1])
        if pd.isna(curr_rsi):
            curr_rsi = 50.0

        adx_val = 0.0
        if df_adx is not None and len(df_adx) >= 30:
            if "adx" in df_adx.columns:
                adx_val = float(df_adx["adx"].iloc[-1])
            else:
                adx_s = calculate_adx_simple(df_adx, ADX_PERIOD)
                if adx_s is not None and not adx_s.empty:
                    adx_val = float(adx_s.iloc[-1])
        if pd.isna(adx_val):
            adx_val = 0.0

        ui_msg = f"[BB:{curr_rsi:.0f}]"
        if observe_only or adx_val < 22.0:
            return {"payload": None, "ui": ui_msg}

        latest = df.iloc[-1]
        last_time = latest["time"]
        if last_time == self.last_signal_time:
            return {"payload": None, "ui": ui_msg}

        rr = STRATEGY_RR.get("BB_BREAKOUT", 2.0)
        sl_dist = max(abs(float(current_price) - float(latest["BBM"])) + 0.5 * atr, atr * 1.5)
        payload = None

        if latest["close"] > latest["BBU"]:
            payload = {
                "signal": "BUY",
                "sl": float(current_price) - sl_dist,
                "tp_price": float(current_price) + sl_dist * rr,
                "confidence": 0.60,
                "comment": "REF_BB_BUY",
            }
            self.last_signal_time = last_time
            self.throttle_timer = time.time()
        elif latest["close"] < latest["BBL"]:
            payload = {
                "signal": "SELL",
                "sl": float(current_price) + sl_dist,
                "tp_price": float(current_price) - sl_dist * rr,
                "confidence": 0.60,
                "comment": "REF_BB_SELL",
            }
            self.last_signal_time = last_time
            self.throttle_timer = time.time()

        return {"payload": payload, "ui": ui_msg}
