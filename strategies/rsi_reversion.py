"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import time
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from core.config import ADX_PERIOD, STRATEGY_RR
from core.indicators import calculate_adx_simple, calculate_rsi_simple


class RSIReversionStrategy(BaseStrategy):
    """Public Reference Adapter: textbook 14-period RSI overbought/oversold reversion."""

    def __init__(self, reversal_model=None, trend_model=None):
        super().__init__("RSI_REVERSION")
        self.reversal_model = reversal_model
        self.trend_model = trend_model
        self.last_rsi_signal_time = None
        self.ai_throttle_timer = 0
        self.dynamic_sl_padding = 2.0

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < 30:
            return {"payload": None, "ui": "[RSI:INIT]"}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        self.dynamic_sl_padding = atr * 1.5

        if "rsi" in df_m5.columns:
            curr_rsi = float(df_m5["rsi"].iloc[-1])
        else:
            rsi_s = calculate_rsi_simple(df_m5["close"], 14)
            curr_rsi = float(rsi_s.iloc[-1]) if len(rsi_s) > 0 else 50.0
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

        ui_msg = f"[RSI-RNG:{curr_rsi:.0f}]" if adx_val < 22.0 else f"[RSI-TRD:{curr_rsi:.0f}]"
        if observe_only or adx_val >= 25.0:
            return {"payload": None, "ui": ui_msg}

        closed = df_m5.iloc[-2]
        cl_time = closed["time"]
        if cl_time == self.last_rsi_signal_time:
            return {"payload": None, "ui": ui_msg}

        rr = STRATEGY_RR.get("RSI_REVERSION", 1.5)
        sl_dist = max(3.5, atr * 1.5)
        payload = None

        if curr_rsi <= 30.0 and closed["close"] > closed["open"]:
            payload = {
                "signal": "BUY",
                "sl": float(current_price) - sl_dist,
                "tp_price": float(current_price) + sl_dist * rr,
                "confidence": 0.60,
                "comment": "REF_RSI_BUY",
            }
            self.last_rsi_signal_time = cl_time
            self.ai_throttle_timer = time.time()
        elif curr_rsi >= 70.0 and closed["close"] < closed["open"]:
            payload = {
                "signal": "SELL",
                "sl": float(current_price) + sl_dist,
                "tp_price": float(current_price) - sl_dist * rr,
                "confidence": 0.60,
                "comment": "REF_RSI_SELL",
            }
            self.last_rsi_signal_time = cl_time
            self.ai_throttle_timer = time.time()

        return {"payload": payload, "ui": ui_msg}
