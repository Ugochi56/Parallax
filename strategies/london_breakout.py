"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import pandas as pd
from strategies.base import BaseStrategy
from core.config import STRATEGY_RR


class LondonBreakoutStrategy(BaseStrategy):
    """Public Reference Adapter: textbook Asian-range breakout at London Open."""

    def __init__(self, ai_model=None):
        super().__init__("LONDON_BREAKOUT")
        self.trend_model = ai_model
        self.asian_high = None
        self.asian_low = None
        self.last_signal_time = None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < 20:
            return {"payload": None, "ui": "[LDN:INIT]"}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        window = df_m5.iloc[-30:-1]
        self.asian_high = float(window["high"].max())
        self.asian_low = float(window["low"].min())
        if observe_only:
            return {"payload": None, "ui": "[LDN:OBS]"}

        closed = df_m5.iloc[-2]
        if closed["time"] == self.last_signal_time:
            return {"payload": None, "ui": "[LDN:SCAN]"}

        rr = STRATEGY_RR.get("LONDON_BREAKOUT", 1.8)
        sl_dist = max(4.0, atr * 1.5)
        if closed["close"] > self.asian_high:
            self.last_signal_time = closed["time"]
            return {"payload": {"signal": "BUY", "sl": float(current_price) - sl_dist, "tp_price": float(current_price) + sl_dist * rr, "confidence": 0.60, "comment": "REF_LDN_BUY"}, "ui": "[LDN:BUY]"}
        if closed["close"] < self.asian_low:
            self.last_signal_time = closed["time"]
            return {"payload": {"signal": "SELL", "sl": float(current_price) + sl_dist, "tp_price": float(current_price) - sl_dist * rr, "confidence": 0.60, "comment": "REF_LDN_SELL"}, "ui": "[LDN:SELL]"}

        return {"payload": None, "ui": "[LDN:SCAN]"}
