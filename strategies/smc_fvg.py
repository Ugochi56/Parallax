"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import pandas as pd
from strategies.base import BaseStrategy
from core.config import STRATEGY_RR


class SMCStrategy(BaseStrategy):
    """Public Reference Adapter: textbook 3-candle Fair Value Gap detector."""

    def __init__(self, ai_model=None):
        super().__init__("SMC_FVG")
        self.trend_model = ai_model
        self.active_fvg = None
        self.denied_times = set()

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < 20:
            return {"payload": None, "ui": "[FVG:INIT]"}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        c1, c3 = df_m5.iloc[-4], df_m5.iloc[-2]
        if c3["low"] > c1["high"]:
            self.active_fvg = {"type": "BUY", "bottom": float(c1["high"]), "top": float(c3["low"]), "time": c3["time"]}
        elif c3["high"] < c1["low"]:
            self.active_fvg = {"type": "SELL", "bottom": float(c3["high"]), "top": float(c1["low"]), "time": c3["time"]}

        if observe_only or self.active_fvg is None or self.active_fvg["time"] in self.denied_times:
            return {"payload": None, "ui": "[FVG:SCAN]"}

        rr = STRATEGY_RR.get("SMC_FVG", 1.8)
        fvg = self.active_fvg
        if fvg["type"] == "BUY" and current_price <= fvg["top"]:
            sl = fvg["bottom"] - atr
            risk = max(1.0, abs(float(current_price) - sl))
            self.denied_times.add(fvg["time"])
            return {"payload": {"signal": "BUY", "sl": sl, "tp_price": float(current_price) + risk * rr, "confidence": 0.60, "comment": "REF_FVG_BUY"}, "ui": "[FVG:BUY]"}
        if fvg["type"] == "SELL" and current_price >= fvg["bottom"]:
            sl = fvg["top"] + atr
            risk = max(1.0, abs(sl - float(current_price)))
            self.denied_times.add(fvg["time"])
            return {"payload": {"signal": "SELL", "sl": sl, "tp_price": float(current_price) - risk * rr, "confidence": 0.60, "comment": "REF_FVG_SELL"}, "ui": "[FVG:SELL]"}

        return {"payload": None, "ui": "[FVG:WAIT]"}
