"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import time
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from core.config import ATR_BASELINE_PERIOD, MIN_DYNAMIC_RR, RISK_REWARD_RATIO, STRATEGY_RR


def map_market_structure_two_tier(df_m5, major_lookback=5, minor_lookback=2):
    """Textbook two-tier swing structure helper for public reference use."""
    if df_m5 is None or len(df_m5) < 15:
        return [], [], {"major_high": None, "major_low": None, "idm_high": None, "idm_low": None}

    minor_struct = []
    highs = df_m5["high"].values
    lows = df_m5["low"].values
    for i in range(minor_lookback, len(df_m5) - minor_lookback):
        if highs[i] == max(highs[i - minor_lookback : i + minor_lookback + 1]):
            minor_struct.append({"type": "HIGH", "price": float(highs[i]), "idx": i})
        if lows[i] == min(lows[i - minor_lookback : i + minor_lookback + 1]):
            minor_struct.append({"type": "LOW", "price": float(lows[i]), "idx": i})

    return [], minor_struct, {"major_high": None, "major_low": None, "idm_high": None, "idm_low": None}


class SMCOrderBlockStrategy(BaseStrategy):
    """Public Reference Adapter for Order-Block zone tracking and mitigation."""

    def __init__(self, ai_model=None):
        super().__init__("SMC_OB")
        self.trend_model = ai_model
        self.active_obs = []
        self.max_obs = 5
        self.last_traded_ob_time = None
        self._warmed_up = False
        self.ai_throttle_timer = 0

    def warmup_order_blocks(self, df):
        if df is None or len(df) < 20:
            self._warmed_up = True
            return
        self._warmed_up = True

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, ai_mode="DEFENSIVE", **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < 20:
            return {"payload": None, "ui": "[OB:INIT]"}
        if not self._warmed_up:
            self.warmup_order_blocks(df_m5)

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        _, minor_struct, _ = map_market_structure_two_tier(df_m5)

        if len(df_m5) >= 3:
            c1, c2 = df_m5.iloc[-3], df_m5.iloc[-2]
            if not observe_only:
                if c1["close"] < c1["open"] and c2["close"] > c1["high"]:
                    cand = {"type": "BUY", "top": float(c1["high"]), "bottom": float(c1["low"]), "time": c1["time"]}
                    if not any(o.get("time") == cand["time"] for o in self.active_obs):
                        self.active_obs.append(cand)
                elif c1["close"] > c1["open"] and c2["close"] < c1["low"]:
                    cand = {"type": "SELL", "top": float(c1["high"]), "bottom": float(c1["low"]), "time": c1["time"]}
                    if not any(o.get("time") == cand["time"] for o in self.active_obs):
                        self.active_obs.append(cand)
                self.active_obs = self.active_obs[-self.max_obs:]

        ui_msg = f"[OB:{len(self.active_obs)}]" if self.active_obs else "[OB_SCAN]"
        if observe_only or not self.active_obs:
            return {"payload": None, "ui": ui_msg}

        closed = df_m5.iloc[-2]
        rr = STRATEGY_RR.get("SMC_OB", RISK_REWARD_RATIO)
        payload = None
        for ob in list(self.active_obs):
            if ob["time"] == self.last_traded_ob_time:
                continue
            if ob["type"] == "BUY" and closed["low"] <= ob["top"] and closed["close"] > closed["open"]:
                sl = float(ob["bottom"] - atr)
                risk_dist = abs(float(current_price) - sl)
                if risk_dist > 0:
                    tp = float(current_price) + risk_dist * rr
                    payload = {"signal": "BUY", "sl": sl, "tp_price": tp, "confidence": 0.65, "comment": "REF_OB_BUY"}
                    self.last_traded_ob_time = ob["time"]
                    self.active_obs.remove(ob)
                    break
            elif ob["type"] == "SELL" and closed["high"] >= ob["bottom"] and closed["close"] < closed["open"]:
                sl = float(ob["top"] + atr)
                risk_dist = abs(sl - float(current_price))
                if risk_dist > 0:
                    tp = float(current_price) - risk_dist * rr
                    payload = {"signal": "SELL", "sl": sl, "tp_price": tp, "confidence": 0.65, "comment": "REF_OB_SELL"}
                    self.last_traded_ob_time = ob["time"]
                    self.active_obs.remove(ob)
                    break

        return {"payload": payload, "ui": ui_msg}
