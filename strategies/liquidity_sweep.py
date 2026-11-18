"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import sys
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from core.config import MIN_DYNAMIC_RR, STRATEGY_RR


class LiquiditySweepStrategy(BaseStrategy):
    """Public Reference Adapter: textbook 5-bar fractal Equal-High / Equal-Low sweep detector."""

    def __init__(self, ai_model=None):
        super().__init__("LIQUIDITY")
        self.reversal_model = ai_model
        self.lookback = 100
        self.proximity_threshold_pct = 0.0005
        self.active_sweeps = set()
        self.armed_sweeps = []
        self.last_closed_candle_time = None
        self.last_traded_sweep_time = None
        self.traded_zones = []

    def find_fractals(self, df):
        swing_highs, swing_lows = [], []
        if df is None or len(df) < 5:
            return swing_highs, swing_lows
        highs = df["high"].values
        lows = df["low"].values
        times = df["time"].values
        for i in range(2, len(df) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
                swing_highs.append({"idx": i, "price": float(highs[i]), "time": times[i]})
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
                swing_lows.append({"idx": i, "price": float(lows[i]), "time": times[i]})
        return swing_highs, swing_lows

    def find_liquidity_pools(self, df, swing_highs, swing_lows, atr):
        eqh_zones, eql_zones = [], []
        invalidation_dist = max(0.5, 0.25 * atr)
        for i in range(len(swing_highs)):
            p1, idx1 = swing_highs[i]["price"], swing_highs[i]["idx"]
            cluster, last_idx = [p1], idx1
            for j in range(i + 1, len(swing_highs)):
                p2 = swing_highs[j]["price"]
                if abs(p1 - p2) / p1 <= self.proximity_threshold_pct:
                    cluster.append(p2)
                    last_idx = max(last_idx, swing_highs[j]["idx"])
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                if not any(abs(avg_price - z) / z <= self.proximity_threshold_pct for z in eqh_zones):
                    prior_highs = df["high"].iloc[last_idx + 1 : -2]
                    if len(prior_highs) == 0 or prior_highs.max() <= avg_price + invalidation_dist:
                        eqh_zones.append(avg_price)

        for i in range(len(swing_lows)):
            p1, idx1 = swing_lows[i]["price"], swing_lows[i]["idx"]
            cluster, last_idx = [p1], idx1
            for j in range(i + 1, len(swing_lows)):
                p2 = swing_lows[j]["price"]
                if abs(p1 - p2) / p1 <= self.proximity_threshold_pct:
                    cluster.append(p2)
                    last_idx = max(last_idx, swing_lows[j]["idx"])
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                if not any(abs(avg_price - z) / z <= self.proximity_threshold_pct for z in eql_zones):
                    prior_lows = df["low"].iloc[last_idx + 1 : -2]
                    if len(prior_lows) == 0 or prior_lows.min() >= avg_price - invalidation_dist:
                        eql_zones.append(avg_price)
        return eqh_zones, eql_zones

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, ai_mode="DEFENSIVE", **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < self.lookback + 5:
            return {"ui": "", "payload": None}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        df_scan = df_m5.iloc[-self.lookback:]
        swing_highs, swing_lows = self.find_fractals(df_scan)
        eqh_zones, eql_zones = self.find_liquidity_pools(df_scan, swing_highs, swing_lows, atr)

        closed_c = df_m5.iloc[-2]
        closed_open = float(closed_c["open"])
        closed_high = float(closed_c["high"])
        closed_low = float(closed_c["low"])
        closed_close = float(closed_c["close"])
        closed_time = closed_c["time"]

        # Age & prune pre-existing armed sweeps on new candle closes
        if self.last_closed_candle_time is not None and self.last_closed_candle_time != closed_time:
            for s in self.armed_sweeps:
                s["bars_aged"] += 1
        self.last_closed_candle_time = closed_time
        self.armed_sweeps = [s for s in self.armed_sweeps if s["bars_aged"] <= 3]

        ui_msg = ""
        payload = None
        rr = STRATEGY_RR.get("LIQUIDITY_SWEEP", 2.0)

        for eqh in eqh_zones:
            if closed_high > eqh:
                reclaimed = closed_close < eqh and closed_close < closed_open
                if observe_only:
                    ui_msg = f"[BSL SWEPT: {eqh:.2f} (OBS)]" if reclaimed else f"[BSL SWEPT: {eqh:.2f} (OBS / WAIT RECLAIM)]"
                    break
                if reclaimed and closed_time != self.last_traded_sweep_time:
                    sl = closed_high + max(1.5, atr * 1.4)
                    risk_dist = abs(sl - float(current_price))
                    if 0 < risk_dist <= 14.0:
                        payload = {"signal": "SELL", "sl": sl, "tp_price": float(current_price) - risk_dist * rr, "confidence": 0.65, "comment": "REF_BSL_SWEEP"}
                        self.last_traded_sweep_time = closed_time
                        ui_msg = f"[BSL SWEPT: {eqh:.2f}]"
                        break

        if payload is None and not ui_msg:
            for eql in eql_zones:
                if closed_low < eql:
                    reclaimed = closed_close > eql and closed_close > closed_open
                    if observe_only:
                        ui_msg = f"[SSL SWEPT: {eql:.2f} (OBS)]" if reclaimed else f"[SSL SWEPT: {eql:.2f} (OBS / WAIT RECLAIM)]"
                        break
                    if reclaimed and closed_time != self.last_traded_sweep_time:
                        sl = closed_low - max(1.5, atr * 1.4)
                        risk_dist = abs(float(current_price) - sl)
                        if 0 < risk_dist <= 14.0:
                            payload = {"signal": "BUY", "sl": sl, "tp_price": float(current_price) + risk_dist * rr, "confidence": 0.65, "comment": "REF_SSL_SWEEP"}
                            self.last_traded_sweep_time = closed_time
                            ui_msg = f"[SSL SWEPT: {eql:.2f}]"
                            break

        if not ui_msg and (eqh_zones or eql_zones):
            above = [x for x in eqh_zones if x >= current_price - 0.50]
            below = [x for x in eql_zones if x <= current_price + 0.50]
            b_str = f"{min(above, key=lambda x: abs(x - current_price)):.1f}" if above else "NONE"
            s_str = f"{min(below, key=lambda x: abs(x - current_price)):.1f}" if below else "NONE"
            ui_msg = f"[LIQ B:{b_str} S:{s_str}]"

        return {"ui": ui_msg, "payload": payload}
