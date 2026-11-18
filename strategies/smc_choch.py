"""Public Reference Strategy Adapter — Live proprietary multi-stage alpha filters and tuned weights are kept in the private production repository."""

import pandas as pd
from strategies.base import BaseStrategy
from core.config import STRATEGY_RR
import core.state_manager as _state_mod


def map_market_structure_two_tier(df_m5, major_lookback=5, minor_lookback=2):
    """Textbook swing high/low mapper for public reference use."""
    if df_m5 is None or len(df_m5) < 15:
        return [], [], {"major_high": None, "major_low": None, "major_highs": [], "major_lows": [], "minor_highs": [], "minor_lows": [], "idm_high": None, "idm_low": None}

    minor = []
    highs = df_m5["high"].values
    lows = df_m5["low"].values
    for i in range(minor_lookback, len(df_m5) - minor_lookback):
        if highs[i] == max(highs[i - minor_lookback : i + minor_lookback + 1]):
            minor.append({"type": "HIGH", "price": float(highs[i]), "idx": i})
        if lows[i] == min(lows[i - minor_lookback : i + minor_lookback + 1]):
            minor.append({"type": "LOW", "price": float(lows[i]), "idx": i})

    m_highs = [x for x in minor if x["type"] == "HIGH"]
    m_lows = [x for x in minor if x["type"] == "LOW"]
    return [], minor, {
        "major_high": m_highs[-1] if m_highs else None,
        "major_low": m_lows[-1] if m_lows else None,
        "major_highs": m_highs,
        "major_lows": m_lows,
        "minor_highs": m_highs,
        "minor_lows": m_lows,
        "idm_high": None,
        "idm_low": None,
    }


class SMCChochStrategy(BaseStrategy):
    """Public Reference Adapter: textbook Change-of-Character swing break detector."""

    def __init__(self, ai_model=None):
        super().__init__("SMC_CHOCH")
        self.ai_model = ai_model
        self.last_traded_choch_time = None
        self.last_traded_break_price = {}

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        observe_only = bool(kwargs.get("observe_only", False))
        if df_m5 is None or len(df_m5) < 30:
            return {"payload": None, "ui": "[CHOCH:INIT]"}

        atr = float(atr) if (atr is not None and not pd.isna(atr) and atr > 0) else 1.0
        _, _, idm = map_market_structure_two_tier(df_m5)
        last_high = idm.get("major_high")
        last_low = idm.get("major_low")
        closed = df_m5.iloc[-2]
        ui_msg = "[CHOCH:SCAN]"

        if observe_only:
            return {"payload": None, "ui": ui_msg}

        rr = STRATEGY_RR.get("SMC_CHOCH", 1.5)
        sl_dist = max(4.0, atr * 1.5)
        symbol = "XAUUSDm"
        payload = None

        if last_high and closed["close"] > last_high["price"] and self.last_traded_break_price.get(symbol) != last_high["price"]:
            payload = {"signal": "BUY", "sl": float(current_price) - sl_dist, "tp_price": float(current_price) + sl_dist * rr, "confidence": 0.60, "comment": "REF_CHOCH_UP"}
            self.last_traded_choch_time = closed["time"]
            self.last_traded_break_price[symbol] = last_high["price"]
        elif last_low and closed["close"] < last_low["price"] and self.last_traded_break_price.get(symbol) != last_low["price"]:
            payload = {"signal": "SELL", "sl": float(current_price) + sl_dist, "tp_price": float(current_price) - sl_dist * rr, "confidence": 0.60, "comment": "REF_CHOCH_DN"}
            self.last_traded_choch_time = closed["time"]
            self.last_traded_break_price[symbol] = last_low["price"]

        return {"payload": payload, "ui": ui_msg}
