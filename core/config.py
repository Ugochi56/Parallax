from decimal import Decimal
import math
import os
import sys
import types

# Shim: create a lightweight fake 'numba' module so pandas_ta loads
# without requiring the real numba/llvmlite C++ DLLs on Windows Server.
_fake_numba = types.ModuleType('numba')
_fake_numba.njit = lambda *a, **kw: (lambda f: f)  # no-op decorator
_fake_numba.jit = lambda *a, **kw: (lambda f: f)
sys.modules.setdefault('numba', _fake_numba)

import MetaTrader5 as mt5

# ==========================================
# ⚙️ GLOBAL CONFIGURATION
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# 👑 MASTER ACCOUNT & TERMINAL LOCK
MASTER_LOGIN = os.getenv("MASTER_LOGIN", "")
MASTER_MT5_PATH = os.getenv("MASTER_MT5_PATH", "")

SYMBOL = "XAUUSDm"
TIMEFRAME_TREND = mt5.TIMEFRAME_H1
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M5
TIMEFRAME_ADX   = mt5.TIMEFRAME_M15  

# 🛡️ 1. VOLATILITY GUARD 
USE_VOLATILITY_GUARD = True
ATR_BASELINE_PERIOD = 100    
VOLATILITY_THRESHOLD = 4.5   

# 🧠 2. DUAL AI SETTINGS
USE_AI_FILTER = True
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parent.parent
TREND_MODEL_FILE = str(_ROOT_DIR / "gold_trend_model_v2.joblib")
REVERSAL_MODEL_FILE = str(_ROOT_DIR / "gold_reversal_model_v2.joblib")
AI_CONFIDENCE_THRESHOLD = 0.55
AI_TREND_THRESHOLD = 0.52
AI_REVERSAL_THRESHOLD = 0.52
AI_LIMIT_ORDER_THRESHOLD = 0.55
EVOLUTION_STRICTNESS = 0.65

# 🛡️ 3. DYNAMIC RISK (SESSION BASED)
SESSION_RISK = {
    "ASIAN": 2.0,             
    "LONDON": 2.5,            
    "NY_LONDON_OVERLAP": 2.5, 
    "NEW_YORK": 2.5,          
    "ROLLOVER": 0.0           
}

# 🚀 3B. SHARP AGGRESSIVE MODE & MANUAL FIXED LOTS
AGGRESSIVE_MODE = True               # Enable Sharp Aggressive Mode for bigger, bolder payouts
SHARP_BASE_RISK_PCT = 2.5            # Base risk per trade (% of equity)
HIGH_CONFIDENCE_AI_BOOST = 2.0       # 2.0x Lot size multiplier when AI confidence >= 65% (5.0% risk on elite signals)
MANUAL_FIXED_LOT = None              # Set to float (e.g. 0.10, 0.50, 1.00) to override auto risk sizing with fixed lots

# 📉 4. AUTO-STRATEGY (ORCHESTRATOR RULES)
AUTO_SWITCH = True       
ADX_TREND_START = 30     
ADX_RANGE_START = 20     
DISPLACEMENT_VELOCITY_OVERRIDE = True     # Fast-track trend regime if explosive price displacement occurs
DISPLACEMENT_3BAR_ATR_MULT = 2.5          # 3-bar net displacement threshold (e.g. 2.5x M5 ATR)
DISPLACEMENT_2BAR_ATR_MULT = 2.0          # 2-bar net displacement threshold (e.g. 2.0x M5 ATR)

# 🏃 5. MANAGEMENT SETTINGS
USE_ATR_SL = True
SL_ATR_MULTIPLIER = 1.5   
USE_TRAILING_STOP = True
TRAIL_START_DOLLARS = 8.00  # Activated at ~2.5x M5 ATR (raised from 5.50 so trailing stop does not choke normal wave pullbacks)  
TRAIL_DIST_DOLLARS = 5.00   # Used as static fallback if ATR calculation fails
ATR_TRAIL_MULTIPLIER = 1.8
USE_TRAILING_TP = True      
TP_TRIGGER_DIST = 2.00      
TP_EXTENSION_DIST = 5.00    

# LIVING SHADOW: Dynamic structural ATR trailing stop behind closed M5 candle pivots
ENABLE_LIVING_SHADOW = True
LIVING_SHADOW_MIN_PROFIT_PCT = 0.55  # Activate once position reaches Break-Even threshold (55% TP progress)
LIVING_SHADOW_ATR_MULT = 1.10        # Dynamic ATR breathing room beyond candle high/low (prevents noise shakeouts)
LIVING_SHADOW_MIN_BUFFER = 2.20      # Minimum dollar buffer on Gold ($2.20)
MAX_DAILY_LOSS_PERCENT = 15.0        # Small account (< $500): 15.0% daily drawdown limit
MID_ACCOUNT_THRESHOLD = 500.0        # $500 - $4,999: 10.0% daily drawdown limit
MID_ACCOUNT_DAILY_LOSS_PERCENT = 10.0
PROP_FIRM_THRESHOLD = 5000.0         # >= $5,000 (prop-firm level): 4.5% daily drawdown (inside 5% prop breach)
PROP_FIRM_DAILY_LOSS_PERCENT = 4.5


def get_daily_loss_limit_pct(equity: float) -> float:
    """Return the active daily drawdown limit (%) for the given starting equity."""
    if isinstance(equity, bool) or (
        type(equity).__module__ == 'numpy' and type(equity).__name__ in ('bool', 'bool_')
    ):
        return PROP_FIRM_DAILY_LOSS_PERCENT
    if isinstance(equity, Decimal):
        if not equity.is_finite():
            return PROP_FIRM_DAILY_LOSS_PERCENT
        if equity >= Decimal(str(PROP_FIRM_THRESHOLD)):
            return PROP_FIRM_DAILY_LOSS_PERCENT
        if equity >= Decimal(str(MID_ACCOUNT_THRESHOLD)):
            return MID_ACCOUNT_DAILY_LOSS_PERCENT
        return MAX_DAILY_LOSS_PERCENT
    try:
        eq = float(equity)
    except (TypeError, ValueError, ArithmeticError):
        return PROP_FIRM_DAILY_LOSS_PERCENT
    if not math.isfinite(eq):
        return PROP_FIRM_DAILY_LOSS_PERCENT
    if eq >= PROP_FIRM_THRESHOLD:
        return PROP_FIRM_DAILY_LOSS_PERCENT
    if eq >= MID_ACCOUNT_THRESHOLD:
        return MID_ACCOUNT_DAILY_LOSS_PERCENT
    return MAX_DAILY_LOSS_PERCENT

# 🧪 6. OPERATION MODES
DRY_RUN = False
CLOSE_ALL_ON_FRIDAY = True
FRIDAY_CLOSE_HOUR = 21
CLOSE_ALL_DAILY = True              # Daily End-of-Day Square-Off: Close remaining intraday trades before evening dead zone
DAILY_CLOSE_ALL_HOUR = 20          # 20:00 UTC (16:00 NY): Auto-close remaining open positions at US cash close
RSI_TREND_CUTOFF_HOUR = 17         # 17:00 UTC (13:00 NY / London Close): Strictly cutoff momentum trend reloads
NEWS_KILL_SWITCH = True
NEWS_HIGH_IMPACT_PRE_MINUTES = 15   # Option A: Blackout 15 mins before High Impact / Major Macro Releases (down from 45m)
NEWS_HIGH_IMPACT_POST_MINUTES = 15  # Option A: Blackout 15 mins after High Impact Releases (down from 45m)
NEWS_MEDIUM_IMPACT_PRE_MINUTES = 10 # Blackout 10 mins before Medium Impact USD Releases (Orange Folder)
NEWS_MEDIUM_IMPACT_POST_MINUTES = 15 # Blackout 15 mins after Medium Impact USD Releases (Orange Folder)
NEWS_NON_USD_HIGH_PRE_MINUTES = 10   # Blackout 10 mins before major non-USD High Impact Releases (GBP, EUR, etc.)
NEWS_NON_USD_HIGH_POST_MINUTES = 10  # Blackout 10 mins after major non-USD High Impact Releases

# 🛡️ OPTION A POST-NEWS SAFETY GUARDS
NEWS_MAX_SPREAD_DOLLARS = 0.35      # Guard 1: Max allowed spread ($0.35 / 350 pts on Gold) in post-news window
NEWS_POST_WINDOW_MINUTES = 45       # Guard window duration after high-impact news
NEWS_DIRECTION_LOCK_MINUTES = 30    # Guard 2: Enforce one-way trend alignment for 30m post-news
NEWS_MIN_IMPULSE_RANGE = 8.00       # Min dollar range on M5 to qualify as directional news displacement
NEWS_MIN_BODY_RATIO = 0.45          # Guard 3: Body must be >= 45% of range; if < 45% (two-way whip/doji), extend blackout to 45m
DYNAMIC_RISK = True
MOVE_BE = True
BE_TRIGGER_PCT = 0.55  # Lock SL to Break-Even (+0.85 pts profit cushion) at 55% TP progress (~1.1R profit, beyond wave pullbacks)
PARTIAL_CLOSE_PCT = 0.50  # Bank 50% profit at Break-Even (>= 55% TP progress), holding remaining size as risk-free runner

# 🌍 7. TIMEZONE & DST SETTINGS
# 📡 8. COPIER SETTINGS
ENABLE_LOCAL_COPIER = False  # Retired in favor of broker-native (Exness Social Trading) and cloud copiers (Social Trader Tools)
AUTO_DST_ADJUST = True

ENABLE_VWAP_REVERSION = False  # Disabled by user preference
ENABLE_SMC_CHOCH = False       # Gutted: structural reversal underperforms on Gold M5 (30% win rate drag)
NOTIFY_REGIME_CHANGES = False  # Disable noisy Telegram regime change broadcasts across channels
NO_NEW_TRADES_AFTER_HOUR = 19  # 19:00 UTC (15:00 NY): Stop opening new positions for all strategies (1h before daily square-off)
ROLLOVER_BLACKOUT_START_MINUTE = 15  # 23:15 UTC (pre-rollover buffer start)
ROLLOVER_BLACKOUT_END_MINUTE = 30    # 00:30 UTC (post-rollover buffer end)
SINGLE_LOSS_DIRECTIONAL_COOLDOWN_MINUTES = 12  # Pause same-direction trading for 12 mins (approx 2 M5 candles) so normal relief rally reloads are not locked out for an hour
CONSECUTIVE_LOSS_COOLDOWN_MINUTES = 45  # Lock strategy direction for 45 mins after 2 consecutive losses (protects against sustained trend reversal traps)

# STANDARD SETTINGS
MAGIC_NUMBER = 999001
DEVIATION = 20
MIN_GAP_SIZE = 0.05
RISK_REWARD_RATIO = 2.0  # Global fallback R:R (used if strategy not in STRATEGY_RR)
STRATEGY_RR = {
    'RSI_REVERSION':    1.2,   # Crisp 1.2R mean-reversion scalp target (banks cash before stall)
    'VWAP_REVERSION':   1.2,   # Same logic - reversion = tight target
    'SMC_CHOCH':        1.5,   # Structural reversal - moderate target
    'SMC_FVG':          1.5,   # Fair value gap - moderate target
    'SMC_OB':           2.0,   # Order block - standard target
    'BB_BREAKOUT':      2.0,   # Breakout momentum - standard target
    'LONDON_BREAKOUT':  2.5,   # Breakout with Asian range - wider target
    'LIQUIDITY_SWEEP':  2.0,   # Sweep plays - standard target
}
MIN_DYNAMIC_RR = 1.2
MAX_CHOCH_SL_DIST = 14.0   # Max SL distance in points (calibrated for 1.0x ATR sweep buffer and tight retests)
MAX_SPREAD_POINTS = 500  # Trial account spreads run 300-400 normally; only block extreme spikes
PENDING_ORDER_EXPIRY_BARS = 6  # Cancel pending orders after 6 x 5min = 30 minutes
ASIAN_OPEN_HOUR = 0       # 00:00 UTC (Tokyo Open)
TRADING_END_HOUR = 20     # 20:00 UTC (NY Cash Close): End of active intraday session

# INDICATOR PARAMS
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_PERIOD = 14

# ⏱️ 8. TIME-BASED VOLATILITY SETTINGS (ANCHORED TO UTC)
KILLZONE_TRACK_START = 0   # 00:00 UTC (Asian Tracking Start)
KILLZONE_TRACK_END = 7     # 07:00 UTC (Asian Tracking End / 08:00 BST London Open)
KILLZONE_EXEC_START = 7    # 07:00 UTC (London Open Execution / 08:00 BST)
KILLZONE_EXEC_END = 11     # 11:00 UTC (London Morning Execution End)
FILTER_LONDON_FAKEOUTS = True

# 🛡️ RISK & EXPOSURE Safeguards
ALLOW_HEDGING = False  # Strictly prevent having BUY and SELL open at the same time
MAX_SINGLE_TRADE_RISK_PCT = 5.5  # Reject trades if broker minimum lot size forces risk above 5.5%
MAX_CONCURRENT_TRADES = 3  # Maximum allowed open positions + active pending orders
FLOATING_DD_BLOCK_PCT = 50  # Block new entries if open positions are >50% of the way toward their SL

# 🐣 9. SMALL ACCOUNT & MICRO/CENT OPTIMIZATIONS
SMALL_ACCOUNT_MODE = True           # Auto-adapts execution for balances < $200 or Cent accounts (XAUUSDc)
SMALL_ACCOUNT_THRESHOLD = 200.0     # Balance threshold ($) under which Small Account Mode triggers

# 🧠 10. DUAL-MODE RSI ARCHITECTURE & AMDA SETTINGS
RSI_RANGE_ADX_MAX = 20.0          # Max M15 ADX for Gear 1 in London/NY (Range Reversion Mode)
RSI_ASIAN_RANGE_ADX_MAX = 28.0    # Max M15 ADX for Gear 1 in Asian Session (<28 Ranging, >=28 Asia Breakout)
RSI_TREND_ADX_MIN = 22.0          # Min M15 ADX for Gear 2 in London/NY (Trend Pullback Mode)
RSI_RUNAWAY_ADX_THRESHOLD = 36.0      # ADX threshold for declaring vertical runaway cascade (raised from 28.0 to prevent shallow retrace traps)
AMDA_MIN_WICK_PCT = 0.25          # Minimum 25% rejection wick required for Range Reversion
RSI_RANGE_MIN_RR = 1.2            # Minimum dynamic R:R to SMA target in Range Reversion Mode (uniform with MIN_DYNAMIC_RR = 1.2)
RSI_PULLBACK_BUY_MIN = 40.0       # Uptrend reload floor: previous RSI was inside 40-50
RSI_PULLBACK_BUY_MAX = 50.0       # Uptrend reload ceiling: current RSI hooks back up
RSI_PULLBACK_SELL_MIN = 50.0      # Downtrend reload floor: previous RSI was inside 50-60
RSI_PULLBACK_SELL_MAX = 60.0      # Downtrend reload ceiling: current RSI rolls back down
RSI_TREND_PULLBACK_RR = 1.5       # Target 1.5R continuation in trend mode
RSI_CLIMAX_LOOKBACK_BARS = 6      # Historical bars to check for post-climax exhaustion veto (30 mins)
RSI_CLIMAX_OVERBOUGHT = 73.0      # True blow-off climax threshold for Uptrend Pullback veto
RSI_CLIMAX_OVERSOLD = 27.0        # True capitulation climax threshold for Downtrend Pullback veto
RSI_TREND_AI_BUY_MIN = 0.50       # Minimum Trend AI probability for Gear 2 BUY continuation
RSI_TREND_AI_SELL_MIN = 0.49      # Minimum Trend AI probability for Gear 2 SELL continuation
USE_AMDA_FILTER = True
AMDA_MAX_RSI_VELOCITY = 15.0      # Max allowed 2-candle RSI drop before declaring Freefall waterfall
AMDA_VOLUME_ABSORPTION_MULT = 1.0 # Minimum tick volume relative to 20-SMA volume

# 🛡️ 11. SMART EXECUTION GUARD (SELL ASYMMETRY & DEFENSE)
USE_SMART_SELL_GUARD = True

# 🎯 12. LIQUIDITY SWEEP REFINEMENTS
LIQ_PRESWEEP_RIBBON = True         # Evaluate M5 ribbon prior to sweep so the sweep impulse does not self-block
LIQ_PRESWEEP_RIBBON_LAG = 4        # Number of bars before sweep candle to sample M5 EMA ribbon
LIQ_HTF_VETO_MIN_ADX = 0.0         # Always veto counter-HTF sweeps when H1 & H4 align (4-mo WF: +$867 net, lower MDD)
LIQ_SL_ATR_MULT = 1.40             # Stop Loss ATR buffer above/below sweep wick (prevents secondary Judas wickouts)
LIQ_MIN_SL_CUSHION = 1.50          # Minimum dollar cushion beyond the extreme sweep wick
LIQ_MIN_SL_DIST = 4.50             # Minimum absolute distance from current market price
GUARD_MIN_RSI_SELL = 38.0           # Block sells when M5 RSI < 38 (oversold floor trap)
GUARD_MAX_RSI_BUY = 62.0            # Block buys when M5 RSI > 62 (overbought ceiling trap)
GUARD_MIN_WICK_PCT = 0.25           # Minimum upper rejection wick required for structural/breakout sells
GUARD_MAX_ADX_COUNTER = 25.0        # Block counter-trend sells if H1 is UP and M15 ADX > 25
SELL_MAX_RISK_PCT = 2.5             # Aligned with session risk budget (2.5% in London/NY)
