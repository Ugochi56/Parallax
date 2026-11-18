import MetaTrader5 as mt5
import pandas as pd
import sys
import time
import os
import joblib
import numpy as np
import threading
import atexit
import math
from datetime import datetime
from core.config import *
from core.state_manager import nexus_state
from core.indicators import calculate_atr_simple, calculate_rsi_simple
from engine.telegram_notifier import notify_trade_opened, notify_trade_closed, notify_veto, notify_boot, broadcast_signal_opened, broadcast_signal_closed, broadcast_trade_update

mt5_lock = threading.RLock()

virtual_positions = {}
virtual_positions_lock = threading.Lock()
virtual_ticket_counter = 1000
live_notified_sl = {}
tracked_live_tickets = {}
_closed_deal_retry_counts = {}
_copied_master_tickets = set()
_partially_closed_tickets = set()



def set_active_magic(new_magic):
    global MAGIC_NUMBER
    MAGIC_NUMBER = new_magic
    if 'core.config' in sys.modules:
        sys.modules['core.config'].MAGIC_NUMBER = new_magic

def connect_mt5(terminal_path=None, is_portable=True, magic_override=None):
    global SYMBOL
    if magic_override is not None:
        set_active_magic(magic_override)

    with mt5_lock:
        master_path = terminal_path or getattr(sys.modules.get('core.config'), 'MASTER_MT5_PATH', r"C:\Program Files\MetaTrader 5\terminal64.exe")
        init_ok = False
        if master_path and os.path.exists(master_path):
            init_ok = mt5.initialize(path=master_path, portable=is_portable)
            if not init_ok:
                # Retry without portable flag in case terminal is installed as non-portable
                init_ok = mt5.initialize(path=master_path)
        if not init_ok:
            init_ok = mt5.initialize()
            
        if not init_ok:
            print(f"\n[ERROR] MT5 Init Failed: {mt5.last_error()} (Path: {master_path})")
            mt5.shutdown()
            sys.exit()
            
        atexit.register(mt5.shutdown)
        print(f"[MT5] Initialized IPC Session: {master_path} | Magic: {MAGIC_NUMBER}")


        # Gold Symbol Auto-Resolution (Supports XAUUSD, XAUUSDm, XAUUSDc, GOLD, XAUUSD.a, etc.)
        if not mt5.symbol_select(SYMBOL, True):
            gold_variants = ["XAUUSD", "XAUUSDm", "XAUUSDc", "GOLD", "XAUUSD.a", "XAUUSD.ecn"]
            resolved = False
            for variant in gold_variants:
                if mt5.symbol_select(variant, True):
                    print(f"\n[SYMBOL RESOLUTION] Primary '{SYMBOL}' unavailable. Auto-selected Gold variant: '{variant}'")
                    SYMBOL = variant
                    # Propagate resolved symbol back to core.config module so ALL importers see the change
                    sys.modules['core.config'].SYMBOL = variant
                    resolved = True
                    break
            if not resolved:
                print(f"\n[ERROR] Failed to select Gold symbol '{SYMBOL}' or any known variants.")
                mt5.shutdown()
                sys.exit()
        account_info = mt5.account_info()
        if account_info:
            print(f"\n[SUCCESS] Connected: {account_info.login} | ${account_info.balance:,.2f}")
            notify_boot(account_info.login, account_info.balance)
            # Dynamic Master Account Auto-Binding
            active_login = getattr(account_info, 'login', 'Unknown')
            active_balance = getattr(account_info, 'balance', 0.0)
            print(f"[SUCCESS] Connected Master Account: #{active_login} | ${active_balance:,.2f}")
        
        # Seed pre-existing positions to prevent re-copying on restart
        try:
            existing_pos = mt5.positions_get(symbol=SYMBOL)
            if existing_pos:
                for p in existing_pos:
                    _copied_master_tickets.add(p.ticket)
        except Exception:
            pass

        # Auto-calibrate spread limit based on broker's actual conditions
        calibrate_spread_limit()

def calibrate_spread_limit():
    """Sample the broker's spread 5 times over 2.5 seconds and set the dynamic limit to 1.5x the average."""
    global dynamic_spread_limit
    with mt5_lock:
        samples = []
        symbol_info = mt5.symbol_info(SYMBOL)
        point = symbol_info.point if symbol_info and symbol_info.point > 0 else 0.001
        
        for _ in range(5):
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                spread_pts = abs(tick.ask - tick.bid) / point
                samples.append(spread_pts)
            time.sleep(0.5)
        
        if samples:
            avg_spread = sum(samples) / len(samples)
            dynamic_spread_limit = int(avg_spread * 1.5)
            print(f"[SPREAD] Auto-calibrated: Avg {avg_spread:.0f} pts -> Limit set to {dynamic_spread_limit} pts")
        else:
            dynamic_spread_limit = MAX_SPREAD_POINTS
            print(f"[SPREAD] Calibration failed, using config default: {MAX_SPREAD_POINTS} pts")

def _parse_positive_finite_float(val):
    if (
        isinstance(val, bool)
        or (type(val).__module__ == 'numpy' and type(val).__name__ in ('bool', 'bool_'))
        or val is None
    ):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError, ArithmeticError):
        return None
    if not math.isfinite(num) or num <= 0.0:
        return None
    return num


def check_daily_drawdown():
    with mt5_lock:
        account = mt5.account_info()
        if not account: return False
        
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 1)
        if rates is None or len(rates) == 0: return False
        
        current_d1_time = int(rates[0]['time'])
        account_equity = _parse_positive_finite_float(getattr(account, 'equity', None))
        if account_equity is None:
            print("[DRAWDOWN] Invalid or non-positive account equity -> FAIL-CLOSED HALT")
            return False

        cached_d1_time = nexus_state.get('last_d1_time', -1)
        
        cached_account_login = nexus_state.get('last_account_login', -1)
        
        # Account switch detection: If account login changed, reset baseline for the new account
        account_switched = (account.login != cached_account_login and cached_account_login != -1)
        if account_switched:
            print(f"\n[DAY] Account Switch Detected ({cached_account_login} -> {account.login})! Resetting Daily Baseline to ${account_equity:.2f}")
        
        # Fresh process start: Re-sync baseline if day changed, account switched, or no state exists
        now_epoch = int(time.time())
        if not hasattr(check_daily_drawdown, '_startup_synced') or account_switched:
            check_daily_drawdown._startup_synced = True
            if current_d1_time != cached_d1_time or cached_d1_time == -1 or account_switched:
                nexus_state.set('daily_start_equity', account_equity)
                nexus_state.set('daily_start_timestamp', now_epoch)
                nexus_state.set('last_d1_time', current_d1_time)
                nexus_state.set('last_account_login', account.login)
                nexus_state.set('is_halted', False)
                cached_d1_time = current_d1_time
                print(f"\n[DAY] New Broker Day (Synced to MT5 Clock)! Account: {account.login} | Equity: ${account_equity:.2f}")
            else:
                if cached_account_login == -1:
                    nexus_state.set('last_account_login', account.login)
                saved_start = _parse_positive_finite_float(nexus_state.get('daily_start_equity', None))
                if saved_start is None:
                    saved_start = account_equity
                    nexus_state.set('daily_start_equity', account_equity)
                print(f"\n[DAY] Resuming Broker Day! Account: {account.login} | Saved Daily Start Equity: ${saved_start:.2f}")
        
        if current_d1_time != cached_d1_time:
            nexus_state.set('daily_start_equity', account_equity)
            nexus_state.set('daily_start_timestamp', now_epoch)
            nexus_state.set('last_d1_time', current_d1_time)
            nexus_state.set('last_account_login', account.login)
            nexus_state.set('is_halted', False)
            cached_d1_time = current_d1_time
            print(f"\n[DAY] New Broker Day! Account: {account.login} | Equity: ${account_equity:.2f} (Synced to MT5 Clock)")
            
        start_equity = _parse_positive_finite_float(nexus_state.get('daily_start_equity', None))
        if start_equity is None:
            start_equity = account_equity
            nexus_state.set('daily_start_equity', account_equity)
        
        # Prop-Firm Standard: Calculate exact realized + floating PnL from NEXUS trades today
        closed_deals = mt5.history_deals_get(current_d1_time, now_epoch + 86400)
        open_positions = mt5.positions_get(symbol=SYMBOL)
        if closed_deals is None or open_positions is None:
            print("[DRAWDOWN] MT5 history/positions unavailable (None) -> FAIL-CLOSED HALT")
            return False

        daily_closed_pnl = 0.0
        if closed_deals:
            pos_magics = {
                getattr(d, 'position_id', 0): d.magic
                for d in closed_deals
                if getattr(d, 'entry', None) == mt5.DEAL_ENTRY_IN and getattr(d, 'magic', 0) != 0
            }
            for d in closed_deals:
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    pid = getattr(d, 'position_id', 0)
                    deal_magic = d.magic if d.magic != 0 else pos_magics.get(pid, 0)
                    if deal_magic == 0 and pid:
                        try:
                            pos_deals = mt5.history_deals_get(position=pid)
                            if pos_deals:
                                for pd_item in pos_deals:
                                    if getattr(pd_item, 'entry', None) == mt5.DEAL_ENTRY_IN and getattr(pd_item, 'magic', 0) != 0:
                                        deal_magic = pd_item.magic
                                        pos_magics[pid] = deal_magic
                                        break
                        except Exception:
                            pass
                    if deal_magic == MAGIC_NUMBER:
                        daily_closed_pnl += (d.profit + d.commission + d.swap + getattr(d, 'fee', 0.0))
                    
        daily_floating_pnl = 0.0
        if open_positions:
            for p in open_positions:
                if p.magic == MAGIC_NUMBER:
                    daily_floating_pnl += (p.profit + getattr(p, 'commission', 0.0) + getattr(p, 'swap', 0.0))
                    
        total_daily_pnl = daily_closed_pnl + daily_floating_pnl
        
        # Calculate Drawdown % directly from trading PnL
        daily_loss_amount = abs(min(0.0, total_daily_pnl))
        loss_percent = (daily_loss_amount / start_equity) * 100 if start_equity > 0 else 0.0
        
        _cfg_mod = sys.modules.get('core.config')
        _limit_fn = getattr(_cfg_mod, 'get_daily_loss_limit_pct', None)
        daily_limit_pct = float(_limit_fn(start_equity)) if callable(_limit_fn) else float(MAX_DAILY_LOSS_PERCENT)

        # Self-Healing: If state was halted by a previous false alarm but equity/drawdown is healthy, auto-unhalt!
        if loss_percent < daily_limit_pct and nexus_state.get('is_halted', False):
            nexus_state.set('is_halted', False)
            print(f"\n[UNHALT] Account is healthy (Daily PnL: ${total_daily_pnl:+.2f}, Drawdown: {loss_percent:.2f}%). Resuming active trading!")

        if loss_percent >= daily_limit_pct:
            if not nexus_state.get('is_halted', False):
                print(f"\n[HALT] DAILY LIMIT HIT! (-{loss_percent:.2f}% / ${daily_loss_amount:.2f} >= {daily_limit_pct:.1f}% limit). EXECUTING GLOBAL LIQUIDATION...")
                nexus_state.set('is_halted', True)
                close_all_positions(reason="DAILY_LIMIT_REACHED")
                cancel_all_pending_orders(reason="DAILY_LIMIT_REACHED")
            else:
                # Reconciliation check: Ensure no matching positions or pending orders survive the halt
                positions_raw = mt5.positions_get()
                matching_positions = [p for p in positions_raw if p.symbol == SYMBOL and p.magic == MAGIC_NUMBER] if positions_raw else []
                if matching_positions:
                    print(f"\n[HALT RECONCILIATION] Warning: {len(matching_positions)} positions active during halt. Retrying close...")
                    close_all_positions(reason="HALT_RECONCILIATION")

                orders_raw = mt5.orders_get(symbol=SYMBOL)
                matching_orders = [o for o in orders_raw if o.magic == MAGIC_NUMBER] if orders_raw else []
                if matching_orders:
                    print(f"\n[HALT RECONCILIATION] Warning: {len(matching_orders)} pending orders active during halt. Retrying cancel...")
                    cancel_all_pending_orders(reason="HALT_RECONCILIATION")

                if not matching_positions and not matching_orders:
                    print(f"\r[HALT] Terminal structurally locked for the day. Wait for reset.".ljust(90), end='')
            return False 
        return True

def check_volatility_guard(df_m5):
    if not USE_VOLATILITY_GUARD: return True, 1.0
    baseline_atr = calculate_atr_simple(df_m5, period=ATR_BASELINE_PERIOD)
    recent_range = (df_m5['high'] - df_m5['low']).rolling(5).mean().iloc[-1]
    if baseline_atr == 0: return True, 1.0
    volatility_ratio = recent_range / baseline_atr
    if volatility_ratio > VOLATILITY_THRESHOLD: return False, volatility_ratio
    return True, volatility_ratio

def get_market_data(symbol, timeframe, n=1000):
    with mt5_lock:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['tick_volume'] = df['tick_volume'].astype(float) 
        return df

def get_dynamic_kelly_risk(base_risk):
    if DRY_RUN: return base_risk
    
    with mt5_lock:
        time_from = datetime.now().astimezone() - pd.Timedelta(hours=24)
        deals = mt5.history_deals_get(time_from, datetime.now().astimezone())
        if deals is None or len(deals) == 0: return base_risk
        
        pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
        # Isolate out-deals (closed positions) by this exact bot
        bot_deals = [d for d in deals if pos_magics.get(d.position_id, d.magic) == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_OUT]
        if len(bot_deals) < 2: return base_risk
        
        # Track discrete win/loss sequences (ignoring tiny scratch BE trades under $0.50)
        outcomes = []
        for d in bot_deals:
            if d.profit > 0.50: outcomes.append(1)
            elif d.profit < -0.50: outcomes.append(-1)
            
        if len(outcomes) >= 3 and sum(outcomes[-3:]) == 3:
            return base_risk * 1.5
        elif len(outcomes) >= 2 and sum(outcomes[-2:]) == -2:
            return base_risk * 0.5
            
        return base_risk

def get_dynamic_evolution_strictness():
    base_strictness = getattr(sys.modules['core.config'], 'EVOLUTION_STRICTNESS', 0.65)
    if DRY_RUN: return base_strictness
    
    with mt5_lock:
        time_from = datetime.now().astimezone() - pd.Timedelta(hours=24)
        deals = mt5.history_deals_get(time_from, datetime.now().astimezone())
        if deals is None or len(deals) == 0: return base_strictness
        
        pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
        bot_deals = [d for d in deals if pos_magics.get(d.position_id, d.magic) == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_OUT]
        if len(bot_deals) < 2: return base_strictness
        
        outcomes = []
        for d in bot_deals:
            if d.profit > 0.50: outcomes.append(1)
            elif d.profit < -0.50: outcomes.append(-1)
            
        if len(outcomes) >= 2 and sum(outcomes[-2:]) == 2:
            return 0.80  # Very Loose (Winning streak)
        elif len(outcomes) >= 2 and sum(outcomes[-2:]) == -2:
            return 0.50  # Ultra-Strict (Losing streak)
            
        return base_strictness

def get_filling_mode(symbol=None):
    if symbol is None:
        symbol = SYMBOL
    info = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_IOC
    filling = info.filling_mode
    if filling & 2:
        return mt5.ORDER_FILLING_IOC
    elif filling & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN

def calculate_position_size(entry_price, sl_price, risk_pct, ai_conf=0.0, signal='BUY', comment_text=""):
    with mt5_lock:
        symbol_info = mt5.symbol_info(SYMBOL)
        account = mt5.account_info()
        if not symbol_info or not account:
            return 0.01

        # Check Manual Fixed Lot Override
        manual_lot = getattr(sys.modules['core.config'], 'MANUAL_FIXED_LOT', None)
        if manual_lot is not None and manual_lot > 0:
            step = symbol_info.volume_step
            if step > 0:
                step_str = str(step).rstrip('0')
                decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                manual_lot = round(math.floor(manual_lot / step + 1e-9) * step, decimals)
            print(f"[SHARP AGGRESSIVE] 🎯 Manual Fixed Lot Applied: {manual_lot} Lots")
            return max(symbol_info.volume_min, min(manual_lot, symbol_info.volume_max))

        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0: return 0.01

        # Unified Session Risk Sizing: Strategy-requested risk is the baseline;
        # AGGRESSIVE_MODE only boosts on high-conviction structural AI signals (not 1.2R RSI scalps).
        effective_risk = risk_pct
        is_rsi_scalp = bool(comment_text and "RSI" in str(comment_text))
        if getattr(sys.modules.get('core.config'), 'AGGRESSIVE_MODE', False) and not is_rsi_scalp:
            if ai_conf >= 0.65:
                # Only boost when AI model has high conviction on elite structural setups
                boost = getattr(sys.modules.get('core.config'), 'HIGH_CONFIDENCE_AI_BOOST', 2.0)
                boosted_risk = risk_pct * boost
                max_single = getattr(sys.modules.get('core.config'), 'MAX_SINGLE_TRADE_RISK_PCT', 5.0)
                effective_risk = min(boosted_risk, max_single)
                print(f"[SHARP AGGRESSIVE] 🚀 High AI Conviction ({ai_conf*100:.1f}%) -> Boosted Risk: {effective_risk:.1f}% (base: {risk_pct:.1f}%)")

        contract_size = symbol_info.trade_contract_size if symbol_info.trade_contract_size > 0 else 100.0
        account_balance = _parse_positive_finite_float(getattr(account, 'balance', None))
        if account_balance is None:
            return 0.01
        lots = (account_balance * (effective_risk / 100)) / (sl_dist * contract_size)
        step = symbol_info.volume_step
        if step > 0:
            step_str = str(step).rstrip('0')
            decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
            # Floor down so quantized volume never rounds up past rigid single-trade risk caps
            lots = round(math.floor(lots / step + 1e-9) * step, decimals)
            
        return max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

def execute_trade(signal, sl_price, risk_pct, magic_num=None, comment_text="", ai_conf=0.0, limit_price=None, stop_price=None, tp_price=None, feature_vector=None, strategy_name=None):
    global virtual_ticket_counter
    if magic_num is None:
        magic_num = MAGIC_NUMBER
    with mt5_lock:
        if nexus_state.get('is_halted', False):
            print("\n[REJECT] DAILY DRAWDOWN HALT ACTIVE: Blocking execute_trade().")
            return False
        positions_raw = mt5.positions_get() or []
        orders_raw = mt5.orders_get() or []
        
        bot_positions = [p for p in positions_raw if p.symbol == SYMBOL and p.magic == magic_num]
        bot_orders = [o for o in orders_raw if o.symbol == SYMBOL and o.magic == magic_num]
        
        # 1. Deduplication Safeguard & Smart Pyramiding
        strat_tag = comment_text.split(':')[0].strip() if ':' in comment_text else (comment_text.split()[0].strip() if comment_text else "")
        if strat_tag:
            is_rsi_trd = strat_tag.startswith("RSI_TRD")
            
            # Cross-Gear Lock between RSI Range and RSI Trend
            if is_rsi_trd:
                for p in bot_positions:
                    p_comm = getattr(p, 'comment', '') or ''
                    if "RSI_RNG" in p_comm:
                        print(f"\n[REJECT] CROSS-GEAR LOCK: Active RSI Range position #{p.ticket} ({p_comm}) blocks RSI Trend reload.")
                        return False
            elif strat_tag.startswith("RSI_RNG"):
                for p in bot_positions:
                    p_comm = getattr(p, 'comment', '') or ''
                    if "RSI_TRD" in p_comm:
                        print(f"\n[REJECT] CROSS-GEAR LOCK: Active RSI Trend position #{p.ticket} ({p_comm}) blocks RSI Range entry.")
                        return False

            if is_rsi_trd:
                # RSI Trend Smart Pyramiding: max 2 active positions
                for o in bot_orders:
                    o_comm = getattr(o, 'comment', '') or ''
                    if "RSI_TRD" in o_comm:
                        print(f"\n[REJECT] DUPLICATE PROTECT: Pending Order #{o.ticket} ({o_comm}) is already active for RSI_TRD.")
                        return False
                
                active_trd_pos = [p for p in bot_positions if "RSI_TRD" in (getattr(p, 'comment', '') or '')]
                if len(active_trd_pos) >= 2:
                    first_p = active_trd_pos[0]
                    print(f"\n[REJECT] DUPLICATE PROTECT: Position #{first_p.ticket} already active (Max 2 for RSI_TRD).")
                    return False
                elif len(active_trd_pos) == 1:
                    existing_p = active_trd_pos[0]
                    p_is_buy = (existing_p.type == mt5.ORDER_TYPE_BUY)
                    is_be_locked = (existing_p.sl >= existing_p.price_open + 0.50) if p_is_buy else (existing_p.sl > 0 and existing_p.sl <= existing_p.price_open - 0.50)
                    if not is_be_locked:
                        print(f"\n[REJECT] PYRAMID SHIELD: Existing RSI_TRD position #{existing_p.ticket} is not yet locked at Break-Even.")
                        return False
                    
                    p_time = getattr(existing_p, 'time', 0)
                    if p_time and (time.time() - p_time < 270):
                        print(f"\n[REJECT] PYRAMID SHIELD: Existing RSI_TRD position #{existing_p.ticket} was opened < 270s ago (same M5 candle). Stacking requires confirmed subsequent candle.")
                        return False
            else:
                # Strict 1-position deduplication for all other strategies (RSI_RNG, SMC_OB, LIQUIDITY_SWEEP, BB_BREAKOUT)
                for p in bot_positions:
                    p_comm = getattr(p, 'comment', '') or ''
                    if strat_tag in p_comm:
                        print(f"\n[REJECT] DUPLICATE PROTECT: Position #{p.ticket} ({p_comm}) is already active for {strat_tag}.")
                        return False
                for o in bot_orders:
                    o_comm = getattr(o, 'comment', '') or ''
                    if strat_tag in o_comm:
                        print(f"\n[REJECT] DUPLICATE PROTECT: Pending Order #{o.ticket} ({o_comm}) is already active for {strat_tag}.")
                        return False

        # 2. Concurrency Limit: Enforce maximum open trades + pending limits
        max_concurrent = getattr(sys.modules.get('core.config'), 'MAX_CONCURRENT_TRADES', 3)
        if len(bot_positions) + len(bot_orders) >= max_concurrent:
            print(f"\n[REJECT] CONCURRENCY LIMIT: {len(bot_positions)} position(s) + {len(bot_orders)} pending order(s) >= {max_concurrent} max limit.")
            return False

        # 2b. Risk-Free Pyramiding Rule: Only allow stacking a 2nd position when all existing
        # open positions are already locked at Break-Even or better, and no pending order is waiting.
        for p in bot_positions:
            p_is_buy = (p.type == mt5.ORDER_TYPE_BUY)
            p_is_safe = (p.sl >= p.price_open) if p_is_buy else (p.sl > 0 and p.sl <= p.price_open)
            if not p_is_safe:
                print(f"\n[REJECT] PYRAMID SHIELD: Open position #{p.ticket} is not yet locked at Break-Even. Blocking stacked exposure.")
                return False
        if len(bot_orders) >= 1:
            print(f"\n[REJECT] PYRAMID SHIELD: Pending order #{bot_orders[0].ticket} is already waiting on the book. Blocking stacked exposure.")
            return False

        # 3. Direction-Lock Safeguard: Prevent conflicting BUY and SELL trades from being open at the same time
        if not getattr(sys.modules.get('core.config'), 'ALLOW_HEDGING', False):
            incoming_direction = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL
            for p in bot_positions:
                if p.type != incoming_direction:
                    dir_existing = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                    print(f"\n[REJECT] DIRECTION CONFLICT! Already have open {dir_existing} position #{p.ticket}. Rejecting opposite {signal} signal.")
                    return False
            for o in bot_orders:
                o_is_buy = o.type in [getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', 2), getattr(mt5, 'ORDER_TYPE_BUY_STOP', 4)]
                if (signal == 'BUY' and not o_is_buy) or (signal == 'SELL' and o_is_buy):
                    dir_existing = "BUY" if o_is_buy else "SELL"
                    print(f"\n[REJECT] DIRECTION CONFLICT! Already have pending {dir_existing} order #{o.ticket}. Rejecting opposite {signal} signal.")
                    return False

        if DYNAMIC_RISK:
            is_reversal = any(keyword in comment_text for keyword in ["CHOCH", "RSI"])
            threshold = AI_REVERSAL_THRESHOLD if is_reversal else AI_TREND_THRESHOLD
            scale = max(0.5, min(2.0, (ai_conf - threshold) / (0.95 - threshold) * 1.5 + 0.5))
            risk_pct = risk_pct * scale

        if risk_pct <= 0: return False
        
        if feature_vector is not None:
            try:
                global _veto_brain_cache, _veto_brain_mtime
                veto_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "evolution_veto.joblib")
                if os.path.exists(veto_path):
                    curr_mtime = os.path.getmtime(veto_path)
                    if globals().get('_veto_brain_cache') is None or globals().get('_veto_brain_mtime', 0.0) != curr_mtime:
                        _veto_brain_cache = joblib.load(veto_path)
                        _veto_brain_mtime = curr_mtime
                    veto_brain = _veto_brain_cache
                    veto_model = veto_brain['model']
                    scaler = veto_brain['scaler']
                    
                    # Handle both numpy arrays (from strategies) and pipe-delimited strings (legacy)
                    if isinstance(feature_vector, str):
                        features = [float(x) for x in feature_vector.split('|')]
                    else:
                        features = [float(x) for x in feature_vector]
                        
                    expected_dim = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 21
                    if len(features) == expected_dim:
                        X_live = scaler.transform(np.array([features]))
                        loss_prob = veto_model.predict_proba(X_live)[0][0]
                        strictness = get_dynamic_evolution_strictness()
                        if loss_prob >= strictness:
                            print(f"\n[VETO] Trade Blocked by Evolution Layer ({loss_prob*100:.0f}% loss prob > {strictness*100:.0f}% strictness).")
                            notify_veto(loss_prob, strictness)
                            return False
                    else:
                        print(f"\n[WARNING] Veto Layer skipped: feature vector dimension mismatch (received {len(features)}, expected {expected_dim}).")
            except Exception as e:
                print(f"\n[WARNING] Veto failsafe triggered: {e}")
                
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None: return False
        
        # Spread Filter: Reject trade if broker spread exceeds dynamic safety threshold
        current_spread = abs(tick.ask - tick.bid)
        symbol_info = mt5.symbol_info(SYMBOL)
        spread_points = current_spread / symbol_info.point if symbol_info and symbol_info.point > 0 else 0
        spread_limit = dynamic_spread_limit if 'dynamic_spread_limit' in globals() else MAX_SPREAD_POINTS
        if spread_points > spread_limit:
            print(f"\n[SPREAD] Trade REJECTED. Spread {spread_points:.0f} pts > {spread_limit} pts limit.")
            return False
        
        price = tick.ask if signal == 'BUY' else tick.bid
        
        # Hybrid Execution Logic: Use Limit Order if AI confidence is extremely high and limit price is valid
        is_limit_order = False
        execution_price = price
        action_type = mt5.TRADE_ACTION_DEAL
        order_type = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL
        
        if limit_price:
            # We only use limit if it's better than current price
            if signal == 'BUY' and limit_price < price:
                is_limit_order = True
                execution_price = limit_price
                action_type = mt5.TRADE_ACTION_PENDING
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
                comment_text += f" LMT {limit_price:.1f}"
            elif signal == 'SELL' and limit_price > price:
                is_limit_order = True
                execution_price = limit_price
                action_type = mt5.TRADE_ACTION_PENDING
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
                comment_text += f" LMT {limit_price:.1f}"
            elif signal == 'BUY' and limit_price >= price:
                # Anti-Chaser Safety Guard: never convert a missed buy limit into a market buy at a worse price
                print(f"\n[ANTI-CHASER] BUY Rejected: Market price ({price:.2f}) has already rallied above Limit Entry ({limit_price:.2f}). Chasing forbidden.")
                return False
            elif signal == 'SELL' and limit_price <= price:
                # Anti-Chaser Safety Guard: never convert a failed sell limit to a market sell at the low
                print(f"\n[ANTI-CHASER] SELL Rejected: Market price ({price:.2f}) has already dumped below Limit Entry ({limit_price:.2f}). Chasing forbidden.")
                return False
                
        elif stop_price:
            is_limit_order = True  # Treat as pending
            execution_price = stop_price
            action_type = mt5.TRADE_ACTION_PENDING
            order_type = mt5.ORDER_TYPE_BUY_STOP if signal == 'BUY' else mt5.ORDER_TYPE_SELL_STOP
            comment_text += f" STP {stop_price:.1f}"
        
        lots = calculate_position_size(execution_price, sl_price, risk_pct, ai_conf=ai_conf, signal=signal, comment_text=comment_text)
        risk_dist = abs(execution_price - sl_price)
        
        # Structural Risk Guard: Prevent broker minimum lot override from destroying the account
        symbol_info = mt5.symbol_info(SYMBOL)
        account = mt5.account_info()
        if not symbol_info or not account:
            print("\n[REJECT] MT5 symbol_info or account_info unavailable -> FAIL-CLOSED.")
            return False

        account_balance = _parse_positive_finite_float(getattr(account, 'balance', None))
        raw_equity = getattr(account, 'equity', account_balance) if hasattr(account, 'equity') else account_balance
        account_equity = _parse_positive_finite_float(raw_equity)
        if account_balance is None or account_equity is None:
            print("\n[REJECT] Invalid or non-positive account balance/equity -> FAIL-CLOSED.")
            return False

        lots_val = _parse_positive_finite_float(lots)
        if lots_val is None:
            print("\n[REJECT] Invalid calculated lot size -> FAIL-CLOSED.")
            return False

        contract_size = _parse_positive_finite_float(getattr(symbol_info, 'trade_contract_size', None)) or 100.0
        actual_monetary_risk = risk_dist * lots_val * contract_size
        actual_risk_pct = (actual_monetary_risk / account_balance) * 100
        if not math.isfinite(actual_risk_pct):
            print("\n[REJECT] Non-finite risk calculation -> FAIL-CLOSED.")
            return False
        _cfg_mod = sys.modules.get('core.config')
        _limit_fn = getattr(_cfg_mod, 'get_daily_loss_limit_pct', None)
        max_daily_loss = float(_limit_fn(account_equity)) if callable(_limit_fn) else float(getattr(_cfg_mod, 'MAX_DAILY_LOSS_PERCENT', MAX_DAILY_LOSS_PERCENT))

        if actual_risk_pct > max_daily_loss:
            print(f"\n[REJECT] INSUFFICIENT EQUITY! The required lot size ({lots}) creates an actual risk of {actual_risk_pct:.1f}% on this trade, which exceeds your global daily limit of {max_daily_loss}%.")
            return False

        # Single-Trade Exposure Guard: Prevent minimum lot size from forcing too much risk on a single trade
        max_trade_risk = getattr(_cfg_mod, 'MAX_SINGLE_TRADE_RISK_PCT', 5.0)
        small_acc_mode = getattr(_cfg_mod, 'SMALL_ACCOUNT_MODE', False)
        small_acc_threshold = getattr(_cfg_mod, 'SMALL_ACCOUNT_THRESHOLD', 200.0)
        is_sharp_aggressive = getattr(_cfg_mod, 'AGGRESSIVE_MODE', False)
        manual_fixed_lot = getattr(_cfg_mod, 'MANUAL_FIXED_LOT', None)

        if manual_fixed_lot is not None or (small_acc_mode and account_balance < small_acc_threshold):
            # Only override risk cap for manual fixed lots or genuinely small accounts (< $200)
            max_trade_risk = max(max_trade_risk, 15.0)

        if actual_risk_pct > max_trade_risk:
            print(f"\n[REJECT] OVER-RISK EXPOSURE! The required lot size ({lots}) creates an actual risk of {actual_risk_pct:.2f}% on this trade, which exceeds your single-trade safety limit of {max_trade_risk}%.")
            return False
        
        # Dynamic Structural RR logic
        tp = None
        if tp_price is not None and risk_dist > 0:
            implied_rr = abs(tp_price - execution_price) / risk_dist
            if implied_rr >= getattr(sys.modules['core.config'], 'MIN_DYNAMIC_RR', 1.2):
                tp = tp_price
                comment_text += f" STR:{implied_rr:.1f}R"
                
        # Fallback to pure math if no valid structural TP exists (per-strategy R:R)
        if tp is None:
            strat_rr = STRATEGY_RR.get(strategy_name, RISK_REWARD_RATIO) if strategy_name else RISK_REWARD_RATIO
            tp = execution_price + (risk_dist * strat_rr) if signal == 'BUY' else execution_price - (risk_dist * strat_rr)
        
        # Pre-flight Limit Order Drift Guard: Reject orders that already exceed max allowable drift
        if is_limit_order and not DRY_RUN:
            tick = mt5.symbol_info_tick(SYMBOL)
            curr_market_price = tick.ask if signal == 'BUY' else (tick.bid if tick else None)
            if curr_market_price:
                init_drift = abs(execution_price - curr_market_price)
                atr_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 14)
                atr_calc = sum(r['high'] - r['low'] for r in atr_rates[-14:]) / 14.0 if (atr_rates is not None and len(atr_rates) >= 14) else 7.0
                max_allowed_drift = max(15.0, 3.0 * atr_calc)
                if init_drift > max_allowed_drift:
                    print(f"\n[REJECT] LIMIT DRIFT EXCEEDED! Proposed limit ${execution_price:.2f} is ${init_drift:.2f} pts from market ${curr_market_price:.2f} (max drift: ${max_allowed_drift:.2f}). Aborting order.")
                    return False

        exec_mode = "LIMIT" if is_limit_order else "MARKET"
        
        if not DRY_RUN:
            _si = mt5.symbol_info(SYMBOL)
            _ai = mt5.account_info()
            if _si and _ai and _ai.balance > 0:
                _contract_size = _si.trade_contract_size if _si.trade_contract_size > 0 else 100.0
                actual_risk_pct = (risk_dist * lots * _contract_size / _ai.balance) * 100
            else:
                actual_risk_pct = risk_pct
        else:
            actual_risk_pct = risk_pct
        
        print(f"\n[EXEC] {exec_mode} ({comment_text}) | Target Risk: {risk_pct:.2f}% | Actual Forced Risk: {actual_risk_pct:.2f}%")
        print(f"       Entry: {execution_price:.2f} | SL: {sl_price:.2f} | TP: {tp:.2f} | Lots: {lots}")
        
        if DRY_RUN:
            virtual_ticket_counter += 1
            with virtual_positions_lock:
                virtual_positions[virtual_ticket_counter] = {
                    'ticket': virtual_ticket_counter,
                    'type': order_type,
                    'price_open': execution_price,
                    'sl': sl_price,
                    'tp': tp,
                    'lots': lots,
                    'magic': magic_num,
                    'comment': comment_text
                }
            print(f"[VIRTUAL] {exec_mode} ORDER PLACED! Ticket: {virtual_ticket_counter}")
            return True
        
        # MT5 strictly enforces a 27-31 character limit on the comment field
        # and will crash if brackets are unbalanced or parsed incorrectly.
        final_comment = str(comment_text)[:27]
        
        expiration = int(time.time()) + (4 * 3600) if action_type == mt5.TRADE_ACTION_PENDING else 0
        request = {
            "action": action_type, "symbol": SYMBOL, "volume": float(lots), 
            "type": order_type, "price": float(execution_price), "sl": float(sl_price), "tp": float(tp), "deviation": int(DEVIATION), 
            "magic": int(magic_num), "comment": final_comment,
            "type_time": mt5.ORDER_TIME_SPECIFIED if action_type == mt5.TRADE_ACTION_PENDING else mt5.ORDER_TIME_GTC, 
            "type_filling": get_filling_mode(SYMBOL) if action_type == mt5.TRADE_ACTION_DEAL else mt5.ORDER_FILLING_RETURN,
        }
        if action_type == mt5.TRADE_ACTION_PENDING:
            request["expiration"] = expiration
        check = mt5.order_check(request)
        if check is None:
            print(f"[ERROR] Pre-flight CHECK FAILED: {mt5.last_error()}")
            return False
        result = mt5.order_send(request)
        if result is None:
            print("[ERROR] Order Failed: mt5.order_send returned None (Network Timeout)")
            return False
        if result.retcode not in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            print(f"[ERROR] {exec_mode} Order Failed (Code {result.retcode}): {result.comment}")
            return False

        print(f"[SUCCESS] {exec_mode} {'Executed' if result.retcode == mt5.TRADE_RETCODE_DONE else 'Placed'}! Ticket: {result.order}")
        
        # Verify and reconcile actual filled volume & price from broker response
        filled_lots = float(result.volume) if hasattr(result, 'volume') and result.volume > 0 else lots
        fill_price = float(result.price) if hasattr(result, 'price') and result.price > 0 else execution_price
        
        if abs(filled_lots - lots) > 0.0001:
            print(f"[RECONCILIATION] Warning: Partial Fill. Requested: {lots:.4f}, Filled: {filled_lots:.4f}")
        if abs(fill_price - execution_price) > 0.0001:
            slippage = abs(fill_price - execution_price)
            print(f"[RECONCILIATION] Slippage detected: {slippage:.4f} points (Requested: {execution_price:.2f}, Filled: {fill_price:.2f})")

        notify_trade_opened(signal, SYMBOL, fill_price, sl_price, tp, filled_lots, result.order, comment_text)
        broadcast_signal_opened(signal, SYMBOL, fill_price, sl_price, tp, filled_lots, comment_text)
        
        # Queue signal for copier daemon (separate process) - ONLY FOR FILLED MARKET DEALS!
        # Pending orders will be dispatched to copier once they are filled in manage_open_positions.
        if action_type == mt5.TRADE_ACTION_DEAL:
            try:
                from engine.copier import queue_copier_signal
                acc = mt5.account_info()
                queue_copier_signal("OPEN", {
                    "ticket": result.order, "signal": signal, "symbol": SYMBOL,
                    "order_mode": "DEAL", "order_type": signal,
                    "price": fill_price, "sl": sl_price, "tp": tp,
                    "lots": filled_lots, "magic": MAGIC_NUMBER,
                    "comment": comment_text,
                    "master_equity": acc.equity if acc else 0
                })
                _copied_master_tickets.add(result.order)
            except Exception as e:
                print(f"[COPIER] ⚠️ Queue error: {e}")
            try:
                from engine.supabase_pusher import sync_all
                threading.Thread(target=sync_all, daemon=True).start()
            except Exception:
                pass
        else:
            print(f"[COPIER] ⏳ Pending order #{result.order} placed on Master ({signal} @ ${fill_price:.2f}). Slaves will copy automatically upon fill execution.")
        
        # Log Full Context for Evolution Engine
        if feature_vector is not None:
            try:
                acc_info = mt5.account_info()
                acc_num = acc_info.login if acc_info else "000000"
                log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'master_decision_log_{acc_num}.csv')
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                balance = acc_info.balance if acc_info else 0.0
                equity = acc_info.equity if acc_info else 0.0
                spread = mt5.symbol_info(SYMBOL).spread if mt5.symbol_info(SYMBOL) else 0.0
                
                feat_str = ",".join([f"{x:.4f}" for x in feature_vector])
                safe_comment = comment_text.replace(",", ";")
                header_str = "Timestamp,Ticket,Signal,Comment,AI_Conf,Balance,Equity,Spread,Features\n"
                data_str = f"{datetime.now()},{result.order},{signal},{safe_comment},{ai_conf:.4f},{balance:.2f},{equity:.2f},{spread:.1f},{feat_str}\n"
                
                header_needed = not os.path.exists(log_file)
                with open(log_file, 'a') as f:
                    if header_needed:
                        f.write(header_str)
                    f.write(data_str)
            except Exception as e:
                print(f"[ERROR] Failed to log full context: {e}")
                
        return True
    
_pending_order_birth = {}  # Internal timestamp tracker: {ticket: unix_ts}

def cancel_all_pending_orders(reason="DAILY_LIMIT_REACHED"):
    """Cancel all resting pending MT5 orders owned by this bot."""
    with mt5_lock:
        if DRY_RUN:
            return 0
        orders = mt5.orders_get(symbol=SYMBOL)
        if not orders:
            return 0
        cancelled = 0
        for order in orders:
            if getattr(order, 'magic', None) != MAGIC_NUMBER:
                continue
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            res = mt5.order_send(req)
            if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                cancelled += 1
                print(f"\n[CANCELLED] Pending Order #{order.ticket} ({reason})")
                _pending_order_birth.pop(order.ticket, None)
                try:
                    from engine.copier import queue_copier_signal
                    queue_copier_signal("CANCEL_ORDER", {"ticket": order.ticket, "symbol": SYMBOL})
                except Exception:
                    pass
        return cancelled

def manage_pending_orders():
    """Cancels Pending Limit Orders that have expired or drifted excessively far from price.
    Exempts Breakout Stop Orders (BUY_STOP/SELL_STOP) from drift cleanup."""
    if DRY_RUN: return
    with mt5_lock:
        if nexus_state.get('is_halted', False):
            cancel_all_pending_orders(reason="DAILY_LIMIT_REACHED")
            return
        orders = mt5.orders_get(symbol=SYMBOL)
        if not orders:
            _pending_order_birth.clear()
            return
        
        now_ts = int(time.time())
        tick = mt5.symbol_info_tick(SYMBOL)
        curr_price = tick.bid if tick else None
        
        # Track any new orders we haven't seen before
        active_tickets = set()
        for order in orders:
            if order.magic != MAGIC_NUMBER: continue
            active_tickets.add(order.ticket)
            if order.ticket not in _pending_order_birth:
                # Use broker's time_setup if valid, otherwise use current time as birth
                broker_ts = getattr(order, 'time_setup', 0)
                _pending_order_birth[order.ticket] = broker_ts if broker_ts > 1000000000 else now_ts
        
        # Prune tracker for orders that no longer exist (filled or manually cancelled)
        stale_keys = [k for k in _pending_order_birth if k not in active_tickets]
        for k in stale_keys:
            del _pending_order_birth[k]
        
        for order in orders:
            if order.magic != MAGIC_NUMBER: continue
            
            # Check if this is a pullback limit order vs a breakout stop trap
            is_pullback_limit = order.type in [getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', 2), getattr(mt5, 'ORDER_TYPE_SELL_LIMIT', 3)]
            expiry_bars = getattr(sys.modules.get('core.config'), 'PENDING_ORDER_EXPIRY_BARS', 12)
            expiry_seconds = expiry_bars * 5 * 60 if is_pullback_limit else (4 * 3600)  # Stop traps valid for 4h Killzone window
            
            # Use internal birth tracker (immune to broker time_setup bugs)
            birth_ts = _pending_order_birth.get(order.ticket, now_ts)
            order_age = now_ts - birth_ts
            
            drift = abs(curr_price - order.price_open) if curr_price else 0.0
            
            # Dynamic drift threshold: 3x ATR instead of hardcoded $25
            atr_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 15)
            if atr_rates is not None and len(atr_rates) >= 14:
                atr_val = sum(r['high'] - r['low'] for r in atr_rates[-14:]) / 14.0
            else:
                atr_val = 7.0  # Fallback for Gold
            max_drift = max(15.0, 3.0 * atr_val)

            # Approach Velocity Guard: Cancel pending limit orders if incoming candle is surging at > 1.8x ATR speed
            is_velocity_spike = False
            velocity_reason = ""
            if is_pullback_limit and atr_rates is not None and len(atr_rates) >= 2 and atr_val > 0:
                c_now = atr_rates[-1]
                c_range = c_now['high'] - c_now['low']
                if c_range > (1.8 * atr_val):
                    is_sell_limit = (order.type == getattr(mt5, 'ORDER_TYPE_SELL_LIMIT', 3))
                    is_buy_limit = (order.type == getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', 2))
                    if is_sell_limit and (c_now['close'] > c_now['open']) and (curr_price is not None and curr_price >= order.price_open - (1.5 * atr_val)):
                        is_velocity_spike = True
                        velocity_reason = f"Freight Train Rocket ({c_range:.1f} pts > 1.8x ATR {1.8*atr_val:.1f})"
                    elif is_buy_limit and (c_now['close'] < c_now['open']) and (curr_price is not None and curr_price <= order.price_open + (1.5 * atr_val)):
                        is_velocity_spike = True
                        velocity_reason = f"Waterfall Dump ({c_range:.1f} pts > 1.8x ATR {1.8*atr_val:.1f})"

            # Only apply drift cancellation to pullback limits that have been alive for at least 60 seconds
            should_cancel = False
            cancel_detail = ""
            if is_pullback_limit:
                if is_velocity_spike:
                    should_cancel = True
                    cancel_detail = f"Velocity Abort: {velocity_reason}"
                elif order_age > expiry_seconds:
                    should_cancel = True
                    cancel_detail = f"Expired TTL (Age: {order_age/60:.1f}m)"
                elif curr_price and drift > max_drift and order_age >= 60:
                    should_cancel = True
                    cancel_detail = f"Excessive Drift (${drift:.2f} > ${max_drift:.2f})"
            elif not is_pullback_limit and (order_age > expiry_seconds):
                should_cancel = True
                cancel_detail = f"Expired TTL (Age: {order_age/60:.1f}m)"

            if should_cancel:
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
                res = mt5.order_send(req)
                if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                    order_kind = "Limit" if is_pullback_limit else "Stop Trap"
                    print(f"\n[TTL] Cancelled {order_kind} #{order.ticket} ({cancel_detail})")
                    _pending_order_birth.pop(order.ticket, None)
                    try:
                        from engine.copier import queue_copier_signal
                        queue_copier_signal("CANCEL_ORDER", {"ticket": order.ticket, "symbol": SYMBOL})
                    except Exception:
                        pass

def manage_open_positions():
    with mt5_lock:
        # Calculate Dynamic ATR Trailing Distance
        dynamic_trail_dist = getattr(sys.modules.get('core.config', sys.modules[__name__]), 'TRAIL_DIST_DOLLARS', 5.00)
        dynamic_trail_start = getattr(sys.modules.get('core.config', sys.modules[__name__]), 'TRAIL_START_DOLLARS', 8.00)
        if getattr(sys.modules.get('core.config', sys.modules[__name__]), 'USE_TRAILING_STOP', True):
            try:
                rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 15)
                if rates is not None and len(rates) >= 15:
                    df_trail = pd.DataFrame(rates)
                    atr_val = calculate_atr_simple(df_trail, period=14)
                    if atr_val > 0:
                        multiplier = getattr(sys.modules.get('core.config', sys.modules[__name__]), 'ATR_TRAIL_MULTIPLIER', 1.8)
                        dynamic_trail_dist = float(atr_val) * multiplier
                        dynamic_trail_start = max(dynamic_trail_start, dynamic_trail_dist + 1.00)
            except Exception as e:
                pass # Fallback to static distance on error
            
        if DRY_RUN:
            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick: return
            symbol_info = mt5.symbol_info(SYMBOL)
            contract_size = symbol_info.trade_contract_size if (symbol_info and symbol_info.trade_contract_size > 0) else 100.0
            closed_tickets = []
            pending_notifications = []
            with virtual_positions_lock:
                for v_id, pos in list(virtual_positions.items()):
                    curr_price = tick.bid if pos['type'] == mt5.ORDER_TYPE_BUY else tick.ask
                    pos_profit_dist = (curr_price - pos['price_open']) if pos['type'] == mt5.ORDER_TYPE_BUY else (pos['price_open'] - curr_price)
                    tp_dist = abs(pos['tp'] - pos['price_open'])
                    
                    if MOVE_BE and tp_dist > 0 and (pos_profit_dist / tp_dist) >= BE_TRIGGER_PCT:
                        if pos.get('be_moved') is None:
                            # Institutional Break-Even Buffer: Covers broker roundturn commission ($0.07/oz) + dynamic exit spread buffer
                            # In Gold (XAU/USD), a Long exits at market Bid. If SL is placed only at entry+commission, the Bid fills at a net loss.
                            live_spread = abs(tick.ask - tick.bid) if (tick and tick.ask > tick.bid) else 0.25
                            comm_friction = 0.07  # $7.00/lot ($0.07/oz) roundturn commission
                            swap_buffer = 0.05    # Overnight financing / safety cushion
                            offset = round(live_spread + comm_friction + swap_buffer, 2)  # Institutional friction: $0.37/oz
                            pos['sl'] = pos['price_open'] + offset if pos['type'] == mt5.ORDER_TYPE_BUY else pos['price_open'] - offset
                            pos['be_moved'] = True
                            if PARTIAL_CLOSE_PCT > 0:
                                closed_lots = pos['lots'] * PARTIAL_CLOSE_PCT
                                pos['lots'] -= closed_lots
                                profit = (pos_profit_dist) * closed_lots * contract_size
                                print(f"\n[PARTIAL] VIRTUAL CLOSE [Ticket {v_id}] | PnL: +{profit:.2f} | BE Hit")
                                pending_notifications.append((v_id, "VIRTUAL PARTIAL CLOSE BE HIT", profit, "BUY" if pos['type'] == mt5.ORDER_TYPE_BUY else "SELL", "VIRTUAL PARTIAL BE HIT"))
                                
                    if pos['type'] == mt5.ORDER_TYPE_BUY:
                        if curr_price <= pos['sl']:
                            profit = (pos['sl'] - pos['price_open']) * pos['lots'] * contract_size
                            print(f"\n[CLOSED] VIRTUAL SL HIT [Ticket {v_id}] | PnL: {profit:.2f}")
                            pending_notifications.append((v_id, "VIRTUAL SL HIT", profit, "BUY", "VIRTUAL SL HIT"))
                            closed_tickets.append(v_id); continue
                        if curr_price >= pos['tp']:
                            profit = (pos['tp'] - pos['price_open']) * pos['lots'] * contract_size
                            print(f"\n[CLOSED] VIRTUAL TP HIT [Ticket {v_id}] | PnL: +{profit:.2f}")
                            pending_notifications.append((v_id, "VIRTUAL TP HIT", profit, "BUY", "VIRTUAL TP HIT"))
                            closed_tickets.append(v_id); continue
                        
                        if USE_TRAILING_STOP and (curr_price - pos['price_open']) > dynamic_trail_start:
                            new_sl = curr_price - dynamic_trail_dist
                            if new_sl > pos['sl'] and new_sl > pos['price_open']: pos['sl'] = new_sl
                        if USE_TRAILING_TP and (pos['tp'] - curr_price) < TP_TRIGGER_DIST:
                             pos['tp'] += TP_EXTENSION_DIST

                    elif pos['type'] == mt5.ORDER_TYPE_SELL:
                        if curr_price >= pos['sl']:
                            profit = (pos['price_open'] - pos['sl']) * pos['lots'] * contract_size
                            print(f"\n[CLOSED] VIRTUAL SL HIT [Ticket {v_id}] | PnL: {profit:.2f}")
                            pending_notifications.append((v_id, "VIRTUAL SL HIT", profit, "SELL", "VIRTUAL SL HIT"))
                            closed_tickets.append(v_id); continue
                        if curr_price <= pos['tp']:
                            profit = (pos['price_open'] - pos['tp']) * pos['lots'] * contract_size
                            print(f"\n[CLOSED] VIRTUAL TP HIT [Ticket {v_id}] | PnL: +{profit:.2f}")
                            pending_notifications.append((v_id, "VIRTUAL TP HIT", profit, "SELL", "VIRTUAL TP HIT"))
                            closed_tickets.append(v_id); continue

                        if USE_TRAILING_STOP and (pos['price_open'] - curr_price) > dynamic_trail_start:
                            new_sl = curr_price + dynamic_trail_dist
                            if (pos['sl'] == 0 or new_sl < pos['sl']) and new_sl < pos['price_open']: pos['sl'] = new_sl
                        if USE_TRAILING_TP and (curr_price - pos['tp']) < TP_TRIGGER_DIST:
                             pos['tp'] -= TP_EXTENSION_DIST
                             
                for t in closed_tickets:
                    del virtual_positions[t]

            for v_id, reason, profit, direction, b_reason in pending_notifications:
                notify_trade_closed(v_id, reason, profit)
                broadcast_signal_closed(direction, SYMBOL, profit, b_reason)
            return
            
        positions_raw = mt5.positions_get()
        positions = [p for p in positions_raw if p.symbol == SYMBOL] if positions_raw else []
        current_tickets = {p.ticket: p for p in positions}
        
        # Track and notify organically closed trades (TP/SL hit on server)
        closed_deal_retry_counts = getattr(sys.modules[__name__], '_closed_deal_retry_counts', {})
        closed_live = [t for t in tracked_live_tickets if t not in current_tickets]
        for ticket in closed_live:
            deals = None
            for _ in range(3):
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    break
                time.sleep(0.1)
            if deals:
                closing_deal = deals[-1]
                profit = sum(d.profit + d.commission + d.swap + d.fee for d in deals) # Total position PnL including swaps/commissions
                
                # Detect whether it was TP, Trailing SL, or Standard SL hit
                comment = getattr(closing_deal, 'comment', '').lower()
                if "tp" in comment:
                    reason = "TP HIT"
                elif "sl" in comment:
                    if profit > 0:
                        reason = "TRAILING SL HIT"
                    else:
                        reason = "SL HIT"
                else:
                    if profit > 0:
                        reason = "TP HIT"
                    else:
                        reason = "SL HIT"
                        
                direction = tracked_live_tickets[ticket]['direction']
                strategy_comment = tracked_live_tickets[ticket].get('comment', 'NEXUS AI')
                notify_trade_closed(ticket, reason, profit)
                broadcast_signal_closed(direction, SYMBOL, profit, strategy_comment, reason)
                try:
                    from engine.copier import queue_copier_signal
                    queue_copier_signal("CLOSE", {"ticket": ticket, "symbol": SYMBOL})
                except Exception:
                    pass
                try:
                    from engine.supabase_pusher import sync_all
                    threading.Thread(target=sync_all, daemon=True).start()
                except Exception:
                    pass
                del tracked_live_tickets[ticket]
                closed_deal_retry_counts.pop(ticket, None)
                _partially_closed_tickets.discard(ticket)
                live_notified_sl.pop(ticket, None)
                _copied_master_tickets.discard(ticket)
            else:
                print(f"[WARNING] History deals for closed ticket {ticket} delayed/not found, holding ticket tracking.")
                if ticket not in closed_deal_retry_counts:
                    closed_deal_retry_counts[ticket] = 0
                closed_deal_retry_counts[ticket] += 1
                if closed_deal_retry_counts[ticket] >= 10:
                    print(f"[CLEANUP] Purged stale closed ticket #{ticket} after 10 retries.")
                    del tracked_live_tickets[ticket]
                    del closed_deal_retry_counts[ticket]
                    _partially_closed_tickets.discard(ticket)
                    live_notified_sl.pop(ticket, None)
                    _copied_master_tickets.discard(ticket)

        if not positions: return
        for pos in positions:
            if pos.magic != MAGIC_NUMBER: continue
            
            # --- AUTO-MIRROR FILLED PENDING ORDERS TO COPIER ---
            if pos.ticket not in _copied_master_tickets:
                try:
                    from engine.copier import queue_copier_signal
                    acc = mt5.account_info()
                    dir_str = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    queue_copier_signal("OPEN", {
                        "ticket": pos.ticket, "signal": dir_str, "symbol": SYMBOL,
                        "order_mode": "DEAL", "order_type": dir_str,
                        "price": pos.price_open, "sl": pos.sl, "tp": pos.tp,
                        "lots": pos.volume, "magic": MAGIC_NUMBER,
                        "comment": getattr(pos, 'comment', 'NEXUS AI'),
                        "master_equity": acc.equity if acc else 0
                    })
                    _copied_master_tickets.add(pos.ticket)
                    print(f"\n[COPIER] 🎯 Master Pending Order #{pos.ticket} FILLED ({dir_str} {pos.volume} @ ${pos.price_open:.2f}) -> Dispatched to Slave Accounts!")
                except Exception as e:
                    print(f"[COPIER] ⚠️ Fill dispatch error: {e}")

            if pos.ticket not in tracked_live_tickets:
                tracked_live_tickets[pos.ticket] = {
                    'direction': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'comment': getattr(pos, 'comment', 'NEXUS AI')
                }
                
            tick_check = mt5.symbol_info_tick(SYMBOL)
            if tick_check is None: continue
            curr_price = tick_check.bid if pos.type == mt5.ORDER_TYPE_BUY else tick_check.ask
            pos_profit_dist = (curr_price - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY else (pos.price_open - curr_price)
            tp_dist = abs(pos.tp - pos.price_open)
            
            symbol_info = mt5.symbol_info(SYMBOL)
            digits = getattr(symbol_info, 'digits', 2) if symbol_info else 2
            point = getattr(symbol_info, 'point', 0.01) if symbol_info else 0.01
            stops_level = getattr(symbol_info, 'stops_level', 0) or 0
            min_stop_dist = max(stops_level * point, 0.50)
            
            # Track live mutable SL/TP across this tick so later blocks (e.g. Trailing Stop)
            # never compare against a stale snapshot and overwrite a tighter Profit Shield SL backward
            current_sl = pos.sl
            current_tp = pos.tp
            
            if MOVE_BE and tp_dist > 0 and (pos_profit_dist / tp_dist) >= BE_TRIGGER_PCT:
                contract_size = symbol_info.trade_contract_size if (symbol_info and symbol_info.trade_contract_size > 0) else 100.0
                
                # Fetch MT5 cumulative commission + swap fees
                commission = getattr(pos, 'commission', 0.0)
                swap = getattr(pos, 'swap', 0.0)
                total_fees = abs(commission) + abs(swap)
                if total_fees == 0.0:
                    total_fees = 0.85 * (pos.volume / 0.01)
                    
                offset_points = total_fees / (contract_size * pos.volume) if (contract_size * pos.volume) > 0 else 0.85
                offset_points = max(0.85, min(offset_points, 1.50))  # Lock +0.85 to +1.50 pts so BE exits net positive after spread
                
                # Clean Break-Even with positive spread-clearing profit cushion
                target_sl = pos.price_open + offset_points if pos.type == mt5.ORDER_TYPE_BUY else pos.price_open - offset_points
                target_sl = round(target_sl, digits)
                
                be_condition = current_sl >= target_sl if pos.type == mt5.ORDER_TYPE_BUY else (current_sl <= target_sl and current_sl > 0)
                if not be_condition:
                    request_sl = {
                        "action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL,
                        "sl": target_sl, "tp": current_tp, "magic": pos.magic
                    }
                    res_sl = mt5.order_send(request_sl)
                    if res_sl is None or res_sl.retcode not in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                        ret_code = res_sl.retcode if res_sl else "None"
                        ret_comment = res_sl.comment if res_sl else "Network Timeout"
                        print(f"[WARNING] BE Move Request Failed for Ticket {pos.ticket} (Code {ret_code}): {ret_comment}")
                    else:
                        prev_sl_val = current_sl
                        current_sl = target_sl
                        live_notified_sl[pos.ticket] = target_sl
                        broadcast_trade_update("BE", SYMBOL, f"{target_sl:.2f}")
                        try:
                            from engine.copier import queue_copier_signal
                            queue_copier_signal("MODIFY", {
                                "action_type": "BE",
                                "ticket": pos.ticket, "symbol": SYMBOL,
                                "master_entry": pos.price_open,
                                "sl": target_sl, "tp": current_tp, "prev_sl": prev_sl_val
                            })
                        except Exception:
                            pass
                    
                # Execute 50% partial close once per position at >= BE_TRIGGER_PCT, even if trailing stop moved SL earlier
                if PARTIAL_CLOSE_PCT > 0 and pos.ticket not in _partially_closed_tickets:
                    vol_min = symbol_info.volume_min if symbol_info else 0.01
                    if pos.volume > vol_min:
                        action_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        _ptick = mt5.symbol_info_tick(SYMBOL)
                        if not _ptick:
                            continue
                        close_price = _ptick.ask if action_type == mt5.ORDER_TYPE_BUY else _ptick.bid
                        vol_to_close = pos.volume * PARTIAL_CLOSE_PCT
                        step = symbol_info.volume_step if symbol_info else 0.01
                        if step > 0:
                            step_str = str(step).rstrip('0')
                            decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                            vol_to_close = round(round(vol_to_close / step) * step, decimals)
                        if vol_to_close >= vol_min:
                             req_partial = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": vol_to_close,
                                "type": action_type, "position": pos.ticket, "price": close_price, "deviation": DEVIATION, "magic": pos.magic,
                                "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(pos.symbol),
                             }
                             res = mt5.order_send(req_partial)
                             if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                                 _partially_closed_tickets.add(pos.ticket)
                                 print(f"\n[PARTIAL] LIVE CLOSE [Ticket {pos.ticket}] | BE Hit")
                                 broadcast_trade_update("PARTIAL", SYMBOL, f"{close_price:.2f}")
                                 try:
                                     from engine.copier import queue_copier_signal
                                     queue_copier_signal("PARTIAL", {
                                         "action_type": "PARTIAL",
                                         "ticket": pos.ticket, "symbol": SYMBOL,
                                         "close_volume": vol_to_close,
                                         "price": close_price
                                     })
                                 except Exception:
                                     pass
                    else:
                        _partially_closed_tickets.add(pos.ticket)
                        print(f"\n[BE_TRIGGER] Skip partial close: ticket {pos.ticket} is already at broker minimum volume ({pos.volume:.2f} lots). Holding full size to TP.")
                                 
            if USE_TRAILING_STOP:
                # --- Adaptive 3-Tier Dynamic Profit Shield ---
                # Tier 1: Climax Blow-Off Spike Lock (Wick >= 40% & Extreme RSI, gated at >= 65% TP)
                # Tier 2: 75% TP Milestone Guard (Locks >= 50% TP distance in pure profit)
                # Tier 3: Momentum Stagnation Lock (3-Candle Stall Shelf + RSI Deceleration, gated at >= 65% TP)
                try:
                    rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 20)
                    if rates_m5 is not None and len(rates_m5) >= 16:
                        df_m5_exhaust = pd.DataFrame(rates_m5)
                        df_m5_exhaust['tr'] = np.maximum(
                            df_m5_exhaust['high'] - df_m5_exhaust['low'],
                            np.maximum(
                                abs(df_m5_exhaust['high'] - df_m5_exhaust['close'].shift(1)),
                                abs(df_m5_exhaust['low'] - df_m5_exhaust['close'].shift(1))
                            )
                        )
                        # Use confirmed closed-candle ATR (iloc[-2]) and RSI (iloc[-2] / iloc[-3])
                        # so live intra-bar tick noise does not re-trigger micro-adjustments on the same closed candle
                        closed_atr = df_m5_exhaust['tr'].rolling(14).mean().iloc[-2]
                        atr_padding = closed_atr if pd.notna(closed_atr) and closed_atr > 0 else 3.5
                        min_exhaust_profit = max(9.0, atr_padding * 2.4)
                        in_shield_zone = (tp_dist > 0 and pos_profit_dist >= 0.55 * tp_dist) or (tp_dist == 0 and pos_profit_dist >= min_exhaust_profit)
                        
                        rsi_m5 = calculate_rsi_simple(df_m5_exhaust['close'], 14)
                        closed_m5_rsi = rsi_m5.iloc[-2] if len(rsi_m5) > 1 else 50.0
                        prev_closed_rsi = rsi_m5.iloc[-3] if len(rsi_m5) > 2 else closed_m5_rsi

                        # Look strictly at confirmed closed candles to eliminate intra-bar false alarms
                        last_closed_candle = df_m5_exhaust.iloc[-2]
                        last_3_closed = df_m5_exhaust.iloc[-4:-1]
                        
                        # ================================================
                        # 1. BUY POSITION PROFIT GUARDS
                        # ================================================
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            target_new_sl = None
                            guard_trigger = ""
                            min_locked_buy_sl = round(pos.price_open + (0.35 * pos_profit_dist), digits)

                            # -- TIER 2: Progressive Milestone Guard --
                            # Only tighten milestone if target is >= 10.0 pts, and enforce minimum 1.0x ATR breathing room
                            min_breathing_dist = max(3.00, atr_padding * 0.85)
                            goal_breathing = max(1.80, atr_padding * 0.45)
                            remaining_to_tp = (current_tp - curr_price) if current_tp > 0 else 999.0
                            # -- TIER 0: 90% Ultra-Harvest Lock (Within $1.50 of TP or >= 90% progress) --
                            if tp_dist > 0 and (pos_profit_dist >= 0.90 * tp_dist or remaining_to_tp <= 1.50):
                                raw_sl = pos.price_open + (0.88 * tp_dist)
                                harvest_sl = round(min(raw_sl, curr_price - 1.20), digits)
                                if (harvest_sl - current_sl) >= 0.25 and harvest_sl > pos.price_open and harvest_sl <= (curr_price - 0.60):
                                    target_new_sl = harvest_sl
                                    guard_trigger = f"90% TP Ultra-Harvest Lock (+88% TP Guaranteed)"
                            elif tp_dist >= 10.0 and pos_profit_dist >= 0.85 * tp_dist:
                                raw_sl = pos.price_open + (0.75 * tp_dist)
                                milestone_sl = round(min(raw_sl, curr_price - goal_breathing), digits)
                                if (milestone_sl - current_sl) >= 0.30 and milestone_sl > pos.price_open and milestone_sl <= (curr_price - goal_breathing):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"85% TP Goal-Line Lock (+75% TP Guaranteed)"
                            elif tp_dist >= 10.0 and pos_profit_dist >= 0.75 * tp_dist:
                                raw_sl = pos.price_open + (0.50 * tp_dist)
                                milestone_sl = round(min(raw_sl, curr_price - min_breathing_dist), digits)
                                if (milestone_sl - current_sl) >= 0.40 and milestone_sl > pos.price_open and milestone_sl <= (curr_price - min_breathing_dist):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"75% TP Milestone Lock (+50% TP Guaranteed)"
                            elif tp_dist >= 8.0 and pos_profit_dist >= 0.65 * tp_dist:
                                raw_sl = pos.price_open + (0.30 * tp_dist)
                                milestone_sl = round(min(raw_sl, curr_price - min_breathing_dist), digits)
                                if (milestone_sl - current_sl) >= 0.40 and milestone_sl > pos.price_open and milestone_sl <= (curr_price - min_breathing_dist):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"65% TP Milestone Lock (+30% TP Guaranteed)"

                            # -- LIVING SHADOW: Structural ATR Trail behind closed M5 pivot lows --
                            if getattr(sys.modules.get('core.config'), 'ENABLE_LIVING_SHADOW', True):
                                shadow_min_pct = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_MIN_PROFIT_PCT', 0.55)
                                if (tp_dist > 0 and pos_profit_dist >= shadow_min_pct * tp_dist) or (tp_dist == 0 and pos_profit_dist >= 5.0):
                                    shadow_mult = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_ATR_MULT', 1.10)
                                    shadow_min_buf = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_MIN_BUFFER', 2.20)
                                    shadow_buf = max(shadow_min_buf, atr_padding * shadow_mult)
                                    c_low_ref = min(last_closed_candle['low'], df_m5_exhaust.iloc[-3]['low'])
                                    cand_sl = round(c_low_ref - shadow_buf, digits)
                                    be_level = round(pos.price_open + 0.50, digits)
                                    cand_sl = max(cand_sl, be_level)
                                    if cand_sl > current_sl and (cand_sl - current_sl) >= 0.35 and cand_sl <= (curr_price - min_stop_dist):
                                        if target_new_sl is None or cand_sl > target_new_sl:
                                            target_new_sl = cand_sl
                                            guard_trigger = f"Living Shadow Trail (${shadow_buf:.2f} buffer)"

                            # -- TIER 1: Climax Blow-Off Spike Guard --
                            # Only arm when >= 65% of TP distance is reached and a confirmed bearish reversal candle closed
                            if in_shield_zone:
                                c = last_closed_candle
                                c_open, c_high, c_low, c_close = c['open'], c['high'], c['low'], c['close']
                                c_range = max(0.01, c_high - c_low)
                                upper_wick = max(0.0, c_high - max(c_open, c_close))
                                upper_wick_pct = upper_wick / c_range
                                
                                # Genuine reversal: candle closed red (bearish) or heavy rejection (>=45%), with extreme overbought RSI
                                is_bearish_reversal_close = (c_close < c_open or upper_wick_pct >= 0.45)
                                if upper_wick_pct >= 0.40 and closed_m5_rsi >= 70.0 and is_bearish_reversal_close:
                                    climax_sl = round(max(min_locked_buy_sl, c_low - max(1.20, atr_padding * 0.50)), digits)
                                    if (climax_sl - current_sl) >= 0.40 and climax_sl >= min_locked_buy_sl and climax_sl <= (curr_price - min_stop_dist):
                                        if target_new_sl is None or climax_sl > target_new_sl:
                                            target_new_sl = climax_sl
                                            guard_trigger = f"Climax Peak (Wick: {upper_wick_pct*100:.0f}%, RSI: {closed_m5_rsi:.0f})"

                            # -- TIER 3: Momentum Stagnation / Sideways Stall Guard --
                            if target_new_sl is None and in_shield_zone:
                                three_highs = last_3_closed['high'].values
                                is_stalling = (three_highs[-1] <= three_highs[0] + 0.30 and three_highs[-2] <= three_highs[0] + 0.30)
                                rsi_decay = (closed_m5_rsi < prev_closed_rsi - 3.0 or closed_m5_rsi < 60.0)
                                if is_stalling and rsi_decay:
                                    shelf_low = round(max(min_locked_buy_sl, last_3_closed['low'].min() - max(1.00, atr_padding * 0.40)), digits)
                                    if (shelf_low - current_sl) >= 0.40 and shelf_low >= min_locked_buy_sl and shelf_low <= (curr_price - min_stop_dist):
                                        target_new_sl = shelf_low
                                        guard_trigger = f"Momentum Stall Shelf (RSI Decel {closed_m5_rsi:.0f})"

                            # Execute SL update if a valid tighter profit-protecting SL was calculated
                            if target_new_sl is not None and (target_new_sl - current_sl) >= 0.40:
                                req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": target_new_sl, "tp": current_tp, "magic": pos.magic}
                                res = mt5.order_send(req)
                                if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                                    prev_sl_val = current_sl
                                    current_sl = target_new_sl
                                    print(f"\n[DYNAMIC PROFIT SHIELD] BUY #{pos.ticket} triggered {guard_trigger} -> Snapped SL to ${target_new_sl:.2f}")
                                    live_notified_sl[pos.ticket] = target_new_sl
                                    broadcast_trade_update("TRAIL", SYMBOL, f"LOCK:{target_new_sl:.2f}")
                                    try:
                                        from engine.copier import queue_copier_signal
                                        queue_copier_signal("MODIFY", {
                                            "action_type": "TRAIL",
                                            "ticket": pos.ticket, "symbol": SYMBOL,
                                            "master_entry": pos.price_open,
                                            "sl": target_new_sl, "tp": current_tp, "prev_sl": prev_sl_val
                                        })
                                    except Exception:
                                        pass
                                else:
                                    ret_code = res.retcode if res else "None"
                                    ret_comment = res.comment if res else "Network Timeout"
                                    print(f"[WARNING] Dynamic Profit Shield SL Request Failed for BUY #{pos.ticket} (Code {ret_code}): {ret_comment}")

                        # ================================================
                        # 2. SELL POSITION PROFIT GUARDS
                        # ================================================
                        elif pos.type == mt5.ORDER_TYPE_SELL:
                            target_new_sl = None
                            guard_trigger = ""
                            min_locked_sell_sl = round(pos.price_open - (0.35 * pos_profit_dist), digits)

                            # -- TIER 2: Progressive Milestone Guard --
                            # Only tighten milestone if target is >= 10.0 pts, and enforce minimum 1.0x ATR breathing room
                            min_breathing_dist = max(3.00, atr_padding * 0.85)
                            goal_breathing = max(1.80, atr_padding * 0.45)
                            remaining_to_tp = (curr_price - current_tp) if current_tp > 0 else 999.0
                            # -- TIER 0: 90% Ultra-Harvest Lock (Within $1.50 of TP or >= 90% progress) --
                            if tp_dist > 0 and (pos_profit_dist >= 0.90 * tp_dist or remaining_to_tp <= 1.50):
                                raw_sl = pos.price_open - (0.88 * tp_dist)
                                harvest_sl = round(max(raw_sl, curr_price + 1.20), digits)
                                if (current_sl == 0.0 or (current_sl - harvest_sl) >= 0.25) and harvest_sl < pos.price_open and harvest_sl >= (curr_price + 0.60):
                                    target_new_sl = harvest_sl
                                    guard_trigger = f"90% TP Ultra-Harvest Lock (+88% TP Guaranteed)"
                            elif tp_dist >= 10.0 and pos_profit_dist >= 0.85 * tp_dist:
                                raw_sl = pos.price_open - (0.75 * tp_dist)
                                milestone_sl = round(max(raw_sl, curr_price + goal_breathing), digits)
                                if (current_sl == 0.0 or (current_sl - milestone_sl) >= 0.30) and milestone_sl < pos.price_open and milestone_sl >= (curr_price + goal_breathing):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"85% TP Goal-Line Lock (+75% TP Guaranteed)"
                            elif tp_dist >= 10.0 and pos_profit_dist >= 0.75 * tp_dist:
                                raw_sl = pos.price_open - (0.50 * tp_dist)
                                milestone_sl = round(max(raw_sl, curr_price + min_breathing_dist), digits)
                                if (current_sl == 0.0 or (current_sl - milestone_sl) >= 0.40) and milestone_sl < pos.price_open and milestone_sl >= (curr_price + min_breathing_dist):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"75% TP Milestone Lock (+50% TP Guaranteed)"
                            elif tp_dist >= 8.0 and pos_profit_dist >= 0.65 * tp_dist:
                                raw_sl = pos.price_open - (0.30 * tp_dist)
                                milestone_sl = round(max(raw_sl, curr_price + min_breathing_dist), digits)
                                if (current_sl == 0.0 or (current_sl - milestone_sl) >= 0.40) and milestone_sl < pos.price_open and milestone_sl >= (curr_price + min_breathing_dist):
                                    target_new_sl = milestone_sl
                                    guard_trigger = f"65% TP Milestone Lock (+30% TP Guaranteed)"

                            # -- LIVING SHADOW: Structural ATR Trail behind closed M5 pivot highs --
                            if getattr(sys.modules.get('core.config'), 'ENABLE_LIVING_SHADOW', True):
                                shadow_min_pct = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_MIN_PROFIT_PCT', 0.55)
                                if (tp_dist > 0 and pos_profit_dist >= shadow_min_pct * tp_dist) or (tp_dist == 0 and pos_profit_dist >= 5.0):
                                    shadow_mult = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_ATR_MULT', 1.10)
                                    shadow_min_buf = getattr(sys.modules.get('core.config'), 'LIVING_SHADOW_MIN_BUFFER', 2.20)
                                    shadow_buf = max(shadow_min_buf, atr_padding * shadow_mult)
                                    c_high_ref = max(last_closed_candle['high'], df_m5_exhaust.iloc[-3]['high'])
                                    cand_sl = round(c_high_ref + shadow_buf, digits)
                                    be_level = round(pos.price_open - 0.50, digits)
                                    cand_sl = min(cand_sl, be_level)
                                    if (current_sl == 0.0 or cand_sl < current_sl) and (current_sl == 0.0 or (current_sl - cand_sl) >= 0.35) and cand_sl >= (curr_price + min_stop_dist):
                                        if target_new_sl is None or cand_sl < target_new_sl:
                                            target_new_sl = cand_sl
                                            guard_trigger = f"Living Shadow Trail (${shadow_buf:.2f} buffer)"

                            # -- TIER 1: Climax Blow-Off Bottom Guard --
                            # Only arm when >= 65% of TP distance is reached and a confirmed bullish reversal candle closed
                            if in_shield_zone:
                                c = last_closed_candle
                                c_open, c_high, c_low, c_close = c['open'], c['high'], c['low'], c['close']
                                c_range = max(0.01, c_high - c_low)
                                lower_wick = max(0.0, min(c_open, c_close) - c_low)
                                lower_wick_pct = lower_wick / c_range
                                
                                # Genuine reversal: candle closed green (bullish) or heavy rejection (>=45%), with extreme oversold RSI
                                is_bullish_reversal_close = (c_close > c_open or lower_wick_pct >= 0.45)
                                if lower_wick_pct >= 0.40 and closed_m5_rsi <= 28.0 and is_bullish_reversal_close:
                                    climax_sl = round(min(min_locked_sell_sl, c_high + max(1.20, atr_padding * 0.50)), digits)
                                    if (current_sl == 0.0 or (current_sl - climax_sl) >= 0.40) and climax_sl <= min_locked_sell_sl and climax_sl >= (curr_price + min_stop_dist):
                                        if target_new_sl is None or climax_sl < target_new_sl:
                                            target_new_sl = climax_sl
                                            guard_trigger = f"Climax Bottom (Wick: {lower_wick_pct*100:.0f}%, RSI: {closed_m5_rsi:.0f})"

                            # -- TIER 3: Momentum Stagnation / Sideways Stall Guard --
                            if target_new_sl is None and in_shield_zone:
                                three_lows = last_3_closed['low'].values
                                is_stalling = (three_lows[-1] >= three_lows[0] - 0.30 and three_lows[-2] >= three_lows[0] - 0.30)
                                rsi_rebound = (closed_m5_rsi > prev_closed_rsi + 3.0 or closed_m5_rsi > 40.0)
                                if is_stalling and rsi_rebound:
                                    shelf_high = round(min(min_locked_sell_sl, last_3_closed['high'].max() + max(1.00, atr_padding * 0.40)), digits)
                                    if (current_sl == 0.0 or (current_sl - shelf_high) >= 0.40) and shelf_high <= min_locked_sell_sl and shelf_high >= (curr_price + min_stop_dist):
                                        target_new_sl = shelf_high
                                        guard_trigger = f"Momentum Stall Shelf (RSI Rebound {closed_m5_rsi:.0f})"

                            # Execute SL update if a valid tighter profit-protecting SL was calculated
                            if target_new_sl is not None and (current_sl == 0.0 or (current_sl - target_new_sl) >= 0.40):
                                req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": target_new_sl, "tp": current_tp, "magic": pos.magic}
                                res = mt5.order_send(req)
                                if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                                    prev_sl_val = current_sl
                                    current_sl = target_new_sl
                                    print(f"\n[DYNAMIC PROFIT SHIELD] SELL #{pos.ticket} triggered {guard_trigger} -> Snapped SL to ${target_new_sl:.2f}")
                                    live_notified_sl[pos.ticket] = target_new_sl
                                    broadcast_trade_update("TRAIL", SYMBOL, f"LOCK:{target_new_sl:.2f}")
                                    try:
                                        from engine.copier import queue_copier_signal
                                        queue_copier_signal("MODIFY", {
                                            "action_type": "TRAIL",
                                            "ticket": pos.ticket, "symbol": SYMBOL,
                                            "master_entry": pos.price_open,
                                            "sl": target_new_sl, "tp": current_tp, "prev_sl": prev_sl_val
                                        })
                                    except Exception:
                                        pass
                                else:
                                    ret_code = res.retcode if res else "None"
                                    ret_comment = res.comment if res else "Network Timeout"
                                    print(f"[WARNING] Dynamic Profit Shield SL Request Failed for SELL #{pos.ticket} (Code {ret_code}): {ret_comment}")
                except Exception:
                    pass

                if pos.type == mt5.ORDER_TYPE_BUY and (curr_price - pos.price_open) > dynamic_trail_start:
                    new_sl = round(curr_price - dynamic_trail_dist, digits)
                    if (new_sl - current_sl) >= 0.50 and new_sl > pos.price_open and new_sl <= (curr_price - min_stop_dist):
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": new_sl, "tp": current_tp, "magic": pos.magic}
                        res = mt5.order_send(req)
                        if res is None or res.retcode not in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                            ret_code = res.retcode if res else "None"
                            ret_comment = res.comment if res else "Network Timeout"
                            print(f"[WARNING] Trailing SL Request Failed for Ticket {pos.ticket} (Code {ret_code}): {ret_comment}")
                        elif res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                            prev_sl_val = current_sl
                            current_sl = new_sl
                            last_sl = live_notified_sl.get(pos.ticket, pos.price_open)
                            if abs(new_sl - last_sl) >= 2.00:
                                live_notified_sl[pos.ticket] = new_sl
                                broadcast_trade_update("TRAIL", SYMBOL, f"{new_sl:.2f}")
                                try:
                                    from engine.copier import queue_copier_signal
                                    queue_copier_signal("MODIFY", {
                                        "action_type": "TRAIL",
                                        "ticket": pos.ticket, "symbol": SYMBOL,
                                        "master_entry": pos.price_open,
                                        "sl": new_sl, "tp": current_tp, "prev_sl": prev_sl_val
                                    })
                                except Exception:
                                    pass
                elif pos.type == mt5.ORDER_TYPE_SELL and (pos.price_open - curr_price) > dynamic_trail_start:
                    new_sl = round(curr_price + dynamic_trail_dist, digits)
                    if (current_sl == 0.0 or (current_sl - new_sl) >= 0.50) and new_sl < pos.price_open and new_sl >= (curr_price + min_stop_dist):
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": new_sl, "tp": current_tp, "magic": pos.magic}
                        res = mt5.order_send(req)
                        if res is None or res.retcode not in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                            ret_code = res.retcode if res else "None"
                            ret_comment = res.comment if res else "Network Timeout"
                            print(f"[WARNING] Trailing SL Request Failed for Ticket {pos.ticket} (Code {ret_code}): {ret_comment}")
                        elif res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                            prev_sl_val = current_sl
                            current_sl = new_sl
                            last_sl = live_notified_sl.get(pos.ticket, pos.price_open)
                            if abs(new_sl - last_sl) >= 2.00:
                                live_notified_sl[pos.ticket] = new_sl
                                broadcast_trade_update("TRAIL", SYMBOL, f"{new_sl:.2f}")
                                try:
                                    from engine.copier import queue_copier_signal
                                    queue_copier_signal("MODIFY", {
                                        "action_type": "TRAIL",
                                        "ticket": pos.ticket, "symbol": SYMBOL,
                                        "master_entry": pos.price_open,
                                        "sl": new_sl, "tp": current_tp, "prev_sl": prev_sl_val
                                    })
                                except Exception:
                                    pass
                                
            # --- Live Trailing TP (Two-Target Structural Jump with Softened SL) ---
            if USE_TRAILING_TP:
                tp_trigger_dist = getattr(sys.modules.get('core.config', sys.modules[__name__]), 'TP_TRIGGER_DIST', 1.00)
                tp_extension_dist = getattr(sys.modules.get('core.config', sys.modules[__name__]), 'TP_EXTENSION_DIST', 5.00)
                
                is_close_to_tp = False
                if pos.type == mt5.ORDER_TYPE_BUY and current_tp > 0:
                    is_close_to_tp = (current_tp - curr_price) < tp_trigger_dist
                elif pos.type == mt5.ORDER_TYPE_SELL and current_tp > 0:
                    is_close_to_tp = (curr_price - current_tp) < tp_trigger_dist
                    
                if is_close_to_tp:
                    try:
                        rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 100)
                        rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 100)
                        df_h1 = pd.DataFrame(rates_h1) if rates_h1 is not None and len(rates_h1) > 0 else None
                        df_m15 = pd.DataFrame(rates_m15) if rates_m15 is not None and len(rates_m15) > 0 else None
                        
                        from core.indicators import map_market_structure
                        
                        target_2 = None
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            h1_targets = []
                            if df_h1 is not None:
                                h1_struct = map_market_structure(df_h1, lookback=2)
                                h1_targets = sorted([p['price'] for p in h1_struct if p['type'] == 'HIGH' and p['price'] > current_tp])
                            
                            m15_targets = []
                            if df_m15 is not None:
                                m15_struct = map_market_structure(df_m15, lookback=2)
                                m15_targets = sorted([p['price'] for p in m15_struct if p['type'] == 'HIGH' and p['price'] > current_tp])
                                
                            if h1_targets:
                                target_2 = h1_targets[0]
                            elif m15_targets:
                                target_2 = m15_targets[0]
                            else:
                                target_2 = current_tp + tp_extension_dist
                            print(f"\n[TP_JUMP_CALC] BUY Ticket {pos.ticket} near TP. H1 High Pivots: {h1_targets}, M15 High Pivots: {m15_targets}. Target chosen: {target_2:.2f}")
                                
                        elif pos.type == mt5.ORDER_TYPE_SELL:
                            h1_targets = []
                            if df_h1 is not None:
                                h1_struct = map_market_structure(df_h1, lookback=2)
                                h1_targets = sorted([p['price'] for p in h1_struct if p['type'] == 'LOW' and p['price'] < current_tp], reverse=True)
                            
                            m15_targets = []
                            if df_m15 is not None:
                                m15_struct = map_market_structure(df_m15, lookback=2)
                                m15_targets = sorted([p['price'] for p in m15_struct if p['type'] == 'LOW' and p['price'] < current_tp], reverse=True)
                                
                            if h1_targets:
                                target_2 = h1_targets[0]
                            elif m15_targets:
                                target_2 = m15_targets[0]
                            else:
                                target_2 = current_tp - tp_extension_dist
                            print(f"\n[TP_JUMP_CALC] SELL Ticket {pos.ticket} near TP. H1 Low Pivots: {h1_targets}, M15 Low Pivots: {m15_targets}. Target chosen: {target_2:.2f}")
                    except Exception as e:
                        target_2 = (current_tp + tp_extension_dist) if pos.type == mt5.ORDER_TYPE_BUY else (current_tp - tp_extension_dist)
                    
                    # Softened SL floor calculation (Target 1 + 1.5x ATR dynamic buffer)
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        new_sl = current_tp - dynamic_trail_dist
                        new_sl = max(current_sl, new_sl)
                        new_sl = round(new_sl, digits)
                        target_2 = round(target_2, digits)
                        
                        if abs(target_2 - current_tp) >= 0.50:
                            req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": new_sl, "tp": target_2, "magic": pos.magic}
                            res = mt5.order_send(req)
                            if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                                prev_sl_val = current_sl
                                current_sl = new_sl
                                current_tp = target_2
                                print(f"\n[TP_JUMP] BUY Ticket {pos.ticket} extended TP to {target_2:.2f}, SL locked at {new_sl:.2f}")
                                broadcast_trade_update("TRAIL", SYMBOL, f"TP_JUMP:{target_2:.2f}")
                                try:
                                    from engine.copier import queue_copier_signal
                                    queue_copier_signal("MODIFY", {
                                        "action_type": "TP_JUMP",
                                        "ticket": pos.ticket, "symbol": SYMBOL,
                                        "master_entry": pos.price_open,
                                        "sl": new_sl, "tp": target_2, "prev_sl": prev_sl_val
                                    })
                                except Exception:
                                    pass
                    elif pos.type == mt5.ORDER_TYPE_SELL:
                        new_sl = current_tp + dynamic_trail_dist
                        if current_sl > 0:
                            new_sl = min(current_sl, new_sl)
                        new_sl = round(new_sl, digits)
                        target_2 = round(target_2, digits)
                        
                        if abs(current_tp - target_2) >= 0.50:
                            req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": SYMBOL, "sl": new_sl, "tp": target_2, "magic": pos.magic}
                            res = mt5.order_send(req)
                            if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                                prev_sl_val = current_sl
                                current_sl = new_sl
                                current_tp = target_2
                                print(f"\n[TP_JUMP] SELL Ticket {pos.ticket} extended TP to {target_2:.2f}, SL locked at {new_sl:.2f}")
                                broadcast_trade_update("TRAIL", SYMBOL, f"TP_JUMP:{target_2:.2f}")
                                try:
                                    from engine.copier import queue_copier_signal
                                    queue_copier_signal("MODIFY", {
                                        "action_type": "TP_JUMP",
                                        "ticket": pos.ticket, "symbol": SYMBOL,
                                        "master_entry": pos.price_open,
                                        "sl": new_sl, "tp": target_2, "prev_sl": prev_sl_val
                                    })
                                except Exception:
                                    pass
                                
        # Clean up live_notified_sl and _copied_master_tickets memory
        active_tickets = {p.ticket for p in positions}
        keys_to_delete = [k for k in live_notified_sl if k not in active_tickets]
        for k in keys_to_delete:
            del live_notified_sl[k]
            
        _copied_master_tickets.intersection_update(active_tickets)

def close_all_positions(reason="FRIDAY_PROTOCOL"):
    with mt5_lock:
        if DRY_RUN:
            with virtual_positions_lock:
                virtual_positions.clear()
            print("\n[LIQUIDATED] DRY_RUN: All Virtual Positions Liquidated.")
            return
            
        positions_raw = mt5.positions_get()
        positions = [p for p in positions_raw if p.symbol == SYMBOL] if positions_raw else []
        if not positions: return
        for pos in positions:
            if pos.magic != MAGIC_NUMBER: continue
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None: continue
            action_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume, 
                "type": action_type, "position": pos.ticket, "price": price, 
                "deviation": DEVIATION, "magic": MAGIC_NUMBER, "comment": reason,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(SYMBOL),
            }
            res = mt5.order_send(request)
            if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                print(f"\n[CLOSED] Ticket #{pos.ticket} ({reason})")
                time.sleep(0.1)
                deals = mt5.history_deals_get(position=pos.ticket)
                profit = sum(d.profit + d.commission + d.swap + d.fee for d in deals) if deals else (pos.profit + getattr(pos, 'swap', 0.0))
                
                orig_dir = "SELL" if pos.type == mt5.ORDER_TYPE_SELL else "BUY"
                if pos.ticket in tracked_live_tickets:
                    orig_dir = tracked_live_tickets[pos.ticket]['direction']
                    del tracked_live_tickets[pos.ticket]
                    _closed_deal_retry_counts.pop(pos.ticket, None)
                    
                notify_trade_closed(pos.ticket, reason, profit)
                broadcast_signal_closed(orig_dir, SYMBOL, profit, reason)
                try:
                    from engine.copier import queue_copier_signal
                    queue_copier_signal("CLOSE", {"ticket": pos.ticket, "symbol": SYMBOL})
                except Exception:
                    pass

def cleanup_stale_pending_orders(current_price, atr):
    """Cancels orphaned pending limit orders if price has drifted excessively far away or order is stale."""
    with mt5_lock:
        try:
            orders_raw = mt5.orders_get() or []
            bot_orders = [o for o in orders_raw if o.symbol == SYMBOL and o.magic == MAGIC_NUMBER]
            now_ts = int(time.time())
            for o in bot_orders:
                dist = abs(current_price - o.price_open)
                max_allowed_dist = max(atr * 4.0, 20.0)
                order_age_s = now_ts - o.time_setup if hasattr(o, 'time_setup') and o.time_setup > 0 else 0
                
                # Cancel if order is older than 2 hours or drifted > max_allowed_dist
                if (order_age_s > 7200 and dist > atr * 2.0) or dist > max_allowed_dist:
                    req_cancel = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": o.ticket,
                    }
                    res = mt5.order_send(req_cancel)
                    if res and res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                        print(f"\n[CLEANUP] Cancelled stale pending order #{o.ticket} (Drift: ${dist:.2f} > ${max_allowed_dist:.2f})")
        except Exception:
            pass
