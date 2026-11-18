import warnings
warnings.filterwarnings("ignore") # Global suppression of all warnings

import time
import sys
import os
import joblib
import shutil
import pandas as pd
import numpy as np
import threading
import argparse
from datetime import datetime, timezone, timedelta

# Add the project root to sys.path so 'core', 'engine' and 'strategies' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable Windows 10/11 ANSI terminal colors/cursor-movement support natively
os.system("")
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.config import *
from core.state_manager import nexus_state
from core.utils import is_broker_dst, get_session_name, get_broker_hour_from_utc
from core.indicators import calculate_atr_simple, calculate_rsi_simple, calculate_adx_simple
from engine.mt5_interface import connect_mt5, check_daily_drawdown, get_market_data, execute_trade, manage_open_positions, manage_pending_orders, check_volatility_guard, close_all_positions, get_dynamic_kelly_risk
from engine.news_filter import fetch_economic_news, is_news_blackout, get_active_blackout, get_post_news_guard_status
from engine.trade_logger import export_trade_ledger
from engine.telegram_notifier import notify_regime_change, update_bot_state, is_paused, start_command_listener
from engine.supabase_pusher import sync_all
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.smc_orderblock import SMCOrderBlockStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.bb_breakout import BBBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.smc_choch import SMCChochStrategy
import MetaTrader5 as mt5
from core.performance_tracker import get_dynamic_strategy_ranking, check_strategy_cooldown, check_global_directional_cooldown

def load_ai_models():
    trend, reversal = None, None
    try:
        if os.path.exists(TREND_MODEL_FILE):
            trend = joblib.load(TREND_MODEL_FILE)
            if hasattr(trend, 'n_jobs'): trend.n_jobs = 1
        else:
            print(f"[WARNING] Trend model file missing: {TREND_MODEL_FILE}")
    except Exception as e:
        print(f"[ERROR] Error loading trend model: {e}")

    try:
        if os.path.exists(REVERSAL_MODEL_FILE):
            reversal = joblib.load(REVERSAL_MODEL_FILE)
            if hasattr(reversal, 'n_jobs'): reversal.n_jobs = 1
        else:
            print(f"[WARNING] Reversal model file missing: {REVERSAL_MODEL_FILE}")
    except Exception as e:
        print(f"[ERROR] Error loading reversal model: {e}")
        
    return trend, reversal

def main(max_iterations=None):
    import core.config as _cfg

    parser = argparse.ArgumentParser(description="NEXUS Quant Engine")
    parser.add_argument("--terminal", type=str, choices=["default", "1", "2"], default="default",
                        help="Terminal instance mode: 'default' (standard MT5), '1' (Terminal_1), or '2' (Terminal_2)")
    parser.add_argument("--mt5-path", type=str, default=None, help="Custom path to terminal64.exe")
    parser.add_argument("--magic", type=int, default=None, help="Custom magic number override")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram command listener polling")
    parser.add_argument("--no-broadcast", action="store_true", help="Disable Telegram channel signal broadcasting")
    parser.add_argument("--broadcast", action="store_true", help="Force enable Telegram channel signal broadcasting")
    args, _ = parser.parse_known_args()

    target_mode = args.terminal
    custom_path = args.mt5_path
    custom_magic = args.magic

    if target_mode == "1":
        instance_tag = "t1"
        target_path = custom_path or os.getenv("MT5_TERMINAL_1_PATH", "")
        active_magic = custom_magic or 999011
        enable_tg_polling = False
        enable_tg_broadcast = bool(args.broadcast)
    elif target_mode == "2":
        instance_tag = "t2"
        target_path = custom_path or os.getenv("MT5_TERMINAL_2_PATH", "")
        active_magic = custom_magic or 999022
        enable_tg_polling = False
        enable_tg_broadcast = bool(args.broadcast)
    else:
        instance_tag = None
        target_path = custom_path or getattr(_cfg, 'MASTER_MT5_PATH', os.getenv("MT5_DEFAULT_PATH", ""))
        active_magic = custom_magic or getattr(_cfg, 'MAGIC_NUMBER', 999001)
        enable_tg_polling = not args.no_telegram
        enable_tg_broadcast = not args.no_broadcast

    # Configure Telegram channel broadcasting per instance
    from engine.telegram_notifier import set_broadcast_enabled
    set_broadcast_enabled(enable_tg_broadcast)

    # 1. Scope persistent state to this instance
    nexus_state.set_instance(instance_tag)

    from core.license_manager import authenticate_nexus_session
    authenticate_nexus_session()

    print("[START] Booting NEXUS Core...")
    print("="*55)
    print(f" 🚀 INSTANCE MODE: [{target_mode.upper()}] | Magic: {active_magic}")
    print(f" 📁 State File: {nexus_state.state_file_path}")
    print("="*55)
    connect_mt5(terminal_path=target_path, is_portable=(target_mode in ["1", "2"]), magic_override=active_magic)
    
    # Re-sync SYMBOL after potential Gold symbol auto-resolution (e.g. XAUUSDm → XAUUSD on FundedNext)
    global SYMBOL, MAGIC_NUMBER
    import core.config as _cfg
    SYMBOL = _cfg.SYMBOL
    MAGIC_NUMBER = active_magic
    _cfg.MAGIC_NUMBER = active_magic
    
    trend_model, reversal_model = load_ai_models()
    if trend_model is None:
        print("[CRITICAL] Engine shutting down due to Missing Intelligence (Joblib)")
        sys.exit()

    # Instantiate Strategies
    strategies = {
        "SMC_OB": SMCOrderBlockStrategy(trend_model),
        "LIQUIDITY_SWEEP": LiquiditySweepStrategy(reversal_model),
        "RSI_REVERSION": RSIReversionStrategy(reversal_model, trend_model),
        "BB_BREAKOUT": BBBreakoutStrategy(trend_model),
    }
    if getattr(sys.modules.get('core.config'), 'ENABLE_SMC_CHOCH', False):
        strategies["SMC_CHOCH"] = SMCChochStrategy(reversal_model)
    if getattr(sys.modules.get('core.config'), 'ENABLE_VWAP_REVERSION', False):
        strategies["VWAP_REVERSION"] = VWAPReversionStrategy()
    
    last_eval_time = 0 
    last_m5_time = None
    last_bar_bucket = -1
    was_observe_only = False
    df_m5, df_h1, df_h4, df_adx = None, None, None, None
    curr_adx, current_atr, curr_rsi = 0, 0, 0
    vol_ratio = 1.0
    is_safe = True
    last_heartbeat_min = -1
    last_equity_check_time = 0
    last_ledger_update_time = 0
    cached_equity = 0.0
    last_deal_count = mt5.history_deals_total(datetime.now(timezone.utc) - timedelta(days=2), datetime.now(timezone.utc) + timedelta(days=1))
    last_supabase_sync = 0
    market_closed_for_weekend = False
    
    prev_strats_str = ""
    prev_session_str = ""
    in_news_blackout = False
    in_daily_dd_halt = False
    last_rank_update = 0
    performance_stats = {}
    sorted_strats_all = []
    
    print("\n" + "="*55)
    print("[NEXUS ONLINE]")
    vwap_tag = ", VWAP" if getattr(sys.modules.get('core.config'), 'ENABLE_VWAP_REVERSION', False) else ""
    choch_tag = ", SMC_CHOCH" if getattr(sys.modules.get('core.config'), 'ENABLE_SMC_CHOCH', False) else ""
    print(f"[ENGINES: SMC_OB, RSI{vwap_tag}, BB_BREAKOUT, LIQUIDITY_SWEEP{choch_tag}]")
    print("="*55 + "\n")

    # Start Telegram Command Listener (only on primary instance to avoid HTTP 409 conflict)
    if enable_tg_polling:
        start_command_listener()
        print("[TELEGRAM] Command listener active. Type /help in Telegram.\n")
    else:
        print(f"[TELEGRAM] Command listener polling bypassed for Instance [{target_mode.upper()}] (Prevents HTTP 409 conflict).\n")

    if enable_tg_broadcast:
        print("[TELEGRAM] Channel signal broadcasting ACTIVE.\n")
    else:
        print(f"[TELEGRAM] Channel broadcasting muted for Instance [{target_mode.upper()}] (Master instance broadcasts only).\n")

    # Initial Sync with Supabase (run in background so boot is instant)
    try:
        threading.Thread(target=sync_all, daemon=True).start()
    except Exception as e:
        print(f"[SUPABASE] Initial sync error: {e}")

    last_logged_error = None
    last_error_time = 0
    consecutive_errors = 0

    iteration = 0
    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        iteration += 1
        try:
            is_daily_halted = not check_daily_drawdown()
            if is_daily_halted and not in_daily_dd_halt:
                now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                print(f"\n[DRAWDOWN] {now_str} Daily Drawdown Protection Active (Passive Observation Mode).")
                in_daily_dd_halt = True
            elif not is_daily_halted and in_daily_dd_halt:
                now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                print(f"\n[DRAWDOWN] {now_str} Daily Drawdown Protection Cleared. Normal execution resumed.")
                in_daily_dd_halt = False

            server_time = datetime.now(timezone.utc)
            utc_hour = server_time.hour
            server_hour = get_broker_hour_from_utc(server_time)
            
            # Build broker-adjusted datetime so weekday and hour are in the same timezone
            dst_offset = 3 if is_broker_dst(server_time) else 2
            broker_time = server_time + timedelta(hours=dst_offset)
            broker_weekday = broker_time.weekday()
            
            is_weekend = (broker_weekday == 5) or \
                         (broker_weekday == 4 and utc_hour >= FRIDAY_CLOSE_HOUR) or \
                         (broker_weekday == 6 and server_hour < 22)
            
            if is_weekend:
                if CLOSE_ALL_ON_FRIDAY and not market_closed_for_weekend:
                    close_all_positions()
                market_closed_for_weekend = True
                print(f"\r[SLEEP] Market Closed for Weekend (Kill-Switch Active)".ljust(90), end='')
                time.sleep(60); continue
            
            if market_closed_for_weekend and not is_weekend:
                market_closed_for_weekend = False

            # (DST shifts are now handled natively inside get_broker_hour_from_utc)
            is_dst_active = is_broker_dst(server_time) if AUTO_DST_ADJUST else False
            
            current_session = get_session_name(server_time)
            sess_map = {"ASIAN": "ASIA", "LONDON": "LDN", "NY_LONDON_OVERLAP": "OLAP", "NEW_YORK": "NY", "ROLLOVER": "ROLL"}
            short_session = sess_map.get(current_session, current_session)
            if prev_session_str and current_session != prev_session_str:
                now_str = server_time.strftime("%H:%M:%S UTC")
                old_short = sess_map.get(prev_session_str, prev_session_str)
                print(f"\n[SESSION] {now_str} Session Switch: {old_short} -> {short_session} (Active Risk: {SESSION_RISK.get(current_session, 0.0):.2f}%)")
            prev_session_str = current_session
            current_risk = SESSION_RISK.get(current_session, 0.0)
            
            if DYNAMIC_RISK and current_risk > 0:
                current_risk = get_dynamic_kelly_risk(current_risk)
            
            dst_tag = "[DST]" if is_dst_active else ""
            # Rollover Blackout: 23:15 to 00:30 UTC/Server time (daily rollover, spread blowout, zero liquidity)
            is_rollover_active = (utc_hour == 23 and server_time.minute >= 15) or (utc_hour == 0 and server_time.minute < 30)

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None: time.sleep(1); continue
            curr_price = tick.bid
            
            now_ts = time.time()
            if now_ts - last_equity_check_time > 60:
                acc_info = mt5.account_info()
                if acc_info: cached_equity = acc_info.equity
                last_equity_check_time = now_ts
                
            # Auto-Log: Trigger CSV ledger update instantly if recent deals increase (trade open/close)
            query_start = datetime.now(timezone.utc) - timedelta(days=2)
            current_deals = mt5.history_deals_total(query_start, datetime.now(timezone.utc) + timedelta(days=1))
            if current_deals is not None and current_deals > last_deal_count:
                export_trade_ledger()
                threading.Thread(target=sync_all, daemon=True).start()
                last_deal_count = current_deals
            elif now_ts - last_ledger_update_time > 900:  # Fallback: Every 15 minutes
                export_trade_ledger()
                last_ledger_update_time = now_ts
                
            # Periodic Supabase Sync: Every 25s if positions are open (real-time floating PnL), else every 5 minutes
            has_open_pos = bool(mt5.positions_total() and mt5.positions_total() > 0)
            sync_interval = 25 if has_open_pos else 300
            if now_ts - last_supabase_sync > sync_interval:
                threading.Thread(target=sync_all, daemon=True).start()
                last_supabase_sync = now_ts
            
            # Active Position & Order Management (runs continuously on every tick to maintain shield & stops)
            manage_open_positions()
            manage_pending_orders()

            # Daily End-of-Day Square-Off (20:00 UTC / 16:00 NY cash close)
            # Closes all open positions so no trades float into dead evening hours or rollover
            if getattr(sys.modules.get('core.config'), 'CLOSE_ALL_DAILY', True):
                daily_close_hour = getattr(sys.modules.get('core.config'), 'DAILY_CLOSE_ALL_HOUR', 20)
                if utc_hour >= daily_close_hour and not is_weekend:
                    if has_open_pos:
                        print(f"\n[DAILY CLOSE] EOD Square-Off triggered at {utc_hour}:00 UTC. Closing open positions...")
                        close_all_positions(reason="DAILY_EOD_SQUAREOFF")
                        has_open_pos = False

            is_off_hours = (utc_hour < ASIAN_OPEN_HOUR or utc_hour >= TRADING_END_HOUR)
            current_bar_bucket = int(server_time.timestamp() // 300)
            is_bar_boundary = (current_bar_bucket != last_bar_bucket)
            waking_from_observe = (was_observe_only and not (is_rollover_active or is_off_hours or in_news_blackout or is_daily_halted))

            now_ts = time.time()
            if is_bar_boundary or waking_from_observe or (now_ts - last_eval_time) > 1.5:
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_ENTRY, 0, 1)
                fetch_economic_news()
                
                if rates is not None and rates[0]['time'] != last_m5_time:
                    df_m5 = get_market_data(SYMBOL, TIMEFRAME_ENTRY, n=500)
                    df_h1 = get_market_data(SYMBOL, TIMEFRAME_TREND, n=100)
                    df_h4 = get_market_data(SYMBOL, mt5.TIMEFRAME_H4, n=300)
                    df_adx = get_market_data(SYMBOL, TIMEFRAME_ADX, n=100)
                    if df_m5 is not None: last_m5_time = df_m5.iloc[-1]['time']
                else:
                    df_m5_light = get_market_data(SYMBOL, TIMEFRAME_ENTRY, n=500)
                    if df_m5_light is not None: df_m5 = df_m5_light
                
                last_eval_time = now_ts
                last_bar_bucket = current_bar_bucket
                
                if df_m5 is not None and df_h1 is not None and df_adx is not None:
                    is_safe, vol_ratio = check_volatility_guard(df_m5)
                    if not is_safe:
                        curr_min = datetime.now().minute
                        if curr_min != last_heartbeat_min:
                            print(f"\n[VOL_GUARD] High Volatility ({vol_ratio:.1f}x) - Reversion Engines Disabled.")
                            last_heartbeat_min = curr_min

                    current_atr = calculate_atr_simple(df_m5, 14)
                    df_m5['rsi'] = calculate_rsi_simple(df_m5['close'], RSI_PERIOD)
                    curr_rsi = df_m5.iloc[-1]['rsi']
                    curr_adx = calculate_adx_simple(df_adx, ADX_PERIOD).iloc[-1]

            if df_m5 is None or df_h1 is None or df_h4 is None:
                print(f"\r[DATA ERROR] Market data returned None for '{SYMBOL}'. Check if symbol exists on this broker server.".ljust(120), end='')
                time.sleep(5); continue
            
            news_hold_str = ""
            if is_news_blackout(df_m5):
                b_info = get_active_blackout() or {}
                b_title = b_info.get('title', 'High-Impact Macro Event')
                b_curr = b_info.get('currency', 'USD')
                b_end = b_info.get('end')
                b_end_str = b_end.strftime('%H:%M UTC') if b_end else 'TBD'
                if not in_news_blackout:
                    now_str = server_time.strftime("%H:%M:%S UTC")
                    print(f"\n[NEWS] {now_str} Blackout Engaged until {b_end_str} | {b_curr}: {b_title} (Passive Observation Active)")
                    in_news_blackout = True
                news_hold_str = f"[NEWS-HOLD->{b_end_str}]"
            else:
                if in_news_blackout:
                    now_str = server_time.strftime("%H:%M:%S UTC")
                    print(f"\n[NEWS] {now_str} Blackout Expired. Normal execution resumed.")
                    in_news_blackout = False

            is_observe_only = is_rollover_active or is_off_hours or in_news_blackout or is_daily_halted
            was_observe_only = is_observe_only
                
            # --- Strategy Routing ---
            regime_icon = "[NEUTRAL]"
            
            # Displacement Velocity Detection: Bypass ADX lag during explosive momentum cascades
            # Use confirmed closed candles (iloc[-2] vs iloc[-4] / iloc[-5]) with closed-bar ATR to prevent intra-bar regime thrashing
            is_velocity_displacement = False
            if getattr(sys.modules.get('core.config'), 'DISPLACEMENT_VELOCITY_OVERRIDE', True):
                if df_m5 is not None and len(df_m5) >= 5:
                    closed_atr = calculate_atr_simple(df_m5.iloc[:-1], 14) if len(df_m5) > 15 else current_atr
                    eval_atr = closed_atr if closed_atr > 0 else current_atr
                    if eval_atr > 0:
                        disp_3 = abs(df_m5['close'].iloc[-2] - df_m5['close'].iloc[-5])
                        disp_2 = abs(df_m5['close'].iloc[-2] - df_m5['close'].iloc[-4])
                        mult_3 = getattr(sys.modules.get('core.config'), 'DISPLACEMENT_3BAR_ATR_MULT', 2.5)
                        mult_2 = getattr(sys.modules.get('core.config'), 'DISPLACEMENT_2BAR_ATR_MULT', 2.0)
                        if disp_3 >= (mult_3 * eval_atr) or disp_2 >= (mult_2 * eval_atr):
                            is_velocity_displacement = True

            # Phase 2: Time-Based Strategy Pruning & Dynamic Regime Selection
            if AUTO_SWITCH:
                if current_session == "ASIAN":
                    if is_velocity_displacement or curr_adx > ADX_TREND_START:
                        active_strats = ["SMC_OB", "LIQUIDITY_SWEEP", "RSI_REVERSION", "BB_BREAKOUT"]
                        regime_icon = "[ASIA-BREAKOUT]"
                    else:
                        active_strats = ["RSI_REVERSION", "LIQUIDITY_SWEEP"]
                        regime_icon = "[ASIAN-MATRIX]"
                        if curr_adx < 18.0:
                            active_strats = ["RSI_REVERSION", "LIQUIDITY_SWEEP"]
                            regime_icon = "[ASIAN-LOWVOL]"
                else:
                    enable_choch = getattr(sys.modules.get('core.config'), 'ENABLE_SMC_CHOCH', False)
                    if curr_adx > ADX_TREND_START or is_velocity_displacement:
                        active_strats = ["SMC_OB", "LIQUIDITY_SWEEP", "BB_BREAKOUT", "RSI_REVERSION"]
                        if enable_choch:
                            active_strats.append("SMC_CHOCH")
                        regime_icon = "[TREND-VEL]" if (is_velocity_displacement and curr_adx <= ADX_TREND_START) else "[TREND]"
                    elif curr_adx < ADX_RANGE_START:
                        range_strats = ["RSI_REVERSION", "LIQUIDITY_SWEEP"]
                        active_strats = range_strats
                        regime_icon = "[RANGE]"
                    else:
                        active_strats = ["SMC_OB", "RSI_REVERSION", "LIQUIDITY_SWEEP", "BB_BREAKOUT"]
                        if enable_choch:
                            active_strats.append("SMC_CHOCH")
                        regime_icon = "[NEUTRAL]"
            else:
                # Fallback when AUTO_SWITCH is off: run all main strategies
                enable_choch = getattr(sys.modules.get('core.config'), 'ENABLE_SMC_CHOCH', False)
                active_strats = ["SMC_OB", "RSI_REVERSION", "LIQUIDITY_SWEEP", "BB_BREAKOUT"]
                if enable_choch:
                    active_strats.append("SMC_CHOCH")
                regime_icon = "[MANUAL-ALL]"

            # Dynamic Performance Ranking League
            current_time = time.time()
            if current_time - last_rank_update >= 3600 or not sorted_strats_all:
                enable_choch = getattr(sys.modules.get('core.config'), 'ENABLE_SMC_CHOCH', False)
                all_league = ["SMC_OB", "LIQUIDITY_SWEEP", "RSI_REVERSION", "BB_BREAKOUT"]
                if enable_choch:
                    all_league.append("SMC_CHOCH")
                if getattr(sys.modules.get('core.config'), 'ENABLE_VWAP_REVERSION', False):
                    all_league.append("VWAP_REVERSION")
                sorted_strats_all, performance_stats = get_dynamic_strategy_ranking(all_league)
                last_rank_update = current_time
                
                now_str = server_time.strftime("%H:%M:%S UTC")
                print(f"\n" + "="*55)
                print(f"[LEAGUE] [NEXUS PERFORMANCE LEAGUE] Live Rankings ({now_str})")
                print(f"="*55)
                for rank, s_name in enumerate(sorted_strats_all, 1):
                    s_stat = performance_stats.get(s_name, {"win_rate": 55.0, "net_profit": 0.0, "total_trades": 0})
                    print(f" {rank}. {s_name.ljust(18)} | Win Rate: {s_stat['win_rate']:.1f}% | PnL: ${s_stat['net_profit']:.2f} | Trades: {s_stat['total_trades']}")
                print("="*55 + "\n")
                
            # Re-order the active_strats list based on the global league table ranking
            active_strats = [s for s in sorted_strats_all if s in active_strats]
            
            # If Volatility Guard is active, remove reversion strategies from the active selection
            if not is_safe:
                reversion_strats = ["RSI_REVERSION", "VWAP_REVERSION"]
                active_strats = [s for s in active_strats if s not in reversion_strats]
            
            ai_mode = "OFFENSIVE" if regime_icon in ("[TREND]", "[TREND-VEL]", "[ASIA-BREAKOUT]") else "DEFENSIVE"

            curr_strats_str = ",".join(active_strats)
            if curr_strats_str != prev_strats_str:
                now_str = server_time.strftime("%H:%M:%S UTC")
                print(f"\n[CHANGE] {now_str} Switched state to {regime_icon}. Active Engines: {' | '.join(active_strats)}")
                notify_regime_change(regime_icon, ' | '.join(active_strats))
                prev_strats_str = curr_strats_str

            # Precompute higher timeframe trend once per tick using confirmed closed candles & EMA 20/50 ribbons
            h1_dir = "FLAT"
            if df_h1 is not None and len(df_h1) >= 50:
                h1_idx = -2 if len(df_h1) >= 2 else -1
                h1_ema_20 = df_h1['close'].ewm(span=20, adjust=False).mean().iloc[h1_idx]
                h1_ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[h1_idx]
                h1_close = df_h1.iloc[h1_idx]['close']
                if h1_close > h1_ema_20 and h1_ema_20 > h1_ema_50:
                    h1_dir = "UP"
                elif h1_close < h1_ema_20 and h1_ema_20 < h1_ema_50:
                    h1_dir = "DOWN"

            h4_dir = "FLAT"
            if df_h4 is not None and len(df_h4) >= 50:
                h4_idx = -2 if len(df_h4) >= 2 else -1
                h4_ema_20 = df_h4['close'].ewm(span=20, adjust=False).mean().iloc[h4_idx]
                h4_ema_50 = df_h4['close'].ewm(span=50, adjust=False).mean().iloc[h4_idx]
                h4_close = df_h4.iloc[h4_idx]['close']
                if h4_close > h4_ema_20 and h4_ema_20 > h4_ema_50:
                    h4_dir = "UP"
                elif h4_close < h4_ema_20 and h4_ema_20 < h4_ema_50:
                    h4_dir = "DOWN"

            curr_trend = "FLAT"
            if h1_dir == "UP" and h4_dir != "DOWN":
                curr_trend = "UP"
            elif h1_dir == "DOWN" and h4_dir != "UP":
                curr_trend = "DOWN"

            status_tags = []
            if is_daily_halted:
                status_tags.append("[DD-HALT]")
            if in_news_blackout:
                status_tags.append(news_hold_str)
            elif is_rollover_active:
                status_tags.append("[WARMUP-ROLL]")
            elif is_off_hours:
                status_tags.append("[WARMUP]")
            display_regime = f"{regime_icon} {' '.join(status_tags)}" if status_tags else regime_icon

            line1 = f"[{short_session}{dst_tag}] {display_regime} T:{curr_trend} | ADX:{curr_adx:.1f} | RSI:{curr_rsi:.1f} | ${cached_equity:,.2f} | {curr_price:.2f}"
            
            # Update Telegram shared state for /status command
            update_bot_state(
                regime=display_regime, adx=curr_adx, rsi=curr_rsi,
                price=curr_price, trend=curr_trend, engines=' | '.join(active_strats),
                drawdown_halted=is_daily_halted, observe_only=is_observe_only
            )

            ui_messages = []

            # Active Position Telemetry Tracker for HUD
            try:
                open_pos_list = mt5.positions_get(symbol=SYMBOL)
                matching_pos = [p for p in open_pos_list if p.magic == MAGIC_NUMBER] if open_pos_list else []
                if matching_pos:
                    p = matching_pos[0]
                    dir_str = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                    tp_dist = abs(curr_price - p.tp) if p.tp > 0 else 0.0
                    pnl_str = f"+${p.profit:.2f}" if p.profit >= 0 else f"-${abs(p.profit):.2f}"
                    if p.tp > 0 and tp_dist <= 2.50:
                        ui_messages.append(f"[POS: {dir_str} | {pnl_str} | TP-DIST: ${tp_dist:.2f} (⚡ TP-JUMP READY)]")
                    else:
                        ui_messages.append(f"[POS: {dir_str} | {pnl_str} | TP-DIST: ${tp_dist:.2f}]")
            except Exception:
                pass
            if active_strats and not is_paused():
                # ORCHESTRATOR: MAXIMUM CONFLUENCE EVALUATOR
                for strat_key in active_strats:
                    strategy = strategies.get(strat_key)
                    if strategy:
                        try:
                            res = strategy.evaluate(
                                df_m5=df_m5, df_h1=df_h1, df_h4=df_h4, df_adx=df_adx, 
                                current_price=curr_price, current_risk=current_risk, atr=current_atr,
                                ai_mode=ai_mode, server_hour=utc_hour,
                                observe_only=is_observe_only
                            )
                        except Exception as strat_err:
                            print(f"\n[STRATEGY ERROR] {strat_key}.evaluate() failed: {strat_err}")
                            continue

                        ui_msg = res.get('ui')
                        if ui_msg and not ui_msg.startswith("[SCAN]") and not ui_msg.startswith("[OB_SCAN]"):
                            ui_messages.append(ui_msg)
                            
                        payload = res.get('payload')
                        
                        if payload and not is_observe_only and not is_daily_halted:
                            sig = payload['signal']
                            is_pending_limit = payload.get('limit_price') is not None
                            is_structural_zone_entry = strat_key in ("SMC_OB", "LIQUIDITY_SWEEP") or is_pending_limit
                            
                            # 0. Universal M5 EMA Scalp Shield (Protects market momentum/reversion entries from trading into momentum cascades)
                            # Exempt pending limit orders, Order Block pullbacks, and Wyckoff Liquidity Sweeps since those intentionally enter at discounted structural extremes.
                            if not is_structural_zone_entry and sig == 'BUY' and df_m5 is not None and len(df_m5) >= 50:
                                is_amda_absorbed = False
                                if strat_key == "RSI_REVERSION" and len(df_m5) >= 3:
                                    last_bar = df_m5.iloc[-2]
                                    prev_bar = df_m5.iloc[-3]
                                    bar_range = max(0.01, last_bar['high'] - last_bar['low'])
                                    lower_wick = max(0.0, min(last_bar['open'], last_bar['close']) - last_bar['low'])
                                    lower_wick_pct = lower_wick / bar_range
                                    is_green_absorption = last_bar['close'] > last_bar['open']
                                    prev_range = max(0.01, prev_bar['high'] - prev_bar['low'])
                                    prev_lower_wick = max(0.0, min(prev_bar['open'], prev_bar['close']) - prev_bar['low'])
                                    is_2bar_absorption = (prev_lower_wick / prev_range >= 0.25) and is_green_absorption and (last_bar['close'] > (prev_bar['high'] + prev_bar['low']) * 0.5)
                                    if (is_green_absorption and lower_wick_pct >= 0.35) or is_2bar_absorption:
                                        is_amda_absorbed = True

                                if not is_amda_absorbed:
                                    ema_21_m5 = df_m5['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                                    ema_50_m5 = df_m5['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                                    m5_close = df_m5.iloc[-1]['close']
                                    m5_open = df_m5.iloc[-1]['open']
                                    
                                    is_suppressed = (m5_close < ema_21_m5) and (m5_close < ema_50_m5)
                                    has_reclaimed_ema21 = (m5_close > ema_21_m5) and (m5_close > m5_open)
                                    
                                    if is_suppressed and not has_reclaimed_ema21:
                                        print(f"\n[EMA_SHIELD] Blocked {strat_key} BUY: M5 price ${m5_close:.2f} below EMA21 (${ema_21_m5:.2f}) & EMA50 (${ema_50_m5:.2f}) without green EMA21 reclaim")
                                        continue

                            elif not is_structural_zone_entry and sig == 'SELL' and df_m5 is not None and len(df_m5) >= 50:
                                # Symmetrical Shield for SELL: Block shorting into runaway green rallies
                                is_amda_exhausted = False
                                if strat_key == "RSI_REVERSION" and len(df_m5) >= 3:
                                    last_bar = df_m5.iloc[-2]
                                    prev_bar = df_m5.iloc[-3]
                                    bar_range = max(0.01, last_bar['high'] - last_bar['low'])
                                    upper_wick = max(0.0, last_bar['high'] - max(last_bar['open'], last_bar['close']))
                                    upper_wick_pct = upper_wick / bar_range
                                    is_red_exhaustion = last_bar['close'] < last_bar['open']
                                    prev_range = max(0.01, prev_bar['high'] - prev_bar['low'])
                                    prev_upper_wick = max(0.0, prev_bar['high'] - max(prev_bar['open'], prev_bar['close']))
                                    is_2bar_exhaustion = (prev_upper_wick / prev_range >= 0.25) and is_red_exhaustion and (last_bar['close'] < (prev_bar['high'] + prev_bar['low']) * 0.5)
                                    if (is_red_exhaustion and upper_wick_pct >= 0.35) or is_2bar_exhaustion:
                                        is_amda_exhausted = True

                                if not is_amda_exhausted:
                                    ema_21_m5 = df_m5['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                                    ema_50_m5 = df_m5['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                                    m5_close = df_m5.iloc[-1]['close']
                                    m5_open = df_m5.iloc[-1]['open']
                                    
                                    is_elevated = (m5_close > ema_21_m5) and (m5_close > ema_50_m5)
                                    has_rejected_ema21 = (m5_close < ema_21_m5) and (m5_close < m5_open)
                                    
                                    if is_elevated and not has_rejected_ema21:
                                        print(f"\n[EMA_SHIELD] Blocked {strat_key} SELL: M5 price ${m5_close:.2f} above EMA21 (${ema_21_m5:.2f}) & EMA50 (${ema_50_m5:.2f}) without red EMA21 rejection")
                                        continue
                            
                            # 1. Strict Higher-Timeframe Trend Enforcement
                            # Reversion strategies manage their own dynamic bounds; for all structural (SMC) & breakout strategies, enforce H1 Session Trend alignment:
                            is_reversion = strat_key in ["RSI_REVERSION", "VWAP_REVERSION", "LIQUIDITY_SWEEP"]
                            if not is_reversion:
                                if sig == 'SELL' and h1_dir == 'UP':
                                    print(f"\n[SESSION BLOCK] Rejected {strat_key} SELL signal. Counter to H1 Session Trend (H1: UP)")
                                    continue
                                elif sig == 'BUY' and h1_dir == 'DOWN':
                                    print(f"\n[SESSION BLOCK] Rejected {strat_key} BUY signal. Counter to H1 Session Trend (H1: DOWN)")
                                    continue
                            
                            # 2. Smart Adaptive Execution Guard (Non-Invasive Sell & Asymmetry Defense)
                            if sig == 'SELL' and getattr(sys.modules.get('core.config'), 'USE_SMART_SELL_GUARD', True):
                                min_rsi_floor = getattr(sys.modules.get('core.config'), 'GUARD_MIN_RSI_SELL', 38.0)
                                # Oversold Floor Trap: Block selling when M5 RSI < 38 (exempt BB_BREAKOUT which manages its own 28-48 momentum band)
                                if strat_key != "BB_BREAKOUT" and curr_rsi < min_rsi_floor:
                                    print(f"\n[GUARD DENIED] Rejected {strat_key} SELL signal: RSI ({curr_rsi:.1f}) < {min_rsi_floor:.1f} (Oversold Floor Trap)")
                                    continue
                                    
                                # Macro Bull Trend Gate: Block shorting into raging Bull runs
                                max_adx = getattr(sys.modules.get('core.config'), 'GUARD_MAX_ADX_COUNTER', 25.0)
                                if (h1_dir == 'UP' and curr_adx >= 25.0) or (h1_dir == 'UP' and h4_dir == 'UP' and curr_adx > max_adx):
                                    print(f"\n[GUARD DENIED] Rejected {strat_key} SELL signal: Counter-trend against raging Bull run (H1:{h1_dir}, H4:{h4_dir}, ADX {curr_adx:.1f})")
                                    continue

                            # Macro Bear Trend Gate: Block buying into raging Bear cascades (ADX >= 25.0 or dual H1+H4 Down)
                            if sig == 'BUY':
                                if getattr(sys.modules.get('core.config'), 'USE_SMART_SELL_GUARD', True):
                                    max_rsi_ceiling = getattr(sys.modules.get('core.config'), 'GUARD_MAX_RSI_BUY', 62.0)
                                    # Exempt BB_BREAKOUT which enforces its own 52-73 momentum expansion band
                                    if strat_key != "BB_BREAKOUT" and curr_rsi > max_rsi_ceiling:
                                        print(f"\n[GUARD DENIED] Rejected {strat_key} BUY signal: RSI ({curr_rsi:.1f}) > {max_rsi_ceiling:.1f} (Overbought Ceiling Trap)")
                                        continue

                                max_adx = getattr(sys.modules.get('core.config'), 'GUARD_MAX_ADX_COUNTER', 25.0)
                                if (h1_dir == 'DOWN' and curr_adx >= 25.0) or (h1_dir == 'DOWN' and h4_dir == 'DOWN' and curr_adx > max_adx):
                                    print(f"\n[GUARD DENIED] Rejected {strat_key} BUY signal: Counter-trend against raging Bear cascade (H1:{h1_dir}, H4:{h4_dir}, ADX {curr_adx:.1f})")
                                    continue

                            # Rollover Entry Blackout Guard: Strictly block entries between 23:15 and 00:30 UTC
                            if (utc_hour == 23 and server_time.minute >= 15) or (utc_hour == 0 and server_time.minute < 30):
                                print(f"\n[TIME BLOCK] Rejected {strat_key} {sig} signal. Rollover Entry Blackout active (23:15-00:30 UTC).")
                                continue

                            # Pre-Rollover Cutoff Guard (No new trades after 21:00 UTC)
                            no_new_after = getattr(sys.modules.get('core.config'), 'NO_NEW_TRADES_AFTER_HOUR', 21)
                            if utc_hour >= no_new_after:
                                print(f"\n[TIME BLOCK] Rejected {strat_key} {sig} signal. New entries blocked after {no_new_after}:00 UTC (Pre-Rollover Protection).")
                                continue

                            # 4a. Global Directional Circuit Breaker (Prevents cross-strategy loss cascades)
                            is_global_cd, rem_global_min = check_global_directional_cooldown(sig)
                            if is_global_cd:
                                print(f"\n[GLOBAL CIRCUIT BREAKER] Blocked {strat_key} {sig} signal: 2 consecutive losses in {sig} direction across bot. Cooling down for {rem_global_min}m.")
                                continue

                            # 4b. Strategy-Specific Loss Cooldown Guard (15m single loss / 45m consecutive loss)
                            is_cooldown, rem_min = check_strategy_cooldown(strat_key, sig)
                            if is_cooldown:
                                print(f"\n[COOLDOWN BLOCK] Rejected {strat_key} {sig} signal. Strategy on {rem_min}m Cooldown after recent loss in {sig} direction.")
                                continue

                            # 4c. Option A Post-News Safety Guards (Spread Gate & One-Way Directional Lock)
                            post_news = get_post_news_guard_status(df_m5)
                            if post_news.get('in_post_window'):
                                elapsed = post_news.get('minutes_elapsed', 0.0)
                                news_title = post_news.get('title', 'USD High Impact')

                                # Guard 1: Post-News Spread Gate ($0.35 max)
                                max_news_spread = getattr(sys.modules.get('core.config'), 'NEWS_MAX_SPREAD_DOLLARS', 0.35)
                                live_tick = mt5.symbol_info_tick(SYMBOL)
                                if live_tick:
                                    live_spread_dlrs = abs(live_tick.ask - live_tick.bid)
                                    if live_spread_dlrs > max_news_spread:
                                        print(f"\n[NEWS SPREAD GATE] Blocked {strat_key} {sig}: Post-news spread (${live_spread_dlrs:.2f}) > ${max_news_spread:.2f} max ({elapsed}m post-{news_title[:20]}). Waiting for liquidity.")
                                        continue

                                # Guard 2: One-Way Directional Lock (Anti-Knife Guard)
                                dir_lock_mins = getattr(sys.modules.get('core.config'), 'NEWS_DIRECTION_LOCK_MINUTES', 30)
                                allowed_bias = post_news.get('directional_bias')
                                if allowed_bias and elapsed <= dir_lock_mins:
                                    if sig != allowed_bias:
                                        print(f"\n[NEWS DIRECTION LOCK] Blocked {strat_key} {sig}: High-impact news displacement was {allowed_bias}. Counter-trend trades blocked for {dir_lock_mins}m post-{news_title[:20]} ({elapsed}m elapsed).")
                                        continue
                            
                            r_pct = payload.get('risk_override', current_risk)
                            trade_success = execute_trade(
                                signal=payload['signal'], 
                                sl_price=payload['sl'], 
                                risk_pct=r_pct, 
                                magic_num=MAGIC_NUMBER, 
                                comment_text=payload['comment'], 
                                ai_conf=payload['confidence'],
                                limit_price=payload.get('limit_price'),
                                stop_price=payload.get('stop_price'),
                                tp_price=payload.get('tp_price'),
                                feature_vector=payload.get('features'),
                                strategy_name=strat_key
                            )
                            if trade_success:
                                # Collision Guard: Prevent other strategies from firing on this same loop iteration
                                break
                            
            line2 = " ".join(ui_messages) if ui_messages else ""
            
            # Combine into an ultra-compact single line
            full_hud = f"\r{line1} || {line2}" if line2 else f"\r{line1}"
            
            # Dynamically fetch the actual width of the user's terminal to prevent wrapping glitches
            term_width = shutil.get_terminal_size((120, 20)).columns
            
            # \033[K clears the entire line so remnants of previous longer strings vanish
            sys.stdout.write(f"\033[K{full_hud[:term_width-1]}")
            sys.stdout.flush()
                
            consecutive_errors = 0
            time.sleep(2.0 if (is_observe_only and not has_open_pos) else 0.5)

        except KeyboardInterrupt:
            mt5.shutdown(); break
        except Exception as e:
            consecutive_errors += 1
            err_str = str(e)
            now_t = time.time()
            if err_str != last_logged_error or (now_t - last_error_time > 60):
                print(f"\n❌ Loop Error: {err_str}")
                last_logged_error = err_str
                last_error_time = now_t
            
            backoff_sleep = min(2 * (2 ** min(consecutive_errors - 1, 5)), 30)
            time.sleep(backoff_sleep)

if __name__ == "__main__":
    main()