import time
import sys
import os
import joblib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from core.config import *
from core.utils import get_session_name, get_broker_hour_from_utc
from core.indicators import calculate_atr_simple, calculate_adx_simple
from engine.mt5_interface import connect_mt5, get_dynamic_kelly_risk
from strategies.smc_fvg import SMCStrategy
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.smc_orderblock import SMCOrderBlockStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.bb_breakout import BBBreakoutStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.smc_choch import SMCChochStrategy

def load_ai_models():
    trend, reversal, veto_model, veto_scaler = None, None, None, None
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Load Trend Model
    try:
        trend_path = os.path.join(root_dir, TREND_MODEL_FILE)
        if os.path.exists(trend_path):
            trend = joblib.load(trend_path)
            if hasattr(trend, 'n_jobs'): trend.n_jobs = 1
    except Exception as e:
        print(f"[BACKTEST] [AI] Failed to load Trend Model: {e}")
        
    # 2. Load Reversal Model
    try:
        reversal_path = os.path.join(root_dir, REVERSAL_MODEL_FILE)
        if os.path.exists(reversal_path):
            reversal = joblib.load(reversal_path)
            if hasattr(reversal, 'n_jobs'): reversal.n_jobs = 1
    except Exception as e:
        print(f"[BACKTEST] [AI] Failed to load Reversal Model: {e}")
        
    # 3. Load Veto Model
    try:
        veto_path = os.path.join(root_dir, "models", "evolution_veto.joblib")
        if os.path.exists(veto_path):
            veto_brain = joblib.load(veto_path)
            veto_model = veto_brain['model']
            veto_scaler = veto_brain['scaler']
            print("[BACKTEST] [AI] Veto Model loaded successfully.")
    except Exception as e:
        print(f"[BACKTEST] [AI] Failed to load Veto Model: {e}")
        
    return trend, reversal, veto_model, veto_scaler

def get_historical_data(symbol, timeframe, num_candles):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def run_backtest(initial_balance=100.0, months=6, use_veto=True):
    print(f"[BACKTEST] Initializing {months} Month Vector Simulation for {SYMBOL}")
    print(f"Starting Balance: ${initial_balance:.2f} | Veto Layer: {'ENABLED' if use_veto else 'DISABLED'}\n")
    
    connect_mt5()
        
    trend_model, reversal_model, veto_model, veto_scaler = load_ai_models()
    
    strategies = {
        "SMC_FVG": SMCStrategy(trend_model),
        "SMC_OB": SMCOrderBlockStrategy(trend_model),
        "LIQUIDITY_SWEEP": LiquiditySweepStrategy(reversal_model),
        "RSI_REVERSION": RSIReversionStrategy(reversal_model, trend_model),
        "VWAP_REVERSION": VWAPReversionStrategy(),
        "BB_BREAKOUT": BBBreakoutStrategy(trend_model),
        "LONDON_BREAKOUT": LondonBreakoutStrategy(trend_model),
        "SMC_CHOCH": SMCChochStrategy(reversal_model)
    }
    
    # Get real contract size from broker for accurate PnL math
    symbol_info = mt5.symbol_info(SYMBOL)
    default_size = 1.0 if SYMBOL.endswith('m') else 100.0
    contract_size = symbol_info.trade_contract_size if symbol_info else default_size
    print(f"[BACKTEST] Contract Size: {contract_size} (Source: {'Broker' if symbol_info else 'Fallback Default'})")
    
    # Calculate approx candles for 6 months (51,840 M5 candles)
    candles_m5 = int(months * 30 * 24 * 12)
    print(f"Executing Full {months}-Month Deep Vector Analysis. Candles: {candles_m5}")
    candles_h1 = int(candles_m5 / 12) + 200
    candles_h4 = int(candles_m5 / 48) + 200
    candles_m15 = int(candles_m5 / 3) + 200
    
    print("[BACKTEST] Downloading Historical Vectors...")
    df_m5 = get_historical_data(SYMBOL, mt5.TIMEFRAME_M5, candles_m5)
    df_h1 = get_historical_data(SYMBOL, mt5.TIMEFRAME_H1, candles_h1)
    df_h4 = get_historical_data(SYMBOL, mt5.TIMEFRAME_H4, candles_h4)
    df_m15 = get_historical_data(SYMBOL, mt5.TIMEFRAME_M15, candles_m15)
    
    if df_m5 is None: return print("Failed to load historical data")
    
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown_pct = 0.0
    open_trades = []
    trade_history = []
    
    # Daily loss protection tracker
    last_date = None
    daily_starting_balance = initial_balance
    daily_loss_halt = False
    
    # Active strategy dynamic league priorities
    sorted_strats_all = ["SMC_FVG", "SMC_OB", "LIQUIDITY_SWEEP", "RSI_REVERSION", "VWAP_REVERSION", "BB_BREAKOUT", "LONDON_BREAKOUT", "SMC_CHOCH"]
    
    h1_idx, h4_idx, m15_idx = 200, 200, 200
    
    print(f"[BACKTEST] Commencing Vector Execution loop over {len(df_m5)} logic blocks...\n")
    start_time = time.time()
    
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    
    def _restore_stdout():
        sys.stdout = old_stdout
        devnull.close()
    import atexit
    atexit.register(_restore_stdout)
    
    for i in range(500, len(df_m5)):
        if i % 500 == 0:
            sys.stdout = old_stdout
            print(f"\r[BACKTEST] Processing vector {i} / {len(df_m5)} ({(i/len(df_m5))*100:.1f}%)", end="")
        sys.stdout = devnull # Aggressive mute
        curr_row = df_m5.iloc[i]
        curr_time = curr_row['time']
        curr_price = curr_row['close']
        
        # Check for day boundary to reset daily loss halt (matches MAX_DAILY_LOSS_PERCENT rule)
        curr_date = curr_time.date()
        if last_date is None or curr_date != last_date:
            daily_starting_balance = balance
            daily_loss_halt = False
            last_date = curr_date
            
        # Check if daily drawdown meets or exceeds global tiered daily loss limit
        if daily_starting_balance > 0:
            daily_loss_pct = (daily_starting_balance - balance) / daily_starting_balance * 100.0
            if daily_loss_pct >= get_daily_loss_limit_pct(daily_starting_balance):
                daily_loss_halt = True
                
        # curr_time from backtest history is already parsed as UTC, so we map it to our static NY-aligned clock
        server_hour = get_broker_hour_from_utc(curr_time)
        
        # Sync multi-timeframe indices efficiently (O(1) approach)
        while h1_idx < len(df_h1) and df_h1.iloc[h1_idx]['time'] <= curr_time: h1_idx += 1
        while h4_idx < len(df_h4) and df_h4.iloc[h4_idx]['time'] <= curr_time: h4_idx += 1
        while m15_idx < len(df_m15) and df_m15.iloc[m15_idx]['time'] <= curr_time: m15_idx += 1
            
        # Manage Open Trades natively
        unrealized_pnl_total = 0.0
        for t in list(open_trades):
            close_price = None
            tp_hit = False
            sl_hit = False
            
            if t['signal'] == 'BUY':
                tp_hit = curr_row['high'] >= t['tp']
                sl_hit = curr_row['low'] <= t['sl']
            else:
                tp_hit = curr_row['low'] <= t['tp']
                sl_hit = curr_row['high'] >= t['sl']
                
            if tp_hit and sl_hit:
                close_price = t['sl']  # Conservative resolution: SL hit if both trigger on same bar
            elif tp_hit:
                close_price = t['tp']
            elif sl_hit:
                close_price = t['sl']
                
            lots = t['lots']
            point_val = contract_size * lots
            spread = curr_row['spread'] * symbol_info.point if (symbol_info and 'spread' in curr_row) else 0.20
            trade_cost = (spread * point_val) + (7.0 * lots)
            
            if close_price:
                if t['signal'] == 'BUY':
                    gross_pnl = (close_price - t['entry']) * point_val
                else:
                    gross_pnl = (t['entry'] - close_price) * point_val
                pnl = gross_pnl - trade_cost
                    
                is_vetoed = t.get('vetoed', False)
                if not is_vetoed:
                    balance += pnl
                
                planned_rr = abs(t['tp'] - t['entry']) / abs(t['entry'] - t['sl']) if abs(t['entry'] - t['sl']) > 0 else 0
                trade_history.append({
                    'Strategy': t['strategy'],
                    'Entry Time': t['entry_time'],
                    'Exit Time': curr_time,
                    'Type': t['signal'],
                    'Lots': t['lots'],
                    'Entry Price': t['entry'],
                    'Exit Price': close_price,
                    'RR': round(planned_rr, 2),
                    'PnL': round(pnl, 2),
                    'Balance': round(balance, 2),
                    'Vetoed': 'Yes' if is_vetoed else 'No',
                    'Veto Prob': f"{t.get('veto_prob', 0.0):.2%}" if is_vetoed else "0.0%"
                })
                open_trades.remove(t)
            else:
                # Mark-to-market unrealized PnL for open trades
                if t['signal'] == 'BUY':
                    u_pnl = (curr_price - t['entry']) * point_val - trade_cost
                else:
                    u_pnl = (t['entry'] - curr_price) * point_val - trade_cost
                if not t.get('vetoed', False):
                    unrealized_pnl_total += u_pnl

        # Mark-to-market Bar Equity & Drawdown tracking
        current_bar_equity = balance + unrealized_pnl_total
        if current_bar_equity > peak_balance:
            peak_balance = current_bar_equity
        if peak_balance > 0:
            dd = (peak_balance - current_bar_equity) / peak_balance * 100.0
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd
                
        # Margin Call Check
        if balance <= 0:
            print(f"\n[MARGIN CALL] Account balance depleted at vector {i}. Halting simulation.")
            break

        # If open real trades exist, we skip entering new ones just like live logic limit (max 1 usually)
        active_real_trades = [t for t in open_trades if not t.get('vetoed', False)]
        if len(active_real_trades) > 0:
            continue
            
        # Skip entering new setups if daily loss limit was reached
        if daily_loss_halt:
            continue
            
        # Dynamic Performance Ranking League (Calculated chronologically based on simulated backtest trades)
        if i % 12 == 0 or not sorted_strats_all:
            cutoff_time = curr_time - timedelta(days=30)
            recent_trades = [t for t in trade_history if t.get('Exit Time') and t['Exit Time'] >= cutoff_time]
            
            stats = {}
            for strat in ["SMC_FVG", "SMC_OB", "LIQUIDITY_SWEEP", "RSI_REVERSION", "VWAP_REVERSION", "BB_BREAKOUT", "LONDON_BREAKOUT", "SMC_CHOCH"]:
                strat_trades = [t for t in recent_trades if t.get('Strategy') == strat]
                if not strat_trades:
                    stats[strat] = {"win_rate": 55.0, "net_profit": 0.0}
                else:
                    wins = sum(1 for t in strat_trades if t['PnL'] > 0.50)
                    losses = sum(1 for t in strat_trades if t['PnL'] < -0.50)
                    total = wins + losses
                    wr = (wins / total * 100.0) if total > 0 else 55.0
                    net_p = sum(t['PnL'] for t in strat_trades)
                    stats[strat] = {"win_rate": wr, "net_profit": net_p}
            
            def get_sort_key(strat_name):
                s_stat = stats.get(strat_name, {"win_rate": 55.0, "net_profit": 0.0})
                return (s_stat["win_rate"], s_stat["net_profit"])
                
            sorted_strats_all = sorted(
                ["SMC_FVG", "SMC_OB", "LIQUIDITY_SWEEP", "RSI_REVERSION", "VWAP_REVERSION", "BB_BREAKOUT", "LONDON_BREAKOUT", "SMC_CHOCH"],
                key=get_sort_key,
                reverse=True
            )
            
        # Strategy Execution Vectoring
        df_m5_slice = df_m5.iloc[i-200 : i+1]
        df_h1_slice = df_h1.iloc[max(0, h1_idx-101) : h1_idx - 1]
        df_h4_slice = df_h4.iloc[max(0, h4_idx-101) : h4_idx - 1]
        df_m15_slice = df_m15.iloc[max(0, m15_idx-101) : m15_idx - 1]
        
        current_atr = calculate_atr_simple(df_m5_slice, ATR_BASELINE_PERIOD)
        curr_adx = calculate_adx_simple(df_m15_slice, ADX_PERIOD).iloc[-1]
        if current_atr == 0: continue
            
        utc_hour = pd.to_datetime(curr_time).hour
        current_session = get_session_name(pd.to_datetime(curr_time))
        current_risk = SESSION_RISK.get(current_session, 0.0)
        
        # Skip Rollover hours (matches live engine sleep behavior)
        if utc_hour < ASIAN_OPEN_HOUR or utc_hour >= TRADING_END_HOUR:
            continue
        if current_risk <= 0:
            continue
        
        # ADX-based strategy routing (mirrors live orchestrator)
        ai_mode = "DEFENSIVE"
        active_strats = ["SMC_FVG", "SMC_OB"]
        
        if current_session == "ASIAN":
            active_strats = ["SMC_FVG", "SMC_OB", "RSI_REVERSION", "VWAP_REVERSION", "LIQUIDITY_SWEEP", "SMC_CHOCH"]
        else:
            if curr_adx > ADX_TREND_START:
                active_strats = ["SMC_FVG", "SMC_OB", "BB_BREAKOUT", "LIQUIDITY_SWEEP", "SMC_CHOCH"]
                ai_mode = "OFFENSIVE"
            elif curr_adx < ADX_RANGE_START:
                active_strats = ["RSI_REVERSION", "VWAP_REVERSION"]
            else:
                active_strats = ["RSI_REVERSION", "VWAP_REVERSION", "BB_BREAKOUT"]

        if KILLZONE_EXEC_START <= server_hour < KILLZONE_EXEC_END:
            active_strats.append("LONDON_BREAKOUT")

        # Re-order the active_strats list based on the global league table ranking
        active_strats = [s for s in sorted_strats_all if s in active_strats]

        for strat_key in active_strats:
            strategy = strategies.get(strat_key)
            if strategy:
                res = strategy.evaluate(
                    df_m5=df_m5_slice, df_h1=df_h1_slice, df_h4=df_h4_slice, df_adx=df_m15_slice, 
                    current_price=curr_price, current_risk=current_risk, atr=current_atr,
                    server_hour=server_hour, ai_mode=ai_mode
                )
                payload = res.get('payload')
                
                if payload:
                    is_vetoed = False
                    veto_probability = 0.0
                    # Evaluate Veto Layer
                    if use_veto and veto_model is not None and veto_scaler is not None and payload.get('features') is not None:
                        features = payload['features']
                        expected_dim = veto_scaler.n_features_in_ if hasattr(veto_scaler, 'n_features_in_') else 21
                        if len(features) == expected_dim:
                            X_scaled = veto_scaler.transform(np.array([features]))
                            loss_prob = veto_model.predict_proba(X_scaled)[0][0]
                            if loss_prob >= EVOLUTION_STRICTNESS:
                                is_vetoed = True
                                veto_probability = loss_prob

                    # Execute theoretical trade
                    signal = payload['signal']
                    sl = payload['sl']
                    # Calculate theoretical TP with dynamic structural overrides
                    entry = curr_price
                    tp = payload.get('tp_price')
                    if tp:
                        structural_rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                        if structural_rr < MIN_DYNAMIC_RR:
                            if signal == 'BUY': tp = entry + ((entry - sl) * 2.0)
                            else: tp = entry - ((sl - entry) * 2.0)
                    else:
                        if signal == 'BUY': tp = entry + ((entry - sl) * RISK_REWARD_RATIO)
                        else: tp = entry - ((sl - entry) * RISK_REWARD_RATIO)
                    
                    # Calculate dynamic lots
                    sl_dist = abs(entry - sl)
                    if sl_dist > 0 and current_risk > 0:
                        risk_amount = balance * (current_risk / 100)
                        lots = max(0.01, round(risk_amount / (sl_dist * contract_size), 2))
                    else:
                        lots = 0.01

                    open_trades.append({
                        'strategy': strat_key,
                        'signal': signal,
                        'entry_time': curr_time,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'lots': lots,
                        'vetoed': is_vetoed,
                        'veto_prob': veto_probability
                    })
                    break # Execute only one setup
                    
        # Print progress every 10,000 candles
        if i % 10000 == 0:
            sys.stdout = old_stdout
            print(f"[SIMULATION] Iterated {i} / {len(df_m5)}... Current Balance: ${balance:.2f}")
            sys.stdout = devnull

    sys.stdout = old_stdout
    devnull.close()

    # Output Statistics
    real_trades = [t for t in trade_history if t.get('Vetoed') == 'No']
    vetoed_trades = [t for t in trade_history if t.get('Vetoed') == 'Yes']
    
    total_real_trades = len(real_trades)
    real_wins = sum(1 for t in real_trades if t['PnL'] > 0)
    real_win_rate = (real_wins / total_real_trades * 100) if total_real_trades > 0 else 0.0
    
    total_vetoed = len(vetoed_trades)
    vetoed_saved_losses = sum(1 for t in vetoed_trades if t['PnL'] < 0)
    vetoed_blocked_wins = sum(1 for t in vetoed_trades if t['PnL'] > 0)
    veto_accuracy = (vetoed_saved_losses / total_vetoed * 100) if total_vetoed > 0 else 0.0
    
    print("\n=======================================================")
    print("[BACKTEST COMPLETE]")
    print("=======================================================")
    print(f"Total Candles Processed   : {len(df_m5)}")
    print(f"Total Real Trades Executed: {total_real_trades}")
    print(f"Real Win Rate             : {real_win_rate:.2f}%")
    print(f"Maximum Drawdown          : {max_drawdown_pct:.2f}%")
    print(f"Initial Balance           : ${initial_balance:.2f}")
    print(f"Final Balance             : ${balance:.2f}")
    print(f"Total Real Profit         : ${(balance - initial_balance):.2f}")
    print(f"Execution Time            : {(time.time() - start_time):.2f} seconds")
    print("-------------------------------------------------------")
    print(f"Total Vetoed Trades       : {total_vetoed}")
    print(f"  - Correctly Blocked (L) : {vetoed_saved_losses}")
    print(f"  - Incorrectly Blocked (W): {vetoed_blocked_wins}")
    print(f"  - Veto Accuracy Rating  : {veto_accuracy:.2f}%")
    print("=======================================================\n")
    
    # Exporting Quantitative CSV 
    if len(trade_history) > 0:
        df_trades = pd.DataFrame(trade_history)
        
        # Adding summary data at bottom 
        summary = pd.DataFrame([{
            'Entry Time': 'SUMMARY',
            'Type': f"Win Rate: {real_win_rate:.2f}%",
            'Entry Price': f"Drawdown: {max_drawdown_pct:.2f}%", 
            'Exit Price': f"Profit: ${(balance - initial_balance):.2f}",
            'PnL': f"Trades: {total_real_trades}",
            'Balance': balance
        }])
        
        # Use pd.concat instead of loc to avoid FutureWarnings
        df_trades = pd.concat([df_trades, pd.DataFrame([{}]), summary], ignore_index=True)
        
        export_path = "data/backtest_results.csv"
        os.makedirs("data", exist_ok=True)
        df_trades.to_csv(export_path, index=False)
        print(f"[EXPORT] Full Historical Ledger saved to -> {export_path}")

if __name__ == "__main__":
    run_backtest(initial_balance=500.0, months=6, use_veto=True)
