import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import MAGIC_NUMBER

def get_strategy_from_comment(comment):
    """
    Classifies the strategy name based on the trade deal comment.
    """
    comment = str(comment).upper()
    if "CHOCH" in comment:
        return "SMC_CHOCH"
    if "OB" in comment:
        return "SMC_OB"
    if "SMC" in comment or "FVG" in comment or "TRAP" in comment or "IFVG" in comment:
        # Keep FVG separate unless it matches OB or CHOCH
        return "SMC_FVG"
    if "RSI" in comment:
        return "RSI_REVERSION"
    if "VWAP" in comment:
        return "VWAP_REVERSION"
    if "BB" in comment or "SQUEEZE" in comment:
        return "BB_BREAKOUT"
    if "KZ_STOP" in comment or "LONDON" in comment:
        return "LONDON_BREAKOUT"
    if "LIQ" in comment or "SWEEP" in comment:
        return "LIQUIDITY_SWEEP"
    return "UNKNOWN"

def track_strategy_performance(lookback_days=30):
    """
    Queries MT5 deal history, parses positions, groups them by strategy, and calculates analytics.
    """
    # Default baseline stats for all strategies
    default_performance = {
        "SMC_CHOCH": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "SMC_OB": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "SMC_FVG": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "LIQUIDITY_SWEEP": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "RSI_REVERSION": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "VWAP_REVERSION": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "BB_BREAKOUT": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0},
        "LONDON_BREAKOUT": {"wins": 0, "losses": 0, "bes": 0, "win_rate": 55.0, "net_profit": 0.0, "total_trades": 0}
    }

    if not mt5.terminal_info():
        return default_performance

    # Query history deals
    time_from = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    time_to = datetime.now(timezone.utc) + timedelta(days=1)
    deals = mt5.history_deals_get(time_from, time_to)
    
    if deals is None or len(deals) == 0:
        return default_performance

    pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
    active_magic = getattr(sys.modules.get('core.config'), 'MAGIC_NUMBER', MAGIC_NUMBER)
    positions = {}
    for d in deals:
        pid = d.position_id
        if pid == 0: 
            continue  # Deposit/Balance transactions
        
        # We only track trades matching our global magic number (resolve from open deal if exit deal magic is 0)
        magic = pos_magics.get(pid, d.magic)
        if magic != active_magic:
            continue
            
        if pid not in positions:
            positions[pid] = {
                'magic': d.magic,
                'symbol': d.symbol,
                'profit': 0.0,
                'commission': 0.0,
                'swap': 0.0,
                'fee': 0.0,
                'comment': ''
            }
            
        positions[pid]['profit'] += d.profit
        positions[pid]['commission'] += d.commission
        positions[pid]['swap'] += d.swap
        positions[pid]['fee'] += getattr(d, 'fee', 0.0)
        
        if d.entry == mt5.DEAL_ENTRY_IN:
            positions[pid]['comment'] = d.comment if d.comment else ""
            positions[pid]['has_open'] = True
        elif d.entry == mt5.DEAL_ENTRY_OUT:
            positions[pid]['has_close'] = True

    # Aggregate performance by strategy type
    strategy_groups = {k: [] for k in default_performance.keys()}
    
    for pid, p in positions.items():
        # Only process closed positions that have complete in/out cycle
        if not p.get('has_close'):
            continue
            
        strat = get_strategy_from_comment(p['comment'])
        if strat in strategy_groups:
            net_pnl = p['profit'] + p['commission'] + p['swap'] + p['fee']
            strategy_groups[strat].append(net_pnl)

    # Compute metrics for each group
    performance = {}
    for strat, pnls in strategy_groups.items():
        if not pnls:
            # Fallback to default baseline
            performance[strat] = default_performance[strat]
            continue
            
        wins = sum(1 for pnl in pnls if pnl > 0.50)
        losses = sum(1 for pnl in pnls if pnl < -0.50)
        bes = sum(1 for pnl in pnls if -0.50 <= pnl <= 0.50)
        
        total_resolved = wins + losses
        win_rate = (wins / total_resolved * 100.0) if total_resolved > 0 else 55.0
        net_profit = sum(pnls)
        
        performance[strat] = {
            "wins": wins,
            "losses": losses,
            "bes": bes,
            "win_rate": round(win_rate, 1),
            "net_profit": round(net_profit, 2),
            "total_trades": len(pnls)
        }
        
    return performance

def get_dynamic_strategy_ranking(active_strats, lookback_days=30):
    """
    Ranks and sorts active_strats based on their live performance.
    """
    stats = track_strategy_performance(lookback_days)
    
    # Sort helper key: prioritizes Net Profit ($ PnL), using Win Rate as tie-breaker
    def get_sort_key(strat_name):
        strat_stats = stats.get(strat_name, {"win_rate": 55.0, "net_profit": 0.0})
        # Handle strategies that might not be in our stats map (fallback to baseline)
        wr = strat_stats.get("win_rate", 55.0)
        profit = strat_stats.get("net_profit", 0.0)
        return (profit, wr)

    # Sort the strategies descending (best performing first)
    sorted_strats = sorted(active_strats, key=get_sort_key, reverse=True)
    return sorted_strats, stats

def check_strategy_cooldown(strategy_name, direction, lookback_hours=3, max_losses=2, cooldown_minutes=None):
    """
    Checks if a strategy is currently in a loss cooldown:
    1. Single-loss cooloff: If the very last trade was a loss in this direction, pause for 15 mins.
    2. Consecutive-loss cooldown: If the last `max_losses` trades were all losses, lock for 45 mins.
    """
    if cooldown_minutes is None:
        cooldown_minutes = getattr(sys.modules.get('core.config'), 'CONSECUTIVE_LOSS_COOLDOWN_MINUTES', 45)
    single_loss_cooldown = getattr(sys.modules.get('core.config'), 'SINGLE_LOSS_DIRECTIONAL_COOLDOWN_MINUTES', 15)

    if not mt5.terminal_info():
        return False, 0
        
    now_utc = datetime.now(timezone.utc)
    cutoff_time = now_utc - timedelta(hours=lookback_hours)
    time_from = now_utc - timedelta(days=max(7, int(lookback_hours // 24) + 2))
    time_to = now_utc + timedelta(minutes=5)
    deals = mt5.history_deals_get(time_from, time_to)
    
    if not deals:
        return False, 0
        
    pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
    active_magic = getattr(sys.modules.get('core.config'), 'MAGIC_NUMBER', MAGIC_NUMBER)
    positions = {}
    for d in deals:
        pid = d.position_id
        if pid == 0:
            continue
        magic = pos_magics.get(pid, d.magic)
        if magic != active_magic:
            continue
        if pid not in positions:
            positions[pid] = {
                'profit': 0.0, 'commission': 0.0, 'swap': 0.0, 'fee': 0.0,
                'comment': '', 'type': '', 'close_time': None
            }
        positions[pid]['profit'] += d.profit
        positions[pid]['commission'] += d.commission
        positions[pid]['swap'] += d.swap
        positions[pid]['fee'] += getattr(d, 'fee', 0.0)
        
        if d.entry == mt5.DEAL_ENTRY_IN:
            positions[pid]['comment'] = d.comment if d.comment else ""
            positions[pid]['type'] = 'BUY' if d.type == mt5.DEAL_TYPE_BUY else 'SELL'
        elif d.entry == mt5.DEAL_ENTRY_OUT:
            positions[pid]['close_time'] = datetime.fromtimestamp(d.time, timezone.utc)
            
    strat_trades = []
    for pid, p in positions.items():
        if not p.get('close_time') or p['close_time'] < cutoff_time:
            continue
        strat = get_strategy_from_comment(p['comment'])
        if strat == strategy_name and p['type'] == direction:
            net_pnl = p['profit'] + p['commission'] + p['swap'] + p['fee']
            strat_trades.append({
                'pnl': net_pnl,
                'close_time': p['close_time']
            })
            
    if not strat_trades:
        return False, 0
        
    # Sort by close time descending (most recent first)
    strat_trades.sort(key=lambda x: x['close_time'], reverse=True)
    
    # 1. Check Consecutive Losses (>= max_losses)
    if len(strat_trades) >= max_losses:
        recent_subset = strat_trades[:max_losses]
        if all(t['pnl'] < -0.50 for t in recent_subset):
            last_loss_time = recent_subset[0]['close_time']
            elapsed_sec = (datetime.now(timezone.utc) - last_loss_time).total_seconds()
            cooldown_sec = cooldown_minutes * 60
            if elapsed_sec < cooldown_sec:
                remaining_min = int((cooldown_sec - elapsed_sec) / 60) + 1
                return True, remaining_min
                
    # 2. Check Single Most Recent Loss (Waterfall protection)
    most_recent = strat_trades[0]
    if most_recent['pnl'] < -0.50:
        elapsed_sec = (datetime.now(timezone.utc) - most_recent['close_time']).total_seconds()
        single_cooldown_sec = single_loss_cooldown * 60
        if elapsed_sec < single_cooldown_sec:
            remaining_min = int((single_cooldown_sec - elapsed_sec) / 60) + 1
            return True, remaining_min

    return False, 0


def check_global_directional_cooldown(direction, lookback_hours=2, max_losses=2, cooldown_minutes=45):
    """
    Global Directional Circuit Breaker:
    Checks if the last `max_losses` (default 2) closed trades across ANY strategies
    in the same `direction` were losses within `lookback_hours`.
    If so, locks out that entire direction across the whole bot for `cooldown_minutes`.
    Prevents cross-strategy loss cascades (e.g. CHOCH loss -> LIQ loss -> OB loss).
    """
    if not mt5.terminal_info():
        return False, 0

    now_utc = datetime.now(timezone.utc)
    cutoff_time = now_utc - timedelta(hours=lookback_hours)
    time_from = now_utc - timedelta(days=max(7, int(lookback_hours // 24) + 2))
    time_to = now_utc + timedelta(minutes=5)
    deals = mt5.history_deals_get(time_from, time_to)

    if not deals:
        return False, 0

    pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
    active_magic = getattr(sys.modules.get('core.config'), 'MAGIC_NUMBER', MAGIC_NUMBER)
    positions = {}
    for d in deals:
        pid = d.position_id
        if pid == 0:
            continue
        magic = pos_magics.get(pid, d.magic)
        if magic != active_magic:
            continue
        if pid not in positions:
            positions[pid] = {
                'profit': 0.0, 'commission': 0.0, 'swap': 0.0, 'fee': 0.0,
                'comment': '', 'type': '', 'close_time': None
            }
        positions[pid]['profit'] += d.profit
        positions[pid]['commission'] += d.commission
        positions[pid]['swap'] += d.swap
        positions[pid]['fee'] += getattr(d, 'fee', 0.0)

        if d.entry == mt5.DEAL_ENTRY_IN:
            positions[pid]['comment'] = d.comment if d.comment else ""
            positions[pid]['type'] = 'BUY' if d.type == mt5.DEAL_TYPE_BUY else 'SELL'
        elif d.entry == mt5.DEAL_ENTRY_OUT:
            positions[pid]['close_time'] = datetime.fromtimestamp(d.time, timezone.utc)

    dir_trades = []
    for pid, p in positions.items():
        if not p.get('close_time') or p['close_time'] < cutoff_time:
            continue
        if p['type'] == direction:
            net_pnl = p['profit'] + p['commission'] + p['swap'] + p['fee']
            dir_trades.append({
                'pnl': net_pnl,
                'close_time': p['close_time']
            })

    if len(dir_trades) < max_losses:
        return False, 0

    # Sort descending by close time (most recent first)
    dir_trades.sort(key=lambda x: x['close_time'], reverse=True)
    recent_subset = dir_trades[:max_losses]

    # If all recent trades in this direction were losses (PnL < -0.50)
    if all(t['pnl'] < -0.50 for t in recent_subset):
        last_loss_time = recent_subset[0]['close_time']
        elapsed_sec = (datetime.now(timezone.utc) - last_loss_time).total_seconds()
        cooldown_sec = cooldown_minutes * 60
        if elapsed_sec < cooldown_sec:
            remaining_min = int((cooldown_sec - elapsed_sec) / 60) + 1
            return True, remaining_min

    return False, 0

