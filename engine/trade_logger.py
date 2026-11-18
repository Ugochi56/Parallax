import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import MAGIC_NUMBER

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def calculate_analytics(df):
    if df.empty:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit": 0.0, "max_drawdown": 0.0}

    total_trades = len(df)
    wins = len(df[df['Outcome'] == 'WIN'])
    losses = len(df[df['Outcome'] == 'LOSS'])
    bes = len(df[df['Outcome'] == 'BE'])

    # Avoid divide by zero
    total_resolved = wins + losses
    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0.0

    net_profit = df['Net Profit'].sum()

    # Calculate Drawdown
    df = df.sort_values('Close Time')
    cumulative_profit = df['Net Profit'].cumsum()
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit
    max_drawdown = drawdown.max()

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "bes": bes,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "max_drawdown": max_drawdown
    }

import threading
_fallback_mt5_lock = threading.RLock()

def _get_mt5_lock():
    mt5_iface = sys.modules.get('engine.mt5_interface')
    if mt5_iface is not None and hasattr(mt5_iface, 'mt5_lock'):
        return mt5_iface.mt5_lock
    return _fallback_mt5_lock

def export_trade_ledger():
    with _get_mt5_lock():
        if not mt5.terminal_info():
            return False

        # Get history from Jan 1st 2020 to now
        time_from = datetime(2020, 1, 1)
        time_to = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(time_from, time_to)
        live_positions = mt5.positions_get()
        account_info = mt5.account_info()
    
    if deals is None or len(deals) == 0:
        return False

    pos_magics = {d.position_id: d.magic for d in deals if d.entry == mt5.DEAL_ENTRY_IN and d.magic != 0}
    positions = {}
    for d in deals:
        pid = d.position_id
        if pid == 0: continue  # Balance/Deposit operations
        
        if pid not in positions:
            positions[pid] = {
                'magic': pos_magics.get(pid, d.magic), 
                'symbol': d.symbol, 
                'profit': 0.0, 
                'commission': 0.0, 
                'swap': 0.0, 
                'fee': 0.0,
                'volume': 0.0,
                'reason': ''
            }
            
        positions[pid]['profit'] += d.profit
        positions[pid]['commission'] += d.commission
        positions[pid]['swap'] += d.swap
        positions[pid]['fee'] += d.fee
        
        if d.entry == mt5.DEAL_ENTRY_IN:
            positions[pid]['open_time'] = datetime.fromtimestamp(d.time)
            positions[pid]['type'] = 'BUY' if d.type == mt5.DEAL_TYPE_BUY else 'SELL'
            positions[pid]['entry_price'] = d.price
            positions[pid]['volume'] = d.volume
            # IN deal holds the magic number natively
            positions[pid]['magic'] = pos_magics.get(pid, d.magic) 
            positions[pid]['reason'] = d.comment if d.comment else "Unknown"
            
        elif d.entry == mt5.DEAL_ENTRY_OUT:
            positions[pid]['close_time'] = datetime.fromtimestamp(d.time)
            positions[pid]['exit_price'] = d.price

    open_pids = {p.ticket for p in live_positions} if live_positions else set()

    active_magic = getattr(sys.modules.get('core.config'), 'MAGIC_NUMBER', MAGIC_NUMBER)
    records = []
    for pid, p in positions.items():
        if p.get('magic') != active_magic: continue
        if 'close_time' not in p or pid in open_pids: continue # Position is still open in MT5
        
        net_profit = p['profit'] + p['commission'] + p['swap'] + p['fee']
        
        outcome = "BE"
        # Using $0.50 as a scratch/break-even threshold
        if net_profit > 0.50: outcome = "WIN"
        elif net_profit < -0.50: outcome = "LOSS"
        
        records.append({
            'Ticket': pid,
            'Symbol': p['symbol'],
            'Type': p.get('type', 'UNKNOWN'),
            'Volume': p.get('volume', 0.0),
            'Open Time': p.get('open_time'),
            'Close Time': p.get('close_time'),
            'Entry Price': round(p.get('entry_price', 0.0), 3),
            'Exit Price': round(p.get('exit_price', 0.0), 3),
            'Gross Profit': round(p['profit'], 2),
            'Fees': round(p['commission'] + p['swap'] + p['fee'], 2),
            'Net Profit': round(net_profit, 2),
            'Outcome': outcome,
            'Reason': p.get('reason', 'Unknown')
        })

    if not records:
        return False

    df_all = pd.DataFrame(records)
    df_all['Close Time'] = pd.to_datetime(df_all['Close Time'])
    
    # Generate Month-Year key for grouping
    df_all['MonthKey'] = df_all['Close Time'].dt.strftime('%b_%Y')  # e.g., 'May_2026'

    os.makedirs(DATA_DIR, exist_ok=True)
    
    acc_num = account_info.login if account_info else "000000"

    for month_key, df_month in df_all.groupby('MonthKey'):
        # Save CSV safely
        csv_path = os.path.join(DATA_DIR, f"nexus_ledger_{acc_num}_{month_key}.csv")
        df_export = df_month.drop(columns=['MonthKey']).sort_values('Close Time', ascending=False)
        try:
            df_export.to_csv(csv_path, index=False)
        except PermissionError:
            print("[LOGGER] ⚠️ PermissionError writing trade ledger CSV")

        # Save Analytics Report safely
        stats = calculate_analytics(df_month)
        report_path = os.path.join(DATA_DIR, f"nexus_report_{acc_num}_{month_key}.txt")
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"=========================================\n")
                f.write(f"NEXUS QUANTITATIVE LEDGER : {month_key}\n")
                f.write(f"=========================================\n\n")
                f.write(f"Total Trades Taken  : {stats['total_trades']}\n")
                f.write(f"Winning Trades      : {stats['wins']}\n")
                f.write(f"Losing Trades       : {stats['losses']}\n")
                f.write(f"Break-Even/Scratch  : {stats['bes']}\n")
                f.write(f"-----------------------------------------\n")
                f.write(f"WIN RATE            : {stats['win_rate']:.2f}%\n")
                f.write(f"MAXIMUM DRAWDOWN    : ${stats['max_drawdown']:.2f}\n")
                f.write(f"NET PROFIT          : ${stats['net_profit']:.2f}\n")
                f.write(f"=========================================\n")
        except PermissionError:
            print(f"[LOGGER] ⚠️ PermissionError writing report: {report_path}")

    return True

if __name__ == "__main__":
    if mt5.initialize():
        print("Running manual ledger export...")
        try:
            success = export_trade_ledger()
            print(f"Export Success: {success}")
        finally:
            mt5.shutdown()
    else:
        print("Failed to initialize MT5")
