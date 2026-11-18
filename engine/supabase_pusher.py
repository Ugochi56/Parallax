import os
import sys
import time
import threading
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Load MT5 connection (imported safe)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Try to load credentials from .env
SUPABASE_URL = ""
SUPABASE_SERVICE_ROLE_KEY = ""
MAGIC_NUMBER = 999001
SYMBOL = "XAUUSDm"

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(_env_path)
except ImportError:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    from core.config import MAGIC_NUMBER, SYMBOL
except ImportError:
    pass

_fallback_mt5_lock = threading.RLock()


def _get_mt5_lock():
    mt5_iface = sys.modules.get('engine.mt5_interface')
    if mt5_iface is not None and hasattr(mt5_iface, 'mt5_lock'):
        return mt5_iface.mt5_lock
    try:
        from engine.mt5_interface import mt5_lock
        return mt5_lock
    except Exception:
        return _fallback_mt5_lock

_last_pushed_trades_state = None

def is_configured():
    """Check if Supabase credentials are set in .env."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

def get_headers(upsert_col=None):
    """Generate headers with service role auth."""
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Connection": "close"
    }
    if upsert_col:
        headers["Prefer"] = f"on-conflict={upsert_col},resolution=merge-duplicates"
    return headers

def make_request(method, url, max_retries=3, backoff_factor=1, **kwargs):
    """Make HTTP requests with automatic retry logic for transient network failures and 5xx errors."""
    for attempt in range(max_retries):
        try:
            # Enforce 15s timeout on API calls to allow for Supabase cold starts
            r = requests.request(method, url, timeout=15, **kwargs)
            if 500 <= r.status_code < 600 or r.status_code in [500, 502, 503, 504]:
                if attempt == max_retries - 1:
                    return r
                time.sleep(backoff_factor * (attempt + 1))
                continue
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff_factor * (attempt + 1))

def push_account_metrics():
    """Pushes live balance and equity to the account_metrics table."""
    if not is_configured():
        return False

    if not mt5:
        return False

    with _get_mt5_lock():
        if not mt5.terminal_info():
            return False
        account_info = mt5.account_info()
        if not account_info:
            return False
        login = account_info.login
        balance = account_info.balance
        equity = account_info.equity
    now_iso = datetime.now(timezone.utc).isoformat()
    
    url = f"{SUPABASE_URL}/rest/v1/account_metrics"
    check_url = f"{url}?account_id=eq.{login}"
    try:
        r = make_request("GET", check_url, headers=get_headers())
        payload = {
            "account_id": login,
            "balance": balance,
            "equity": equity,
            "last_updated": now_iso
        }
        
        if r.status_code == 200 and len(r.json()) > 0:
            # Update (PATCH)
            make_request("PATCH", check_url, headers=get_headers(), json=payload)
        else:
            # Insert (POST)
            make_request("POST", url, headers=get_headers(), json=payload)
        return True
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print("[SUPABASE] ⏳ Cloud connection reset / waking up (will retry on next interval)")
        return False
    except Exception as e:
        print(f"[SUPABASE] Error pushing metrics: {e}")
        return False

def categorize_strategy(comment):
    """Determine strategy name from MT5 deal comment."""
    c = str(comment).upper()
    if "CHOCH" in c: return "SMC_CHOCH"
    if "OB" in c: return "SMC_OB"
    if "SMC" in c or "FVG" in c: return "SMC_FVG"
    if "RSI" in c: return "RSI_REVERSION"
    if "VWAP" in c: return "VWAP_REVERSION"
    if "LONDON" in c or "KZ_STOP" in c: return "LONDON_BREAKOUT"
    if "LIQ" in c or "SWEEP" in c: return "LIQUIDITY_SWEEP"
    if "BB" in c: return "BB_BREAKOUT"
    return "OTHER"

def push_closed_trades():
    """Fetches closed MT5 deals AND live open positions, syncing both to Supabase."""
    if not is_configured():
        return False

    if not mt5:
        return False

    url = f"{SUPABASE_URL}/rest/v1/trades"
    active_magic = getattr(sys.modules.get('core.config'), 'MAGIC_NUMBER', MAGIC_NUMBER)

    with _get_mt5_lock():
        if not mt5.terminal_info():
            return False
        # 1. Fetch live open positions to distinguish open vs fully closed trades
        live_positions = mt5.positions_get()
        positions_fetch_ok = (live_positions is not None)
        open_pids = {p.ticket for p in live_positions} if live_positions else set()
        acc_info = mt5.account_info()
        current_account_login = acc_info.login if acc_info else 0

        # 2. Fetch history deals from MT5
        time_from = datetime(2020, 1, 1)
        time_to = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(time_from, time_to)
    
    positions = {}
    if deals is not None and len(deals) > 0:
        # Reconstruct closed positions (summing PnL across all partial & final exit deals)
        pos_magics = {d.position_id: d.magic for d in deals if d.entry == 0 and d.magic != 0}
        for d in deals:
            pid = d.position_id
            if pid == 0: continue
            
            if pid not in positions:
                positions[pid] = {
                    'ticket': pid,
                    'symbol': d.symbol,
                    'magic': pos_magics.get(pid, d.magic),
                    'profit': 0.0,
                    'commission': 0.0,
                    'swap': 0.0,
                    'fee': 0.0,
                    'volume': 0.0,
                    'comment': ''
                }
            positions[pid]['profit'] += d.profit
            positions[pid]['commission'] += d.commission
            positions[pid]['swap'] += d.swap
            positions[pid]['fee'] += d.fee
            
            if d.entry == 0:  # mt5.DEAL_ENTRY_IN
                positions[pid]['open_time'] = datetime.fromtimestamp(d.time, timezone.utc).isoformat()
                positions[pid]['type'] = 'BUY' if d.type == 0 else 'SELL'
                positions[pid]['entry_price'] = d.price
                positions[pid]['volume'] = d.volume
                positions[pid]['magic'] = pos_magics.get(pid, d.magic)
                positions[pid]['comment'] = d.comment if d.comment else "Unknown"
            elif d.entry == 1:  # mt5.DEAL_ENTRY_OUT
                positions[pid]['close_time'] = datetime.fromtimestamp(d.time, timezone.utc).isoformat()
                positions[pid]['exit_price'] = d.price
            
    # Compile records to insert/upsert
    new_records = []
    
    # 2a. Add closed positions from deals
    for pid, p in positions.items():
        if p.get('magic') != active_magic:
            continue
        # Position must have at least one exit deal AND no longer be in open MT5 positions
        if 'close_time' not in p or pid in open_pids:
            continue  # Still open in MT5
            
        net_profit = p['profit'] + p['commission'] + p['swap'] + p['fee']
        outcome = "BE"
        if net_profit > 0.50:
            outcome = "WIN"
        elif net_profit < -0.50:
            outcome = "LOSS"
            
        new_records.append({
            'ticket': pid,
            'account_id': current_account_login,
            'symbol': p['symbol'],
            'type': p.get('type', 'BUY'),
            'volume': round(p['volume'], 2),
            'open_time': p.get('open_time', p.get('close_time', '1970-01-01T00:00:00+00:00')),
            'close_time': p['close_time'],
            'entry_price': round(p.get('entry_price', 0.0), 3),
            'exit_price': round(p.get('exit_price', 0.0), 3),
            'net_profit': round(net_profit, 2),
            'fees': round(p['commission'] + p['swap'] + p['fee'], 2),
            'strategy': categorize_strategy(p.get('comment', 'Unknown')),
            'outcome': outcome
        })

    # 2b. Add LIVE OPEN positions for this account only (no cross-account phantom mirroring)
    valid_open_tickets = set()
    if live_positions:
        for pos in live_positions:
            if pos.magic != active_magic:
                continue
            open_iso = datetime.fromtimestamp(pos.time, timezone.utc).isoformat()
            strat = categorize_strategy(pos.comment if pos.comment else 'Unknown')
            valid_open_tickets.add(pos.ticket)

            new_records.append({
                'ticket': pos.ticket,
                'account_id': current_account_login,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': round(pos.volume, 2),
                'open_time': open_iso,
                'close_time': '1970-01-01T00:00:00+00:00',  # Sentinel for OPEN trade (satisfies NOT NULL constraint)
                'entry_price': round(pos.price_open, 3),
                'exit_price': round(pos.price_current, 3),
                'net_profit': round(pos.profit + pos.swap, 2),
                'fees': round(pos.swap, 2),
                'strategy': strat,
                'outcome': 'OPEN'
            })
        
    if not new_records:
        return False

    # Deduplicate by ticket so ON CONFLICT never receives duplicate keys in a single POST batch
    seen_tickets = set()
    deduped_records = []
    for rec in new_records:
        if rec['ticket'] not in seen_tickets:
            seen_tickets.add(rec['ticket'])
            deduped_records.append(rec)
    new_records = deduped_records
        
    global _last_pushed_trades_state
    current_state = tuple(sorted((r['ticket'], r['close_time'], r['net_profit']) for r in new_records))
    if _last_pushed_trades_state == current_state:
        return True  # No trade state changes, skip redundant network call & log spam!
        
    # 3. Clean up any orphaned 'OPEN' rows for THIS account that are no longer open in MT5
    if positions_fetch_ok and current_account_login:
        try:
            check_open_url = f"{url}?outcome=eq.OPEN&account_id=eq.{current_account_login}"
            resp = make_request("GET", check_open_url, headers=get_headers())
            if resp.status_code == 200 and resp.json():
                for row in resp.json():
                    if row['ticket'] not in valid_open_tickets:
                        make_request("DELETE", f"{url}?ticket=eq.{row['ticket']}&outcome=eq.OPEN&account_id=eq.{current_account_login}", headers=get_headers())
        except Exception:
            pass

    # 4. Bulk insert/upsert to Supabase when trade state changes
    try:
        r = make_request("POST", url, headers=get_headers(upsert_col="ticket"), json=new_records)
        if r.status_code in [200, 201]:
            _last_pushed_trades_state = current_state
            return True
        elif r.status_code == 400 and "account_id" in r.text:
            # Fallback if account_id column does not exist in Supabase table schema yet
            clean_records = [{k: v for k, v in record.items() if k != 'account_id'} for record in new_records]
            r_fallback = make_request("POST", url, headers=get_headers(upsert_col="ticket"), json=clean_records)
            if r_fallback.status_code in [200, 201]:
                _last_pushed_trades_state = current_state
                return True
            else:
                print(f"[SUPABASE] Push failed: {r_fallback.status_code} - {r_fallback.text}")
                return False
        else:
            print(f"[SUPABASE] Push failed: {r.status_code} - {r.text}")
            return False
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print("[SUPABASE] ⏳ Trades sync connection reset / timed out (will retry on next interval)")
        return False
    except Exception as e:
        print(f"[SUPABASE] Error inserting trades: {e}")
        return False

_sync_lock = threading.Lock()

def sync_all():
    """Convenience function to push both metrics and trades (non-blocking if a sync is already in flight)."""
    if not is_configured():
        return False
    if not _sync_lock.acquire(blocking=False):
        return False
    try:
        m_success = push_account_metrics()
        t_success = push_closed_trades()
        return m_success or t_success
    finally:
        _sync_lock.release()

if __name__ == "__main__":
    # Test execution
    if mt5 and mt5.initialize():
        print("[SUPABASE] Running standalone test sync...")
        try:
            sync_all()
        finally:
            mt5.shutdown()
    else:
        print("[SUPABASE] Standalone run failed: MT5 not initialized.")
