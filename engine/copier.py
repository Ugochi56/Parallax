"""
NEXUS CORE — MASTER-SLAVE TRADE COPIER ENGINE
Signal queue writer for the copier daemon. The master bot calls queue_copier_signal()
to write trade events to data/copier_queue.jsonl. The copier_daemon.py (separate process)
picks them up and executes on slave accounts via its own MT5 terminal.
"""

import sys
import threading
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import os
import uuid
from datetime import datetime

_copier_file_lock = threading.RLock()

SLAVES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'client_slaves.json')
QUEUE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'copier_queue.jsonl')

def load_slave_accounts():
    """Loads configured client slave accounts from JSON file."""
    if not os.path.exists(SLAVES_FILE):
        default_data = []
        os.makedirs(os.path.dirname(SLAVES_FILE), exist_ok=True)
        with open(SLAVES_FILE, 'w') as f:
            json.dump(default_data, f, indent=4)
        return default_data
    try:
        with open(SLAVES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[COPIER] ⚠️ Error loading {SLAVES_FILE}: {e}")
        return []

def add_slave_account(client_name, login, password, server, risk_mode="proportional", risk_multiplier=1.0):
    """Utility helper to add a new client account to the copier registry."""
    slaves = load_slave_accounts()
    
    # Check if already exists
    for s in slaves:
        if s['login'] == int(login):
            s['client_name'] = client_name
            s['password'] = password
            s['server'] = server
            s['risk_mode'] = risk_mode
            s['risk_multiplier'] = risk_multiplier
            s['active'] = True
            with open(SLAVES_FILE, 'w') as f:
                json.dump(slaves, f, indent=4)
            print(f"[COPIER] ✅ Updated existing client account: {client_name} ({login})")
            return
            
    new_slave = {
        "client_name": client_name,
        "login": int(login),
        "password": password,
        "server": server,
        "risk_mode": risk_mode,
        "risk_multiplier": risk_multiplier,
        "active": True,
        "added_at": datetime.now().isoformat()
    }
    slaves.append(new_slave)
    with open(SLAVES_FILE, 'w') as f:
        json.dump(slaves, f, indent=4)
    print(f"[COPIER] 🎉 Registered new client account: {client_name} ({login})")

def queue_copier_signal(signal_type, trade_data):
    """
    Appends a trade signal to the copier queue file.
    Disabled when ENABLE_LOCAL_COPIER is False to eliminate file I/O and terminal spam.
    """
    try:
        from core.config import ENABLE_LOCAL_COPIER
        if not ENABLE_LOCAL_COPIER:
            return
    except Exception:
        pass
    signal = {
        "id": uuid.uuid4().hex[:8],
        "ts": datetime.now().isoformat(),
        "type": signal_type,
        "action_type": trade_data.get("action_type"),
        "order_mode": trade_data.get("order_mode"),
        "order_type": trade_data.get("order_type"),
        "ticket": trade_data.get("ticket"),
        "signal": trade_data.get("signal"),
        "symbol": trade_data.get("symbol", "XAUUSD"),
        "price": trade_data.get("price"),
        "master_entry": trade_data.get("master_entry"),
        "sl": trade_data.get("sl"),
        "tp": trade_data.get("tp"),
        "prev_sl": trade_data.get("prev_sl"),
        "lots": trade_data.get("lots"),
        "close_pct": trade_data.get("close_pct"),
        "close_volume": trade_data.get("close_volume"),
        "magic": trade_data.get("magic"),
        "comment": trade_data.get("comment", ""),
        "master_equity": trade_data.get("master_equity", 0),
        "reason": trade_data.get("reason"),
        "done": False
    }
    # Remove None values to keep signals clean
    signal = {k: v for k, v in signal.items() if v is not None}
    
    try:
        with _copier_file_lock:
            os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
            with open(QUEUE_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(signal) + '\n')
        print(f"[COPIER] 📡 Queued {signal_type} for #{trade_data.get('ticket', 'ALL')}")
    except Exception as e:
        print(f"[COPIER] ⚠️ Failed to write signal to queue: {e}")

if __name__ == '__main__':
    print("=== NEXUS MASTER-SLAVE TRADE COPIER ===")
    print(f"Queue File: {QUEUE_FILE}")
    slaves = load_slave_accounts()
    print(f"Total Registered Client Accounts: {len(slaves)}")
    for s in slaves:
        print(f"  • Client: {s.get('client_name')} | Login: {s.get('login')} | Server: {s.get('server')} | Active: {s.get('active')}")
    
    # Show pending signals
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'r') as f:
            lines = f.readlines()
        pending = sum(1 for l in lines if '"done": false' in l.lower() or '"done":false' in l.lower())
        print(f"\nPending Signals in Queue: {pending}")
    else:
        print("\nNo queue file yet (no trades have been signaled).")
