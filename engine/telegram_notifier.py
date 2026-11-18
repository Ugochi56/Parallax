import requests
import threading
import time
import MetaTrader5 as mt5
import sys
import os
import json
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import SYMBOL, MAGIC_NUMBER

# ==========================================
# 📱 TELEGRAM NOTIFICATION ENGINE
# ==========================================
# Sends real-time alerts AND listens for commands from your phone.
# All sends are queued and dispatched by a single background worker thread.

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
TELEGRAM_ENABLED = False
SIGNAL_CHANNEL_PRIVATE = ""
SIGNAL_CHANNEL_PUBLIC = ""
PAYMENT_LINK = ""
PUBLIC_SIGNALS_PER_DAY = 5

_msg_queue = queue.Queue()
_worker_thread = None

def _get_mt5_lock():
    try:
        from engine.mt5_interface import mt5_lock
        return mt5_lock
    except Exception:
        return threading.RLock()

def _telegram_worker():
    while True:
        try:
            item = _msg_queue.get()
            if item is None:
                break
            url, payload, timeout, max_retries = item
            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json=payload, timeout=timeout)
                    if res.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(1)
            _msg_queue.task_done()
        except Exception:
            pass

def _ensure_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_telegram_worker, daemon=True)
        _worker_thread.start()

def _enqueue_post(url, payload, timeout=12, max_retries=1):
    _ensure_worker()
    _msg_queue.put((url, payload, timeout, max_retries))

try:
    from engine.telegram_credentials import *
    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    pass  # No credentials file - notifications disabled

def _get_base_url():
    try:
        from engine import telegram_credentials as creds
        token = creds.TELEGRAM_BOT_TOKEN
    except ImportError:
        token = TELEGRAM_BOT_TOKEN
    return f"https://api.telegram.org/bot{token}"

_bot_state_lock = threading.Lock()
_public_state_lock = threading.Lock()

# Daily public signal counter (persisted to disk across bot restarts)
PUBLIC_SIGNAL_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'public_signal_state.json')

def _load_public_signal_state():
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    default_state = {"count": 0, "day": today, "broadcasted_orders": []}
    if not os.path.exists(PUBLIC_SIGNAL_STATE_FILE):
        return default_state
    try:
        with open(PUBLIC_SIGNAL_STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            if state.get("day") != today:
                state = {"count": 0, "day": today, "broadcasted_orders": []}
            return state
    except Exception as e:
        return default_state

def _save_public_signal_state(state):
    with _public_state_lock:
        try:
            os.makedirs(os.path.dirname(PUBLIC_SIGNAL_STATE_FILE), exist_ok=True)
            tmp_file = PUBLIC_SIGNAL_STATE_FILE + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            os.replace(tmp_file, PUBLIC_SIGNAL_STATE_FILE)
        except Exception as e:
            pass

# Shared state - goldvx.py writes these, commands read them
_bot_state = {
    "regime": "BOOTING",
    "adx": 0.0,
    "rsi": 0.0,
    "price": 0.0,
    "trend": "FLAT",
    "engines": "",
    "paused": False,
    "drawdown_halted": False,
    "observe_only": False,
}

def update_bot_state(**kwargs):
    """Called by goldvx.py to keep the shared state current."""
    with _bot_state_lock:
        _bot_state.update(kwargs)

def is_paused():
    """Check if trading is paused via Telegram command."""
    with _bot_state_lock:
        return _bot_state.get("paused", False)

# ==========================================
# 📤 OUTBOUND NOTIFICATIONS (Bot → You)
# ==========================================

def send_telegram(text):
    """Non-blocking Telegram notification. Returns immediately."""
    if not TELEGRAM_ENABLED:
        return
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    _enqueue_post(f"{_get_base_url()}/sendMessage", payload, timeout=12, max_retries=1)

# ==========================================
# 📦 PRE-BUILT NOTIFICATION TEMPLATES
# ==========================================

def notify_boot(account_login, balance):
    send_telegram(
        f"🟢 <b>NEXUS CORE ONLINE</b>\n"
        f"Account: <code>{account_login}</code>\n"
        f"Balance: <b>${balance:,.2f}</b>\n"
        f"Engines: 8 | AI Models: 3\n\n"
        f"Type /help for available commands."
    )

def notify_trade_opened(direction, symbol, price, sl, tp, lots, ticket, strategy):
    arrow = "🟩 BUY" if direction == "BUY" else "🟥 SELL"
    send_telegram(
        f"{arrow} <b>TRADE OPENED</b>\n"
        f"Symbol: {symbol} @ <b>{price:.2f}</b>\n"
        f"Lot Size: {lots}\n"
        f"SL: {sl:.2f} | TP: {tp:.2f}\n"
        f"Strategy: <code>{strategy}</code>\n"
        f"Ticket: <code>{ticket}</code>"
    )

def notify_trade_closed(ticket, reason, pnl=None):
    if pnl is not None:
        outcome = f"${pnl:+.2f} (🟢 Profit)" if pnl > 0.0 else (f"${pnl:+.2f} (⚪ Break-Even)" if abs(pnl) < 1e-4 else f"${pnl:+.2f} (🔴 Loss)")
    else:
        outcome = "Closed"
    emoji = "💰" if pnl and pnl > 0 else ("⚖️" if pnl is not None and abs(pnl) < 1e-4 else "🔻")
    send_telegram(
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"Ticket: <code>{ticket}</code>\n"
        f"Reason: {reason}\n"
        f"PnL: <b>{outcome}</b>"
    )

def notify_copier_summary(master_ticket, action, symbol, price, slave_results):
    """Sends ONE clean, compact summary alert listing all client copier executions."""
    if not slave_results:
        return
    dir_str = "BUY" if (action == 0 or action == "BUY") else "SELL"
    arrow = "🟩 BUY" if dir_str == "BUY" else "🟥 SELL"
    lines = [
        f"⚡ <b>COPIER SUMMARY ALERT</b>\n",
        f"🎯 Master Trade #{master_ticket} ({arrow} @ {price:.2f})",
        f"✅ Mirrored to <b>{len(slave_results)} Client Accounts</b> (&lt; 15ms):\n"
    ]
    for r in slave_results:
        lines.append(f"• <b>{r['name']}</b> (<code>{r['login']}</code>): {r['lots']} Lots")
    send_telegram("\n".join(lines))

def notify_copier_close_summary(master_ticket, closed_clients):
    """Sends ONE clean summary alert when slave trades are closed."""
    if not closed_clients:
        return
    lines = [
        f"🏁 <b>COPIER CLOSE SUMMARY</b>\n",
        f"🎯 Master Trade #{master_ticket} Closed!",
        f"✅ Closed on <b>{len(closed_clients)} Client Accounts</b> (&lt; 15ms):\n"
    ]
    for c in closed_clients:
        lines.append(f"• <b>{c['name']}</b> (<code>{c['login']}</code>)")
    send_telegram("\n".join(lines))

def notify_veto(loss_prob, strictness):
    send_telegram(
        f"🛡️ <b>AI VETO - TRADE BLOCKED</b>\n"
        f"Loss Probability: <b>{loss_prob*100:.0f}%</b>\n"
        f"Strictness Threshold: {strictness*100:.0f}%\n"
        f"The Evolution Layer protected your capital."
    )

def notify_regime_change(regime, engines):
    if not getattr(sys.modules.get('core.config'), 'NOTIFY_REGIME_CHANGES', False):
        return
    emoji_map = {"[TREND]": "📈", "[RANGE]": "📊", "[NEUTRAL]": "⚖️", "[ASIAN-RANGE]": "🌏"}
    emoji = emoji_map.get(regime, "🔄")
    msg = (
        f"{emoji} <b>REGIME CHANGE</b>\n"
        f"New State: <b>{regime}</b>\n"
        f"Active Engines: <code>{engines}</code>"
    )
    # 1. Admin Feed
    send_telegram(msg)
    
    # 2. VIP Channel (Full Details)
    if SIGNAL_CHANNEL_PRIVATE:
        if isinstance(SIGNAL_CHANNEL_PRIVATE, list):
            for ch in SIGNAL_CHANNEL_PRIVATE:
                _send_to_channel(ch, f"🧠 <b>NEXUS AI UPDATE</b>\n{msg}")
        else:
            _send_to_channel(SIGNAL_CHANNEL_PRIVATE, f"🧠 <b>NEXUS AI UPDATE</b>\n{msg}")
            
    # 3. Public Channel Teaser (Marketing FOMO)
    if SIGNAL_CHANNEL_PUBLIC:
        cta = f"\n\nTrade with the AI: {PAYMENT_LINK}" if PAYMENT_LINK else ""
        pub_msg = (
            f"🧠 <b>VIP AI INSIGHT</b>\n"
            f"Market Regime Shift Detected: <b>{regime}</b>\n"
            f"<i>The AI has automatically swapped its active trading engines to adapt to the new market conditions.</i>"
            f"{cta}"
        )
        _send_to_channel(SIGNAL_CHANNEL_PUBLIC, pub_msg)

def notify_daily_summary(equity, day_pnl, trades_today):
    emoji = "📈" if day_pnl >= 0 else "📉"
    send_telegram(
        f"{emoji} <b>DAILY SUMMARY</b>\n"
        f"Equity: <b>${equity:,.2f}</b>\n"
        f"Day PnL: <b>${day_pnl:+.2f}</b>\n"
        f"Trades Today: {trades_today}"
    )

def notify_drawdown_warning(current_loss_pct):
    send_telegram(
        f"🚨 <b>DRAWDOWN ALERT</b>\n"
        f"Daily Loss: <b>{current_loss_pct:.1f}%</b>\n"
        f"Approaching safety limit. Bot may pause trading."
    )

def notify_news_blackout(event_name):
    send_telegram(
        f"📰 <b>NEWS BLACKOUT ACTIVE</b>\n"
        f"Event: <code>{event_name}</code>\n"
        f"Trading paused until news impact clears."
    )

def notify_error(error_msg):
    send_telegram(
        f"⚠️ <b>NEXUS ERROR</b>\n"
        f"<code>{error_msg}</code>"
    )

# ==========================================
# 📥 INBOUND COMMANDS (You → Bot)
# ==========================================

def _handle_command(command):
    """Process a Telegram command and return the response text."""
    cmd = command.strip().lower()

    if cmd == "/help" or cmd == "/start":
        return (
            "<b>NEXUS COMMAND CENTER</b>\n\n"
            "/subscribe - Get premium signal access\n"
            "/check - Check subscription status\n"
            "/status - Current bot state\n"
            "/sentiment - Market sentiment report\n"
            "/equity - Balance and equity\n"
            "/trades - Open positions\n"
            "/today - Today's P&L\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/kill - Emergency close all\n"
            "/help - Show this menu"
        )

    elif cmd == "/status":
        with _bot_state_lock:
            state = _bot_state.copy()
        if state.get("drawdown_halted") and state.get("paused"):
            paused_tag = "🛑 DD-HALT (PAUSED)"
        elif state.get("drawdown_halted"):
            paused_tag = "🛑 DD-HALT (OBSERVE)"
        elif state.get("paused"):
            paused_tag = "⏸️ PAUSED"
        elif state.get("observe_only"):
            paused_tag = "👁️ OBSERVE ONLY"
        else:
            paused_tag = "▶️ ACTIVE"
        return (
            f"📊 <b>NEXUS STATUS</b>\n\n"
            f"State: <b>{paused_tag}</b>\n"
            f"Regime: <b>{state['regime']}</b>\n"
            f"Trend: {state['trend']}\n"
            f"ADX: {state['adx']:.1f} | RSI: {state['rsi']:.0f}\n"
            f"Price: <b>{state['price']:.2f}</b>\n"
            f"Engines: <code>{state['engines']}</code>"
        )

    elif cmd == "/equity":
        try:
            with _get_mt5_lock():
                acc = mt5.account_info()
            if acc:
                margin_pct = (acc.margin / acc.equity * 100) if acc.equity > 0 else 0
                return (
                    f"💵 <b>ACCOUNT INFO</b>\n\n"
                    f"Balance: <b>${acc.balance:,.2f}</b>\n"
                    f"Equity: <b>${acc.equity:,.2f}</b>\n"
                    f"Free Margin: ${acc.margin_free:,.2f}\n"
                    f"Margin Used: {margin_pct:.1f}%\n"
                    f"Floating P&L: <b>${acc.profit:+,.2f}</b>"
                )
            return "⚠️ Could not fetch account info."
        except Exception:
            return "⚠️ MT5 connection error."

    elif cmd == "/trades":
        try:
            with _get_mt5_lock():
                positions_raw = mt5.positions_get()
            positions = [p for p in positions_raw if p.symbol == SYMBOL] if positions_raw else []
            if not positions:
                return "📭 No open positions."
            
            lines = [f"📋 <b>OPEN POSITIONS</b> ({len(positions)})\n"]
            for pos in positions:
                direction = "BUY" if pos.type == 0 else "SELL"
                emoji = "🟩" if pos.type == 0 else "🟥"
                pnl_emoji = "✅" if pos.profit >= 0 else "❌"
                lines.append(
                    f"{emoji} <b>{direction}</b> {pos.volume} lots @ {pos.price_open:.2f}\n"
                    f"   SL: {pos.sl:.2f} | TP: {pos.tp:.2f}\n"
                    f"   {pnl_emoji} PnL: <b>${pos.profit:+.2f}</b>\n"
                    f"   Ticket: <code>{pos.ticket}</code>\n"
                )
            return "\n".join(lines)
        except Exception:
            return "⚠️ Could not fetch positions."

    elif cmd == "/today":
        try:
            from datetime import datetime, timedelta
            now = datetime.now()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            with _get_mt5_lock():
                acc = mt5.account_info()
                deals = mt5.history_deals_get(start_of_day, now + timedelta(hours=1))
                positions_raw = mt5.positions_get()
            
            closed_count = 0
            day_pnl = 0.0
            if deals:
                pos_magics = {
                    getattr(d, 'position_id', 0): d.magic
                    for d in deals
                    if getattr(d, 'entry', None) == 0 and getattr(d, 'magic', 0) != 0
                }
                for d in deals:
                    deal_magic = d.magic if d.magic != 0 else pos_magics.get(getattr(d, 'position_id', 0), 0)
                    if deal_magic == MAGIC_NUMBER and d.entry == 1:  # Exit deals only
                        closed_count += 1
                        day_pnl += d.profit + d.commission + d.swap + getattr(d, 'fee', 0.0)
            
            positions = [p for p in positions_raw if p.symbol == SYMBOL] if positions_raw else []
            open_count = len(positions)
            floating = sum(p.profit for p in positions)
            
            equity = acc.equity if acc else 0
            
            return (
                f"📅 <b>TODAY'S SUMMARY</b>\n\n"
                f"Equity: <b>${equity:,.2f}</b>\n"
                f"Closed Trades: {closed_count}\n"
                f"Realized P&L: <b>${day_pnl:+.2f}</b>\n"
                f"Open Trades: {open_count}\n"
                f"Floating P&L: <b>${floating:+.2f}</b>\n"
                f"Total Day P&L: <b>${day_pnl + floating:+.2f}</b>"
            )
        except Exception as e:
            return f"⚠️ Error: {e}"

    elif cmd == "/pause":
        with _bot_state_lock:
            _bot_state["paused"] = True
        return "⏸️ <b>TRADING PAUSED</b>\nNEXUS will stop opening new trades.\nOpen positions will continue to be managed.\nType /resume to restart."

    elif cmd == "/resume":
        with _bot_state_lock:
            _bot_state["paused"] = False
        return "▶️ <b>TRADING RESUMED</b>\nNEXUS is back online and scanning for setups."

    elif cmd == "/kill":
        try:
            with _get_mt5_lock():
                positions_raw = mt5.positions_get()
                positions = [p for p in positions_raw if p.symbol == SYMBOL and p.magic == MAGIC_NUMBER] if positions_raw else []
                if not positions:
                    return "📭 No positions to close."
                
                closed = 0
                for pos in positions:
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if not tick:
                        continue
                    action_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume,
                        "type": action_type, "position": pos.ticket, "price": price,
                        "deviation": 20, "magic": MAGIC_NUMBER, "comment": "TG_KILL",
                        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    res = mt5.order_send(request)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
            
            with _bot_state_lock:
                _bot_state["paused"] = True
            return (
                f"🔴 <b>EMERGENCY CLOSE</b>\n"
                f"Closed {closed}/{len(positions)} positions.\n"
                f"Trading has been paused.\n"
                f"Type /resume to restart."
            )
        except Exception as e:
            return f"⚠️ Kill failed: {e}"

    elif cmd == "/sentiment":
        try:
            import pandas as pd
            import pandas_ta as ta
            
            # Fetch latest data for M5, M15, H1, H4 (increased to 300 for EMA 200 lookback)
            with _get_mt5_lock():
                rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 300)
                rates_m15 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 100)
                rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 300)
                rates_h4 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 300)
            
            if rates_m5 is None or rates_h1 is None or rates_h4 is None or len(rates_m5) == 0 or len(rates_h1) == 0 or len(rates_h4) == 0:
                return "⚠️ Could not fetch sentiment rates from MT5."
                
            df_m5 = pd.DataFrame(rates_m5)
            df_m15 = pd.DataFrame(rates_m15) if (rates_m15 is not None and len(rates_m15) > 0) else df_m5
            df_h1 = pd.DataFrame(rates_h1)
            df_h4 = pd.DataFrame(rates_h4)
            
            m5_close = df_m5['close'].iloc[-1]
            
            # M5 Calculations with safety fallbacks
            m5_rsi_s = ta.rsi(df_m5['close'], length=14)
            m5_rsi = m5_rsi_s.iloc[-1] if m5_rsi_s is not None and not m5_rsi_s.empty else 50.0
            
            m5_ema_50_s = ta.ema(df_m5['close'], length=50)
            m5_ema_50 = m5_ema_50_s.iloc[-1] if m5_ema_50_s is not None and not m5_ema_50_s.empty else m5_close
            
            m5_ema_200_s = ta.ema(df_m5['close'], length=200)
            m5_ema_200 = m5_ema_200_s.iloc[-1] if m5_ema_200_s is not None and not m5_ema_200_s.empty else m5_close
            
            # H1 Calculations (EMA 20/50 ribbon aligned with trading engine)
            h1_idx = -2 if len(df_h1) >= 2 else -1
            h1_close = df_h1['close'].iloc[h1_idx]
            h1_rsi_s = ta.rsi(df_h1['close'], length=14)
            h1_rsi = h1_rsi_s.iloc[-1] if h1_rsi_s is not None and not h1_rsi_s.empty else 50.0
            h1_ema_20 = df_h1['close'].ewm(span=20, adjust=False).mean().iloc[h1_idx]
            h1_ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[h1_idx]
            
            # H4 Calculations (EMA 20/50 ribbon aligned with trading engine)
            h4_idx = -2 if len(df_h4) >= 2 else -1
            h4_close = df_h4['close'].iloc[h4_idx]
            h4_rsi_s = ta.rsi(df_h4['close'], length=14)
            h4_rsi = h4_rsi_s.iloc[-1] if h4_rsi_s is not None and not h4_rsi_s.empty else 50.0
            h4_ema_20 = df_h4['close'].ewm(span=20, adjust=False).mean().iloc[h4_idx]
            h4_ema_50 = df_h4['close'].ewm(span=50, adjust=False).mean().iloc[h4_idx]
            
            # M15 ADX for trend strength measurement (matches orchestrator & strategies)
            m15_adx_s = ta.adx(df_m15['high'], df_m15['low'], df_m15['close'], length=14)
            m5_adx = m15_adx_s['ADX_14'].iloc[-1] if m15_adx_s is not None and 'ADX_14' in m15_adx_s.columns else 20.0
            
            # M5 ATR for distance buffer calculation
            m5_atr_s = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
            m5_atr = m5_atr_s.iloc[-1] if m5_atr_s is not None and not m5_atr_s.empty else 0.0
            
            # ──── Honest Multi-Tier Trend Classification (EMA 20/50 Ribbon) ────
            h1_bull = (h1_close > h1_ema_20 and h1_ema_20 > h1_ema_50)
            h1_bear = (h1_close < h1_ema_20 and h1_ema_20 < h1_ema_50)
            h4_bull = (h4_close > h4_ema_20 and h4_ema_20 > h4_ema_50)
            h4_bear = (h4_close < h4_ema_20 and h4_ema_20 < h4_ema_50)
            h1_ema_dist = abs(h1_close - h1_ema_50)
            h4_ema_dist = abs(h4_close - h4_ema_50)
            ema_buffer = max(2.0, m5_atr * 0.4)  # Minimum $2 or 0.4x ATR distance required
            
            h1_trend = "BULLISH" if h1_bull else ("BEARISH" if h1_bear else "FLAT")
            h4_trend = "BULLISH" if h4_bull else ("BEARISH" if h4_bear else "FLAT")
            
            # Overall Sentiment Bias (Honest 3-Tier System)
            both_bullish = h1_bull and h4_bull
            both_bearish = h1_bear and h4_bear
            has_strong_adx = m5_adx >= 22.0
            has_momentum_rsi = (h1_rsi > 55 and h4_rsi > 52) if both_bullish else (h1_rsi < 45 and h4_rsi < 48) if both_bearish else False
            has_ema_clearance = (h1_ema_dist >= ema_buffer and h4_ema_dist >= ema_buffer)
            
            if both_bullish and has_strong_adx and has_momentum_rsi and has_ema_clearance:
                macro_bias = "🟢 STRONG BULLISH (UP)"
            elif both_bearish and has_strong_adx and has_momentum_rsi and has_ema_clearance:
                macro_bias = "🔴 STRONG BEARISH (DOWN)"
            elif both_bullish and has_ema_clearance:
                macro_bias = "🟢 MILD BULLISH (Leaning UP)"
            elif both_bearish and has_ema_clearance:
                macro_bias = "🔴 MILD BEARISH (Leaning DOWN)"
            elif h1_bull or (h4_bull and not h1_bear):
                macro_bias = "🟡 WEAK BULLISH (Transition / Ranging)"
            elif h1_bear or (h4_bear and not h1_bull):
                macro_bias = "🟡 WEAK BEARISH (Transition / Ranging)"
            else:
                macro_bias = "⚪ NEUTRAL / FLAT"
            
            return (
                f"🧠 <b>NEXUS MARKET SENTIMENT</b>\n\n"
                f"Asset: <b>{SYMBOL}</b>\n"
                f"Price: <b>${m5_close:.2f}</b>\n"
                f"M5 Volatility (ATR): <b>{m5_atr:.2f}</b>\n\n"
                f"📈 <b>MACRO STRUCTURE:</b>\n"
                f"Bias: <b>{macro_bias}</b>\n"
                f"H4 Trend: {h4_trend} | RSI: {h4_rsi:.0f}\n"
                f"H1 Trend: {h1_trend} | RSI: {h1_rsi:.0f}\n"
                f"ADX: {m5_adx:.0f} | EMA Clearance: {'✅' if has_ema_clearance else '❌'}\n\n"
                f"⚡ <b>SCALP STRUCTURE (M5):</b>\n"
                f"RSI: {m5_rsi:.0f}\n"
                f"EMA 50: {'Above' if m5_close > m5_ema_50 else 'Below'}\n"
                f"EMA 200: {'Above' if m5_close > m5_ema_200 else 'Below'}"
            )
        except Exception as e:
            return f"⚠️ Sentiment calculation failed: {e}"

    elif cmd.startswith("/genkey"):
        # Format: /genkey <name> [days] [mt5_account]
        parts = command.strip().split()
        if len(parts) < 2:
            return "<b>Usage:</b> <code>/genkey &lt;Name&gt; [Days=30] [MT5_Account]</code>\n<b>Example:</b> <code>/genkey John 30 12345678</code>"

        name = parts[1]
        days = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30
        account = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None

        import secrets
        import string
        from datetime import datetime, timedelta, timezone

        raw_key = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(12))
        key = f"NEXUS-{raw_key[:4]}-{raw_key[4:8]}-{raw_key[8:]}"

        exp_dt = datetime.now(timezone.utc) + timedelta(days=days)
        exp_iso = exp_dt.isoformat()

        from core.config import SUPABASE_URL, SUPABASE_ANON_KEY
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            return "⚠️ Supabase configuration missing."

        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        payload = {
            "license_key": key,
            "customer_name": name,
            "is_active": True,
            "expires_at": exp_iso,
            "mt5_account": account
        }

        try:
            res = requests.post(f"{SUPABASE_URL}/rest/v1/licenses", json=payload, headers=headers, timeout=10)
            if res.status_code in [200, 201]:
                acc_str = str(account) if account else "Unbound (Any MT5 Account)"
                return (
                    f"🔑 <b>NEXUS LICENSE GENERATED</b>\n\n"
                    f"<b>License Key:</b> <code>{key}</code>\n"
                    f"<b>Customer:</b> {name}\n"
                    f"<b>Valid For:</b> {days} Days (until {exp_iso[:10]})\n"
                    f"<b>MT5 Account Lock:</b> {acc_str}\n\n"
                    f"<i>Copy and send this key to your buyer!</i>"
                )
            else:
                return f"❌ Failed to save key in DB: {res.text}"
        except Exception as e:
            return f"❌ Error generating key: {e}"

    else:
        return f"❓ Unknown command: <code>{cmd}</code>\nType /help to see available commands."


def _handle_public_command(command, chat_id, username):
    """Handle commands available to ALL users (not just the owner)."""
    cmd = command.strip().lower()
    
    if cmd == "/subscribe":
        return "SUBSCRIBE_CHOICES"
    
    elif cmd == "/check":
        from engine.subscription_manager import check_subscription
        is_active, message = check_subscription(chat_id)
        status = "ACTIVE" if is_active else "INACTIVE"
        return f"<b>Subscription Status: {status}</b>\n{message}"
    
    elif cmd == "/start" or cmd == "/help":
        return "WELCOME_WITH_BUTTONS"  # Special flag handled by listener
    
    return None  # Not a public command


def _send_subscription_choices(chat_id):
    """Send the plan selection options to the user with inline buttons."""
    from engine.subscription_manager import check_subscription
    
    if TELEGRAM_CHAT_ID and str(chat_id) == str(TELEGRAM_CHAT_ID):
        _send_to_chat(chat_id, "👑 <b>Admin/Owner Access</b>\nYou have unlimited lifetime access to NEXUS Premium. You don't need to pay! Tap /check to view your status.")
        return
        
    is_active, _ = check_subscription(chat_id)
    if is_active:
        _send_to_chat(chat_id, "You already have an active subscription! Tap /check to see details.")
        return
        
    try:
        requests.post(f"{_get_base_url()}/sendMessage", json={
            "chat_id": chat_id,
            "text": (
                "🏆 <b>NEXUS Premium Signals</b>\n\n"
                "Select your preferred subscription tier:\n\n"
                f"1. 📅 <b>Weekly Access</b> (7 Days) - <b>${WEEKLY_SUBSCRIPTION_PRICE:.0f} USDT</b>\n"
                f"2. ⚡ <b>Monthly Access</b> (30 Days) - <b>${SUBSCRIPTION_PRICE:.0f} USDT</b>\n\n"
                "Includes:\n"
                "- Real-time gold signals (Entry, SL, TP)\n"
                "- 8 Strategy Engines & Machine Learning models\n"
                "- Native AI Veto risk protection"
            ),
            "parse_mode": "HTML",
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": f"📅 Weekly (${WEEKLY_SUBSCRIPTION_PRICE:.0f} USDT)", "callback_data": "sub_weekly"},
                        {"text": f"⚡ Monthly (${SUBSCRIPTION_PRICE:.0f} USDT)", "callback_data": "sub_monthly"}
                    ],
                    [{"text": "💬 DM Owner Directly (@NexusQuantSupport)", "url": "https://t.me/NexusQuantSupport"}],
                    [{"text": " Check My Subscription", "callback_data": "check"}]
                ]
            }
        }, timeout=5)
    except Exception:
        pass


def _send_invoice_response(chat_id, username, plan):
    """Build the subscription invoice and send it to the user."""
    from engine.subscription_manager import create_invoice, check_subscription
    
    if TELEGRAM_CHAT_ID and str(chat_id) == str(TELEGRAM_CHAT_ID):
        _send_to_chat(chat_id, "👑 <b>Admin/Owner Access</b>\nYou have unlimited lifetime access to NEXUS Premium. You don't need to pay! Tap /check to view your status.")
        return
        
    is_active, _ = check_subscription(chat_id)
    if is_active:
        _send_to_chat(chat_id, "You already have an active subscription! Tap /check to see details.")
        return
        
    pay_url, error = create_invoice(chat_id, username, plan)
    if error:
        _send_to_chat(chat_id, f"Could not create invoice: {error}")
        return
        
    plan_name = "Weekly Access" if plan == "weekly" else "Monthly Access"
    price = WEEKLY_SUBSCRIPTION_PRICE if plan == "weekly" else SUBSCRIPTION_PRICE
    duration = 7 if plan == "weekly" else 30
    
    _send_to_chat(chat_id, (
        f"<b>NEXUS Premium - {plan_name}</b>\n\n"
        f"Price: <b>${price:.0f} USDT</b>\n"
        f"Duration: <b>{duration} days</b>\n\n"
        f"Pay here: {pay_url}\n\n"
        f"After payment, you will automatically receive your private channel invite link."
    ))


def _command_listener():
    """Background thread that polls Telegram for new commands every 2 seconds."""
    last_update_id = 0
    
    while True:
        try:
            response = requests.get(
                f"{_get_base_url()}/getUpdates",
                params={"offset": last_update_id + 1, "timeout": 10},
                timeout=15
            )
            data = response.json()
            
            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    last_update_id = update["update_id"]
                    
                    # Handle button taps (callback queries)
                    callback = update.get("callback_query")
                    if callback:
                        cb_chat_id = str(callback["from"]["id"])
                        cb_username = callback["from"].get("username", "")
                        cb_data = callback.get("data", "")
                        cb_id = callback.get("id", "")
                        
                        # Acknowledge the button tap
                        try:
                            requests.post(f"{_get_base_url()}/answerCallbackQuery", json={"callback_query_id": cb_id}, timeout=3)
                        except Exception:
                            pass
                        
                        if cb_data == "subscribe":
                            _send_subscription_choices(cb_chat_id)
                        elif cb_data == "sub_weekly":
                            _send_invoice_response(cb_chat_id, cb_username, "weekly")
                        elif cb_data == "sub_monthly":
                            _send_invoice_response(cb_chat_id, cb_username, "monthly")
                        elif cb_data == "check":
                            from engine.subscription_manager import check_subscription
                            is_active, message = check_subscription(cb_chat_id)
                            status = "ACTIVE" if is_active else "INACTIVE"
                            _send_to_chat(cb_chat_id, f"<b>Subscription Status: {status}</b>\n{message}")
                        elif cb_data == "free_channel":
                            _send_to_chat(cb_chat_id, "Join our free signals channel: @NexusQuantSignals")
                        continue
                    
                    # Handle text messages
                    message = update.get("message", {})
                    chat_id = str(message.get("chat", {}).get("id", ""))
                    username = message.get("from", {}).get("username", "")
                    text = message.get("text", "")
                    
                    if not text.startswith("/"):
                        continue
                    
                    # Try public commands first (available to everyone)
                    public_reply = _handle_public_command(text, chat_id, username)
                    if public_reply == "WELCOME_WITH_BUTTONS":
                        _send_welcome_with_buttons(chat_id)
                        continue
                    elif public_reply == "SUBSCRIBE_CHOICES":
                        _send_subscription_choices(chat_id)
                        continue
                    elif public_reply:
                        _send_to_chat(chat_id, public_reply)
                        continue
                    
                    # Admin commands (owner only)
                    if chat_id == TELEGRAM_CHAT_ID:
                        reply = _handle_command(text)
                        send_telegram(reply)
        
        except Exception:
            pass  # Network blip - silently retry
        
        time.sleep(2)


def _send_welcome_with_buttons(chat_id):
    """Send the welcome message with inline buttons."""
    payload = {
        "chat_id": chat_id,
        "text": (
            "<b>Welcome to NEXUS Gold Signals</b>\n\n"
            "AI-powered gold trading signals built with:\n\n"
            "- 3 Machine Learning Models\n"
            "- 8 Strategy Engines\n"
            "- Self-Learning AI Veto System\n"
            "- Institutional Fakeout Detection\n\n"
            "Get real-time BUY/SELL signals with exact Entry, Stop Loss, and Take Profit levels."
        ),
        "parse_mode": "HTML",
        "reply_markup": {
            "inline_keyboard": [
                [{"text": "🚀 Get Premium Access", "callback_data": "subscribe"}],
                [{"text": "💬 DM Owner Directly (@NexusQuantSupport)", "url": "https://t.me/NexusQuantSupport"}],
                [{"text": "Check My Subscription", "callback_data": "check"}]
            ]
        }
    }
    _enqueue_post(f"{_get_base_url()}/sendMessage", payload, timeout=5, max_retries=1)


def _send_to_chat(chat_id, text):
    """Send a message to a specific chat ID."""
    if not TELEGRAM_ENABLED:
        return
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    _enqueue_post(f"{_get_base_url()}/sendMessage", payload, timeout=5, max_retries=1)


def start_command_listener():
    """Launch the command listener and payment monitor as daemon threads."""
    listener = threading.Thread(target=_command_listener, daemon=True)
    listener.start()
    
    # Start weekly review scheduler
    weekly_scheduler = threading.Thread(target=_weekly_scheduler_loop, daemon=True)
    weekly_scheduler.start()
    
    # Start payment monitor
    try:
        from engine.subscription_manager import start_payment_monitor
        if TELEGRAM_BOT_TOKEN and SIGNAL_CHANNEL_PRIVATE:
            start_payment_monitor(TELEGRAM_BOT_TOKEN, SIGNAL_CHANNEL_PRIVATE)
    except Exception:
        pass


# ==========================================
# 📡 SIGNAL CHANNEL BROADCASTING
# ==========================================

_broadcast_enabled = True

def set_broadcast_enabled(enabled: bool):
    """Enable or disable channel signal broadcasting (used for multi-instance separation)."""
    global _broadcast_enabled
    _broadcast_enabled = bool(enabled)

def is_broadcast_enabled() -> bool:
    """Return whether channel signal broadcasting is enabled."""
    return _broadcast_enabled

def _send_to_channel(channel_id, text):
    """Send a message to a Telegram channel (supports single ID or list of IDs)."""
    if not TELEGRAM_ENABLED or not channel_id or not _broadcast_enabled:
        return
        
    # Convert single channel ID to list for uniform handling
    channels = channel_id if isinstance(channel_id, list) else [channel_id]
    
    for ch in channels:
        payload = {
            "chat_id": ch,
            "text": text,
            "parse_mode": "HTML"
        }
        _enqueue_post(f"{_get_base_url()}/sendMessage", payload, timeout=12, max_retries=2)


def broadcast_signal_opened(direction, symbol, price, sl, tp, lots, strategy, confidence=None):
    """Broadcast a trade signal to both private and public channels."""
    if not _broadcast_enabled:
        return
    arrow = "🟩 BUY" if direction == "BUY" else "🟥 SELL"
    
    # -- PRIVATE CHANNEL: Full details for paying subscribers (NO lot size) --
    if SIGNAL_CHANNEL_PRIVATE:
        conf_text = f"\nAI Confidence: <b>{confidence:.0f}%</b>" if confidence else ""
        _send_to_channel(SIGNAL_CHANNEL_PRIVATE, (
            f"{arrow} <b>SIGNAL - {symbol}</b>\n\n"
            f"Entry: <b>{price:.2f}</b>\n"
            f"Stop Loss: <b>{sl:.2f}</b>\n"
            f"Take Profit: <b>{tp:.2f}</b>\n"
            f"Strategy: <code>{strategy}</code>"
            f"{conf_text}\n\n"
            f"Powered by NEXUS Core AI"
        ))
    
    # -- PUBLIC CHANNEL: Free teaser channel with full Entry/SL/TP (NO lot size) --
    if SIGNAL_CHANNEL_PUBLIC:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        state = _load_public_signal_state()
        
        if state.get("day") != today:
            state["count"] = 0
            state["day"] = today
            state["broadcasted_orders"] = []
        
        if state.get("count", 0) < PUBLIC_SIGNALS_PER_DAY:
            state["count"] = state.get("count", 0) + 1
            _save_public_signal_state(state)
            
            cta = f"\n\nWant real-time auto-trading? Get VIP access:\n{PAYMENT_LINK}" if PAYMENT_LINK else ""
            conf_text = f"\nAI Confidence: <b>{confidence:.0f}%</b>" if confidence else ""
            
            _send_to_channel(SIGNAL_CHANNEL_PUBLIC, (
                f"{arrow} <b>SIGNAL - {symbol}</b>\n\n"
                f"Entry: <b>{price:.2f}</b>\n"
                f"Stop Loss: <b>{sl:.2f}</b>\n"
                f"Take Profit: <b>{tp:.2f}</b>\n"
                f"Strategy: <code>{strategy}</code>"
                f"{conf_text}\n\n"
                f"Signal {state['count']}/{PUBLIC_SIGNALS_PER_DAY} today"
                f"{cta}\n\n"
                f"Powered by NEXUS Core AI"
            ))


def broadcast_signal_closed(direction, symbol, pnl, strategy, reason=None):
    """Broadcast a closed trade result to both channels."""
    if not _broadcast_enabled:
        return
    # If reason is not explicitly passed, infer it from strategy comment (especially for virtual hits)
    if not reason:
        reason = strategy
        
    if pnl is not None:
        if pnl > 0:
            if reason and any(k in reason.upper() for k in ["TP", "TAKE PROFIT", "TAKE_PROFIT"]):
                result_word = "TP HIT 🟢"
            elif reason and any(k in reason.upper() for k in ["TRAILING", "TRAIL", "BE", "BREAK-EVEN", "BREAK_EVEN"]):
                result_word = "TRAILING SL HIT 🟢"
            else:
                result_word = "PROFIT 🟢"
        elif pnl < 0:
            if reason and any(k in reason.upper() for k in ["SL", "STOP LOSS", "STOP_LOSS"]):
                result_word = "STOP LOSS HIT 🔴"
            else:
                result_word = "LOSS 🔴"
        else:
            result_word = "BREAK-EVEN ⚪"
    else:
        result_word = "CLOSED"
        
    # -- PRIVATE CHANNEL: Full result --
    if SIGNAL_CHANNEL_PRIVATE:
        _send_to_channel(SIGNAL_CHANNEL_PRIVATE, (
            f"TRADE CLOSED - {symbol}\n\n"
            f"Result: <b>{result_word}</b>\n"
            f"Strategy: <code>{strategy}</code>\n\n"
            f"Powered by NEXUS Core AI"
        ))
    
    # -- PUBLIC CHANNEL: Just the result (builds credibility) --
    if SIGNAL_CHANNEL_PUBLIC:
        cta = f"\n\nGet real-time signals: {PAYMENT_LINK}" if PAYMENT_LINK else ""
        
        _send_to_channel(SIGNAL_CHANNEL_PUBLIC, (
            f"CLOSED - {result_word}\n\n"
            f"Strategy: <code>{strategy}</code>\n"
            f"Result: <b>{result_word}</b>"
            f"{cta}\n\n"
            f"Powered by NEXUS Core AI"
        ))


def broadcast_trade_update(update_type, symbol, price_str):
    """
    Broadcasts mid-trade management updates to the VIP channel, and teaser updates to the Public channel.
    update_type: 'BE', 'PARTIAL', or 'TRAIL'
    """
    if not _broadcast_enabled:
        return
    # -- PRIVATE CHANNEL MESSAGE --
    if SIGNAL_CHANNEL_PRIVATE:
        priv_msg = ""
        if update_type == "BE":
            priv_msg = f"🛡 <b>UPDATE: {symbol}</b>\nStop Loss moved to Break-Even (Risk-Free) at {price_str}"
        elif update_type == "PARTIAL":
            priv_msg = f"💰 <b>UPDATE: {symbol}</b>\nSecured Partial Profits at {price_str}"
        elif update_type == "TRAIL":
            if price_str.startswith("TP_JUMP:"):
                target_price = price_str.split(":")[1]
                priv_msg = f"🚀 <b>UPDATE: {symbol}</b>\nTake Profit extended to structural Pivot point: <b>{target_price}</b> (Stop Loss locked in profit)"
            else:
                priv_msg = f"📈 <b>UPDATE: {symbol}</b>\nTrailing Stop Loss moved to {price_str}"
            
        if priv_msg:
            if isinstance(SIGNAL_CHANNEL_PRIVATE, list):
                for ch in SIGNAL_CHANNEL_PRIVATE:
                    _send_to_channel(ch, priv_msg)
            else:
                _send_to_channel(SIGNAL_CHANNEL_PRIVATE, priv_msg)
                
    # -- PUBLIC CHANNEL TEASER (FOMO) --
    if SIGNAL_CHANNEL_PUBLIC:
        pub_msg = ""
        cta = f"\n\nTrade like a VIP: {PAYMENT_LINK}" if PAYMENT_LINK else ""
        if update_type == "BE":
            pub_msg = f"🛡 <b>VIP UPDATE: {symbol}</b>\nThe AI just moved the VIP trade to Break-Even (Risk-Free)!{cta}"
        elif update_type == "PARTIAL":
            pub_msg = f"💰 <b>VIP UPDATE: {symbol}</b>\nThe AI just secured Partial Profits for VIP members!{cta}"
        elif update_type == "TRAIL" and price_str.startswith("TP_JUMP:"):
            pub_msg = f"🚀 <b>VIP UPDATE: {symbol}</b>\nThe AI just extended its Take Profit to the next structural pivot target to capture a massive run!{cta}"
            
        if pub_msg:
            _send_to_channel(SIGNAL_CHANNEL_PUBLIC, pub_msg)

def broadcast_retrain_success():
    """Broadcasts a hype message about AI model retraining to all channels."""
    msg = (
        "🧠 <b>NEXUS AI UPDATE</b>\n\n"
        "<i>System Maintenance Complete.</i>\n"
        "The Quantitative Neural Networks have successfully finished their weekly retraining cycle on the latest XAUUSD data.\n\n"
        "Trend Model: <b>Optimized</b> 🟢\n"
        "Reversal Model: <b>Calibrated</b> 🟢\n\n"
        "The AI is online and ready for the upcoming sessions."
    )
    
    if SIGNAL_CHANNEL_PRIVATE:
        if isinstance(SIGNAL_CHANNEL_PRIVATE, list):
            for ch in SIGNAL_CHANNEL_PRIVATE:
                _send_to_channel(ch, msg)
        else:
            _send_to_channel(SIGNAL_CHANNEL_PRIVATE, msg)
            
    if SIGNAL_CHANNEL_PUBLIC:
        cta = f"\n\nTrade alongside the AI: {PAYMENT_LINK}" if PAYMENT_LINK else ""
        _send_to_channel(SIGNAL_CHANNEL_PUBLIC, msg + cta)

def broadcast_weekly_review():
    """Calculate weekly stats and send separate reviews to public and private channels."""
    from datetime import datetime, timedelta
    
    # Calculate start of the week (Monday 00:00)
    now = datetime.now()
    monday = now - timedelta(days=now.weekday())
    monday_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        with _get_mt5_lock():
            deals = mt5.history_deals_get(monday_start, now)
    except Exception:
        deals = None
        
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    if deals:
        pos_magics = {
            getattr(d, 'position_id', 0): d.magic
            for d in deals
            if getattr(d, 'entry', None) == 0 and getattr(d, 'magic', 0) != 0
        }
        for d in deals:
            deal_magic = d.magic if d.magic != 0 else pos_magics.get(getattr(d, 'position_id', 0), 0)
            if deal_magic == MAGIC_NUMBER and d.entry == 1: # Exit deal
                pnl = d.profit + d.commission + d.swap + getattr(d, 'fee', 0.0)
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                    
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    roi_pct = 0.0
    try:
        with _get_mt5_lock():
            acc = mt5.account_info()
        if acc and total_pnl != 0:
            starting_bal = acc.balance - total_pnl
            if starting_bal > 0:
                roi_pct = (total_pnl / starting_bal) * 100
    except Exception:
        pass
    
    # -- PUBLIC CHANNEL MESSAGE --
    if SIGNAL_CHANNEL_PUBLIC:
        cta = f"\n\nUpgrade to Premium for next week's setups: {PAYMENT_LINK}" if PAYMENT_LINK else ""
        public_msg = (
            f"📊 <b>NEXUS AI - WEEKLY REVIEW</b>\n\n"
            f"Another week of automated trading completed. Here is how the AI performed:\n\n"
            f"Total Trades: <b>{total_trades}</b>\n"
            f"Win Rate: <b>{win_rate:.1f}%</b> ({wins}W / {losses}L)\n"
            f"Weekly Return: <b>{roi_pct:+.2f}%</b>\n\n"
            f"You received a few free signals this week. VIP members received all {total_trades} trades automatically."
            f"{cta}"
        )
        _send_to_channel(SIGNAL_CHANNEL_PUBLIC, public_msg)
        
    # -- PRIVATE CHANNEL MESSAGE --
    if SIGNAL_CHANNEL_PRIVATE:
        private_msg = (
            f"🏆 <b>VIP WEEKLY REVIEW</b>\n\n"
            f"Great week, Nexus Premium! Here are our final numbers:\n\n"
            f"Total Trades: <b>{total_trades}</b>\n"
            f"Win Rate: <b>{win_rate:.1f}%</b> ({wins}W / {losses}L)\n"
            f"Weekly Return: <b>{roi_pct:+.2f}%</b>\n\n"
            f"Rest up this weekend. The AI will resume scanning the markets on Sunday night! 🤖📈"
        )
        _send_to_channel(SIGNAL_CHANNEL_PRIVATE, private_msg)


def _weekly_scheduler_loop():
    """Background thread to trigger the weekly review automatically on Saturday morning."""
    from datetime import datetime
    last_run_week = -1
    
    while True:
        try:
            now = datetime.now()
            # Trigger on Saturday at 10:00 AM server time
            if now.weekday() == 5 and now.hour == 10:
                current_week = now.isocalendar()[1]
                if current_week != last_run_week:
                    broadcast_weekly_review()
                    last_run_week = current_week
        except Exception:
            pass
        time.sleep(60)
