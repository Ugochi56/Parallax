import requests
import threading
import json
import os
import time
import tempfile
from datetime import datetime, timedelta

# ==========================================
# 💳 CRYPTOBOT SUBSCRIPTION MANAGER
# ==========================================
# Handles automated crypto payments via CryptoBot Crypto Pay API.
# Creates invoices, tracks payments, auto-invites to private channel.

CRYPTOBOT_API_TOKEN = ""
SUBSCRIPTION_PRICE = 29.0       # USD equivalent in USDT
SUBSCRIPTION_DAYS = 30           # Duration per payment
WEEKLY_SUBSCRIPTION_PRICE = 9.0  # USD equivalent in USDT
WEEKLY_SUBSCRIPTION_DAYS = 7     # Duration per payment
SIGNAL_CHANNEL_PRIVATE = ""

TELEGRAM_CHAT_ID = ""

try:
    from engine.telegram_credentials import (
        CRYPTOBOT_API_TOKEN, SUBSCRIPTION_PRICE, SUBSCRIPTION_DAYS,
        WEEKLY_SUBSCRIPTION_PRICE, WEEKLY_SUBSCRIPTION_DAYS,
        SIGNAL_CHANNEL_PRIVATE, TELEGRAM_CHAT_ID
    )
except (ImportError, AttributeError):
    pass

_CRYPTO_API = "https://pay.crypt.bot/api"
_HEADERS = {"Crypto-Pay-API-Token": CRYPTOBOT_API_TOKEN}

# Subscriber database file
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SUBS_FILE = os.path.join(_DATA_DIR, "subscribers.json")
_PENDING_FILE = os.path.join(_DATA_DIR, "pending_invoices.json")


_json_lock = threading.RLock()

def _load_json(path):
    with _json_lock:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


def _save_json(path, data):
    with _json_lock:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8')
        try:
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            temp_file.close()
            os.replace(temp_file.name, path)
        except Exception:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
            raise


def create_invoice(user_chat_id, username="", plan="monthly"):
    """Create a CryptoBot USDT invoice and return the pay URL."""
    if not CRYPTOBOT_API_TOKEN:
        return None, "Subscription system not configured."
        
    if plan == "weekly":
        amount = WEEKLY_SUBSCRIPTION_PRICE
        duration_days = WEEKLY_SUBSCRIPTION_DAYS
        description = f"NEXUS Gold Signals - {WEEKLY_SUBSCRIPTION_DAYS} Day Access"
    else:
        amount = SUBSCRIPTION_PRICE
        duration_days = SUBSCRIPTION_DAYS
        description = f"NEXUS Gold Signals - {SUBSCRIPTION_DAYS} Day Access"
        
    # Check for existing pending invoice to prevent spam
    with _json_lock:
        pending = _load_json(_PENDING_FILE)
        for inv_id, inv_data in pending.items():
            if inv_data.get("chat_id") == str(user_chat_id) and inv_data.get("duration_days") == duration_days:
                try:
                    created = datetime.fromisoformat(str(inv_data.get("created", "")))
                except Exception:
                    continue
                if (datetime.now() - created).total_seconds() < 3600:
                    pay_url = inv_data.get("pay_url")
                    if pay_url:
                        return pay_url, None
    
    try:
        response = requests.post(f"{_CRYPTO_API}/createInvoice", headers=_HEADERS, json={
            "currency_type": "crypto",
            "asset": "USDT",
            "amount": str(amount),
            "description": description,
            "payload": json.dumps({"chat_id": user_chat_id, "username": username, "duration_days": duration_days}),
            "expires_in": 3600  # Invoice expires in 1 hour
        }, timeout=10)
        
        data = response.json()
        
        if data.get("ok"):
            invoice = data["result"]
            invoice_id = str(invoice["invoice_id"])
            pay_url = invoice.get("mini_app_invoice_url") or invoice.get("pay_url", "")
            
            # Save pending invoice
            with _json_lock:
                pending = _load_json(_PENDING_FILE)
                pending[invoice_id] = {
                    "chat_id": str(user_chat_id),
                    "username": username,
                    "amount": amount,
                    "duration_days": duration_days,
                    "created": datetime.now().isoformat(),
                    "status": "pending",
                    "pay_url": pay_url
                }
                _save_json(_PENDING_FILE, pending)
            
            return pay_url, None
        else:
            return None, f"API Error: {data.get('error', {}).get('name', 'Unknown')}"
    
    except Exception as e:
        return None, f"Connection error: {e}"


def check_pending_invoices(bot_token, private_channel_id):
    """Check all pending invoices and process any that have been paid."""
    if not CRYPTOBOT_API_TOKEN:
        return
    
    with _json_lock:
        pending = _load_json(_PENDING_FILE)
    
    if not pending:
        return
    
    invoice_ids = list(pending.keys())
    
    # Chunk invoice IDs to avoid URI Too Long errors (DoS protection)
    chunk_size = 50
    for i in range(0, len(invoice_ids), chunk_size):
        chunk = invoice_ids[i:i + chunk_size]
        
        try:
            response = requests.get(f"{_CRYPTO_API}/getInvoices", headers=_HEADERS, params={
                "invoice_ids": ",".join(chunk)
            }, timeout=10)
            
            data = response.json()
            
            if not data.get("ok"):
                continue
            
            for invoice in data["result"]["items"]:
                inv_id = str(invoice["invoice_id"])
                status = invoice.get("status", "")
                
                if (status == "paid" or status == "expired"):
                    with _json_lock:
                        # Re-load pending to avoid overwriting updates from other threads
                        current_pending = _load_json(_PENDING_FILE)
                        if inv_id in current_pending:
                            if status == "paid":
                                chat_id = current_pending[inv_id]["chat_id"]
                                username = current_pending[inv_id].get("username", "")
                                duration_days = current_pending[inv_id].get("duration_days", SUBSCRIPTION_DAYS)
                                
                                # Activate subscription
                                _activate_subscriber(chat_id, username, bot_token, private_channel_id, duration_days)
                            
                            # Re-load again, delete, and save
                            current_pending = _load_json(_PENDING_FILE)
                            if inv_id in current_pending:
                                del current_pending[inv_id]
                                _save_json(_PENDING_FILE, current_pending)
        
        except Exception:
            pass


def _activate_subscriber(chat_id, username, bot_token, private_channel_id, duration_days=30):
    """Add a subscriber and send them a private channel invite link."""
    # Save to subscribers database
    with _json_lock:
        subs = _load_json(_SUBS_FILE)
        current_expiry = None
        if str(chat_id) in subs and subs[str(chat_id)].get("active", True):
            try:
                current_expiry = datetime.fromisoformat(subs[str(chat_id)]["expires"])
            except Exception:
                pass
                
        if current_expiry and current_expiry > datetime.now():
            expiry = (current_expiry + timedelta(days=duration_days)).isoformat()
        else:
            expiry = (datetime.now() + timedelta(days=duration_days)).isoformat()
            
        subs[str(chat_id)] = {
            "username": username,
            "subscribed": datetime.now().isoformat(),
            "expires": expiry,
            "active": True
        }
        _save_json(_SUBS_FILE, subs)
    
    # Generate one-time invite link for private channel
    try:
        channels = private_channel_id if isinstance(private_channel_id, list) else [private_channel_id]
        invite_links = []
        for ch in channels:
            invite_response = requests.post(
                f"https://api.telegram.org/bot{bot_token}/createChatInviteLink",
                json={
                    "chat_id": ch,
                    "member_limit": 1,
                    "name": f"NEXUS-{chat_id}",
                    "expire_date": int(time.time()) + 86400  # Link expires in 24 hours
                },
                timeout=5
            )
            invite_data = invite_response.json()
            invite_link = invite_data.get("result", {}).get("invite_link", "")
            if invite_link:
                invite_links.append(invite_link)
                
        if not invite_links:
            raise Exception("No invite links generated")
            
        # Send welcome message with invite link(s)
        expires_str = datetime.fromisoformat(expiry).strftime("%B %d, %Y")
        links_text = f"Join your private channel here:\n{invite_links[0]}" if len(invite_links) == 1 else "Join your private channels here:\n" + "\n".join(invite_links)
        
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={
            "chat_id": chat_id,
            "text": (
                f"<b>Payment Confirmed!</b>\n\n"
                f"Welcome to NEXUS Gold Signals Premium!\n\n"
                f"{links_text}\n\n"
                f"Your access expires: <b>{expires_str}</b>\n"
                f"You will be notified before expiry to renew."
            ),
            "parse_mode": "HTML"
        }, timeout=5)
    
    except Exception:
        # Fallback: notify owner to manually add user (wrapped in try-except to prevent crashes)
        try:
            requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Payment confirmed! Please contact @NexusQuantSupport for your channel access.",
                "parse_mode": "HTML"
            }, timeout=5)
        except Exception:
            pass


def check_subscription(chat_id):
    """Check if a user has an active subscription."""
    if TELEGRAM_CHAT_ID and str(chat_id) == str(TELEGRAM_CHAT_ID):
        return True, "👑 Lifetime Owner Access enabled."
        
    with _json_lock:
        subs = _load_json(_SUBS_FILE)
    sub = subs.get(str(chat_id))
    
    if not sub:
        return False, "No subscription found."
    
    try:
        expiry = datetime.fromisoformat(str(sub.get("expires", "")))
    except Exception:
        return False, "Invalid subscription expiry date."
    if datetime.now() > expiry:
        return False, f"Your subscription expired on {expiry.strftime('%B %d, %Y')}. Type /subscribe to renew."
    
    days_left = (expiry - datetime.now()).days
    return True, f"Active! Expires {expiry.strftime('%B %d, %Y')} ({days_left} days remaining)."


def get_subscriber_count():
    """Get total active subscriber count."""
    with _json_lock:
        subs = _load_json(_SUBS_FILE)
    active = 0
    for chat_id, sub in subs.items():
        try:
            expiry = datetime.fromisoformat(str(sub.get("expires", "")))
        except Exception:
            continue
        if datetime.now() < expiry:
            active += 1
    return active


def check_expiring_subscribers(bot_token):
    """Notify subscribers whose access expires in 3 days or less."""
    with _json_lock:
        subs = _load_json(_SUBS_FILE)
    
    for chat_id, sub in subs.items():
        try:
            expiry = datetime.fromisoformat(str(sub.get("expires", "")))
        except Exception:
            continue
        rem_sec = (expiry - datetime.now()).total_seconds()
        days_left = (expiry - datetime.now()).days
        
        if rem_sec > 0 and 0 <= days_left <= 3 and sub.get("active", True):
            today_str = datetime.now().date().isoformat()
            if sub.get("last_expiry_notification") == today_str:
                continue
                
            try:
                response = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": (
                        f"<b>Subscription Expiring Soon!</b>\n\n"
                        f"Your NEXUS Premium access expires in <b>{days_left} day(s)</b>.\n"
                        f"Type /subscribe to renew your subscription."
                    ),
                    "parse_mode": "HTML"
                }, timeout=5)
                
                if response.status_code == 200:
                    with _json_lock:
                        # Re-load to avoid race condition overwrite
                        current_subs = _load_json(_SUBS_FILE)
                        if chat_id in current_subs:
                            current_subs[chat_id]["last_expiry_notification"] = today_str
                            _save_json(_SUBS_FILE, current_subs)
            except Exception:
                pass


def kick_expired_subscribers(bot_token, private_channel_id):
    """Kick users whose subscription has expired and unban them to allow rejoins."""
    with _json_lock:
        subs = _load_json(_SUBS_FILE)
    
    channels = private_channel_id if isinstance(private_channel_id, list) else [private_channel_id]
    
    for chat_id, sub in list(subs.items()):
        if not sub.get("active", True):
            continue
            
        try:
            expiry = datetime.fromisoformat(str(sub.get("expires", "")))
        except Exception:
            continue
        if datetime.now() > expiry:
            # Subscription is expired! Kick them!
            try:
                for ch in channels:
                    # Ban removes them from the channel
                    requests.post(f"https://api.telegram.org/bot{bot_token}/banChatMember", json={
                        "chat_id": ch,
                        "user_id": int(chat_id),
                        "revoke_messages": False
                    }, timeout=5)
                    
                    # Unban immediately so they can rejoin via a new link later
                    requests.post(f"https://api.telegram.org/bot{bot_token}/unbanChatMember", json={
                        "chat_id": ch,
                        "user_id": int(chat_id),
                        "only_if_banned": True
                    }, timeout=5)
                
                # Notify the user
                requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": "🔴 <b>Subscription Expired</b>\n\nYour NEXUS Premium access has ended and you have been removed from the private channel. Type /subscribe to renew and rejoin at any time!",
                    "parse_mode": "HTML"
                }, timeout=5)
                
                with _json_lock:
                    current_subs = _load_json(_SUBS_FILE)
                    if chat_id in current_subs:
                        current_subs[chat_id]["active"] = False
                        _save_json(_SUBS_FILE, current_subs)
            except Exception:
                pass


def _payment_monitor(bot_token, private_channel_id):
    """Background thread that monitors pending payments every 30 seconds."""
    while True:
        try:
            check_pending_invoices(bot_token, private_channel_id)
            check_expiring_subscribers(bot_token)
            kick_expired_subscribers(bot_token, private_channel_id)
        except Exception:
            pass
        time.sleep(30)


def start_payment_monitor(bot_token, private_channel_id):
    """Launch the payment monitor as a daemon thread."""
    monitor = threading.Thread(
        target=_payment_monitor,
        args=(bot_token, private_channel_id),
        daemon=True
    )
    monitor.start()
