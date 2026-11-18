# ==========================================
# TELEGRAM CREDENTIALS (Loaded from .env)
# ==========================================
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; fall back to system env vars

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Signal Channels
SIGNAL_CHANNEL_PRIVATE = os.getenv("SIGNAL_CHANNEL_PRIVATE", "")
SIGNAL_CHANNEL_PUBLIC = os.getenv("SIGNAL_CHANNEL_PUBLIC", "")

# Call-to-action link on public channel signals (points to bot for payment)
PAYMENT_LINK = os.getenv("PAYMENT_LINK", "https://t.me/NexusQuantBot")

# How many teaser signals per day on the public channel
PUBLIC_SIGNALS_PER_DAY = int(os.getenv("PUBLIC_SIGNALS_PER_DAY", "5"))

# CryptoBot Crypto Pay API
CRYPTOBOT_API_TOKEN = os.getenv("CRYPTOBOT_API_TOKEN", "")
SUBSCRIPTION_PRICE = float(os.getenv("SUBSCRIPTION_PRICE", "29.0"))
SUBSCRIPTION_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))

WEEKLY_SUBSCRIPTION_PRICE = float(os.getenv("WEEKLY_SUBSCRIPTION_PRICE", "9.0"))
WEEKLY_SUBSCRIPTION_DAYS = 7
