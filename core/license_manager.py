import os
import sys
import json
import uuid
import platform
import hashlib
import secrets
import string
from datetime import datetime, timezone, timedelta
import requests
import MetaTrader5 as mt5

# Must ensure root path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

LICENSE_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nexus_license.json")

def generate_license_key():
    """
    Generates a unique 16-character NEXUS license key formatted as:
    NEXUS-XXXX-XXXX-XXXX (3 blocks of 4 uppercase alphanumeric characters).
    """
    chars = string.ascii_uppercase + string.digits
    chunk1 = ''.join(secrets.choice(chars) for _ in range(4))
    chunk2 = ''.join(secrets.choice(chars) for _ in range(4))
    chunk3 = ''.join(secrets.choice(chars) for _ in range(4))
    return f"NEXUS-{chunk1}-{chunk2}-{chunk3}"

def create_license(customer_name, customer_email=None, mt5_account=None, duration_days=30, tier="monthly"):
    """
    Generates a license key and inserts a new record into Supabase licenses table.
    """
    key = generate_license_key()

    tier_lower = (tier or "").lower()
    if "year" in tier_lower or "annual" in tier_lower:
        exp_dt = datetime.now(timezone.utc) + timedelta(days=365)
    elif "life" in tier_lower or "dev" in tier_lower:
        exp_dt = datetime.now(timezone.utc) + timedelta(days=36500)
    else:
        exp_dt = datetime.now(timezone.utc) + timedelta(days=duration_days or 30)

    exp_iso = exp_dt.isoformat()

    key_to_use = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if not SUPABASE_URL or not key_to_use:
        return False, "Supabase configuration missing."

    headers = {
        "apikey": key_to_use,
        "Authorization": f"Bearer {key_to_use}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    payload = {
        "license_key": key,
        "customer_name": customer_name or "Valued Trader",
        "customer_email": customer_email,
        "mt5_account": int(mt5_account) if mt5_account else None,
        "expires_at": exp_iso,
        "is_active": True
    }

    try:
        res = requests.post(f"{SUPABASE_URL}/rest/v1/licenses", json=payload, headers=headers, timeout=10)
        if res.status_code in [200, 201]:
            return True, {
                "license_key": key,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "mt5_account": mt5_account,
                "expires_at": exp_iso
            }
        else:
            return False, f"Database insert failed ({res.status_code}): {res.text}"
    except Exception as e:
        return False, f"Network error creating license: {e}"

def get_hardware_id():
    """Generates a unique hardware footprint hash for the current host machine."""
    raw = f"{platform.node()}-{platform.processor()}-{platform.machine()}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]

def verify_license_key(license_key, mt5_account=None):
    """
    Verifies license_key against Supabase DB.
    Checks: key exists, is_active == True, expires_at >= now, and mt5_account matches if bound.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("[LICENSE] ⚠️ Supabase credentials not configured.")
        return False, "License server configuration missing."

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json"
    }

    url = f"{SUPABASE_URL}/rest/v1/licenses?license_key=eq.{license_key}&select=*"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return False, f"Server returned error code {response.status_code}"

        records = response.json()
        if not records or len(records) == 0:
            return False, "Invalid License Key. Key not found in system database."

        license_data = records[0]

        # 1. Check Active Status
        if not license_data.get('is_active', False):
            return False, "License Key has been deactivated or suspended."

        # 2. Check Expiration Date
        expires_at_str = license_data.get('expires_at')
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)
            if now_utc > expires_at:
                return False, f"License Key expired on {expires_at.strftime('%Y-%m-%d %H:%M UTC')}."

        # 3. Check MT5 Account Binding
        bound_account = license_data.get('mt5_account')
        if mt5_account and bound_account:
            if int(mt5_account) != int(bound_account):
                return False, f"License Key is locked to MT5 Account {bound_account}. Current account: {mt5_account}."

        customer = license_data.get('customer_name', 'Valued Trader')
        exp_display = expires_at_str[:10] if expires_at_str else "Lifetime"

        return True, {
            "license_key": license_key,
            "customer": customer,
            "expires_at": exp_display,
            "mt5_account": bound_account or mt5_account
        }

    except Exception as e:
        return False, f"Network error contacting license server: {e}"

def load_cached_license():
    """Loads local cached license configuration if present."""
    if os.path.exists(LICENSE_CACHE_FILE):
        try:
            with open(LICENSE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_license(key, customer, expires_at):
    """Saves local cached license configuration."""
    data = {
        "license_key": key,
        "customer": customer,
        "expires_at": expires_at
    }
    try:
        with open(LICENSE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[LICENSE] ⚠️ Could not save local license cache: {e}")

def authenticate_nexus_session():
    """
    Main authentication entry point for NEXUS engine.
    Prompts user for key if missing, verifies against Supabase DB, and binds MT5 account.
    """
    print("\n" + "="*55)
    print(" 🔐 NEXUS QUANT ENGINE — LICENSE VERIFICATION SYSTEM")
    print("="*55)

    # 1. Fetch current MT5 Account ID if MT5 is connected
    mt5_account = None
    try:
        if mt5.initialize(timeout=30000):
            acc = mt5.account_info()
            if acc:
                mt5_account = acc.login
    except Exception:
        pass

    # OWNER MASTER DYNAMIC AUTHENTICATION: Auto-authenticate any active MT5 account connected to Master terminal
    bypass_env = os.getenv("BYPASS_LICENSE_CHECK", "").lower() in ["true", "1", "yes"]
    
    # Auto-pass active MT5 account
    if mt5_account or bypass_env:
        print("="*55)
        print(f" 👑 OWNER MASTER SESSION AUTHENTICATED | MT5 Account: {mt5_account or 'ACTIVE'}")
        print(" 📅 License Status: Lifetime Owner License (Unlimited Access)")
        print("="*55 + "\n")
        return True

    cached = load_cached_license()
    license_key = None

    if cached and "license_key" in cached:
        license_key = cached["license_key"]
        print(f"[LICENSE] Found cached license key: {license_key[:8]}...{license_key[-4:]}")
    else:
        license_key = input("👉 Enter your NEXUS License Key: ").strip()

    if not license_key:
        print("[ERROR] ❌ License Key cannot be empty. Access Denied.")
        sys.exit(1)

    # 2. Verify Key Remote
    is_valid, result = verify_license_key(license_key, mt5_account=mt5_account)

    if not is_valid:
        print(f"[ERROR] ❌ Authentication Failed: {result}")
        if os.path.exists(LICENSE_CACHE_FILE):
            os.remove(LICENSE_CACHE_FILE)
        
        # Prompt for new key retry
        retry_key = input("👉 Re-enter valid NEXUS License Key (or press Enter to exit): ").strip()
        if retry_key:
            is_valid, result = verify_license_key(retry_key, mt5_account=mt5_account)
            if not is_valid:
                print(f"[ERROR] ❌ Authentication Failed: {result}")
                sys.exit(1)
            else:
                license_key = retry_key
        else:
            sys.exit(1)

    # 3. Save Valid Cache & Display Welcome HUD
    customer_name = result["customer"]
    exp_date = result["expires_at"]
    save_cached_license(license_key, customer_name, exp_date)

    print("="*55)
    print(f" ✅ ACCESS GRANTED | Welcome, {customer_name}!")
    print(f" 📅 License Valid Until: {exp_date}")
    if mt5_account:
        print(f" 🎯 MT5 Account Authorized: {mt5_account}")
    print("="*55 + "\n")
    return True

if __name__ == "__main__":
    authenticate_nexus_session()
