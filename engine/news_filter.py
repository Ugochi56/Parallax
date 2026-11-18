import os
import sys
import json
import requests
import time
from datetime import datetime, timezone, timedelta
from core.config import *

daily_news_data = []
last_news_fetch_day = ""
api_failed_lockdown = False
last_fetch_attempt_time = 0
active_blackout_info = None

def get_active_blackout():
    """Returns a dict describing the event that triggered the current blackout, or None."""
    return active_blackout_info

def fetch_economic_news():
    global daily_news_data, last_news_fetch_day, api_failed_lockdown, last_fetch_attempt_time
    curr_day = datetime.now().strftime("%Y-%m-%d")
    
    if curr_day == last_news_fetch_day: 
        return
        
    # Rate limit fetch attempts on failure (retry every 10 minutes)
    current_time = time.time()
    if api_failed_lockdown and (current_time - last_fetch_attempt_time < 600):
        return
        
    last_fetch_attempt_time = current_time
    _ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_file = os.path.join(_ROOT_DIR, "data", "news_cache.json")
    
    # Check local cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if cached.get("System Cache Date") == curr_day:
                daily_news_data = cached.get("Macroeconomic Calendar", [])
                last_news_fetch_day = curr_day
                api_failed_lockdown = False
                print("\n[CACHE] Loaded ForexFactory Calendar from local cache.")
                return
        except Exception: 
            pass
    
    print(f"\n[NEWS] Fetching ForexFactory Macro Calendar from Database API...")
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        import random
        time.sleep(random.uniform(1.5, 3.0))
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        
        raw_data = response.json()
        
        # Completely restructure json to layman terms
        color_map = {
            "High": "🔴 High Threat Level (Red Folder)",
            "Medium": "🟠 Medium Threat Level (Orange Folder)",
            "Low": "Low Threat Level",
            "Non": "⚪ Non-Economic"
        }
        
        formatted_data = []
        for e in raw_data:
            orig_impact = e.get("impact", "")
            
            try:
                dt = datetime.fromisoformat(e.get('date', '').replace('Z', '+00:00'))
                human_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except Exception:
                human_time = e.get('date', 'Unknown')
                
            readable_event = {
                "Event Name": e.get("title", "Unknown"),
                "Currency Affected": e.get("country", "Unknown"),
                "Date and Time": human_time,
                "Danger Level": color_map.get(orig_impact, orig_impact),
                "Expected Result (Forecast)": e.get("forecast", ""),
                "Previous Result": e.get("previous", ""),
                "_bot_raw_date": e.get("date", ""),
                "_bot_raw_impact": orig_impact
            }
            formatted_data.append(readable_event)
            
        daily_news_data = formatted_data
        last_news_fetch_day = curr_day
        api_failed_lockdown = False
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "System Cache Date": curr_day, 
                "Notice": "This file contains the macroeconomic outlook for the entire trading week.",
                "Macroeconomic Calendar": daily_news_data
            }, f, indent=4)
            
        print(f"[SUCCESS] News Database Fetched & Cached Successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch news API: {e}")
        # Fallback to whatever news cache is on disk (even if stale) to prevent blocking bot trading
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                daily_news_data = cached.get("Macroeconomic Calendar", [])
                last_news_fetch_day = curr_day
                api_failed_lockdown = True
                print(f"[WARNING] Using stale news cache fallback due to API fetch failure.")
                return
            except Exception:
                pass
        
        # If cache load also fails or is empty, we must lockdown to prevent trading blindly into news
        api_failed_lockdown = True

import pandas as pd

def _analyze_news_candle(df_m5, event_time_utc):
    """Analyzes the M5 candle closest to event_time_utc for displacement range and body ratio."""
    if df_m5 is None or len(df_m5) == 0:
        return None
    try:
        times = pd.to_datetime(df_m5['time'], utc=True)
        diffs = (times - event_time_utc).abs()
        nearest_idx = diffs.idxmin()
        if diffs.loc[nearest_idx].total_seconds() > 600:
            return None
        bar = df_m5.loc[nearest_idx]
        rng = float(bar['high'] - bar['low'])
        body = abs(float(bar['close'] - bar['open']))
        ratio = body / rng if rng > 0 else 0.0
        is_bear = float(bar['close']) < float(bar['open'])
        return {
            'bar': bar,
            'time': bar['time'],
            'range': rng,
            'body': body,
            'ratio': ratio,
            'is_bear': is_bear,
            'direction': 'SELL' if is_bear else 'BUY'
        }
    except Exception:
        return None

def get_post_news_guard_status(df_m5=None):
    """
    Evaluates whether the market is currently inside the 45-minute post-news guard window
    following a Tier 1 USD macro event, and computes directional bias (Guard 2).
    """
    if not NEWS_KILL_SWITCH:
        return {'in_post_window': False, 'directional_bias': None, 'minutes_elapsed': 0.0, 'title': ''}

    now_utc = datetime.now(timezone.utc)
    config_mod = sys.modules.get('core.config')
    post_window_s = getattr(config_mod, 'NEWS_POST_WINDOW_MINUTES', 45) * 60
    high_post_s = getattr(config_mod, 'NEWS_HIGH_IMPACT_POST_MINUTES', 15) * 60
    min_rng = getattr(config_mod, 'NEWS_MIN_IMPULSE_RANGE', 8.00)
    min_ratio = getattr(config_mod, 'NEWS_MIN_BODY_RATIO', 0.45)
    high_vol_keywords = ['claims', 'cpi', 'ppi', 'pce', 'payroll', 'fomc', 'fed', 'gdp', 'retail sales', 'interest rate', 'inflation']

    for event in daily_news_data:
        impact = event.get('_bot_raw_impact')
        currency = event.get('Currency Affected')
        event_name = (event.get('Event Name') or '').lower()
        is_key_usd = (currency == 'USD' and (impact == 'High' or any(k in event_name for k in high_vol_keywords)))
        if not is_key_usd:
            continue

        try:
            raw_dt_str = event.get('_bot_raw_date', '').replace('Z', '+00:00')
            dt = datetime.fromisoformat(raw_dt_str)
            event_time_utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
            time_diff = (now_utc - event_time_utc).total_seconds()

            # Active in the post-release window (between 0 and 45 minutes)
            if 0 <= time_diff <= post_window_s:
                minutes_elapsed = round(time_diff / 60.0, 1)
                bias = None
                candle_info = _analyze_news_candle(df_m5, event_time_utc)
                if candle_info and candle_info['range'] >= min_rng and candle_info['ratio'] >= min_ratio:
                    bias = candle_info['direction']

                return {
                    'in_post_window': True,
                    'minutes_elapsed': minutes_elapsed,
                    'event_time': event_time_utc,
                    'title': event.get('Event Name', 'USD High Impact'),
                    'directional_bias': bias,
                    'candle_info': candle_info
                }
        except Exception:
            pass

    return {'in_post_window': False, 'directional_bias': None, 'minutes_elapsed': 0.0, 'title': ''}

def is_news_blackout(df_m5=None):
    """
    Tiered XAUUSD news blackout with Option A Anti-Whipsaw Guard.
    Tier 1 (USD high impact):
        NEWS_HIGH_IMPACT_PRE/POST_MINUTES (default 15/15).
        Guard 3 (Anti-Whipsaw Filter): If 15-45m post-release and news candle had
        range >= $10 with body < 45% (two-way whip/doji), blackout is extended to 45m.
    Tier 2 (USD medium impact & speeches): 10m pre / 15m post.
    Tier 3 (non-USD high impact): 10m pre / 10m post.
    """
    global active_blackout_info
    if not NEWS_KILL_SWITCH:
        active_blackout_info = None
        return False
    if api_failed_lockdown:
        active_blackout_info = {
            'title': 'News API unavailable (safety lockdown)',
            'currency': 'ALL',
            'tier': 'LOCKDOWN',
            'start': None,
            'end': None
        }
        return True

    now_utc = datetime.now(timezone.utc)
    non_usd_majors = ['EUR', 'GBP', 'AUD', 'CAD', 'CHF', 'JPY', 'NZD', 'CNY', 'ALL']
    high_vol_keywords = ['claims', 'cpi', 'ppi', 'pce', 'payroll', 'fomc', 'fed', 'gdp', 'retail sales', 'interest rate', 'inflation']
    speech_keywords = ['speaks', 'testifies', 'speech', 'press conference']

    config_mod = sys.modules.get('core.config')
    high_pre = getattr(config_mod, 'NEWS_HIGH_IMPACT_PRE_MINUTES', 15) * 60
    high_post = getattr(config_mod, 'NEWS_HIGH_IMPACT_POST_MINUTES', 15) * 60
    post_window_s = getattr(config_mod, 'NEWS_POST_WINDOW_MINUTES', 45) * 60
    min_ratio = getattr(config_mod, 'NEWS_MIN_BODY_RATIO', 0.45)
    med_pre = getattr(config_mod, 'NEWS_MEDIUM_IMPACT_PRE_MINUTES', 10) * 60
    med_post = getattr(config_mod, 'NEWS_MEDIUM_IMPACT_POST_MINUTES', 15) * 60
    nonusd_pre = getattr(config_mod, 'NEWS_NON_USD_HIGH_PRE_MINUTES', 10) * 60
    nonusd_post = getattr(config_mod, 'NEWS_NON_USD_HIGH_POST_MINUTES', 10) * 60

    for event in daily_news_data:
        impact = event.get('_bot_raw_impact')
        currency = event.get('Currency Affected')
        event_name = (event.get('Event Name') or '').lower()

        is_speech = any(k in event_name for k in speech_keywords)
        is_key_usd_release = (currency == 'USD' and any(k in event_name for k in high_vol_keywords))

        tier = None
        if currency == 'USD':
            if is_speech:
                if 'press conference' in event_name and impact == 'High':
                    tier = 'USD-HIGH'
                elif impact in ('High', 'Medium'):
                    tier = 'USD-SPEECH'
            elif impact == 'High' or is_key_usd_release:
                tier = 'USD-HIGH'
            elif impact == 'Medium':
                tier = 'USD-MEDIUM'
        elif currency in non_usd_majors and impact == 'High':
            tier = 'NON-USD-HIGH'

        if tier is None:
            continue

        if tier == 'USD-HIGH':
            pre_s, post_s = high_pre, high_post
        elif tier in ('USD-MEDIUM', 'USD-SPEECH'):
            pre_s, post_s = med_pre, med_post
        else:
            pre_s, post_s = nonusd_pre, nonusd_post

        try:
            raw_dt_str = event.get('_bot_raw_date', '').replace('Z', '+00:00')
            dt = datetime.fromisoformat(raw_dt_str)
            event_time_utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
            time_diff = (now_utc - event_time_utc).total_seconds()

            # Primary Blackout Window (-pre_s <= time_diff <= post_s)
            if -pre_s <= time_diff <= post_s:
                active_blackout_info = {
                    'title': event.get('Event Name', 'Unknown'),
                    'currency': currency,
                    'tier': tier,
                    'event_time': event_time_utc,
                    'start': event_time_utc - timedelta(seconds=pre_s),
                    'end': event_time_utc + timedelta(seconds=post_s),
                }
                return True

            # Guard 3: Anti-Whipsaw Filter (Extend blackout up to 45m if news bar is a violent two-way doji/whip)
            if tier == 'USD-HIGH' and post_s < time_diff <= post_window_s and df_m5 is not None:
                candle_info = _analyze_news_candle(df_m5, event_time_utc)
                if candle_info and candle_info['range'] >= 10.0 and candle_info['ratio'] < min_ratio:
                    active_blackout_info = {
                        'title': f"{event.get('Event Name', 'USD High Impact')} [WHIPSAW GUARD: Body {candle_info['ratio']*100:.0f}% < 45%]",
                        'currency': currency,
                        'tier': 'USD-HIGH-WHIPSAW',
                        'event_time': event_time_utc,
                        'start': event_time_utc - timedelta(seconds=pre_s),
                        'end': event_time_utc + timedelta(seconds=post_window_s),
                    }
                    return True
        except Exception:
            pass

    active_blackout_info = None
    return False
