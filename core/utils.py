from datetime import datetime, timedelta, timezone
from core.config import *

def is_broker_dst(date_obj):
    """Calculates if the given UTC date is inside European Broker Daylight Saving Time (last Sunday of March to last Sunday of October)."""
    if not AUTO_DST_ADJUST:
        return False
    year = date_obj.year
    # Last Sunday of March
    march_last = datetime(year, 3, 31, 1, 0, 0)
    march_last = march_last - timedelta(days=(march_last.weekday() + 1) % 7)
    # Last Sunday of October
    oct_last = datetime(year, 10, 31, 1, 0, 0)
    oct_last = oct_last - timedelta(days=(oct_last.weekday() + 1) % 7)
    
    naive_date = date_obj.replace(tzinfo=None)
    return march_last <= naive_date < oct_last

def get_broker_hour_from_utc(utc_time):
    """
    Converts a raw UTC datetime into a static 'New York Aligned' Broker Hour.
    New York Close (5:00 PM NY) is universally anchored to Hour 0 (Midnight).
    """
    utc_hour = utc_time.hour
    if is_broker_dst(utc_time):
        # Summer (EEST / UTC+3): To map UTC 21:00 (5pm NY) to Hour 0, we add 3
        return (utc_hour + 3) % 24
    else:
        # Winter (EET / UTC+2): To map UTC 22:00 (5pm NY) to Hour 0, we add 2
        return (utc_hour + 2) % 24

def get_session_name(time_or_hour):
    """
    Returns the Institutional Forex Market Session dynamically adjusted for DST.
    Accepts either a datetime object (UTC) or an integer hour (UTC).
    
    True Global Market Schedule (UTC):
    - Summer (DST / Current):
        * ASIAN:   00:00 - 07:00 UTC (01:00 - 08:00 BST)
        * LONDON:  07:00 - 12:00 UTC (08:00 - 13:00 BST)
        * OVERLAP: 12:00 - 16:30 UTC (13:00 - 17:30 BST / 08:00 - 12:30 EDT)
        * NY:      16:30 - 21:00 UTC (12:30 - 17:00 EDT)
        * ROLL:    21:00 - 23:59 UTC (NY Close / High Spreads)
    - Winter (Standard):
        * ASIAN:   00:00 - 08:00 UTC (00:00 - 08:00 GMT)
        * LONDON:  08:00 - 13:00 UTC (08:00 - 13:00 GMT)
        * OVERLAP: 13:00 - 17:30 UTC (13:00 - 17:30 GMT / 08:00 - 12:30 EST)
        * NY:      17:30 - 22:00 UTC (12:30 - 17:00 EST)
        * ROLL:    22:00 - 23:59 UTC (NY Close / High Spreads)
    """
    now_dt = time_or_hour if isinstance(time_or_hour, datetime) else datetime.now(timezone.utc)
    in_dst = is_broker_dst(now_dt) if AUTO_DST_ADJUST else True
    
    ldn_open = 7 if in_dst else 8
    overlap_open = 12 if in_dst else 13
    ny_close = 21 if in_dst else 22
    
    if isinstance(time_or_hour, datetime):
        utc_minutes = time_or_hour.hour * 60 + time_or_hour.minute
    else:
        hour = int(time_or_hour)
        utc_minutes = hour * 60
        
    if 0 <= utc_minutes < ldn_open * 60:
        return "ASIAN"
    elif ldn_open * 60 <= utc_minutes < overlap_open * 60:
        return "LONDON"
    elif overlap_open * 60 <= utc_minutes < int((overlap_open + 4.5) * 60):
        return "NY_LONDON_OVERLAP"
    elif int((overlap_open + 4.5) * 60) <= utc_minutes < ny_close * 60:
        return "NEW_YORK"
    else:
        return "ROLLOVER"
