"""
Helpful utilities for working with datetime values.
"""
import datetime
from enum import Enum, auto

class DatetimeRounding(Enum):
    """An enumeration for types of datetime rounding"""
    NEAREST_HOUR = auto()
    NEAREST_MINUTE = auto()

def round(dt, rounding=None):
    """Round a datetime value using specified rounding method.

    Args:
        dt: `datetime` value to be rounded.
        rounding: `DatetimeRounding` value representing rounding method.
    """
    if rounding is DatetimeRounding.NEAREST_HOUR:
        return round_to_hour(dt)
    elif rounding is DatetimeRounding.NEAREST_MINUTE:
        return round_to_minute(dt)
    else:
        return dt

def round_to_hour(dt):
    """Round `datetime` value to nearest hour.

    Args:
        dt: `datetime` value to be rounded.

    Returns:
        Rounded `datetime` value.
    """
    base_dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, 0, 0)
    if dt.minute >= 30:
        return base_dt + datetime.timedelta(hours=1)
    return base_dt

def round_to_minute(dt):
    """Round `datetime` value to nearest minute.

    Args:
        dt: `datetime` value to be rounded.

    Returns:
        Rounded `datetime` value.
    """
    base_dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0)
    if dt.second >= 30:
        return base_dt + datetime.timedelta(minutes=1)
    return base_dt

