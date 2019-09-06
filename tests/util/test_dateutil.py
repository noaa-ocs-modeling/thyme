import datetime

from thyme.util import dateutil

def test_datetime_rounding_enum():
    """Verify that enums are unique"""
    assert dateutil.DatetimeRounding.NEAREST_HOUR != dateutil.DatetimeRounding.NEAREST_MINUTE

def test_round():
    """Test generic rounding"""
    dt = datetime.datetime(2009,1,3,9,30,46)
    dt_rounded = datetime.datetime(2009,1,3,10,0,0)
    assert dateutil.round(dt, dateutil.DatetimeRounding.NEAREST_HOUR) == dt_rounded

    dt = datetime.datetime(2009,1,3,9,30,46)
    dt_rounded = datetime.datetime(2009,1,3,9,31,0)
    assert dateutil.round(dt, dateutil.DatetimeRounding.NEAREST_MINUTE) == dt_rounded

def test_round_to_hour():
    """Test rounding up/down to nearest hour"""
    dt = datetime.datetime(2009,1,3,9,30,46)
    dt_rounded = datetime.datetime(2009,1,3,10,0,0)
    assert dateutil.round_to_hour(dt) == dt_rounded

    dt = datetime.datetime(2009,1,3,9,29,14)
    dt_rounded = datetime.datetime(2009,1,3,9,0,0)
    assert dateutil.round_to_hour(dt) == dt_rounded

def test_round_to_minute():
    """Test rounding up/down to nearest minute"""
    dt = datetime.datetime(2009,1,3,9,30,46)
    dt_rounded = datetime.datetime(2009,1,3,9,31,0)
    assert dateutil.round_to_minute(dt) == dt_rounded

    dt = datetime.datetime(2009,1,3,9,30,14)
    dt_rounded = datetime.datetime(2009,1,3,9,30,0)
    assert dateutil.round_to_minute(dt) == dt_rounded
