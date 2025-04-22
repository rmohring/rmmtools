import datetime
from types import SimpleNamespace

import pandas as pd


dow_map = dict(zip(range(7), ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]))
dow_map.update({-1: "all"})
ONE_DAY_UTC = 86400


## Time and Date functions
def convert_utc_timestamp_to_local_dttm(ts: int, offset: int = 0) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ts) + datetime.timedelta(seconds=offset)


def convert_dttm_to_utc_timestamp(dttm, offset: int = 0, day_offset: int = 0) -> int:
    return int(
        datetime.datetime.timestamp(
            dttm
            + datetime.timedelta(seconds=offset)
            + datetime.timedelta(days=day_offset)
        )
    )


def convert_dttm_to_dateint(dttm):
    dttm_str = str(dttm)
    year = int(dttm_str[0:4])
    month = int(dttm_str[5:7])
    day = int(dttm_str[8:10])
    return int(pd.Timestamp(year, month, day).strftime("%Y%m%d"))


def convert_dateint_to_timestamp(dateint: int):
    dateint_str = str(dateint)
    year = int(dateint_str[0:4])
    month = int(dateint_str[4:6])
    day = int(dateint_str[-2:])
    return pd.Timestamp(year, month, day)


def get_year_from_dateint(dateint: int) -> int:
    return int(str(dateint)[0:4])


def dateint_add_days(dateint: int, add_days: int):
    ts = convert_dateint_to_timestamp(dateint=dateint) + pd.Timedelta(
        add_days, unit="days"
    )
    return int(ts.strftime("%Y%m%d"))


def get_now_local_and_utc(
    local_fmt="%Y-%m-%d %H:%M:%S.%f", utc_fmt="%Y-%m-%dT%H:%M:%S.%fZ"
) -> SimpleNamespace:
    now = SimpleNamespace()
    # tz0 = datetime.datetime.today().astimezone()
    now.utc = pd.Timestamp.today(tz="UTC")

    local_tz = datetime.datetime.today().astimezone()

    # # In the unlikely chance this crosses a time change event...
    # if local_tz.tzname() != tz0.tzname():
    #     now.utc = pd.Timestamp.today(tz="UTC")

    now.local = now.utc.astimezone(local_tz.tzinfo)

    now.dttmstr_utc = now.utc.strftime(utc_fmt)
    now.dttmstr_local = now.local.strftime(local_fmt)

    now.dateint_utc = int(now.utc.strftime("%Y%m%d"))
    now.dateint_local = int(now.local.strftime("%Y%m%d"))

    return now


def get_now_utc() -> pd.Timestamp:
    now = get_now_local_and_utc()
    return now.utc


def get_now_local() -> pd.Timestamp:
    now = get_now_local_and_utc()
    return now.local


def get_now_dateint_utc() -> int:
    now = get_now_local_and_utc()
    return now.dateint_utc


def get_now_dateint_local() -> int:
    now = get_now_local_and_utc()
    return now.dateint_local


def get_now_dttmstr_local(fmt="%Y-%m-%d %H:%M:%S.%f") -> str:
    if fmt is not None:
        now = get_now_local_and_utc(local_fmt=fmt)
    else:
        now = get_now_local_and_utc()
    return now.dttmstr_local


def get_now_dttmstr_utc(fmt=None) -> str:
    if fmt is not None:
        now = get_now_local_and_utc(utc_fmt=fmt)
    else:
        now = get_now_local_and_utc()

    return now.dttmstr_utc
