# 节假日 / 极端天气特征（实验）— 在 lib_feat.build 之上追加
import sys
from pathlib import Path

_COMP = Path(__file__).resolve().parent.parent
_TUNE = Path(__file__).resolve().parent
for _p in (_COMP, _TUNE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
import pandas as pd

from lib_feat import build as build_base, TIME, TARGET

LONG_HOL_MIN_DAYS = 3
MAX_LONG_HOLS = 8
HOL_PRE_DAYS = 14
HOL_V1_PRE_DAYS = 3
HOL_V1_POST_DAYS = 2
EXTREME_WEATHER_COLS = ['温度(℃)', '湿度(%)', '平均风速(m/s)']
HOLIDAY_V1 = ['hol_active', 'hol_days', 'hol_lagy']
HOLIDAY_V1_LEGACY = ['days_to_holiday', 'days_from_holiday', 'lag_holiday_y']
HOLIDAY_V2 = [f'hol{k}_{s}' for k in range(1, MAX_LONG_HOLS + 1) for s in ('days', 'lagy')]
HOLIDAY_V2B = ['hol_active', 'hol_days', 'hol_lagy']
HOLIDAY_V3 = ['hol_type', 'hol_days_v3', 'hol_lagy_rank', 'hol_lagy_cal']


def _holiday_episodes(is_hol, min_len=1):
    is_hol = np.asarray(is_hol, dtype=bool)
    eps, i, n = [], 0, len(is_hol)
    while i < n:
        if is_hol[i]:
            j = i + 1
            while j < n and is_hol[j]:
                j += 1
            if j - i >= min_len:
                eps.append((i, j - 1))
            i = j
        else:
            i += 1
    return eps


def _episodes_by_year(dates, episodes):
    by_year = {}
    for s, e in episodes:
        y = pd.Timestamp(dates[s]).year
        by_year.setdefault(y, []).append((s, e))
    for y in by_year:
        by_year[y].sort(key=lambda x: x[0])
    return by_year


def _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps, post_steps=0):
    yr = pd.Timestamp(dates[i]).year
    if i < s - pre_steps or i > e + post_steps:
        return 0.0
    prev = by_year.get(yr - 1, [])
    if rank >= len(prev):
        return 0.0
    s2, e2 = prev[rank]
    j = s2 + (i - s)
    if 0 <= j < len(y) and j <= e2:
        return float(y[j])
    return 0.0


def _episode_hol_type(s, e, dates):
    ds = pd.Timestamp(dates[s])
    if ds.month in (1, 2) or (ds.month == 12 and ds.day >= 20):
        return 1
    if ds.month in (5, 10):
        return 3
    return 2


def _lag_same_calendar_day(i, dates, y, dt_lookup):
    di = pd.Timestamp(dates[i])
    target = di - pd.DateOffset(years=1)
    for key in (target, target.floor('15min')):
        j = dt_lookup.get(key)
        if j is not None and 0 <= j < len(y):
            return float(y[j])
    return 0.0


def _nearest_long_episode(i, year_eps, dates, pre_steps, post_steps=0):
    di = pd.Timestamp(dates[i])
    best = None
    for rank, (s, e) in enumerate(year_eps):
        if i < s - pre_steps or i > e + post_steps:
            continue
        ds = pd.Timestamp(dates[s])
        de = pd.Timestamp(dates[e])
        if s <= i <= e:
            dist = 0.0
        elif i < s:
            dist = float((ds - di).days)
        else:
            dist = float((di - de).days)
        hol_type = _episode_hol_type(s, e, dates)
        cand = (dist, -rank, rank, s, e, hol_type)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    _, _, rank, s, e, hol_type = best
    ds = pd.Timestamp(dates[s])
    de = pd.Timestamp(dates[e])
    if s <= i <= e:
        days = 0.0
    elif i < s:
        days = max(0.0, float((ds - di).days))
    else:
        days = -max(1.0, float((di - de).days))
    return rank, s, e, days, hol_type


def _add_holiday_compact(df, pre_days, post_days, min_len=LONG_HOL_MIN_DAYS):
    df = df.copy()
    dt = pd.to_datetime(df[TIME])
    is_hol = df['IS_FDJJR'].fillna(0).astype(float) > 0.5
    n = len(df)
    y = df[TARGET].values.astype(float)
    dates = dt.values
    pre_steps = pre_days * 96
    post_steps = post_days * 96
    eps = _holiday_episodes(is_hol, min_len=min_len)
    by_year = _episodes_by_year(dates, eps)
    active = np.zeros(n)
    days_col = np.zeros(n)
    lag_col = np.zeros(n)
    for i in range(n):
        yr = pd.Timestamp(dates[i]).year
        hit = _nearest_long_episode(i, by_year.get(yr, []), dates, pre_steps, post_steps)
        if hit is None:
            continue
        rank, s, e, days, _ = hit
        active[i] = 1.0
        days_col[i] = days
        lag_col[i] = _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps, post_steps)
    df['hol_active'] = active
    df['hol_days'] = days_col
    df['hol_lagy'] = lag_col
    return df


def hol_active_mask(df, pre_days=HOL_V1_PRE_DAYS, post_days=HOL_V1_POST_DAYS, min_len=LONG_HOL_MIN_DAYS):
    tmp = _add_holiday_compact(df[[TIME, TARGET, 'IS_FDJJR']].copy(), pre_days, post_days, min_len)
    return tmp['hol_active'].values.astype(bool)


def add_extreme_weather_flags(df, train_df, q_low=0.05, q_high=0.95):
    df = df.copy()
    tr = train_df[EXTREME_WEATHER_COLS].astype(float)
    q05, q95 = tr.quantile(q_low), tr.quantile(q_high)
    extreme = np.zeros(len(df), dtype=bool)
    for c in EXTREME_WEATHER_COLS:
        v = df[c].astype(float).values
        extreme |= (v <= q05[c]) | (v >= q95[c])
    df['bool_extreme'] = extreme.astype(float)
    return df


def add_holiday_v1_legacy(df):
    df = df.copy()
    dt = pd.to_datetime(df[TIME])
    is_hol = df['IS_FDJJR'].fillna(0).astype(float) > 0.5
    n = len(df)
    y = df[TARGET].values.astype(float)
    dates = dt.values
    days_to = np.full(n, 999.0)
    days_from = np.zeros(n)
    lag_hy = np.zeros(n)
    eps = _holiday_episodes(is_hol, min_len=1)
    by_year = _episodes_by_year(dates, eps)
    pre_steps = HOL_PRE_DAYS * 96
    for s, e in eps:
        ds = pd.Timestamp(dates[s])
        for i in range(n):
            di = pd.Timestamp(dates[i])
            if s <= i <= e:
                days_to[i] = min(days_to[i], 0.0)
                days_from[i] = max(days_from[i], (di - ds).days)
            elif i < s:
                d = (ds - di).days
                if d >= 0:
                    days_to[i] = min(days_to[i], float(d))
    for i in range(n):
        yr = pd.Timestamp(dates[i]).year
        for rank, (s, e) in enumerate(by_year.get(yr, [])):
            lag = _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps)
            if lag > 0 or (s - pre_steps <= i <= e):
                lag_hy[i] = lag
                break
    df['days_to_holiday'] = days_to
    df['days_from_holiday'] = days_from
    df['lag_holiday_y'] = lag_hy
    return df


def add_holiday_v1(df, min_len=LONG_HOL_MIN_DAYS):
    return _add_holiday_compact(df, HOL_V1_PRE_DAYS, HOL_V1_POST_DAYS, min_len)


def add_holiday_v2(df, max_hols=MAX_LONG_HOLS, min_len=LONG_HOL_MIN_DAYS):
    df = df.copy()
    dt = pd.to_datetime(df[TIME])
    is_hol = df['IS_FDJJR'].fillna(0).astype(float) > 0.5
    n = len(df)
    y = df[TARGET].values.astype(float)
    dates = dt.values
    pre_steps = HOL_PRE_DAYS * 96
    eps = _holiday_episodes(is_hol, min_len=min_len)
    by_year = _episodes_by_year(dates, eps)
    for k in range(1, max_hols + 1):
        days_col = np.zeros(n)
        lag_col = np.zeros(n)
        rank = k - 1
        for yr, year_eps in by_year.items():
            if rank >= len(year_eps):
                continue
            s, e = year_eps[rank]
            ds = pd.Timestamp(dates[s])
            for i in range(n):
                if i < s - pre_steps or i > e:
                    continue
                di = pd.Timestamp(dates[i])
                days_col[i] = 0.0 if s <= i <= e else max(0.0, float((ds - di).days))
                lag_col[i] = _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps)
        df[f'hol{k}_days'] = days_col
        df[f'hol{k}_lagy'] = lag_col
    return df


def add_holiday_v2b(df, min_len=LONG_HOL_MIN_DAYS):
    return _add_holiday_compact(df, HOL_PRE_DAYS, 0, min_len)


def add_holiday_v3(df, min_len=LONG_HOL_MIN_DAYS):
    df = df.copy()
    dt = pd.to_datetime(df[TIME])
    is_hol = df['IS_FDJJR'].fillna(0).astype(float) > 0.5
    n = len(df)
    y = df[TARGET].values.astype(float)
    dates = dt.values
    pre_steps = HOL_PRE_DAYS * 96
    dt_lookup = {pd.Timestamp(t): i for i, t in enumerate(dt)}
    eps = _holiday_episodes(is_hol, min_len=min_len)
    by_year = _episodes_by_year(dates, eps)
    hol_type = np.zeros(n)
    days_col = np.zeros(n)
    lag_rank = np.zeros(n)
    lag_cal = np.zeros(n)
    for i in range(n):
        yr = pd.Timestamp(dates[i]).year
        year_eps = by_year.get(yr, [])
        hit = _nearest_long_episode(i, year_eps, dates, pre_steps, 0)
        if hit is None:
            continue
        rank, s, e, days, htype = hit
        hol_type[i] = float(htype)
        days_col[i] = days
        if htype == 3:
            lag_cal[i] = _lag_same_calendar_day(i, dates, y, dt_lookup)
            if lag_cal[i] <= 0:
                lag_rank[i] = _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps, 0)
        else:
            lag_rank[i] = _lag_same_holiday_y(i, s, e, rank, by_year, dates, y, pre_steps, 0)
    df['hol_type'] = hol_type
    df['hol_days_v3'] = days_col
    df['hol_lagy_rank'] = lag_rank
    df['hol_lagy_cal'] = lag_cal
    return df


def build(df, full=True, holiday=None):
    """baseline build + 可选 holiday=v1|v1_legacy|v2|v2b|v3"""
    df, feats = build_base(df, full=full)
    if not full or not holiday:
        return df, feats
    if holiday == 'v1':
        df = add_holiday_v1(df)
    elif holiday == 'v1_legacy':
        df = add_holiday_v1_legacy(df)
    elif holiday == 'v2':
        df = add_holiday_v2(df)
    elif holiday == 'v2b':
        df = add_holiday_v2b(df)
    elif holiday == 'v3':
        df = add_holiday_v3(df)
    feats = [c for c in df.columns if c not in (TIME, TARGET, '_src')]
    df[feats] = df[feats].ffill().fillna(0)
    return df, feats
