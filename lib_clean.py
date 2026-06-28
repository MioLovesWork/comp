# [B] 数据清洗 — 记忆：去重→补15min→IQR截尾→插值
import pandas as pd

TIME, TARGET, FREQ = 'date', '全口径发购', '15min'


def load(path_train, path_test, path_fallback=None):
    """读 hist + test；无 xlsx 时用 cleaned_df.csv"""
    try:
        h = pd.read_excel(path_train)
    except FileNotFoundError:
        h = pd.read_csv(path_fallback, parse_dates=[TIME])
    try:
        t = pd.read_excel(path_test)
    except FileNotFoundError:
        t = h.tail(2880).copy(); h = h.iloc[:-2880]
    h[TIME] = pd.to_datetime(h[TIME]); t[TIME] = pd.to_datetime(t[TIME])
    h['_src'], t['_src'] = 'hist', 'test'
    return clean(h), clean(t)


def clean(df):
    df = df.drop(columns=[c for c in df if str(c).startswith('Unnamed')], errors='ignore')
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.drop_duplicates(TIME, keep='first').set_index(TIME)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=FREQ))
    if TARGET in df:
        s = df[TARGET]; q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        ok = s[(s >= q1-3*iqr) & (s <= q3+3*iqr)]; df[TARGET] = s.clip(ok.min(), ok.max())
    return df.interpolate('time').ffill().bfill().reset_index(names=TIME)
