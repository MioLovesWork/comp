# [B] 数据清洗 — 记忆：去重→补15min→IQR截尾→插值
from pathlib import Path

import pandas as pd

TIME, TARGET, FREQ = 'date', '全口径发购', '15min'
DATA_DIR = Path(__file__).parent / 'data'
PATH_TRAIN = DATA_DIR / '实验数据1.xlsx'
PATH_TEST = DATA_DIR / 'data_test1.xlsx'
PATH_CLEANED = DATA_DIR / 'cleaned_df.csv'  # output only (from clean())


def load(path_train=PATH_TRAIN, path_test=PATH_TEST):
    """读 hist + test xlsx，再 clean"""
    h = pd.read_excel(path_train)
    t = pd.read_excel(path_test)
    h[TIME] = pd.to_datetime(h[TIME])
    t[TIME] = pd.to_datetime(t[TIME])
    h['_src'], t['_src'] = 'hist', 'test'
    return clean(h), clean(t)


def save_cleaned(df, path=PATH_CLEANED):
    """写出清洗后的 hist（供调试/复用，非比赛输入）"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def clean(df):
    df = df.drop(columns=[c for c in df if str(c).startswith('Unnamed')], errors='ignore')
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.drop_duplicates(TIME, keep='first').set_index(TIME)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=FREQ))
    if TARGET in df:
        s = df[TARGET]; q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        ok = s[(s >= q1-3*iqr) & (s <= q3+3*iqr)]; df[TARGET] = s.clip(ok.min(), ok.max())
    return df.infer_objects(copy=False).interpolate('time').ffill().bfill().reset_index(names=TIME)
