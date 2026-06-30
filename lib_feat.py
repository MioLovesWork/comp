# [C] 特征工程（比赛）— 记忆：时8 + 滞4 + 滑2 + 傅10 = 24新 + 7原 = 31
import numpy as np
import pandas as pd

TIME, TARGET = 'date', '全口径发购'
ORIG = ['降水(mm)', '湿度(%)', '温度(℃)', '平均风速(m/s)', 'DATA_WEEK', 'IS_ZM', 'IS_FDJJR']
LAG_YEAR = 365 * 96  # 365 days @ 15min


def build(df, full=True):
    """full=False→7维 debug；full=True→31维 baseline"""
    if not full:
        return df, ORIG.copy()
    dt = df[TIME]
    df = df.copy()
    df['hour'] = dt.dt.hour.astype(float)
    df['dayofweek'] = dt.dt.dayofweek.astype(float)
    df['month'] = dt.dt.month.astype(float)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(float)
    for lag in (96, 192, 672, LAG_YEAR):
        df[f'lag_{lag}'] = df[TARGET].shift(lag)
    s = df[TARGET].shift(1)
    df['rolling_mean_96'] = s.rolling(96, min_periods=1).mean()
    df['rolling_mean_672'] = s.rolling(672, min_periods=1).mean()
    td = dt.dt.hour*4 + dt.dt.minute//15
    tw = df['dayofweek']*96 + td
    for k in range(1, 4):
        df[f'fourier_daily_sin_{k}'] = np.sin(2*np.pi*k*td/96)
        df[f'fourier_daily_cos_{k}'] = np.cos(2*np.pi*k*td/96)
    for k in range(1, 3):
        df[f'fourier_weekly_sin_{k}'] = np.sin(2*np.pi*k*tw/672)
        df[f'fourier_weekly_cos_{k}'] = np.cos(2*np.pi*k*tw/672)
    feats = [c for c in df.columns if c not in (TIME, TARGET, '_src')]
    df[feats] = df[feats].ffill().fillna(0)
    return df, feats
