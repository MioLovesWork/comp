# [A/B/C] 竞赛核心 — 无 GPU、无调参，读 params.json
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge

TIME, TARGET = 'date', '全口径发购'
RNN_DEFAULT = dict(seq=96, train_rows=20000, units=64, epochs=20,
                   batch_size=512, lr=1e-3, dropout=0.2)


def load_params(path='params.json'):
    with open(Path(__file__).parent / path, encoding='utf-8') as f:
        return json.load(f)


def split(df, val_start, feats):
    """[A] hist 按 val_start 切 train/val；test=_src=='test'"""
    h, t = df[df['_src'] == 'hist'], df[df['_src'] == 'test']
    tr, va = h[h[TIME] < val_start], h[h[TIME] >= val_start]
    pack = lambda d, y=True: (d[feats], d[TARGET] if y else None, d[TIME].values)
    return {'train': pack(tr), 'val': pack(va), 'test': pack(t, False), 'feats': feats}


def train_tree(name, X_tr, y_tr, X_va, y_va):
    """[A] lgb/xgb/cat"""
    p = dict(n_estimators=500, learning_rate=0.1, random_state=42)
    if name == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMRegressor(max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    elif name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBRegressor(max_depth=6, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, verbosity=0, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(depth=6, iterations=500, learning_rate=0.1, verbose=0, random_seed=42)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    return m


def mape(y, p):
    return mean_absolute_percentage_error(y, p)


def compare(results):
    df = pd.DataFrame([{'model': k, 'mape%': mape(v['y'], v['p']) * 100} for k, v in results.items()])
    print(df.sort_values('mape%').to_string(index=False)); return df


def stack(preds_val, y_val, preds_test, alpha=1.0):
    """[B] Ridge 融合"""
    n = len(y_val)
    Xv = np.column_stack([np.asarray(p)[-n:] for p in preds_val.values()])
    Xt = np.column_stack([np.asarray(p).flatten() for p in preds_test.values()])
    meta = Ridge(alpha=alpha).fit(Xv, y_val)
    return meta.predict(Xv), meta.predict(Xt), meta


def _win(X, y, seq, sy):
    ys = sy.transform(y.reshape(-1, 1)).flatten()
    n = len(X) - seq
    Xw = np.empty((n, seq, X.shape[1] + 1), np.float32)
    for i in range(n):
        j = i + seq
        Xw[i] = np.column_stack([X[i:j], ys[i:j].reshape(-1, 1)])
    return Xw, ys[seq:]


def _build_rnn(kind, seq, n_in, p):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    RNN = LSTM if kind == 'lstm' else GRU
    m = Sequential([
        RNN(p['units'], return_sequences=True, input_shape=(seq, n_in)),
        Dropout(p['dropout']),
        RNN(p['units'] // 2),
        Dropout(p['dropout']),
        Dense(32, activation='relu'),
        Dense(1),
    ])
    m.compile(Adam(p['lr']), 'mse')
    return m


def train_rnn(kind, X_tr, y_tr, X_va, y_va, sy, params=None):
    """[C] 返回 model, 验证集原始尺度预测"""
    from tensorflow.keras.callbacks import EarlyStopping
    p = {**RNN_DEFAULT, **(params or {})}
    if p.get('train_rows') and len(X_tr) > p['train_rows']:
        X_tr, y_tr = X_tr[-p['train_rows']:], y_tr[-p['train_rows']:]
    seq = p['seq']
    Xw, yw = _win(np.asarray(X_tr, np.float32), np.asarray(y_tr, np.float32), seq, sy)
    Xvw, yvw = _win(np.asarray(X_va, np.float32), np.asarray(y_va, np.float32), seq, sy)
    m = _build_rnn(kind, seq, Xw.shape[2], p)
    m.fit(Xw, yw, validation_data=(Xvw, yvw), epochs=p['epochs'], batch_size=p['batch_size'],
          callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
    p_scaled = m.predict(Xvw, verbose=0).flatten()
    return m, sy.inverse_transform(p_scaled.reshape(-1, 1)).flatten()


def predict_rnn_test(model, X, y, seq, start, n, sy):
    """[C] test 滚动预测"""
    X, y = np.asarray(X, np.float32), y.copy().astype(np.float64)
    out = []
    for i in range(start, start + n):
        ys = sy.transform(y[i - seq:i].reshape(-1, 1)).flatten()
        win = np.column_stack([X[i - seq:i], ys.reshape(-1, 1)]).reshape(1, seq, -1)
        p = sy.inverse_transform(model.predict(win, verbose=0)).flatten()[0]
        out.append(p); y[i] = p
    return np.array(out)
