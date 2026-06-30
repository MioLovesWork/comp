# Keras RNN（非比赛，jingsai 可选）+ RNN 特征裁剪
import sys
from pathlib import Path

_COMP = Path(__file__).resolve().parent.parent
_TUNE = Path(__file__).resolve().parent
for _p in (_COMP, _TUNE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
from lib_comp import mape

TIME, TARGET = 'date', '全口径发购'
RNN_DROP = ['lag_96', 'lag_192', 'rolling_mean_96', 'rolling_mean_672']
RNN_FOURIER = (
    [f'fourier_daily_sin_{k}' for k in range(1, 4)]
    + [f'fourier_daily_cos_{k}' for k in range(1, 4)]
    + [f'fourier_weekly_sin_{k}' for k in range(1, 3)]
    + [f'fourier_weekly_cos_{k}' for k in range(1, 3)]
)
RNN_DEFAULT = dict(seq=96, train_rows=20000, units=64, epochs=20,
                   batch_size=512, lr=1e-3, dropout=0.2)


def build_rnn_feats(all_feats, profile='v1'):
    drop = set(RNN_DROP)
    if profile == 'v2':
        drop |= set(RNN_FOURIER)
    return [c for c in all_feats if c not in drop]


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
    X, y = np.asarray(X, np.float32), y.copy().astype(np.float64)
    out = []
    for i in range(start, start + n):
        ys = sy.transform(y[i - seq:i].reshape(-1, 1)).flatten()
        win = np.column_stack([X[i - seq:i], ys.reshape(-1, 1)]).reshape(1, seq, -1)
        p = sy.inverse_transform(model.predict(win, verbose=0)).flatten()[0]
        out.append(p)
        y[i] = p
    return np.array(out)
