# [C] dinov3 PyTorch GPU tuning — do NOT import in work.ipynb (jingsai)
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from lib_clean import load
from lib_feat import build, TIME, TARGET
from lib_model import split, mape, load_params, RNN_DEFAULT

PATH_TRAIN = '../data/实验数据1.xlsx'
PATH_TEST = '../data/data_test1.xlsx'
PATH_FB = '../data/cleaned_df.csv'


def setup_torch():
    """[dinov3] check PyTorch CUDA"""
    ok = torch.cuda.is_available()
    dev = torch.device('cuda' if ok else 'cpu')
    print(f'[tune] env={Path(sys.prefix).name}  py={sys.version.split()[0]}')
    print(f'[tune] torch={torch.__version__}  device={dev}' +
          (f'  ({torch.cuda.get_device_name(0)})' if ok else '  (CPU fallback)'))
    return dev


def save_params(params, path='params.json'):
    out = Path(__file__).parent / path
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f'[tune] saved -> {out}')


def _win(X, y, seq, sy):
    ys = sy.transform(y.reshape(-1, 1)).flatten()
    n = len(X) - seq
    Xw = np.empty((n, seq, X.shape[1] + 1), np.float32)
    for i in range(n):
        j = i + seq
        Xw[i] = np.column_stack([X[i:j], ys[i:j].reshape(-1, 1)])
    return Xw, ys[seq:]


class RNNNet(nn.Module):
    """Mirror lib_model Keras: RNN->Drop->RNN->Drop->Dense32->Dense1"""

    def __init__(self, kind, n_in, p):
        super().__init__()
        RNN = nn.LSTM if kind == 'lstm' else nn.GRU
        u = p['units']
        self.rnn1 = RNN(n_in, u, batch_first=True)
        self.rnn2 = RNN(u, u // 2, batch_first=True)
        self.drop = nn.Dropout(p['dropout'])
        self.head = nn.Sequential(nn.Linear(u // 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn1(x)
        o = self.drop(o)
        o, _ = self.rnn2(o)
        o = self.drop(o[:, -1])
        return self.head(o).squeeze(-1)


def train_rnn_torch(kind, X_tr, y_tr, X_va, y_va, sy, params=None, device=None):
    """[dinov3] train on GPU, return val preds (original scale)"""
    p = {**RNN_DEFAULT, **(params or {})}
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if p.get('train_rows') and len(X_tr) > p['train_rows']:
        X_tr, y_tr = X_tr[-p['train_rows']:], y_tr[-p['train_rows']:]
    seq = p['seq']
    Xw, yw = _win(np.asarray(X_tr, np.float32), np.asarray(y_tr, np.float32), seq, sy)
    Xvw, yvw = _win(np.asarray(X_va, np.float32), np.asarray(y_va, np.float32), seq, sy)

    model = RNNNet(kind, Xw.shape[2], p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p['lr'])
    loss_fn = nn.MSELoss()
    Xw_t = torch.tensor(Xw, device=device)
    yw_t = torch.tensor(yw, device=device)
    Xvw_t = torch.tensor(Xvw, device=device)
    yvw_t = torch.tensor(yvw, device=device)

    best_loss, best_state, wait = float('inf'), None, 0
    bs = p['batch_size']
    for ep in range(p['epochs']):
        model.train()
        idx = torch.randperm(len(Xw_t), device=device)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xw_t[b]), yw_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xvw_t), yvw_t).item()
        if vl < best_loss:
            best_loss, wait = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xvw_t).cpu().numpy()
    return sy.inverse_transform(pred.reshape(-1, 1)).flatten()


def prep_rnn_data(df_all, val_start):
    df, feats = build(df_all, full=True)
    ds = split(df, val_start, feats)
    ht = pd.concat([df[df['_src'] == 'hist'], df[df['_src'] == 'test']]).reset_index(drop=True)
    n_hist = (df['_src'] == 'hist').sum()
    n_tr = len(ds['train'][0])
    hist = ht['_src'] == 'hist'
    sx = StandardScaler().fit(ht.loc[hist, feats].values)
    sy = StandardScaler().fit(ht.loc[hist, [TARGET]].values)
    X = sx.transform(ht[feats].values).astype(np.float32)
    y = ht[TARGET].ffill().fillna(0).values
    return dict(X=X, y=y, n_tr=n_tr, n_hist=n_hist, sy=sy, yva=ds['val'][1])


def tune_rnn(data, n_trials=30, kind='gru', device=None):
    X, y, n_tr, n_hist, sy, yva = (
        data['X'], data['y'], data['n_tr'], data['n_hist'], data['sy'], data['yva'],
    )

    def objective(trial):
        p = dict(
            seq=trial.suggest_categorical('seq', [96, 192]),
            train_rows=trial.suggest_categorical('train_rows', [10000, 20000]),
            units=trial.suggest_categorical('units', [32, 64]),
            epochs=15,
            batch_size=512,
            lr=trial.suggest_categorical('lr', [1e-3, 5e-4]),
            dropout=trial.suggest_categorical('dropout', [0.1, 0.2]),
        )
        i0 = max(0, n_tr - p['seq'])
        try:
            pva = train_rnn_torch(kind, X[:n_tr], y[:n_tr], X[i0:n_hist], y[i0:n_hist],
                                  sy, params=p, device=device)
            return mape(yva, pva[-len(yva):])
        except Exception as e:
            print(f'  trial fail: {e}')
            return 1.0

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**RNN_DEFAULT, **study.best_params, 'epochs': 20}
    print(f'[tune] best MAPE={study.best_value*100:.2f}%  params={best}')
    return best


def run_tune_pipeline(n_trials=30, kind='gru', device=None):
    device = device or setup_torch()
    base = load_params()
    df_hist, df_test = load(PATH_TRAIN, PATH_TEST, PATH_FB)
    df_all = pd.concat([df_hist, df_test]).sort_values(TIME).reset_index(drop=True)
    data = prep_rnn_data(df_all, base['val_start'])
    base['rnn'] = tune_rnn(data, n_trials=n_trials, kind=kind, device=device)
    base['rnn_models'] = [kind]
    save_params(base)
    return base
