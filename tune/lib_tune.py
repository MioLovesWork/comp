# dinov3 PyTorch GPU 调参 — 勿在 work.ipynb (jingsai) 引用
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

COMP = Path(__file__).resolve().parent.parent
TUNE = Path(__file__).resolve().parent
for _p in (COMP, TUNE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from lib_clean import load, PATH_TRAIN, PATH_TEST
from lib_feat import build, TIME, TARGET
from lib_model import RNN_DEFAULT, build_rnn_feats


def setup_torch():
    ok = torch.cuda.is_available()
    dev = torch.device('cuda' if ok else 'cpu')
    print(f'[tune] env={Path(sys.prefix).name}  py={sys.version.split()[0]}')
    print(f'[tune] torch={torch.__version__}  device={dev}' +
          (f'  ({torch.cuda.get_device_name(0)})' if ok else '  (CPU fallback)'))
    return dev


def save_params(params, path='params.json'):
    out = COMP / path
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


def train_rnn_torch_full(kind, X_tr, y_tr, X_va, y_va, sy, params=None, device=None):
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
    pva = sy.inverse_transform(pred.reshape(-1, 1)).flatten()
    return model, pva


def train_rnn_torch(kind, X_tr, y_tr, X_va, y_va, sy, params=None, device=None):
    _, pva = train_rnn_torch_full(kind, X_tr, y_tr, X_va, y_va, sy, params=params, device=device)
    return pva


def predict_rnn_roll_torch(model, X, y, seq, start, n, sy, device=None):
    device = device or next(model.parameters()).device
    X = np.asarray(X, np.float32)
    y = y.copy().astype(np.float64)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(start, start + n):
            ys = sy.transform(y[i - seq:i].reshape(-1, 1)).flatten()
            win = np.column_stack([X[i - seq:i], ys.reshape(-1, 1)]).astype(np.float32)
            win_t = torch.tensor(win, device=device).unsqueeze(0)
            p = sy.inverse_transform(model(win_t).cpu().numpy().reshape(-1, 1)).flatten()[0]
            out.append(float(p))
            y[i] = p
    return np.array(out)


def _weekly_test_mape(times, y_true, y_pred):
    df = pd.DataFrame({TIME: pd.to_datetime(times), 'y': y_true, 'p': y_pred})
    df = df.dropna(subset=['y'])
    if len(df) == 0:
        return {}
    t0 = df[TIME].min()
    out = {}
    for w, g in df.groupby((df[TIME] - t0).dt.days // 7):
        out[f'mape_test_w{int(w)}'] = float(mape(g['y'], g['p']) * 100)
    return out


def diagnose_rnn_torch(data, df_test, rnn_params, kinds=('gru', 'lstm'), device=None,
                       fast_epochs=8):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, n_tr, n_hist, sy, yva = (
        data['X'], data['y'], data['n_tr'], data['n_hist'], data['sy'], data['yva'],
    )
    tva, dte, n_test = data['tva'], data['dte'], data['n_test']
    yva = np.asarray(yva, np.float64)
    p = {**RNN_DEFAULT, **(rnn_params or {}), 'epochs': fast_epochs}
    seq, i0 = p['seq'], max(0, n_tr - p['seq'])
    n_val = len(yva)
    if TARGET in df_test.columns:
        by_pos = df_test[TARGET].values
        merged = pd.DataFrame({TIME: pd.to_datetime(dte)}).merge(
            df_test[[TIME, TARGET]].assign(**{TIME: pd.to_datetime(df_test[TIME])}),
            on=TIME, how='left',
        )
        align_ok = bool(np.allclose(by_pos, merged[TARGET].values, equal_nan=True))
        print(f'[diag] test TARGET align(pos vs TIME)={align_ok}')
    rows, val_frames, test_frames = [], [], []
    for kind in kinds:
        print(f'[diag] training {kind} (epochs={fast_epochs})...')
        model, pva = train_rnn_torch_full(
            kind, X[:n_tr], y[:n_tr], X[i0:n_hist], y[i0:n_hist], sy, params=p, device=device,
        )
        p_teacher = pva[-n_val:]
        p_roll_val = predict_rnn_roll_torch(model, X, y.copy(), seq, n_tr, n_val, sy, device)
        p_roll_test = predict_rnn_roll_torch(model, X, y.copy(), seq, n_hist, n_test, sy, device)
        row = dict(
            kind=kind,
            mape_val_teacher=float(mape(yva, p_teacher) * 100),
            mape_val_roll=float(mape(yva, p_roll_val) * 100),
            mape_test_roll=float('nan'),
        )
        if TARGET in df_test.columns:
            yt_pos = df_test[TARGET].values
            row['mape_test_roll'] = float(mape(yt_pos, p_roll_test) * 100)
            yt_merge = merged[TARGET].values
            row['mape_test_roll_merge'] = float(mape(yt_merge, p_roll_test) * 100)
            row.update(_weekly_test_mape(dte, yt_merge, p_roll_test))
        rows.append(row)
        val_frames.append(pd.DataFrame({
            TIME: pd.to_datetime(tva),
            f'{kind}_teacher_val': p_teacher,
            f'{kind}_roll_val': p_roll_val,
        }))
        test_frames.append(pd.DataFrame({
            TIME: pd.to_datetime(dte),
            f'{kind}_roll_test': p_roll_test,
        }))
    diag_df = pd.DataFrame(rows)
    val_df = val_frames[0]
    for vf in val_frames[1:]:
        val_df = val_df.merge(vf, on=TIME, how='outer')
    test_df = test_frames[0]
    for tf in test_frames[1:]:
        test_df = test_df.merge(tf, on=TIME, how='outer')
    if TARGET in df_test.columns:
        test_df = test_df.merge(
            df_test[[TIME, TARGET]].assign(**{TIME: pd.to_datetime(df_test[TIME])}),
            on=TIME, how='left',
        )
    print('[diag] summary:')
    print(diag_df.to_string(index=False))
    return diag_df, val_df, test_df


def prep_rnn_data(df_all, val_start, rnn_profile='v1'):
    df, feats = build(df_all, full=True)
    feats_rnn = build_rnn_feats(feats, rnn_profile)
    ds = split(df, val_start, feats)
    ht = pd.concat([df[df['_src'] == 'hist'], df[df['_src'] == 'test']]).reset_index(drop=True)
    n_hist = (df['_src'] == 'hist').sum()
    n_tr = len(ds['train'][0])
    hist = ht['_src'] == 'hist'
    sx = StandardScaler().fit(ht.loc[hist, feats_rnn].values)
    sy = StandardScaler().fit(ht.loc[hist, [TARGET]].values)
    X = sx.transform(ht[feats_rnn].values).astype(np.float32)
    y = ht[TARGET].ffill().fillna(0).values
    return dict(X=X, y=y, n_tr=n_tr, n_hist=n_hist, sy=sy, yva=ds['val'][1],
                tva=ds['val'][2], dte=ds['test'][2], n_test=len(ds['test'][0]),
                feats_rnn=feats_rnn, rnn_profile=rnn_profile)


def eval_rnn_roll_val(kind, X, y, n_tr, n_hist, yva, sy, params, device=None):
    yva = np.asarray(yva, np.float64)
    seq = params['seq']
    i0 = max(0, n_tr - seq)
    model, _ = train_rnn_torch_full(
        kind, X[:n_tr], y[:n_tr], X[i0:n_hist], y[i0:n_hist], sy, params=params, device=device,
    )
    p_roll = predict_rnn_roll_torch(model, X, y.copy(), seq, n_tr, len(yva), sy, device)
    return float(mape(yva, p_roll)), p_roll


def tune_rnn(data, n_trials=30, kind='gru', device=None):
    X, y, n_tr, n_hist, sy, yva = (
        data['X'], data['y'], data['n_tr'], data['n_hist'], data['sy'], data['yva'],
    )
    profile = data.get('rnn_profile', 'v1')
    print(f'[tune] profile={profile}  feats={X.shape[1]}  trials={n_trials}')
    print('[tune] objective=mape_val_roll (rolling, not teacher-forcing)')

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
        try:
            score, _ = eval_rnn_roll_val(
                kind, X, y, n_tr, n_hist, yva, sy, params=p, device=device,
            )
            return score
        except Exception as e:
            print(f'  trial fail: {e}')
            return 1.0

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**RNN_DEFAULT, **study.best_params, 'epochs': 20}
    print(f'[tune] profile={profile}  best roll_val MAPE={study.best_value*100:.2f}%  params={best}')
    return best, study.best_value


def tune_rnn_profiles(df_all, val_start, n_trials=30, kind='gru', device=None,
                      profiles=('v1', 'v2')):
    rows = []
    best_profile, best_params = None, None
    best_mape = float('inf')
    for profile in profiles:
        data = prep_rnn_data(df_all, val_start, rnn_profile=profile)
        params, mape_val = tune_rnn(data, n_trials=n_trials, kind=kind, device=device)
        rows.append(dict(profile=profile, mape_val_roll=mape_val * 100,
                         n_feats=len(data['feats_rnn']), **params))
        if mape_val < best_mape:
            best_mape, best_profile, best_params = mape_val, profile, params
    cmp = pd.DataFrame(rows).sort_values('mape_val_roll')
    print('[tune] profile comparison (roll_val MAPE):')
    print(cmp.to_string(index=False))
    print(f'[tune] winner profile={best_profile}  roll_val MAPE={best_mape*100:.2f}%')
    return best_profile, best_params, cmp


def run_tune_pipeline(n_trials=30, kind='gru', device=None, profiles=('v1', 'v2')):
    device = device or setup_torch()
    base = load_params()
    df_hist, df_test = load(PATH_TRAIN, PATH_TEST)
    df_all = pd.concat([df_hist, df_test]).sort_values(TIME).reset_index(drop=True)
    profile, rnn_params, cmp = tune_rnn_profiles(
        df_all, base['val_start'], n_trials=n_trials, kind=kind, device=device, profiles=profiles,
    )
    base['rnn_profile'] = profile
    base['rnn'] = rnn_params
    base['rnn_models'] = [kind]
    save_params(base)
    return base, cmp
