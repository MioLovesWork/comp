# 竞赛 baseline — 口诀：params → load → feat → trees → stack
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

from lib_clean import load, save_cleaned, PATH_TRAIN, PATH_TEST, TIME, TARGET
from lib_feat import build

__all__ = [
    'TIME', 'TARGET', 'PATH_TRAIN', 'PATH_TEST',
    'load_params', 'prepare', 'split', 'train_tree', 'train_trees',
    'mape', 'compare', 'stack', 'stack_v1', 'run_baseline',
]


def load_params(path='params.json'):
    with open(Path(__file__).parent / path, encoding='utf-8') as f:
        return json.load(f)


def prepare(path_train=PATH_TRAIN, path_test=PATH_TEST, save_hist=False):
    """[B] 洗 + 拼接（lag 必须先 concat）"""
    df_hist, df_test = load(path_train, path_test)
    if save_hist:
        save_cleaned(df_hist)
    df_all = pd.concat([df_hist, df_test]).sort_values(TIME).reset_index(drop=True)
    return df_hist, df_test, df_all


def split(df, val_start, feats):
    """hist 按 val_start 切 train/val；test=_src=='test'"""
    h, t = df[df['_src'] == 'hist'], df[df['_src'] == 'test']
    tr, va = h[h[TIME] < val_start], h[h[TIME] >= val_start]
    pack = lambda d, y=True: (d[feats], d[TARGET] if y else None, d[TIME].values)
    return {'train': pack(tr), 'val': pack(va), 'test': pack(t, False), 'feats': feats}


def train_tree(name, X_tr, y_tr, X_va, y_va):
    """lgb / xgb / cat"""
    p = dict(n_estimators=500, learning_rate=0.1, random_state=42)
    if name == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMRegressor(max_depth=6, num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    elif name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBRegressor(max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=50, verbosity=0, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(depth=6, iterations=500, learning_rate=0.1,
                              verbose=0, random_seed=42)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    return m


def mape(y, p):
    return mean_absolute_percentage_error(y, p)


def compare(results):
    """results: {name: {'y', 'p'}}"""
    df = pd.DataFrame([{'model': k, 'mape%': mape(v['y'], v['p']) * 100}
                       for k, v in results.items()])
    print(df.sort_values('mape%').to_string(index=False))
    return df


def stack(preds_val, y_val, preds_test, alpha=1.0):
    """Ridge 融合 val/test 预测"""
    n = len(y_val)
    Xv = np.column_stack([np.asarray(p)[-n:] for p in preds_val.values()])
    Xt = np.column_stack([np.asarray(p).flatten() for p in preds_test.values()])
    meta = Ridge(alpha=alpha).fit(Xv, y_val)
    return meta.predict(Xv), meta.predict(Xt), meta


def train_trees(df_all, val_start, trees, full=True):
    """训三棵树 → res_val/res_test + 数据包"""
    df, feats = build(df_all, full=full)
    ds = split(df, val_start, feats)
    Xtr, ytr, _ = ds['train']
    Xva, yva, _ = ds['val']
    Xte, _, dte = ds['test']
    res_val, res_test, models = {}, {}, {}
    for name in trees:
        m = train_tree(name, Xtr, ytr, Xva, yva)
        models[name] = m
        res_val[name] = m.predict(Xva)
        res_test[name] = m.predict(Xte)
    return dict(
        df=df, feats=feats, ds=ds, models=models,
        res_val=res_val, res_test=res_test,
        yva=yva, dte=dte, Xte=Xte,
    )


def stack_v1(pack, alpha=1.0):
    """仅树 stack"""
    return stack(pack['res_val'], pack['yva'], pack['res_test'], alpha=alpha)


def run_baseline(params=None, full=True, out_csv='final_trees.csv'):
    """一键：洗 → 特征 → 树 → stack_v1"""
    p = load_params() if params is None else params
    _, _, df_all = prepare()
    pack = train_trees(df_all, p['val_start'], p['trees'], full=full)
    pva, pte, meta = stack_v1(pack, alpha=p['stack_alpha'])
    if out_csv:
        pd.DataFrame({TIME: pack['dte'], TARGET: pte}).to_csv(out_csv, index=False)
    return dict(pack=pack, pva=pva, pte=pte, meta=meta,
                mape_val=mape(pack['yva'], pva) * 100)
