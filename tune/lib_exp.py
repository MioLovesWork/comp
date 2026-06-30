# 树模型实验 — 长训、分段 MAPE、v4_1 门控（非比赛）
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

from lib_comp import split, stack, mape
from lib_feat import TIME, TARGET
from lib_feat_ext import build, hol_active_mask, add_extreme_weather_flags

TREE_MAX_ROUNDS = 1500
TREE_ES_ROUNDS = 100


def train_tree_long(name, X_tr, y_tr, X_va, y_va,
                    max_rounds=TREE_MAX_ROUNDS, es_rounds=TREE_ES_ROUNDS):
    p = dict(n_estimators=max_rounds, learning_rate=0.1, random_state=42)
    best_iter = None
    if name == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMRegressor(max_depth=6, num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        best_iter = m.best_iteration_
    elif name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBRegressor(max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=es_rounds, verbosity=0, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = getattr(m, 'best_iteration', None)
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(depth=6, iterations=max_rounds, learning_rate=0.1,
                              verbose=0, random_seed=42)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=es_rounds)
        best_iter = m.get_best_iteration()
    return m, best_iter


def make_seg_lookup(df_all):
    _df_seg, _ = build(df_all, full=True, holiday='v1')
    seg = _df_seg[[TIME, 'hol_active', 'hol_days', 'IS_FDJJR']].copy()
    seg[TIME] = pd.to_datetime(seg[TIME])
    return seg


def merge_y(dte, df_src):
    return pd.DataFrame({TIME: pd.to_datetime(dte)}).merge(
        df_src[[TIME, TARGET]].assign(**{TIME: pd.to_datetime(df_src[TIME])}),
        on=TIME, how='left',
    )[TARGET].values


def segment_masks(dte, seg_lookup):
    dte = pd.to_datetime(dte)
    n = len(dte)
    aux = pd.DataFrame({TIME: dte}).merge(seg_lookup, on=TIME, how='left')
    return {
        'full': np.ones(n, dtype=bool),
        'may1_5': (dte >= pd.Timestamp('2025-05-01')) & (dte <= pd.Timestamp('2025-05-05 23:59:59')),
        'fdjjr': aux['IS_FDJJR'].fillna(0).astype(float).values > 0.5,
        'pre7': (aux['hol_active'].fillna(0).astype(float).values > 0.5)
                & (aux['hol_days'].fillna(999).values > 0)
                & (aux['hol_days'].fillna(999).values <= 7),
    }


def segment_mape(y, pred, masks, prefix):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {}
    for seg, m in masks.items():
        if m.sum() == 0:
            continue
        out[f'{prefix}_{seg}_mape'] = mape(y[m], pred[m]) * 100
    return out


def run_trees(df_all, df_test, val_start, trees, params, holiday_mode, label, seg_lookup):
    df, feats = build(df_all, full=True, holiday=holiday_mode)
    ds = split(df, val_start, feats)
    Xtr, ytr, _ = ds['train']
    Xva, yva, dva = ds['val']
    Xte, _, dte = ds['test']
    res_val, res_test, iters = {}, {}, {}
    for name in trees:
        m, bi = train_tree_long(name, Xtr, ytr, Xva, yva)
        res_val[name] = m.predict(Xva)
        res_test[name] = m.predict(Xte)
        iters[name] = bi
    pva, pte, _ = stack(res_val, yva, res_test, alpha=params['stack_alpha'])
    row = dict(variant=label, holiday=holiday_mode, n_feats=len(feats),
               stack_val_mape=mape(yva, pva) * 100, lgb_best_iter=iters.get('lgb'))
    for name in trees:
        row[f'{name}_val_mape'] = mape(yva, res_val[name]) * 100
    row.update(segment_mape(yva, pva, segment_masks(dva, seg_lookup), 'stack_val'))
    if TARGET in df_test.columns:
        yt = merge_y(dte, df_test)
        row['stack_test_mape'] = mape(yt, pte) * 100
        for name in trees:
            row[f'{name}_test_mape'] = mape(yt, res_test[name]) * 100
        row.update(segment_mape(yt, pte, segment_masks(dte, seg_lookup), 'stack_test'))
    pack = dict(dte=dte, yva=yva, pte=pte, res_test=res_test, feats=feats, iters=iters)
    return row, pack


def _split_bool(df, val_start):
    ts = pd.to_datetime(val_start)
    h = df[df['_src'] == 'hist']
    tr = h[pd.to_datetime(h[TIME]) < ts]
    va = h[pd.to_datetime(h[TIME]) >= ts]
    te = df[df['_src'] == 'test']
    return tr, va, te


def _gate_mix(p_base, p_exp, mask, w):
    out = np.asarray(p_base, dtype=float).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = w * p_base[m] + (1.0 - w) * p_exp[m]
    return out


def _search_gate_weight(p_base, p_exp, y, mask, weights=None):
    weights = weights if weights is not None else np.linspace(0.2, 0.8, 13)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.5, float('nan')
    best_w, best_m = 0.5, float('inf')
    for w in weights:
        pred = _gate_mix(p_base, p_exp, mask, w)
        m_ = mape(y[mask], pred[mask]) * 100
        if m_ < best_m:
            best_m, best_w = m_, w
    return best_w, best_m


def _v4_prepare(df_all, val_start):
    df, feats = build(df_all, full=True, holiday=None)
    tr_df, va_df, te_df = _split_bool(df, val_start)
    df = df.copy()
    df['bool_holiday'] = hol_active_mask(df).astype(float)
    df = add_extreme_weather_flags(df, tr_df)
    tr_df, va_df, te_df = _split_bool(df, val_start)
    ds = split(df, val_start, feats)
    Xtr, ytr, _ = ds['train']
    Xva, yva, dva = ds['val']
    Xte, _, dte = ds['test']
    flags = dict(
        bh_tr=tr_df['bool_holiday'].values.astype(bool),
        bh_va=va_df['bool_holiday'].values.astype(bool),
        bh_te=te_df['bool_holiday'].values.astype(bool),
        be_tr=tr_df['bool_extreme'].values.astype(bool),
        be_va=va_df['bool_extreme'].values.astype(bool),
        be_te=te_df['bool_extreme'].values.astype(bool),
    )
    return df, feats, Xtr, ytr, Xva, yva, dva, Xte, dte, flags


def _train_expert(mask_tr, Xtr, ytr, mask_va, Xva, yva, Xte, tag, expert_tree='lgb', min_n=96):
    if mask_tr.sum() < min_n:
        mu = float(np.mean(ytr))
        print(f'  [{tag}] train_n={mask_tr.sum()} 过少 → 常数 {mu:.0f}')
        return np.full(len(Xva), mu), np.full(len(Xte), mu), None
    X_es, y_es = Xva, yva
    if mask_va.sum() >= 48:
        X_es, y_es = Xva[mask_va], yva[mask_va]
    m, bi = train_tree_long(expert_tree, Xtr[mask_tr], ytr[mask_tr], X_es, y_es)
    return m.predict(Xva), m.predict(Xte), bi


def _v4_train_base_experts(ctx, trees, params, expert_tree='lgb'):
    Xtr, ytr, Xva, yva, Xte, flags = (
        ctx[k] for k in ('Xtr', 'ytr', 'Xva', 'yva', 'Xte', 'flags')
    )
    res_val, res_test, iters = {}, {}, {}
    for name in trees:
        m, bi = train_tree_long(name, Xtr, ytr, Xva, yva)
        res_val[name] = m.predict(Xva)
        res_test[name] = m.predict(Xte)
        iters[name] = bi
    p_base_va, p_base_te, meta = stack(res_val, yva, res_test, alpha=params['stack_alpha'])
    bh_tr, bh_va = flags['bh_tr'], flags['bh_va']
    be_tr, be_va = flags['be_tr'], flags['be_va']
    p_hol_va, p_hol_te, hol_iter = _train_expert(
        bh_tr, Xtr, ytr, bh_va, Xva, yva, Xte, 'hol', expert_tree,
    )
    p_ext_va, p_ext_te, ext_iter = _train_expert(
        be_tr, Xtr, ytr, be_va & ~bh_va, Xva, yva, Xte, 'ext', expert_tree,
    )
    return dict(
        res_val=res_val, res_test=res_test, iters=iters,
        p_base_va=p_base_va, p_base_te=p_base_te,
        p_hol_va=p_hol_va, p_hol_te=p_hol_te,
        p_ext_va=p_ext_va, p_ext_te=p_ext_te,
        hol_iter=hol_iter, ext_iter=ext_iter, meta=meta,
    )


def run_v4_1(df_all, df_test, val_start, trees, params, seg_lookup,
             label='v4_1', expert_tree='lgb'):
    df, feats, Xtr, ytr, Xva, yva, dva, Xte, dte, flags = _v4_prepare(df_all, val_start)
    ctx = dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, flags=flags)
    m = _v4_train_base_experts(ctx, trees, params, expert_tree)
    bh_va, bh_te = flags['bh_va'], flags['bh_te']
    be_va, be_te = flags['be_va'], flags['be_te']
    m_ext_va = be_va & ~bh_va
    m_ext_te = be_te & ~bh_te
    w_ext, ext_seg_mape = _search_gate_weight(m['p_base_va'], m['p_ext_va'], yva, m_ext_va)
    pva = _gate_mix(m['p_base_va'], m['p_ext_va'], m_ext_va, w_ext)
    w_hol, hol_seg_mape = _search_gate_weight(m['p_base_va'], m['p_hol_va'], yva, bh_va)
    pva = _gate_mix(pva, m['p_hol_va'], bh_va, w_hol)
    pte = _gate_mix(m['p_base_te'], m['p_ext_te'], m_ext_te, w_ext)
    pte = _gate_mix(pte, m['p_hol_te'], bh_te, w_hol)
    row = dict(
        variant=label, holiday='v4_1', n_feats=len(feats),
        stack_val_mape=mape(yva, pva) * 100,
        lgb_best_iter=m['iters'].get('lgb'),
        hol_expert_iter=m['hol_iter'], ext_expert_iter=m['ext_iter'],
        hol_train_n=int(flags['bh_tr'].sum()), ext_train_n=int(flags['be_tr'].sum()),
        gate_w_hol=w_hol, gate_w_ext=w_ext,
        val_hol_seg_mape=hol_seg_mape, val_ext_seg_mape=ext_seg_mape,
    )
    for name in trees:
        row[f'{name}_val_mape'] = mape(yva, m['res_val'][name]) * 100
    row.update(segment_mape(yva, pva, segment_masks(dva, seg_lookup), 'stack_val'))
    if TARGET in df_test.columns:
        yt = merge_y(dte, df_test)
        row['stack_test_mape'] = mape(yt, pte) * 100
        for name in trees:
            row[f'{name}_test_mape'] = mape(yt, m['res_test'][name]) * 100
        row.update(segment_mape(yt, pte, segment_masks(dte, seg_lookup), 'stack_test'))
    print(f'  gate w_hol={w_hol:.2f} w_ext={w_ext:.2f}  '
          f'val_hol_seg={hol_seg_mape:.2f}% val_ext_seg={ext_seg_mape:.2f}%')
    pack = dict(dte=dte, yva=yva, pte=pte, res_test=m['res_test'], feats=feats,
                iters=m['iters'], w_hol=w_hol, w_ext=w_ext)
    return row, pack
