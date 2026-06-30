# 竞赛记忆卡

> 目标：预测 **15 分钟粒度** 负荷 `全口径发购`。

---

## 一、全员必背（30 秒版）

| 项目 | 内容 |
|------|------|
| **口诀** | `params → prepare → trees → stack` |
| **环境** | `jingsai` · CPU · 跑 `work.ipynb` |
| **输入** | `data/实验数据1.xlsx`（hist）+ `data/data_test1.xlsx`（5 月 test） |
| **输出** | `final_trees.csv`（列：`date`, `全口径发购`） |
| **配置** | `params.json`：`val_start` / `trees` / `stack_alpha` |
| **铁律** | **hist 与 test 必须先拼接，再做 lag/滚动特征**（否则 5 月 lag 断档） |

```bash
conda activate jingsai
cd comp
jupyter notebook work.ipynb   # 从上到下 Run All
```

---

## 二、竞赛在做什么（流程图）

```
params.json
    ↓
[B] prepare()          洗数据 + hist∥test 拼接 → df_all
    ↓
[C] build(full=False)  7 维 debug 树（先跑通）
    ↓
[C] build(full=True)   31 维全特征
    ↓
[A] train_trees()      lgb + xgb + cat（各 500 轮，ES=50）
    ↓
[A] stack_v1()         Ridge 融合三棵树
    ↓
final_trees.csv        提交用预测（5 月 2880 点）
```

**数据切分（默认 `val_start = 2025-04-03`）**

| 集合 | 来源 | 时间范围（默认） | 用途 |
|------|------|------------------|------|
| train | `_src=='hist'` 且 `< val_start` | ~2022-01 ~ 2025-04-02 | 训树 |
| val | `_src=='hist'` 且 `≥ val_start` | 2025-04-03 ~ 2025-04-30 | early stop + stack 拟合 |
| test | `_src=='test'` | 2025-05-01 ~ 2025-05-30 | 最终预测评估 |

---

## 三、文件地图（比赛）

| 文件 | 标签 | 职责 |
|------|------|------|
| `lib_clean.py` | **[B]** | 读 xlsx、清洗、打 `_src` |
| `lib_feat.py` | **[C]** | `build()`：7 维 / 31 维特征 |
| `lib_comp.py` | **[A]** | 切分、三棵树、`stack`、一键 `run_baseline()` |
| `work.ipynb` | — | 比赛主流程（按 cell 顺序跑） |
| `params.json` | — | 唯一配置文件 |

---

## 四、分人记忆（谁背什么）

### 1. 数据同学 — **[B] `lib_clean.py`**

**口诀：去重 → 补 15min → IQR 截尾 → 插值**

| 步骤 | 做什么 |
|------|--------|
| 去重 | 按 `date` 去重，保留第一条 |
| 补全 | `reindex` 到连续 15min 网格 |
| IQR | 目标列 `全口径发购`：3×IQR 外截到边界 |
| 插值 | `interpolate('time')` + `ffill` + `bfill` |

**必记常量**

- `TIME = 'date'`，`TARGET = '全口径发购'`，`FREQ = '15min'`
- hist 打 `_src='hist'`，test 打 `_src='test'`
- `prepare()` = `load()` + `concat(hist, test)`（**lag 的前置条件**）

**API**

```python
from lib_comp import prepare
df_hist, df_test, df_all = prepare(save_hist=True)
```

---

### 2. 特征同学 — **[C] `lib_feat.py`**

**口诀：时 8 + 滞 4 + 滑 2 + 傅 10 = 24 新 + 7 原 = 31**

| 类型 | 列 |
|------|-----|
| **7 原** | `降水(mm)` `湿度(%)` `温度(℃)` `平均风速(m/s)` `DATA_WEEK` `IS_ZM` `IS_FDJJR` |
| **时 8** | `hour` `dayofweek` `month` `hour_sin/cos` `dow_sin/cos` `is_weekend` |
| **滞 4** | `lag_96` `lag_192` `lag_672` `lag_35040`（35040 = 365×96） |
| **滑 2** | `rolling_mean_96` `rolling_mean_672` |
| **傅 10** | 日周期 sin/cos k=1,2,3 + 周周期 sin/cos k=1,2 |

**必记**

- `build(df, full=False)` → 仅 7 原（debug）
- `build(df, full=True)` → 31 维（正式）
- 缺失：`ffill` + `fillna(0)`

---

### 3. 模型同学 — **[A] `lib_comp.py`**

**口诀：三树各训 → val 上 early stop → Ridge stack**

**三棵树（默认相同骨架）**

| 模型 | 关键参数 |
|------|----------|
| lgb | `max_depth=6`, `num_leaves=31`, `n_estimators=500`, `lr=0.1`, ES=50 |
| xgb | `max_depth=6`, `n_estimators=500`, ES=50 |
| cat | `depth=6`, `iterations=500`, ES=50 |

**stack**

- 元学习器：`Ridge(alpha=stack_alpha)`，默认 `alpha=1.0`
- 在 **val 预测** 上拟合权重，再预测 test

**API**

```python
pack = train_trees(df_all, params['val_start'], params['trees'], full=True)
pva, pte, meta = stack_v1(pack, alpha=params['stack_alpha'])
```

**一键（脚本调试用）**

```python
from lib_comp import run_baseline
run_baseline()  # → final_trees.csv
```

---

### 4. 集成 / 提交同学 — **`work.ipynb`**

**按 cell 顺序，不要跳步**

1. 读 `params.json`，import `lib_comp`
2. `prepare()` — 洗 + 拼接
3. `train_trees(..., full=False)` — debug 7 维（确认环境 OK）
4. `train_trees(..., full=True)` — 正式 31 维
5. `stack_v1(pack)` — 写 **`final_trees.csv`**
6. 可选：汇总 `pred_test_all.csv`、画最近 7 天曲线

**提交检查清单**

- [ ] Kernel = `jingsai`
- [ ] `final_trees.csv` 有 2880 行（5 月 15min）
- [ ] 列名 exactly：`date`, `全口径发购`

---

## 五、`params.json` 字段说明

```json
{
  "val_start": "2025-04-03",
  "trees": ["lgb", "xgb", "cat"],
  "stack_alpha": 1.0
}
```

| 字段 | 含义 | 谁改 |
|------|------|------|
| `val_start` | hist 训练/验证分界日 | 队长 / 模型同学（短 val 贴 5 月前） |
| `trees` | 基模型列表与顺序 | 模型同学 |
| `stack_alpha` | Ridge 正则 | 模型同学（一般保持 1.0） |

---

## 六、指标与评估

- **指标**：MAPE（`sklearn.metrics.mean_absolute_percentage_error`）
- **val MAPE**：stack 在 4 月验证段上的误差（调参看这个）
- **test MAPE**：5 月有标签时本地对比；**比赛不以 test 为准则改切分**

`compare()` 打印单模型 val MAPE 排序，便于看三棵树谁强。

---

## 七、常见坑

1. **先 concat 再 `build()`** — 否则 test 段 `lag_35040` 等为 NaN/0。
2. **val / test 不要混** — train 只用 hist 的 `< val_start`；test 行不参与训练。
3. **改 `val_start` 会改变整链** — 训练样本、ES 轮数、stack 权重全变，test 预测也会变。

---

## 八、一句话分工速查

| 同学 | 背一句 | 负责文件 |
|------|--------|----------|
| 数据 | 去重补格 IQR 插值，hist test 先拼 | `lib_clean.py` |
| 特征 | 时 8 滞 4 滑 2 傅 10，共 31 维 | `lib_feat.py` |
| 模型 | lgb xgb cat 各 500 轮，Ridge stack | `lib_comp.py` |
| 提交 | params → prepare → trees → stack → csv | `work.ipynb` |

**全队同一句话：`params → prepare → trees → stack → final_trees.csv`**

---

---

# 附录：非比赛实验（`comp/tune/`）

> 以下仅供本地探索，**不上交、不写入比赛默认路径**。验证稳定后由负责人单独合入 `work.ipynb` / `lib_feat.py`。

## 环境与入口

```bash
conda activate dinov3
cd comp/tune
pip install -r requirements_extra.txt
```

| Notebook | 做什么 |
|----------|--------|
| `test.ipynb` | 节假日特征 v1/v2b/v3、v4_1 门控；树 **1500 轮** |
| `tune.ipynb` | RNN Optuna 调参（PyTorch GPU） |

## 实验代码文件

| 文件 | 职责 |
|------|------|
| `lib_feat_ext.py` | 节假日特征（在 baseline `build()` 上追加） |
| `lib_exp.py` | 长训树、分段 MAPE、v4_1 专家门控 |
| `lib_model.py` | Keras RNN（jingsai 可选，比赛已不用） |
| `lib_tune.py` | PyTorch RNN 训练与 Optuna |

## 实验同学必记

- **口诀**：`dinov3` + `cd tune` + `test.ipynb` / `tune.ipynb`
- `VAL_START_OVERRIDE` 只改 notebook 内切分，**不要用 test 反推 `val_start` 上比赛**
- 节假日方案对比分段：`full` / `may1_5` / `fdjjr` / `pre7`
- 实验树默认 `n_estimators=1500`，比赛仍为 **500**
- 日常提交 **不要** import `tune/` 下任何模块

## 实验常见坑

1. **`comp/test.ipynb` 若为旧副本** — 以 `tune/test.ipynb` 为准。
2. **改 `val_start` 同样会改变实验链** — 与比赛逻辑一致，仅用于对比，别用 test 过拟合切分。
3. **RNN 已从 `work.ipynb` 移除** — 仅在 `tune/` 研究，比赛环境无需 PyTorch。

## 实验分工速查

| 方向 | 背一句 | 文件 |
|------|--------|------|
| 节假日 | v1/v2b/v3 对比，合入前看五一分段 | `lib_feat_ext.py` + `test.ipynb` |
| 门控专家 | v4_1：平时 stack，假期/极端窗口混专家 | `lib_exp.py` |
| RNN | dinov3 GPU 调参，不写回比赛 | `lib_tune.py` + `tune.ipynb` |
