# Optiver Trading at the Close —— 特征工程文档

> 本文档整理自 [`training/optiver-258-lgb-submit.ipynb`](training/optiver-258-lgb-submit.ipynb) 中的
> `feature_pipeline()`，共 **258 个入模特征**。所有特征名与数量均由脚本
> [`feature_list_audit.py`](feature_list_audit.py) 实际执行该流水线后导出，非人工整理。
>
> 编写目的：作为**同类竞赛项目的特征工程参考**——这套"相对化"特征构造思路
> （时间差分 / 时间累计 / 横截面偏离 / 全局标度）可迁移到任何"多标的 × 日内高频 × 截面预测"的场景。

---

## 1. 背景与数据

**赛题**：Kaggle *Optiver - Trading at the Close*（2023）。预测纳斯达克收盘集合竞价阶段，
个股在未来 60 秒的收益表现。

| 项目 | 取值 | 说明 |
|---|---|---|
| 标的数 | **200**（`stock_id` 0–199） | 每只股票每个时间点一行 |
| 时间粒度 | **55 个桶**，`seconds_in_bucket` = 0–540，步长 10 | 每 10 秒一个快照，窗口约 9 分钟 |
| 训练期 | `date_id` 0–480（481 个交易日） | 官方 `train.csv` |
| 标签 | `target` | 未来 60 秒的股票收益相对市场收益的**残差** × 10000 |

### 1.1 原始字段

| 字段 | 含义 | 备注 |
|---|---|---|
| `stock_id` | 股票标识 | |
| `date_id` | 交易日序号 | |
| `seconds_in_bucket` | 当日桶内秒数 | 0–540，步长 10 |
| `imbalance_size` | 集合竞价未成交的失衡股数 | |
| `imbalance_buy_sell_flag` | 失衡方向 | +1 买 / −1 卖 / 0 均衡 |
| `reference_price` | 参考价 | |
| `matched_size` | 已撮合股数 | |
| `far_price` | 交叉竞价理论价（远） | **约 55% 缺失** |
| `near_price` | 交叉竞价理论价（近） | **约 55% 缺失** |
| `bid_price` / `ask_price` | 最优买价 / 卖价 | |
| `bid_size` / `ask_size` | 最优买量 / 卖量 | |
| `wap` | 加权平均价 | |
| `target` | 预测目标 | 训练集有，测试集需模拟 API 揭示 |

> `far_price` / `near_price` 仅在集合竞价的特定阶段有值，缺失率约 55%。
> 这会向下游传导：所有含这两个字段的不平衡、差分、偏离特征都会同步缺失。

---

## 2. 流水线总览

`feature_pipeline()` 是唯一的特征入口，内部严格顺序调用 9 个阶段。
**顺序不可调换**——第 1 阶段产出的 `_eng` 特征会被第 4/5/6 阶段按名字筛选后二次复用。

| # | 阶段 | 函数 | 新增 | 累计 |
|---|---|---|---|---|
| — | 原始表（读入后 `drop(['row_id','time_id'])`） | — | 15 | 15 |
| 1 | 手工四则运算 | `feature_engineering` | **11** | 26 |
| 2 | 盘口量不平衡 `_imb_sz_` | `compute_imbalances(..., '_sz_')` | **6** | 32 |
| 3 | 价格不平衡 `_imb_pr_` | `compute_imbalances(..., '_pr_')` | **15** | 47 |
| 4 | 日内差分 `_diff_lag{k}` | `create_diff_lagged_features_within_date_revised` | **133** | 180 |
| 5 | 日内累计 `_cumsum` | `create_cumsum_features` | **18** | 198 |
| 6 | 横截面偏离 | `create_deviation_within_seconds` | **25** | 223 |
| 7 | 目标滞后 `lag{k}_target` | `lag_function` | **12** | 235 |
| 8 | 全局统计 `global_*` | `map_global` | **20** | 255 |
| 9 | 目标滞后统计 | `calculate_stat_lag` | **5** | 260 |
| | **剔除** `date_id`、`target` | | **−2** | **258** |

**剔除规则**（在 notebook cell 11 定义）：

```python
excluded_columns = ['row_id', 'date_id', 'time_id', 'target', 'stock_return']
lgb_features = [col for col in train_eng.columns if col not in excluded_columns]
```

> `row_id` / `time_id` 已在读入时 drop；`stock_return` 不存在于公开 `train.csv`，属防御性写法。

### 2.1 三个模型的字段差异

| 模型 | 特征数 | 类别特征 | 备注 |
|---|---|---|---|
| LGB | **258** | `seconds_in_bucket` | `stock_id` 作为数值特征直接入模 |
| NN | 257 | `seconds_in_bucket` | 额外剔除 `stock_id`；数值特征做 `StandardScaler` |
| RNN | 257 | `seconds_in_bucket` | 同上，另按 `window_size=3` 切序列 |

> **设计要点**：树模型（LGB）**不做标准化**，保留原始量纲，让分裂点自行寻找阈值；
> 仅 NN / RNN 需要 `StandardScaler`。缺失值用**训练集中位数**填充。

---

## 3. 各特征族详解

**记号约定**

- $s$ = `stock_id`，$d$ = `date_id`，$t$ = `seconds_in_bucket`
- $X(s,d,t)$ = 某原始字段在 (股票, 日, 秒桶) 处的取值
- $k$ = 滞后步数，**1 步 = 10 秒**
- $I(\cdot)$ = 指示函数

---

### 3.1 手工四则运算特征 —— 11 个

由 `feature_engineering()` 直接构造，全部是盘口一阶量之间**有明确微观结构含义**的比值或差。

| 特征名 | 公式 | 设计意图 |
|---|---|---|
| `spread_eng` | $\text{ask\_price} - \text{bid\_price}$ | 买卖价差，流动性的直接度量 |
| `volume_eng` | $\text{bid\_size} + \text{ask\_size}$ | 盘口总量 |
| `volumne_imbalance_eng` | $\text{bid\_size} - \text{ask\_size}$ | 盘口量净失衡（**注意原代码拼写为 `volumne`**） |
| `weighted_imbalance_eng` | $\text{imbalance\_size} \times \text{imb\_buy\_sell\_flag}$ | **带符号**的失衡量；原注释标为 "very important" |
| `imbalance_ratio` | $\dfrac{\text{imbalance\_size}}{\text{matched\_size}}$ | 失衡量相对成交量 |
| `price_spread_near_far` | $\text{near\_price} - \text{far\_price}$ | 近远端理论价差 |
| `price_wap_difference_eng` | $\text{reference\_price} - \text{wap}$ | 参考价对成交均价偏离 |
| `bid_ask_ratio` | $\dfrac{\text{bid\_size}}{\text{ask\_size}}$ | 盘口量比 |
| `imbalance_to_bid_ratio_eng` | $\dfrac{\text{imbalance\_size}}{\text{bid\_size}}$ | 失衡量相对买盘 |
| `imbalance_to_ask_ratio_eng` | $\dfrac{\text{imbalance\_size}}{\text{ask\_size}}$ | 失衡量相对卖盘 |
| `matched_size_to_total_size_ratio_eng` | $\dfrac{\text{matched\_size}}{\text{bid\_size} + \text{ask\_size}}$ | 成交占盘口比 |

> **复用机制**：上表 11 个中有 **8 个以 `_eng` 结尾**（除 `imbalance_ratio`、`price_spread_near_far`、
> `bid_ask_ratio` 之外的 8 个）。流水线用子串匹配 `"_eng" in feature` 把它们筛出来，
> 在第 4/5/6 阶段当作"新原始字段"再做差分、累计、偏离。
> **这是本流水线最值得借鉴的工程手法：把构造出的高阶特征当作一等公民迭代加工。**

---

### 3.2 盘口不平衡特征 —— 21 个

核心公式（对任意两个字段 $A, B$）：

$$
\text{imb}(A,B) = \frac{A - B}{A + B}
$$

实现上对字段集合做 **C(n,2) 两两组合**，并按字典序排序保证命名稳定：

```python
for col1, col2 in combinations(columns, 2):
    col1, col2 = sorted([col1, col2])
    total = df[col1] + df[col2]
    df[f'{col1}_{col2}_imb{prefix}'] = (df[col1] - df[col2]).divide(total, fill_value=np.nan)
```

| 分组 | 源字段集 | 组合数 | 前缀 |
|---|---|---|---|
| **量不平衡** | `imbalance_size`, `matched_size`, `bid_size`, `ask_size` | C(4,2) = **6** | `_imb_sz_` |
| **价格不平衡** | `reference_price`, `far_price`, `near_price`, `bid_price`, `ask_price`, `wap` | C(6,2) = **15** | `_imb_pr_` |

命名示例：`ask_size_bid_size_imb_sz_`、`far_price_reference_price_imb_pr_`。

> **归一化价值**：$\frac{A-B}{A+B}$ 天然落在 $[-1, 1]$，把不同量纲的字段（价格 ~100、股数 ~10⁶）
> 拉到同一尺度，且对"整体放大/缩小"免疫。这比裸差值 $A-B$ 更适合跨股票、跨时段泛化。
>
> **注意**：价格不平衡 $\frac{P_1-P_2}{P_1+P_2}$ 在价格量级下数值极小（分母约 200，分子约 0.01 量级），
> 树模型仍可用，但若换用线性模型需要额外缩放。

---

### 3.3 日内差分特征 —— 133 个

$$
\Delta_k X(s,d,t) = X(s,d,t) - X(s,d,t-k)
$$

实现：按 `(stock_id, date_id)` 分组后 `shift(k)`。

```python
lagged_df = df.groupby(['stock_id', 'date_id'])[columns_to_lag].shift(periods=lag)
new_column = df[column] - lagged_df[column]
```

| 项 | 取值 |
|---|---|
| 源字段数 | **19** = 11 个原始字段 + 8 个 `_eng` 特征 |
| 滞后步长 | **7** 个：`[1, 2, 3, 6, 12, 18, 24]` |
| 合计 | 19 × 7 = **133** |

**步长的时间含义**（1 步 = 10 秒）：

| k | 1 | 2 | 3 | 6 | 12 | 18 | 24 |
|---|---|---|---|---|---|---|---|
| 实际秒数 | 10s | 20s | 30s | **1min** | **2min** | **3min** | **4min** |

> **尺度设计的意图**：步长在对数尺度上近似均匀（1,2,3,6,12,18,24），覆盖
> 短时噪声（10–30 秒）到中期趋势（1–4 分钟）多个尺度，且 6 的倍数（6/12/18/24）对齐整分钟。
> 整个收盘阶段仅 9 分钟，24 步（4 分钟）已接近可用历史的一半。
>
> **⚠️ 实现依赖**：`groupby().shift()` 按**行序**位移，而非按 `seconds_in_bucket` 显式对齐。
> 这要求同一 `(stock_id, date_id)` 内数据已按 `seconds_in_bucket` 升序排列。
> 若上游数据未排序，滞后的语义会被静默破坏——**迁移到其他项目时必须加显式排序**。

---

### 3.4 日内累计特征 —— 18 个

$$
C_X(s,d,t) = \sum_{t' \le t} X(s,d,t')
$$

实现：按 `(stock_id, date_id)` 分组做 `cumsum()`。

| 项 | 取值 |
|---|---|
| 源字段 | 4 个量字段 + 6 个 `_imb_sz_` + 8 个 `_eng` |
| 合计 | **18** |

**设计意图**：刻画**当日截至目前**的成交量与失衡累积进程。
单点快照无法区分"刚刚开始的放量"和"已持续 3 分钟的放量"，累计量则携带了这个状态信息。
配合 `seconds_in_bucket` 使用，模型可自行推算"单位时间增量"。

---

### 3.5 横截面偏离特征 —— 25 个

$$
D_X(s,d,t) = X(s,d,t) - \operatorname{median}_{s'}\big[X(s',d,t)\big]
$$

实现：按 `(date_id, seconds_in_bucket)` 分组取中位数，再逐行相减。

```python
grouped_median = df.groupby(['date_id', 'seconds_in_bucket'])[feature].transform('median')
df[f'deviation_from_median_{feature}'] = df[feature] - grouped_median
```

| 项 | 取值 |
|---|---|
| 源字段 | 11 个原始字段 + 8 个 `_eng` + 6 个 `_imb_sz_` |
| 合计 | **25** |

> **这是本方案最有价值的特征族之一。** 目标 `target` 本身就是"相对市场的残差收益"，
> 而该特征族把每个输入也转换成"相对全市场同期中位数的偏离"——**让特征与标签在同一坐标系下表达**。
> 同时它自动消除了市场级的共同因子（大盘整体放量、整体波动），让模型专注个股特异信号。
>
> 用**中位数**而非均值，对极端值与停牌股票的缺失更稳健。
>
> **⚠️ 注意**：分组键不含 `stock_id`，因此同一时刻全市场约 200 只股票同组。
> 若某时刻只有部分股票有数据，截面基准会随之漂移。

---

### 3.6 目标滞后特征 —— 12 个

$$
\text{lag}_k\text{\_target}(s,d,t) = \text{target}(s,\, d-k,\, t)
$$

实现：以 `(stock_id, seconds_in_bucket)` 为分组、按 `date_id` 排序后 `shift(k)`。

```python
df_indexed = df.set_index(['stock_id', 'seconds_in_bucket', 'date_id'])
df_indexed[f'lag{k}_target'] = df_indexed.groupby(
    level=['stock_id', 'seconds_in_bucket'])[col].shift(k)
```

**关键区别**：与 3.3 的**日内**差分不同，这里是**跨日**滞后——
取"同一只股票、同一秒桶、前 $k$ 个交易日"的目标值，$k \in [1, 12]$。

> **为何同秒对齐**：收盘竞价的微观结构随时间桶剧烈变化（越接近收盘，失衡越剧烈）。
> 用"昨天的同一秒"作参照，比"昨天的同一时刻但不区分秒"精确得多。
>
> **⚠️ 缺失模式**：交易日前 12 天内的样本，其 `lag12_target` 及部分高阶滞后为 NaN；
> 首个交易日（`date_id=0`）的全部 12 个滞后特征均缺失。

---

### 3.7 全局统计特征 —— 20 个

按 `stock_id` 在**全量数据**上聚合，刻画每只股票的静态标度（对同一只股票是常数）。

| 特征名 | 公式 |
|---|---|
| `global_{mean,median,std,min,max,q25,q75}_bid_size` | 对 `bid_size` 的对应聚合（7 个） |
| `global_{mean,median,std,min,max,q25,q75}_ask_size` | 对 `ask_size` 的对应聚合（7 个） |
| `global_median_size` | $\operatorname{median}(\text{bid\_size}) + \operatorname{median}(\text{ask\_size})$ |
| `global_std_size` | $\operatorname{std}(\text{bid\_size}) + \operatorname{std}(\text{ask\_size})$ |
| `global_ptp_size` | $\max(\text{bid\_size}) - \min(\text{bid\_size})$ |
| `global_median_price` | $\operatorname{median}(\text{bid\_price}) + \operatorname{median}(\text{ask\_price})$ |
| `global_std_price` | $\operatorname{std}(\text{bid\_price}) + \operatorname{std}(\text{ask\_price})$ |
| `global_ptp_price` | $\max(\text{bid\_price}) - \min(\text{ask\_price})$ |

实现方式是先构造成字典再 `map` 回原表：

```python
aggregated_dic = aggregated_features_dic(train)   # 在 cell 4 一次性算好
def map_global(df, dict):
    for key, value in dict.items():
        df_[f"global_{key}"] = df_["stock_id"].map(value.to_dict())
```

> **作用**：本质是给树模型一个**股票身份的连续型嵌入**。
> 不同股票的价格量级、盘口规模差异巨大（高价股 vs 低价股），
> 有了 `global_*` 后，模型可以把"绝对量"解读为"相对该股常态的偏离"。
>
> **⚠️ 三点隐患**（详见第 5 节）：
> 1. `global_ptp_price` 用 `bid_price` 的最大值减 `ask_price` 的最小值，**买卖两侧混用**，无干净的微观结构含义；
> 2. `median_price` / `std_price` 是买卖两侧统计量**相加**，不是价差，量纲上也不是价格；
> 3. 最关键——在全量数据（含验证期）上计算，构成**跨期信息泄漏**。

---

### 3.8 目标滞后统计特征 —— 5 个

对 3.6 的 12 个 `lag{k}_target` 做**行内**（`axis=1`）统计：

| 特征名 | 公式 |
|---|---|
| `target_mean` | $\frac{1}{12}\sum_{k=1}^{12} \text{lag}_k\text{\_target}$ |
| `target_std_dev` | $\operatorname{std}_{k}(\text{lag}_k\text{\_target})$ |
| `target_variance` | $\operatorname{var}_{k}(\text{lag}_k\text{\_target})$ |
| `target_median` | $\operatorname{median}_{k}(\text{lag}_k\text{\_target})$ |
| `target_range` | $\max_k - \min_k$ |

> **作用**：把 12 个离散滞后压缩成"该股票近 12 日的表现形态"——
> 均值代表历史收益水平，标准差/极差代表历史波动性。这是**时序特征的降维池化**，
> 用 5 个特征替代 12 个，降低维度并提升泛化。
>
> **推测**：notebook 的 `comments` 单元格对比了 "258 feat" 与 "253 feat" 两版配置，
> 差值为 5，很可能正是这一族是否存在。**此为推断，未在代码中直接验证。**

---

## 4. 可迁移的设计经验

这套流水线有四点值得直接搬到同类项目中：

**① 四类"相对化"视角构成完整坐标系。**
绝对量对金融数据几乎无用——价格 100 和 10 的股票、放量 100 万股和 1 万股的日子，
没有共同的比较基准。该方案用四种方式把绝对量转成相对量：

| 视角 | 相对谁 | 实现 |
|---|---|---|
| 时间差分 | 自己的过去 | `_diff_lag{k}` |
| 时间累计 | 自己的当日历史 | `_cumsum` |
| 横截面偏离 | 同期的其他股票 | `deviation_from_median_*` |
| 全局标度 | 自己的历史常态 | `global_*` |

**② 特征可迭代。** 把 8 个 `_eng` 特征当作新的"原始字段"再次做差分/累计/偏离，
用一行 `[c for c in df.columns if "_eng" in c]` 实现自动筛选。
这让"手工特征"和"派生特征"没有层级壁垒——**新加一个手工特征会自动获得 20+ 个派生版本**。

**③ 用集合运算代替手工枚举。**
`compute_imbalances` 接收任意字段列表，自动做 C(n,2) 组合并生成规范命名
（字典序排序保证可复现）。新增一个字段只需加进列表，自动获得与所有已有字段的配对特征。
**这是"低成本扩展特征空间"最有效的模式。**

**④ 建模阶段不做标准化。**
树模型对单调变换免疫，保留原始量纲反而让分裂点更易解释、无需保存 scaler 状态；
只有 NN/RNN 才 `StandardScaler`。特征工程与模型选择应当解耦考虑。

---

## 5. 数据泄漏与实现陷阱

迁移到其他项目时，以下几处**必须修正或至少知悉**：

### 5.1 全局统计特征存在跨期泄漏（最严重）

```python
aggregated_dic = aggregated_features_dic(train)   # cell 4：用完整 train 计算
...
train_data = split_by_date(train_eng, dates_train)   # cell 11：之后才切分
test_data  = split_by_date(train_eng, dates_test)
```

`global_*` 在**全量训练集**（含验证期 `date_id` 391–480）上计算，然后映射进验证集。
这意味着验证集的特征里混入了验证期自身的统计信息，**验证分数会偏乐观**。

notebook 的 `dates_train = [0, 480]`（全量）与 `[0, 390] / [391, 480]`（公开验证）两种配置下，
泄漏程度不同——前者无验证集概念，后者存在泄漏。

**修正方式**：只在训练期上计算 `global_*`，再映射到验证/测试期。

### 5.2 日内差分依赖行序，非显式时间对齐

`groupby(['stock_id','date_id']).shift(k)` 按行位移。若数据未按 `seconds_in_bucket` 升序，
`_diff_lag1` 就不再是"10 秒前"。**迁移时必须显式 `sort_values(['stock_id','date_id','seconds_in_bucket'])`**，
或改用按桶号显式对齐的写法。

### 5.3 除零与无穷大

- `imbalance_ratio` 分母 `matched_size` 可为 0；
- `compute_imbalances` 的 $\frac{A-B}{A+B}$ 在 $A=B=0$ 时得到 `0/0`。

代码在流水线末尾统一兜底：

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

但 **NaN 本身未在此处理**，而是留到建模阶段用训练集中位数填充
（`train_eng.fillna(medians, inplace=True)`）。注意 `.fillna()` 是**原地**修改，
在 NN/RNN 分支中会污染后续 LGB 分支共用的 `train_eng`。

### 5.4 缺失字段的下游传导

`far_price` / `near_price` 缺失约 55%，导致：
- 含它们的 **6 个价格不平衡**特征（`far_price_*` / `near_price_*` 配对）同步缺失；
- `price_spread_near_far` 及其 **7 个差分**、**1 个累计**、**1 个偏离**版本同步缺失。

由于 `shift()` 会把缺失向后传播，高阶滞后特征的缺失率可能远超 55%——**建议建模前实测各特征缺失率**。

### 5.5 目标滞后的冷启动

`lag{k}_target` 前 12 个交易日全为 NaN，`date_id = 0` 完全无历史。
若用时间序列切分验证，训练集首段样本的这些特征基本无用。

### 5.6 命名拼写错误

`volumne_imbalance_eng` 中 `volumne` 是 `volume` 的**拼写错误**（原代码如此）。
该名字作为字符串被下游逻辑引用（`"_eng" in feature` 以及所有派生特征名），
**重命名需全链路同步修改**，不能只改一处。

### 5.7 未启用函数的实现缺陷

`create_autocorrelation_features`（未启用）存在实现问题：

```python
df_copy[f'{column}_autocorr_lag{lag}'] = df_copy[column].corrwith(lagged_series)
```

`Series.corrwith(Series)` 返回的是**一个标量**（整列与滞后列的总体皮尔逊相关系数），
而非逐行特征。赋值后该列恒为常数，**没有预测价值**。
若想实现"滚动自相关"，应使用 `rolling(window).corr(...)`。
虽然当前未启用，但迁移时若照搬会踩坑。

---

## 6. 已定义但未启用的特征函数

notebook 中另有 5 个函数定义了却从未被 `feature_pipeline()` 调用（经全 notebook 调用点扫描确认）。
它们代表**预留的扩展方向**：

| 函数 | 产出公式 | 状态与说明 |
|---|---|---|
| `create_features_to_start_optimized` | $X(s,d,t) - X(s,d,\text{open})$，即相对**当日首个桶**的偏移 | 未启用。与 `_diff_lag` 互补：给出相对开盘的**累积**偏移基准 |
| `compute_percentage_difference` | $\dfrac{A-B}{B} \times 100$ | 未启用。与 `_imb_pr_` 是同一族的不同归一化方式（百分比差 vs 对称归一化），可作 A/B 对照 |
| `calculate_stat` | 对任意列集合做行内 `mean/std/var/median/range` | 未启用。是 3.8 `calculate_stat_lag` 的**泛化版本**（可作用于价格字段而非仅 target 滞后） |
| `create_autocorrelation_features` | 名义上为自相关特征 | 未启用，且**实现有误**（见 5.7） |
| `flatten_outliers_y_train` | $\text{clip}(y, Q_{0.01}, Q_{0.99})$ | 未启用。作用于**标签**而非特征：分位数截断以抑制 `target` 的极端值 |

> **另外两处被注释掉的开关**（在 `feature_pipeline` 内）：
> - `deviation_cols` 中 `#+ imb_features_price` 被注释 → 价格不平衡**未参与**横截面偏离（差 15 个特征）；
> - `colsample_bytree` 与 `num_leaves` 在 LGB 参数中有多组候选，`comments` 单元格记录了对应的验证分数对比。

---

## 7. 如何复现特征清单

[`feature_list_audit.py`](feature_list_audit.py) 会：
1. 从 notebook 中**原样提取** cell 3/4/5/6 的函数定义并执行；
2. 用符合 Optiver 表结构的合成数据跑通 `feature_pipeline()`；
3. 逐阶段追踪新增列，输出精确的特征名与数量。

```bash
python feature_list_audit.py
```

**实测输出**：

```text
流水线输出总列数            : 260
LGB 实际入模特征数          : 258
  其中 seconds_in_bucket 作为 categorical
```

258 这个数字正是 notebook 文件名中 "258" 的来源。

> 合成数据仅用于**提取特征结构与命名**，数值无意义。要复现分数需下载官方
> `train.csv`（约 5 GB）并按 cell 2 的路径放置。

---

## 8. 参考

- 竞赛主页：<https://www.kaggle.com/competitions/optiver-trading-at-the-close>
- 方案来源：Kaggle 公开 notebook，作者 `nimashahbazi`（见 [README.md](README.md)）
- 榜单成绩：LSTM 5.3508 / ConvNet 5.3439（公众榜 MAE，越低越好）

---
## 附录 A：258 个入模特征全量清单

本附录由 `_feat_extract.py` 实际执行 notebook 中的 `feature_pipeline()` 后导出，非手工整理，可直接用于比对。

下表按特征族分组。命名规则中 `{X}` 表示源字段、`{k}` 表示滞后步数、`{A}/{B}` 表示配对字段（按字典序排序后的两个字段名）。

#### A. 原始行情字段（直接入模，未做变换） —— 11 个

> 保留原始量纲，供树模型自行切分。

```text
imbalance_size
matched_size
bid_size
ask_size
reference_price
far_price
near_price
bid_price
ask_price
wap
imbalance_buy_sell_flag
```

#### B. 标识与类别字段 —— 2 个

> `stock_id` 仅 LGB 使用（NN/RNN 会剔除）；`seconds_in_bucket` 全程作为 categorical 特征。

```text
stock_id
seconds_in_bucket
```

#### C. 手工四则运算特征 —— 11 个

> 由 `feature_engineering()` 直接构造。其中 8 个以 `_eng` 结尾者会被第 4/5/6 阶段二次复用。

```text
spread_eng
volume_eng
volumne_imbalance_eng
imbalance_ratio
price_spread_near_far
price_wap_difference_eng
weighted_imbalance_eng
bid_ask_ratio
imbalance_to_bid_ratio_eng
imbalance_to_ask_ratio_eng
matched_size_to_total_size_ratio_eng
```

#### D. 盘口量不平衡特征（`_imb_sz_`） —— 6 个

> 4 个 size 字段两两配对，C(4,2)=6。

```text
imbalance_size_matched_size_imb_sz_
bid_size_imbalance_size_imb_sz_
ask_size_imbalance_size_imb_sz_
bid_size_matched_size_imb_sz_
ask_size_matched_size_imb_sz_
ask_size_bid_size_imb_sz_
```

#### E. 价格不平衡特征（`_imb_pr_`） —— 15 个

> 6 个价格字段两两配对，C(6,2)=15。

```text
far_price_reference_price_imb_pr_
near_price_reference_price_imb_pr_
bid_price_reference_price_imb_pr_
ask_price_reference_price_imb_pr_
reference_price_wap_imb_pr_
far_price_near_price_imb_pr_
bid_price_far_price_imb_pr_
ask_price_far_price_imb_pr_
far_price_wap_imb_pr_
bid_price_near_price_imb_pr_
ask_price_near_price_imb_pr_
near_price_wap_imb_pr_
ask_price_bid_price_imb_pr_
bid_price_wap_imb_pr_
ask_price_wap_imb_pr_
```

#### F. 日内差分特征（`_diff_lag{k}`） —— 133 个

> 19 个源字段 × 7 个滞后步长 [1, 2, 3, 6, 12, 18, 24] = 133。k 为**行位移**，数据为 10 秒粒度，故 k=1/2/3/6/12/18/24 对应 10s/20s/30s/1min/2min/3min/4min。

按源字段分组（每个源字段各有 7 个滞后版本）：

```text
imbalance_size                         -> imbalance_size_diff_lag1, imbalance_size_diff_lag2, imbalance_size_diff_lag3, imbalance_size_diff_lag6, imbalance_size_diff_lag12, imbalance_size_diff_lag18, imbalance_size_diff_lag24
matched_size                           -> matched_size_diff_lag1, matched_size_diff_lag2, matched_size_diff_lag3, matched_size_diff_lag6, matched_size_diff_lag12, matched_size_diff_lag18, matched_size_diff_lag24
bid_size                               -> bid_size_diff_lag1, bid_size_diff_lag2, bid_size_diff_lag3, bid_size_diff_lag6, bid_size_diff_lag12, bid_size_diff_lag18, bid_size_diff_lag24
ask_size                               -> ask_size_diff_lag1, ask_size_diff_lag2, ask_size_diff_lag3, ask_size_diff_lag6, ask_size_diff_lag12, ask_size_diff_lag18, ask_size_diff_lag24
reference_price                        -> reference_price_diff_lag1, reference_price_diff_lag2, reference_price_diff_lag3, reference_price_diff_lag6, reference_price_diff_lag12, reference_price_diff_lag18, reference_price_diff_lag24
far_price                              -> far_price_diff_lag1, far_price_diff_lag2, far_price_diff_lag3, far_price_diff_lag6, far_price_diff_lag12, far_price_diff_lag18, far_price_diff_lag24
near_price                             -> near_price_diff_lag1, near_price_diff_lag2, near_price_diff_lag3, near_price_diff_lag6, near_price_diff_lag12, near_price_diff_lag18, near_price_diff_lag24
bid_price                              -> bid_price_diff_lag1, bid_price_diff_lag2, bid_price_diff_lag3, bid_price_diff_lag6, bid_price_diff_lag12, bid_price_diff_lag18, bid_price_diff_lag24
ask_price                              -> ask_price_diff_lag1, ask_price_diff_lag2, ask_price_diff_lag3, ask_price_diff_lag6, ask_price_diff_lag12, ask_price_diff_lag18, ask_price_diff_lag24
wap                                    -> wap_diff_lag1, wap_diff_lag2, wap_diff_lag3, wap_diff_lag6, wap_diff_lag12, wap_diff_lag18, wap_diff_lag24
imbalance_buy_sell_flag                -> imbalance_buy_sell_flag_diff_lag1, imbalance_buy_sell_flag_diff_lag2, imbalance_buy_sell_flag_diff_lag3, imbalance_buy_sell_flag_diff_lag6, imbalance_buy_sell_flag_diff_lag12, imbalance_buy_sell_flag_diff_lag18, imbalance_buy_sell_flag_diff_lag24
spread_eng                             -> spread_eng_diff_lag1, spread_eng_diff_lag2, spread_eng_diff_lag3, spread_eng_diff_lag6, spread_eng_diff_lag12, spread_eng_diff_lag18, spread_eng_diff_lag24
volume_eng                             -> volume_eng_diff_lag1, volume_eng_diff_lag2, volume_eng_diff_lag3, volume_eng_diff_lag6, volume_eng_diff_lag12, volume_eng_diff_lag18, volume_eng_diff_lag24
volumne_imbalance_eng                  -> volumne_imbalance_eng_diff_lag1, volumne_imbalance_eng_diff_lag2, volumne_imbalance_eng_diff_lag3, volumne_imbalance_eng_diff_lag6, volumne_imbalance_eng_diff_lag12, volumne_imbalance_eng_diff_lag18, volumne_imbalance_eng_diff_lag24
price_wap_difference_eng               -> price_wap_difference_eng_diff_lag1, price_wap_difference_eng_diff_lag2, price_wap_difference_eng_diff_lag3, price_wap_difference_eng_diff_lag6, price_wap_difference_eng_diff_lag12, price_wap_difference_eng_diff_lag18, price_wap_difference_eng_diff_lag24
weighted_imbalance_eng                 -> weighted_imbalance_eng_diff_lag1, weighted_imbalance_eng_diff_lag2, weighted_imbalance_eng_diff_lag3, weighted_imbalance_eng_diff_lag6, weighted_imbalance_eng_diff_lag12, weighted_imbalance_eng_diff_lag18, weighted_imbalance_eng_diff_lag24
imbalance_to_bid_ratio_eng             -> imbalance_to_bid_ratio_eng_diff_lag1, imbalance_to_bid_ratio_eng_diff_lag2, imbalance_to_bid_ratio_eng_diff_lag3, imbalance_to_bid_ratio_eng_diff_lag6, imbalance_to_bid_ratio_eng_diff_lag12, imbalance_to_bid_ratio_eng_diff_lag18, imbalance_to_bid_ratio_eng_diff_lag24
imbalance_to_ask_ratio_eng             -> imbalance_to_ask_ratio_eng_diff_lag1, imbalance_to_ask_ratio_eng_diff_lag2, imbalance_to_ask_ratio_eng_diff_lag3, imbalance_to_ask_ratio_eng_diff_lag6, imbalance_to_ask_ratio_eng_diff_lag12, imbalance_to_ask_ratio_eng_diff_lag18, imbalance_to_ask_ratio_eng_diff_lag24
matched_size_to_total_size_ratio_eng   -> matched_size_to_total_size_ratio_eng_diff_lag1, matched_size_to_total_size_ratio_eng_diff_lag2, matched_size_to_total_size_ratio_eng_diff_lag3, matched_size_to_total_size_ratio_eng_diff_lag6, matched_size_to_total_size_ratio_eng_diff_lag12, matched_size_to_total_size_ratio_eng_diff_lag18, matched_size_to_total_size_ratio_eng_diff_lag24
```

#### G. 日内累计特征（`_cumsum`） —— 18 个

> 按 (stock_id, date_id) 对当日已发生的秒级观测做累加，刻画当日成交量/失衡的累计进程。

```text
imbalance_size_cumsum
matched_size_cumsum
bid_size_cumsum
ask_size_cumsum
imbalance_size_matched_size_imb_sz__cumsum
bid_size_imbalance_size_imb_sz__cumsum
ask_size_imbalance_size_imb_sz__cumsum
bid_size_matched_size_imb_sz__cumsum
ask_size_matched_size_imb_sz__cumsum
ask_size_bid_size_imb_sz__cumsum
spread_eng_cumsum
volume_eng_cumsum
volumne_imbalance_eng_cumsum
price_wap_difference_eng_cumsum
weighted_imbalance_eng_cumsum
imbalance_to_bid_ratio_eng_cumsum
imbalance_to_ask_ratio_eng_cumsum
matched_size_to_total_size_ratio_eng_cumsum
```

#### H. 横截面偏离特征（`deviation_from_median_*`） —— 25 个

> 按 (date_id, seconds_in_bucket) 分组取**跨股票中位数**，再做差——衡量个股相对全市场的同步偏离。

```text
deviation_from_median_imbalance_size
deviation_from_median_matched_size
deviation_from_median_bid_size
deviation_from_median_ask_size
deviation_from_median_reference_price
deviation_from_median_far_price
deviation_from_median_near_price
deviation_from_median_bid_price
deviation_from_median_ask_price
deviation_from_median_wap
deviation_from_median_imbalance_buy_sell_flag
deviation_from_median_spread_eng
deviation_from_median_volume_eng
deviation_from_median_volumne_imbalance_eng
deviation_from_median_price_wap_difference_eng
deviation_from_median_weighted_imbalance_eng
deviation_from_median_imbalance_to_bid_ratio_eng
deviation_from_median_imbalance_to_ask_ratio_eng
deviation_from_median_matched_size_to_total_size_ratio_eng
deviation_from_median_imbalance_size_matched_size_imb_sz_
deviation_from_median_bid_size_imbalance_size_imb_sz_
deviation_from_median_ask_size_imbalance_size_imb_sz_
deviation_from_median_bid_size_matched_size_imb_sz_
deviation_from_median_ask_size_matched_size_imb_sz_
deviation_from_median_ask_size_bid_size_imb_sz_
```

#### I. 目标滞后特征（`lag{k}_target`） —— 12 个

> 跨日滞后：同 `stock_id`、同 `seconds_in_bucket`、前 k 个交易日的 target。

```text
lag1_target
lag2_target
lag3_target
lag4_target
lag5_target
lag6_target
lag7_target
lag8_target
lag9_target
lag10_target
lag11_target
lag12_target
```

#### J. 全局统计特征（`global_*`） —— 20 个

> 按 `stock_id` 在**全量数据**上聚合得到的个股标度刻画，对同一只股票为常数。

```text
global_mean_bid_size
global_median_bid_size
global_std_bid_size
global_min_bid_size
global_max_bid_size
global_q25_bid_size
global_q75_bid_size
global_mean_ask_size
global_median_ask_size
global_std_ask_size
global_min_ask_size
global_max_ask_size
global_q25_ask_size
global_q75_ask_size
global_median_size
global_std_size
global_ptp_size
global_median_price
global_std_price
global_ptp_price
```

#### K. 目标滞后统计特征 —— 5 个

> 对 12 个 `lag{k}_target` 做**行内**（axis=1）统计，概括该股票近 12 日的历史表现形态。

```text
target_mean
target_std_dev
target_variance
target_median
target_range
```

### 附录校验

- 分类合计：**258** 个
- 流水线实际 LGB 入模特征数：**258** 个
- 一致性：✅ 完全一致
