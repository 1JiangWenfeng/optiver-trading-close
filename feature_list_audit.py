# -*- coding: utf-8 -*-
"""
从 optiver-258-lgb-submit.ipynb 中提取并实跑特征流水线，输出精确的特征清单。

做三件事：
  1. 原样执行 notebook 的 cell 3/4/5/6（函数定义 + aggregated_dic），保证与原文一致；
  2. 用符合 Optiver 表结构的合成数据跑通 feature_pipeline()；
  3. 逐阶段追踪新增列，输出每个特征族的精确名称与数量。

合成数据只用于提取**特征结构与命名**，数值无意义。
用法：python feature_list_audit.py
"""
import json
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "training", "optiver-258-lgb-submit.ipynb")
# Notebook 中实际用于计算全局统计特征的 cell 编号
CELLS_WITH_FUNCTIONS = (3, 4, 5, 6)


def make_synthetic_train(seed=0, n_stocks=3, n_dates=3):
    """构造一份符合 Optiver train.csv 表结构的数据（读入后已 drop row_id / time_id）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for stock_id in range(n_stocks):
        for date_id in range(n_dates):
            for sib in range(0, 550, 10):
                # far_price / near_price 按官方数据特征保留约 55% 缺失
                missing_quote = sib % 100 == 0
                rows.append(dict(
                    stock_id=stock_id, date_id=date_id, seconds_in_bucket=sib,
                    imbalance_size=rng.normal(1e6, 1e5),
                    imbalance_buy_sell_flag=rng.integers(-1, 2),
                    reference_price=rng.normal(100, 1),
                    matched_size=rng.uniform(1e5, 5e6),
                    far_price=np.nan if missing_quote else rng.normal(100, 1),
                    near_price=np.nan if missing_quote else rng.normal(100, 1),
                    bid_price=rng.normal(100, 1),
                    bid_size=rng.uniform(1e4, 1e6),
                    ask_price=rng.normal(100, 1),
                    ask_size=rng.uniform(1e4, 1e6),
                    wap=rng.normal(100, 1),
                    target=rng.normal(0, 5),
                ))
    return pd.DataFrame(rows)


def load_notebook_namespace(train):
    """执行 notebook 中的函数定义 cell，返回其命名空间。"""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    ns = {"pd": pd, "np": np, "os": os, "pickle": pickle,
          "combinations": combinations, "train": train}
    for i in CELLS_WITH_FUNCTIONS:
        exec("".join(nb["cells"][i]["source"]), ns)
    return ns


def main():
    train = make_synthetic_train()
    ns = load_notebook_namespace(train)
    raw_cols = ns["raw_cols"]
    columns_sizes = ns["columns_sizes"]
    columns_prices = ns["columns_prices"]
    diff_lags = ns["diff_lags"]

    stages = []
    df = train.copy()

    def step(name, fn, note=""):
        """执行一步并记录新增列。"""
        nonlocal df
        before = set(df.columns)
        df = fn(df)
        stages.append((name, [c for c in df.columns if c not in before], note))

    # 1. 手工四则运算（feature_engineering 内部，需单独追踪以便分离 _eng 特征）
    before = set(df.columns)
    df = ns["feature_engineering"](df)
    manual_all = [c for c in df.columns if c not in before]
    manual_eng = [c for c in manual_all if "_eng" in c]
    stages.append((
        "1. feature_engineering —— 手工四则运算特征", manual_all,
        f"其中 {len(manual_eng)} 个以 _eng 结尾，会被后续阶段二次复用；"
        f"{len(manual_all) - len(manual_eng)} 个为独立比率"))

    # 2/3. 盘口量、价格不平衡
    step("2. compute_imbalances(columns_sizes, '_sz_')",
         lambda d: ns["compute_imbalances"](d, columns_sizes, prefix="_sz_"),
         f"{len(columns_sizes)} 个 size 字段两两组合 C({len(columns_sizes)},2)="
         f"{len(columns_sizes)*(len(columns_sizes)-1)//2}")

    step("3. compute_imbalances(columns_prices, '_pr_')",
         lambda d: ns["compute_imbalances"](d, columns_prices, prefix="_pr_"),
         f"{len(columns_prices)} 个价格字段两两组合 C({len(columns_prices)},2)="
         f"{len(columns_prices)*(len(columns_prices)-1)//2}")

    # 4. 日内差分：源字段 = 原始字段 + _eng 特征
    imb_features_size = [c for c in df.columns if "_sz_" in c]
    diff_lag_cols = raw_cols + manual_eng
    step("4. create_diff_lagged_features_within_date_revised",
         lambda d: ns["create_diff_lagged_features_within_date_revised"](d, diff_lag_cols, diff_lags),
         f"{len(diff_lag_cols)} 列 × {len(diff_lags)} 个滞后 = {len(diff_lag_cols)*len(diff_lags)}")

    # 5. 日内累计
    cumsum_columns = columns_sizes + imb_features_size + manual_eng
    step("5. create_cumsum_features",
         lambda d: ns["create_cumsum_features"](d, cumsum_columns),
         f"{len(cumsum_columns)} 列日内累加")

    # 6. 横截面偏离
    deviation_cols = raw_cols + manual_eng + imb_features_size
    step("6. create_deviation_within_seconds",
         lambda d: ns["create_deviation_within_seconds"](d, deviation_cols),
         f"{len(deviation_cols)} 列减去同秒截面中位数")

    # 7. 目标滞后
    step("7. lag_function(['target'], 1..12)",
         lambda d: ns["lag_function"](d, ["target"], ns["target_lags"]),
         f"{ns['num_of_target_lags']} 个目标滞后")

    # 8. 全局统计
    step("8. map_global",
         lambda d: ns["map_global"](d, ns["aggregated_dic"]),
         f"{len(ns['aggregated_dic'])} 个 stock 级全局统计")

    # 9. 目标滞后统计
    step("9. calculate_stat_lag(12)",
         lambda d: ns["calculate_stat_lag"](d, num_lags=ns["num_of_target_lags"]),
         "对 12 个 target 滞后做行内统计")

    # ---- 汇总 ----
    excluded = ["row_id", "date_id", "time_id", "target", "stock_return"]
    lgb_features = [c for c in df.columns if c not in excluded]

    print("=" * 78)
    print(f"{'阶段':<50}{'新增列数':>10}")
    print("=" * 78)
    total = 0
    for name, added, _ in stages:
        print(f"{name:<50}{len(added):>10}")
        total += len(added)
    print("-" * 78)
    print(f"{'合计新增特征':<50}{total:>10}")
    print("=" * 78)
    print(f"流水线输出总列数            : {len(df.columns)}")
    print(f"LGB 实际入模特征数          : {len(lgb_features)}")
    print(f"  其中 seconds_in_bucket 作为 categorical")
    print()
    for name, added, note in stages:
        print(f"\n--- {name}  ({len(added)}) ---")
        if note:
            print(f"    备注: {note}")
        show = added if len(added) <= 22 else added[:14] + [f"...(共 {len(added)} 个)"] + added[-4:]
        for c in show:
            print("      ", c)

    dump_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_list_258.json")
    json.dump({"n_features": len(lgb_features),
               "stages": {n: a for n, a, _ in stages},
               "lgb_features": lgb_features},
              open(dump_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[已导出机器可读特征清单 {dump_path}]")


if __name__ == "__main__":
    main()
