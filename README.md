# Optiver 收盘竞价预测（LSTM 与 ConvNet）

本仓库包含两个用于 Kaggle [Optiver Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview) 竞赛的机器学习模型：长短期记忆网络（LSTM）模型和卷积神经网络（ConvNet）模型。两个模型都不做特征工程，前者仅使用原始特征，后者仅使用原始特征加少量不平衡（imbalance）特征，用于预测股票收益率。

## LSTM 模型

LSTM 模型使用原始特征并叠加目标值滞后项（target lag）。该模型侧重数据的时序结构，在不引入大量特征工程的前提下捕捉时间维度上的模式。

- **成绩**：公开排行榜（public leaderboard）得分 5.3508。
- **特征**：原始特征 + 目标值滞后项。
- **Notebook**：实现见 Kaggle 用户名 `nimashahbazi` 下的 notebook。

## ConvNet 模型

ConvNet 模型则仅使用不平衡特征配合原始特征。

- **成绩**：公开排行榜得分 5.3439，主要收益来自残差部分。
- **特征**：包含不平衡（imbalance）特征。
- **改进建议**：再加入全局股票映射（global stock mapping）等若干特征，成绩可较容易提升至 5.33X。
- **Notebook**：实现同样发布在 Kaggle 用户名 `nimashahbazi` 下。

## 相关文档

- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)：`optiver-258-lgb-submit` 的特征工程文档，含 258 个入模特征的分类清单、公式说明，以及可迁移的设计经验与数据泄漏陷阱。
- [feature_list_258.json](feature_list_258.json)：机器可读的特征清单，按流水线阶段分组。
- [feature_list_audit.py](feature_list_audit.py)：实跑 notebook 流水线以复现上述特征清单的审计脚本。
