"""
工具 3：自动化训练 LightGBM + SHAP 可解释分析

设计理念：这个工具回答"哪些因素驱动了目标变量"类的归因问题。
用 LightGBM 是因为它对类别特征和缺失值天然友好、训练快，适合 Agent 场景下的
"即问即答"式建模；用 SHAP 而非简单的 feature_importance，是因为 SHAP 值
具有可加性和博弈论基础，能精确回答"这个特征把预测值推高/推低了多少"，
这是金融场景（如信贷审批）里合规可解释性的硬性要求。
"""
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示器环境下渲染图片
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import OUTPUT_DIR


def _is_classification(y: pd.Series) -> bool:
    """简单启发式：唯一值数量少（<=10）且都是整数，判定为分类任务"""
    unique_vals = y.dropna().unique()
    return len(unique_vals) <= 10 and pd.api.types.is_integer_dtype(y.dropna())


def train_ml_model(df: pd.DataFrame, target_column: str, feature_columns: List[str],
                    output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """
    工具主入口：训练 LightGBM 模型 + 计算 SHAP 特征重要性，并保存可视化图表。

    返回:
        {
            "success": bool,
            "task_type": "classification" | "regression",
            "metrics": {...},
            "feature_importance": [{"feature": ..., "mean_abs_shap": ...}, ...],  # 按重要性降序
            "shap_plot_path": 保存的 SHAP summary plot 路径,
            "top_insight": 自然语言总结最重要的驱动因素
        }
    """
    missing_cols = [c for c in [target_column] + feature_columns if c not in df.columns]
    if missing_cols:
        return {"success": False, "error": f"以下列不存在于数据中: {missing_cols}，可用列: {list(df.columns)}"}

    work_df = df[[target_column] + feature_columns].dropna()
    if len(work_df) < 30:
        return {"success": False, "error": f"有效样本量过少（{len(work_df)} 条，去除缺失值后），无法可靠训练模型，至少需要 30 条。"}

    # 类别特征自动编码（LightGBM 原生支持 category dtype）
    #
    # 注意：不能只判断 `X[col].dtype == object`。较新版本的 pandas（2.x 后期起）
    # 对纯字符串列可能推断为新的 "str" dtype 或 pyarrow-backed "string" dtype，
    # 而不再是传统的 numpy "object" dtype，导致这个判断漏检，字符串列原样传给
    # LightGBM 就会报 "pandas dtypes must be int, float or bool" 错误。
    # 这里改用"排除法"：只要不是数值类型、也不是布尔类型，就当作类别特征处理，
    # 兼容 object / str / string / category 等各种字符串类 dtype，且不依赖具体 pandas 版本。
    X = work_df[feature_columns].copy()
    for col in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])):
            X[col] = X[col].astype("category")
    y = work_df[target_column]

    is_clf = _is_classification(y)
    task_type = "classification" if is_clf else "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if is_clf and y.nunique() > 1 else None,
    )

    if is_clf:
        model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    random_state=42, verbose=-1)
    else:
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   random_state=42, verbose=-1)

    model.fit(X_train, y_train, categorical_feature="auto")

    # ---- 评估指标 ----
    metrics = {}
    if is_clf:
        preds = model.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        if y.nunique() == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, proba))
    else:
        preds = model.predict(X_test)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["r2"] = float(r2_score(y_test, preds))

    # ---- SHAP 归因分析 ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 分类任务的 shap_values 对二分类可能返回 list [class0, class1] 或直接是 class1 的矩阵，做兼容处理
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ---- 保存 SHAP summary plot ----
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"shap_summary_{target_column}.png")
    plt.figure()
    shap.summary_plot(shap_matrix, X_test, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    top_feature = importance_df.iloc[0]
    top_insight = (
        f"在预测 '{target_column}' 的过程中，'{top_feature['feature']}' 是影响力最大的特征"
        f"（平均 |SHAP值| = {top_feature['mean_abs_shap']:.4f}），"
        f"其次是 '{importance_df.iloc[1]['feature']}'（如果特征数 >= 2）。"
    )

    return {
        "success": True,
        "task_type": task_type,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "feature_importance": importance_df.to_dict(orient="records"),
        "shap_plot_path": plot_path,
        "top_insight": top_insight,
    }


if __name__ == "__main__":
    # 离线冒烟测试
    np.random.seed(1)
    n = 800
    balance = np.random.exponential(50000, n)
    credit = np.random.normal(650, 80, n)
    complaints = np.random.poisson(0.5, n)
    logins = np.random.randint(1, 30, n)
    churn_logit = -0.00002 * balance - 0.004 * credit + 0.6 * complaints + 2
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = np.random.binomial(1, churn_prob)

    test_df = pd.DataFrame({
        "account_balance": balance, "credit_score": credit,
        "complaint_count": complaints, "last_login_days": logins,
        "has_churned": churned,
    })

    result = train_ml_model(test_df, "has_churned",
                             ["account_balance", "credit_score", "complaint_count", "last_login_days"])
    print("Task type:", result.get("task_type"))
    print("Metrics:", result.get("metrics"))
    print("Feature importance:")
    for row in result.get("feature_importance", []):
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    print("Insight:", result.get("top_insight"))
    print("Plot saved at:", result.get("shap_plot_path"))
