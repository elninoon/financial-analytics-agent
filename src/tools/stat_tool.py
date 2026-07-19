"""
工具 2：自动化数理统计检验

设计理念：这个工具本身不依赖 LLM——统计推断的计算过程必须是确定性、可复现的，
LLM 只负责"决定调用哪种检验、传什么参数"，真正的数值计算交给 scipy，
避免让大模型"编造"p 值（LLM 生成数字的可靠性远不如调用真实统计库）。
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


def _interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    if p_value < alpha:
        return f"p = {p_value:.4g} < {alpha}，差异具有统计学显著性，可以拒绝原假设。"
    return f"p = {p_value:.4g} >= {alpha}，未观察到统计学显著差异，不能拒绝原假设。"


def run_ttest(df: pd.DataFrame, group_column: str, value_column: str) -> Dict[str, Any]:
    """独立样本 t 检验：适用于 group_column 恰好有 2 个分组的场景"""
    groups = df[group_column].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"t 检验要求分组变量恰好有 2 个取值，但 {group_column} 有 {len(groups)} 个取值，请改用 anova。")

    g1 = df[df[group_column] == groups[0]][value_column].dropna()
    g2 = df[df[group_column] == groups[1]][value_column].dropna()

    # Welch's t-test：不假设两组方差相等，比经典 Student's t 检验更稳健
    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)

    return {
        "test_type": "Welch's t-test（独立样本，不假设方差齐性）",
        "group_column": group_column,
        "value_column": value_column,
        "group_means": {str(groups[0]): float(g1.mean()), str(groups[1]): float(g2.mean())},
        "group_sizes": {str(groups[0]): int(len(g1)), str(groups[1]): int(len(g2))},
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "conclusion": _interpret_p_value(p_value),
    }


def run_anova(df: pd.DataFrame, group_column: str, value_column: str) -> Dict[str, Any]:
    """单因素方差分析：适用于 group_column 有 3 个及以上分组的场景"""
    groups = df[group_column].dropna().unique()
    samples = [df[df[group_column] == g][value_column].dropna() for g in groups]

    f_stat, p_value = stats.f_oneway(*samples)

    return {
        "test_type": "单因素方差分析 (One-way ANOVA)",
        "group_column": group_column,
        "value_column": value_column,
        "group_means": {str(g): float(s.mean()) for g, s in zip(groups, samples)},
        "group_sizes": {str(g): int(len(s)) for g, s in zip(groups, samples)},
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "conclusion": _interpret_p_value(p_value),
    }


def run_chi2(df: pd.DataFrame, group_column: str, value_column: str) -> Dict[str, Any]:
    """卡方独立性检验：适用于两个分类变量之间是否存在关联"""
    contingency_table = pd.crosstab(df[group_column], df[value_column])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "test_type": "卡方独立性检验 (Chi-square test of independence)",
        "group_column": group_column,
        "value_column": value_column,
        "contingency_table": contingency_table.to_dict(),
        "chi2_statistic": float(chi2_stat),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "conclusion": _interpret_p_value(p_value),
    }


def run_correlation(df: pd.DataFrame, value_column: str, group_column: str = None) -> Dict[str, Any]:
    """
    Pearson 相关性分析：group_column 在此处复用为第二个数值变量名，
    用于计算 value_column 与 group_column 两个连续变量之间的线性相关性。
    """
    if group_column is None:
        raise ValueError("相关性分析需要传入两个数值列（value_column 与 group_column）")

    x = df[value_column].dropna()
    y = df[group_column].dropna()
    aligned = pd.concat([x, y], axis=1).dropna()

    r, p_value = stats.pearsonr(aligned[value_column], aligned[group_column])

    return {
        "test_type": "Pearson 相关性分析",
        "variable_x": value_column,
        "variable_y": group_column,
        "correlation_coefficient": float(r),
        "p_value": float(p_value),
        "conclusion": (
            f"相关系数 r = {r:.4f}（{'正相关' if r > 0 else '负相关'}，"
            f"{'弱' if abs(r) < 0.3 else '中等' if abs(r) < 0.6 else '强'}相关）。"
            + _interpret_p_value(p_value)
        ),
    }


_DISPATCH = {
    "ttest": run_ttest,
    "anova": run_anova,
    "chi2": run_chi2,
    "correlation": run_correlation,
}


def run_statistical_test(df: pd.DataFrame, test_type: str, value_column: str, group_column: str = None) -> Dict[str, Any]:
    """工具主入口，供 Agent 统一调用"""
    if test_type not in _DISPATCH:
        return {"success": False, "error": f"不支持的检验类型: {test_type}，可选: {list(_DISPATCH.keys())}"}

    if value_column not in df.columns:
        return {"success": False, "error": f"列 '{value_column}' 不存在于当前数据中，可用列: {list(df.columns)}"}
    if group_column and group_column not in df.columns:
        return {"success": False, "error": f"列 '{group_column}' 不存在于当前数据中，可用列: {list(df.columns)}"}

    try:
        result = _DISPATCH[test_type](df, group_column, value_column) if test_type != "correlation" \
            else _DISPATCH[test_type](df, value_column, group_column)
        result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # 离线冒烟测试：构造一份模拟数据直接测试四种检验
    np.random.seed(0)
    test_df = pd.DataFrame({
        "has_churned": np.random.binomial(1, 0.3, 500),
        "account_balance": np.random.exponential(50000, 500),
        "region": np.random.choice(["East", "West", "North"], 500),
        "credit_score": np.random.normal(650, 80, 500),
    })

    print("=== t 检验 ===")
    print(run_statistical_test(test_df, "ttest", "account_balance", "has_churned"))

    print("\n=== ANOVA ===")
    print(run_statistical_test(test_df, "anova", "account_balance", "region"))

    print("\n=== 卡方检验 ===")
    print(run_statistical_test(test_df, "chi2", "region", "has_churned"))

    print("\n=== 相关性分析 ===")
    print(run_statistical_test(test_df, "correlation", "account_balance", "credit_score"))
