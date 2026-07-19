"""
工具 4：将分析结果动态渲染为 PDF 报告

使用 reportlab（纯 Python，无需系统级依赖如 wkhtmltopdf），
把 Agent 在整个 ReAct 循环中积累的"分析轨迹"（SQL 查询、统计检验结果、
SHAP 图表）组装成一份结构化、可交付给业务方的 PDF。
"""
import os
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, PageBreak)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import OUTPUT_DIR
from src.utils.markdown_pdf import render_markdown

# ==================== 中文字体注册 ====================
# reportlab 默认的 Helvetica/Times 字体不含 CJK 字形，中文会被渲染成方块。
# STSong-Light 是 Adobe 提供、reportlab 内置支持的 CID 字体，无需额外字体文件即可正确显示中文。
_CJK_FONT_NAME = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(_CJK_FONT_NAME))

# STSong-Light 只有一种字重（没有内置的 CJK 粗体变体）。如果不做这一步映射，
# Markdown 里的 **加粗** 转成 <b> 标签后，reportlab 会去找 "STSong-Light-Bold"
# 这个不存在的字体从而报错。这里把粗体/斜体都映射回同一个字体，保证不崩溃
# （视觉上不会真的变粗，这是内置 CJK CID 字体的已知限制）。
pdfmetrics.registerFontFamily(
    _CJK_FONT_NAME,
    normal=_CJK_FONT_NAME, bold=_CJK_FONT_NAME,
    italic=_CJK_FONT_NAME, boldItalic=_CJK_FONT_NAME,
)


def _build_styles():
    styles = getSampleStyleSheet()
    for style_name in ["Normal", "Title", "Heading1", "Heading2", "BodyText"]:
        styles[style_name].fontName = _CJK_FONT_NAME

    styles.add(ParagraphStyle(name="ReportTitle", fontName=_CJK_FONT_NAME, fontSize=20, leading=26,
                               spaceAfter=16, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="SectionHeading", fontName=_CJK_FONT_NAME, fontSize=14, leading=18,
                               spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a3c6e")))
    styles.add(ParagraphStyle(name="SubHeading", fontName=_CJK_FONT_NAME, fontSize=12, leading=16,
                               spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2c5282")))
    styles.add(ParagraphStyle(name="BodyTextCN", fontName=_CJK_FONT_NAME, fontSize=10.5, leading=16))
    styles.add(ParagraphStyle(name="BodyTextBullet", fontName=_CJK_FONT_NAME, fontSize=10.5, leading=16,
                               leftIndent=14, spaceAfter=3))
    return styles


def _render_tool_step(step: Dict[str, Any], styles, elements: List):
    """把 Agent 执行轨迹中的一步（一次工具调用+结果）渲染为报告的一个小节"""
    tool_name = step.get("tool")
    result = step.get("result", {})

    if tool_name == "query_database":
        elements.append(Paragraph(f"数据查询：{step.get('input', {}).get('natural_language_query', '')}", styles["BodyTextCN"]))
        if result.get("sql"):
            elements.append(Paragraph(f"<font face='Courier' size=8>SQL: {result['sql']}</font>", styles["BodyTextCN"]))
        rows = result.get("rows", [])[:10]
        if rows:
            columns = result.get("columns", list(rows[0].keys()))
            table_data = [columns] + [[str(row.get(c, "")) for c in columns] for row in rows]
            t = Table(table_data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ]))
            elements.append(Spacer(1, 6))
            elements.append(t)

    elif tool_name == "run_statistical_test":
        elements.append(Paragraph(f"<b>{result.get('test_type', '统计检验')}</b>", styles["BodyTextCN"]))
        elements.append(Paragraph(result.get("conclusion", ""), styles["BodyTextCN"]))

    elif tool_name == "train_ml_model":
        elements.append(Paragraph(result.get("top_insight", ""), styles["BodyTextCN"]))
        metrics = result.get("metrics", {})
        if metrics:
            metrics_str = "，".join(f"{k}={v:.4f}" for k, v in metrics.items())
            elements.append(Paragraph(f"模型表现：{metrics_str}", styles["BodyTextCN"]))
        plot_path = result.get("shap_plot_path")
        if plot_path and os.path.exists(plot_path):
            elements.append(Spacer(1, 6))
            elements.append(RLImage(plot_path, width=14 * cm, height=9 * cm))

    elements.append(Spacer(1, 10))


def generate_report(title: str, summary: str, execution_trace: List[Dict[str, Any]] = None,
                     output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """
    工具主入口：生成最终 PDF 报告。

    参数:
        title: 报告标题
        summary: Agent 对整个分析过程的自然语言总结
        execution_trace: Agent 在 ReAct 循环中记录的所有工具调用及结果列表，
                          每项形如 {"tool": "run_statistical_test", "input": {...}, "result": {...}}
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_filename = "".join(c for c in title if c.isalnum() or c in " _-").strip().replace(" ", "_") or "report"
    pdf_path = os.path.join(output_dir, f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    styles = _build_styles()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                             leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)

    elements = []
    elements.append(Paragraph(title, styles["ReportTitle"]))
    elements.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyTextCN"]))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("分析结论摘要", styles["SectionHeading"]))
    elements.extend(render_markdown(summary, styles, _CJK_FONT_NAME))
    elements.append(Spacer(1, 10))

    if execution_trace:
        elements.append(Paragraph("详细分析过程", styles["SectionHeading"]))
        for step in execution_trace:
            _render_tool_step(step, styles, elements)

    doc.build(elements)

    return {"success": True, "pdf_path": pdf_path}


if __name__ == "__main__":
    # 回归测试：直接用真实场景中出问题的 Markdown 摘要内容（标题/加粗/表格/分割线/emoji 混合）
    # 验证 Markdown 渲染层是否能正确解析，而不是把 ##、**、| 原样打印出来。
    markdown_summary = """## 一、分析概述
本次分析基于 1,000 条客户数据，综合运用统计检验和机器学习方法，对客户流失（Churn）的主要驱动因素进行了系统性归因分析。

---

## 二、流失客户 vs 非流失客户账户余额差异检验

### 检验结果

| 分组 | 样本量 | 平均账户余额 |
|------|--------|-------------|
| 流失客户（has_churned=1） | 375 | ¥31,800 |
| 未流失客户（has_churned=0） | 625 | ¥60,937 |

- **t 统计量**：-11.01
- **p 值**：1.06 × 10⁻²⁶（远小于 0.05）

### 结论

✅ **流失客户与未流失客户的账户余额存在极显著的统计学差异。**

### 特征重要性排名（按平均 |SHAP| 值）

| 排名 | 特征 | 平均 |SHAP| | 业务含义 |
|------|------|------------|----------|
| 1 | complaint_count | 4.875 | 投诉次数——流失的最强信号 |
| 2 | transaction_count_3m | 4.741 | 近3个月交易活跃度 |

### 💡 业务建议

1. **投诉预警机制**：对产生 2 次及以上投诉的客户立即启动挽留流程
2. **活跃度监控**：对近3月交易次数 < 8 次的客户进行定向激活营销
"""

    fake_trace = [
        {
            "tool": "query_database",
            "input": {"natural_language_query": "对比流失与非流失客户的平均账户余额"},
            "result": {
                "sql": "SELECT has_churned, AVG(account_balance) as avg_balance FROM customer_profile GROUP BY has_churned",
                "columns": ["has_churned", "avg_balance"],
                "rows": [{"has_churned": 0, "avg_balance": 60274.88}, {"has_churned": 1, "avg_balance": 31920.99}],
            },
        },
        {
            "tool": "run_statistical_test",
            "input": {"test_type": "ttest"},
            "result": {"test_type": "Welch's t-test", "conclusion": "p < 0.001，差异具有统计学显著性。"},
        },
    ]
    result = generate_report(
        title="客户流失驱动因素分析报告（Markdown渲染回归测试）",
        summary=markdown_summary,
        execution_trace=fake_trace,
    )
    print(result)

    # 自动校验：确认输出 PDF 里不再残留原始 Markdown 语法符号
    import fitz
    doc = fitz.open(result["pdf_path"])
    full_text = "\n".join(page.get_text() for page in doc)
    assert "##" not in full_text, "PDF 中残留了未解析的 Markdown 标题符号 ##"
    assert "**" not in full_text, "PDF 中残留了未解析的 Markdown 加粗符号 **"
    assert "投诉次数" in full_text and "4.875" in full_text, "表格内容未正确渲染"
    print("\n✅ Markdown 渲染回归测试通过：标题/加粗/表格/分割线均已正确解析，无残留语法符号")
