"""
轻量级 Markdown -> ReportLab Flowables 渲染器

背景：LLM 生成的 `summary` 文本天然带有 Markdown 语法（# 标题、**加粗**、
表格、--- 分割线、emoji），而 reportlab 的 Paragraph 不会解析 Markdown，
直接塞进去就是把 `##`、`**`、`|` 这些符号原样打印出来。

这里手写一个"够用就好"的最小化 Markdown 解析器（不引入 markdown/mistune 等
第三方依赖），覆盖 LLM 报告场景里最常见的几种元素：
  - # / ## / ### 标题
  - **加粗**、`行内代码`
  - | a | b | 表格（含表头分隔行 |---|---|）
  - - 或 * 或 1. 开头的列表
  - --- 水平分割线
  - emoji（reportlab 内置 CJK 字体不含 emoji 字形，直接剔除，避免变成方块/缺字符）
"""
import re
from typing import List

from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.units import cm


# emoji 常见 unicode 区间：misc symbols/dingbats、emoji 主区块、区域指示符（国旗）
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


def _escape_xml(text: str) -> str:
    """reportlab Paragraph 用类 XML 标签做富文本标记，普通文本里的 &/</> 必须先转义，
    否则要么被误当成标签解析报错，要么显示错乱。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _convert_inline_markdown(text: str, font_name: str) -> str:
    """
    把一行文本里的行内 Markdown（**加粗**、`代码`）转换成 reportlab 支持的标记标签。
    注意：必须先转义 XML 特殊字符，再插入 <b>/<font> 标签，否则标签本身也会被转义掉。
    """
    text = _strip_emoji(text)
    text = _escape_xml(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", rf"<font face='Courier' size=9>\1</font>", text)
    return text


def _split_table_row(line: str) -> List[str]:
    """按 | 切分表格行，去掉首尾因为行首/行尾 | 产生的空字符串"""
    cells = line.strip().split("|")
    if cells and cells[0].strip() == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return [c.strip() for c in cells]


def _is_table_separator(line: str) -> bool:
    """判断是否是表头分隔行，如 |------|------|"""
    cells = _split_table_row(line)
    return len(cells) > 0 and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _normalize_row(row: List[str], n_cols: int) -> List[str]:
    """把一行的单元格数量对齐到目标列数：多出的合并进最后一列，不够的用空字符串补齐"""
    if len(row) > n_cols:
        return row[: n_cols - 1] + [" ".join(row[n_cols - 1:])]
    if len(row) < n_cols:
        return row + [""] * (n_cols - len(row))
    return row


def _build_table_flowable(table_lines: List[str], font_name: str) -> Table:
    header_raw = _split_table_row(table_lines[0])
    data_rows_raw = [_split_table_row(line) for line in table_lines[2:]]  # 跳过表头 + 分隔行

    # 容错关键点：表头本身也可能因为单元格内容里恰好含有 "|" 字符（如 "平均 |SHAP| 值"）
    # 而被切出比实际列数更多的片段。这里不盲目信任表头的列数，而是以"数据行的列数"里
    # 出现次数最多的那个作为目标列数（数据行通常远多于 1 行、格式更规整、更可信），
    # 表头和所有数据行最终都按这个目标列数做合并/补齐，保证整张表列对齐。
    if data_rows_raw:
        col_counts = [len(r) for r in data_rows_raw]
        n_cols = max(set(col_counts), key=col_counts.count)
    else:
        n_cols = len(header_raw)

    header = _normalize_row(header_raw, n_cols)
    data_rows = [_normalize_row(row, n_cols) for row in data_rows_raw]

    styles_sheet = _make_cell_style(font_name)
    header_cells = [Paragraph(_convert_inline_markdown(h, font_name), styles_sheet["header"]) for h in header]
    body_cells = [[Paragraph(_convert_inline_markdown(cell, font_name), styles_sheet["body"]) for cell in row]
                  for row in data_rows]

    table_data = [header_cells] + body_cells
    t = Table(table_data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _make_cell_style(font_name: str):
    from reportlab.lib.styles import ParagraphStyle
    return {
        "header": ParagraphStyle(name="TableHeader", fontName=font_name, fontSize=8.5,
                                  textColor=colors.white, leading=11),
        "body": ParagraphStyle(name="TableBody", fontName=font_name, fontSize=8.5, leading=11),
    }


def render_markdown(md_text: str, styles, font_name: str) -> List:
    """
    主入口：把一段 Markdown 文本解析成 reportlab flowables 列表。

    参数:
        md_text: 原始 Markdown 字符串（通常来自 LLM 生成的 summary）
        styles: report_tool._build_styles() 返回的样式表，需包含
                'SectionHeading' / 'BodyTextCN' / 'BodyTextBullet'
        font_name: CJK 字体名，用于表格单元格等需要单独指定字体的地方
    """
    if not md_text:
        return []

    lines = md_text.replace("\r\n", "\n").split("\n")
    flowables = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # 水平分割线
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", line):
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cccccc")))
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # 标题
        heading_match = re.match(r"(#{1,4})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            content = _convert_inline_markdown(heading_match.group(2), font_name)
            style_name = "SectionHeading" if level <= 2 else "SubHeading"
            flowables.append(Paragraph(content, styles[style_name]))
            i += 1
            continue

        # 表格（连续以 | 开头的行，且第二行是分隔行）
        if line.startswith("|") and i + 1 < n and _is_table_separator(lines[i + 1].strip()):
            table_lines = [line]
            j = i + 1
            while j < n and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            flowables.append(Spacer(1, 4))
            flowables.append(_build_table_flowable(table_lines, font_name))
            flowables.append(Spacer(1, 8))
            i = j
            continue

        # 无序/有序列表项
        bullet_match = re.match(r"[-*]\s+(.*)", line)
        ordered_match = re.match(r"(\d+)\.\s+(.*)", line)
        if bullet_match or ordered_match:
            if bullet_match:
                prefix = "•  "
                content = _convert_inline_markdown(bullet_match.group(1), font_name)
            else:
                prefix = f"{ordered_match.group(1)}.  "
                content = _convert_inline_markdown(ordered_match.group(2), font_name)
            flowables.append(Paragraph(f"{prefix}{content}", styles["BodyTextBullet"]))
            i += 1
            continue

        # 普通段落
        content = _convert_inline_markdown(line, font_name)
        flowables.append(Paragraph(content, styles["BodyTextCN"]))
        i += 1

    return flowables
