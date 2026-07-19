"""
Prompt 模版与工具 Schema 定义
这是整个 Agent"可控性"的核心：通过 Function-Calling Schema 强制 LLM 以结构化 JSON
输出下一步要调用的工具，而不是自由发挥的文本。
"""

# ==================== 工具 Schema（OpenAI Function-Calling 格式） ====================
# 每新增一个 tools/ 下的模块，都要在这里注册对应的 schema，Agent 才能"看到"它。

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "将自然语言问题转换为 SQL 并在金融数据库中执行查询，返回结构化的表格数据。"
                "适用于任何需要从数据库中提取、筛选、聚合客户/贷款数据的场景。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "natural_language_query": {
                        "type": "string",
                        "description": "用自然语言描述你想从数据库中获取什么数据，例如'流失客户和未流失客户的平均账户余额对比'"
                    }
                },
                "required": ["natural_language_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_statistical_test",
            "description": (
                "对已获取的数据执行自动化统计推断，支持 t 检验、方差分析(ANOVA)、卡方检验、"
                "相关性分析。用于验证'某分组之间是否存在显著差异'类的问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["ttest", "anova", "chi2", "correlation"],
                        "description": "统计检验类型",
                    },
                    "group_column": {
                        "type": "string",
                        "description": "分组变量的列名（分类变量），如 has_churned",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "被检验的数值变量列名，如 account_balance",
                    },
                },
                "required": ["test_type", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_ml_model",
            "description": (
                "在已获取的数据上自动训练 LightGBM 分类/回归模型，并用 SHAP 计算特征重要性，"
                "适用于'哪些因素驱动了流失/违约'这类归因分析问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "预测目标列名，如 has_churned 或 is_default",
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "用作特征的列名列表",
                    },
                },
                "required": ["target_column", "feature_columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "将本轮分析过程中产生的所有结论、图表和数据整理为一份 PDF 报告，作为分析任务的最终交付物。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "报告标题"},
                    "summary": {"type": "string", "description": "对本次分析结论的自然语言总结"},
                },
                "required": ["title", "summary"],
            },
        },
    },
]


# ==================== 系统 Prompt ====================

def build_system_prompt(db_schema_description: str) -> str:
    """
    构建注入了数据库元数据的系统 Prompt。
    这是防止 LLM"瞎写 SQL"的关键：把真实 schema 喂给它，而不是让它凭空猜表结构。
    """
    return f"""你是一名资深的金融数据分析师 Agent，擅长使用统计学和机器学习方法回答业务问题。

你可以访问以下数据库，这是数据库的真实结构（表名、字段名、字段类型），
你生成的一切 SQL 都必须严格基于以下 schema，禁止引用不存在的表或字段：

{db_schema_description}

你的工作方式遵循 ReAct 范式：
1. Thought: 先思考当前问题需要哪个工具、需要什么参数
2. Action: 调用一个工具（严格按照给定的 function schema）
3. Observation: 你会收到工具的执行结果
4. 重复以上过程，直到你认为已经收集到足够信息回答用户问题
5. 最后调用 generate_report 生成最终交付物，或直接用文字总结回答

调用 generate_report 时，summary 参数会被解析为 Markdown 并渲染进 PDF，请遵守以下格式约束：
- 表格单元格内容本身不要包含 "|" 字符（例如指代绝对值时写"平均SHAP值"而不是"平均|SHAP|值"），
  否则会被误判为额外的表格分隔符导致列错位
- 不要使用 emoji（✅ 🔑 📊 等），PDF 使用的中文字体不含 emoji 字形，会显示为缺失字符

注意事项：
- 每次只调用一个工具，等待结果后再决定下一步
- 如果某个工具调用失败，分析失败原因并调整参数重试，而不是放弃
- 涉及"是否有显著差异/影响"类问题，必须使用 run_statistical_test 而非主观判断
- 涉及"哪些因素驱动了 XX"类问题，优先使用 train_ml_model 做归因分析
- 保持分析的统计学严谨性，不要夸大或臆造结论
"""


REACT_LOOP_INSTRUCTION = (
    "请基于以上工具的执行结果，继续你的分析。如果信息已经足够回答用户的原始问题，"
    "请调用 generate_report 生成报告；否则请继续调用下一个必要的工具。"
)
