# 智能金融数据分析 Agent (Financial Analytics Agent)

一个基于 ReAct 架构的 LLM Agent，能够自主调用 **Text-to-SQL / 统计推断 / 机器学习归因 / PDF 报告生成**
四类工具，端到端完成"自然语言问题 -> 数据洞察 -> 交付报告"的金融业务分析流程。

## 架构

```
financial-analytics-agent/
├── config/config.py          # LLM API 密钥、模型、超时等配置（从环境变量读取）
├── init_db.py                # 生成本地模拟金融数据库（SQLite，含统计学规律）
├── src/
│   ├── agent.py              # Agent 核心：ReAct 循环控制器
│   ├── prompts.py            # 工具 Schema 定义 + 系统 Prompt（含 schema 注入）
│   ├── tools/
│   │   ├── db_tool.py        # 工具1：Text-to-SQL（schema 自省 + 元数据感知）
│   │   ├── stat_tool.py      # 工具2：自动化统计检验（t/ANOVA/卡方/相关性，scipy 计算）
│   │   ├── ml_tool.py        # 工具3：LightGBM + SHAP 归因分析
│   │   └── report_tool.py    # 工具4：动态渲染 PDF 报告（含中文 CJK 字体支持）
│   └── utils/sandbox.py      # 安全沙箱：SQL 白名单校验 + 只读连接 + 超时保护
├── tests_mock_run.py         # 离线 Mock 集成测试（无需真实 API Key 即可验证全链路）
├── main.py                   # CLI 入口
└── data/finance.db           # 模拟数据库（客户画像 / 行为 / 贷款三张表）
```

## 核心技术设计

1. **Text-to-SQL 的元数据感知**：`db_tool.introspect_schema()` 在每次生成 SQL 前自省数据库真实结构
   （表名、字段、类型、外键、样例数据），注入 Prompt，避免 LLM 凭空捏造字段名。

2. **代码/SQL 沙箱隔离**：`utils/sandbox.py` 对 LLM 生成的 SQL 做正则白名单校验（仅允许单条 SELECT，
   禁止 DROP/DELETE/PRAGMA 等），并在只读模式 (`mode=ro`) 的数据库连接上执行，双重防线防止误删/注入。

3. **统计计算不依赖 LLM**：LLM 只负责"决定调用哪种检验"，真正的数值计算交给 `scipy.stats`，
   避免大模型编造 p 值这类不可靠行为。

4. **SHAP 可解释性而非简单特征重要性**：金融归因分析（如信贷违约驱动因素）需要满足合规可解释性要求，
   SHAP 值的可加性能精确回答"某特征把预测值推高/推低了多少"。

5. **结构化输出**：所有工具通过 OpenAI Function-Calling Schema 定义（`prompts.TOOL_SCHEMAS`），
   强制 LLM 以合法 JSON 输出下一步动作，而不是自由文本。

## 快速开始

```bash
pip install -r requirements.txt

# 1. 初始化模拟数据库
python init_db.py

# 2. 不依赖真实 API Key 的离线全链路测试（验证四个工具+沙箱是否打通）
python tests_mock_run.py

# 3. 配置密钥：复制 .env.example 为 .env，并填入真实密钥
cp .env.example .env
# 用编辑器打开 .env，填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME
# .env 已被 .gitignore 排除，不会被提交到 Git，可放心填写真实密钥

# 4. 运行（支持 OpenAI / Qwen / DeepSeek 等 OpenAI 兼容端点）
python main.py "分析客户流失的主要驱动因素，并检验流失客户和非流失客户的账户余额是否有显著差异，最后生成一份报告"
```

> 也可以不用 `.env`，直接用系统环境变量（`export LLM_API_KEY=...`），二者等效。
> **切勿**把真实密钥直接写死在 `config/config.py` 里再提交到 Git——该文件会被版本控制跟踪。

生成的 PDF 报告保存在 `output/` 目录下。

## 已验证的测试结果（Mock LLM，真实数据）

| 步骤 | 工具 | 结果 |
|---|---|---|
| 1 | query_database | 沙箱内安全执行 JOIN 查询，1000 行客户数据 |
| 2 | run_statistical_test (ttest) | 流失/非流失客户账户余额差异 p ≈ 1.06e-26，高度显著 |
| 3 | train_ml_model (LightGBM+SHAP) | AUC = 0.979，account_balance/complaint_count 为最重要驱动因素 |
| 4 | generate_report | 成功生成含数据表 + SHAP 图表的中文 PDF 报告 |

## 下一步可扩展方向

- 引入向量数据库支持多轮对话中的长期记忆
- 增加更多统计检验类型（非参数检验、生存分析等，适配违约时间预测场景）
- 支持多工具并行调用（当前为单步串行，保证轨迹清晰可控）
- 增加 Web UI（Streamlit/Gradio）替代当前 CLI
