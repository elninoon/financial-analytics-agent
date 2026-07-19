"""
工具 1：Text-to-SQL 自动化数据提取引擎

核心难点：LLM 并不天然知道你的数据库长什么样。
解决方案：启动时自省 (introspect) SQLite 的 sqlite_master + PRAGMA table_info，
          生成结构化的 schema 描述，注入到每次生成 SQL 的 Prompt 中（元数据感知机制）。
"""
import sqlite3
import pandas as pd
from typing import Dict, Any

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import DB_PATH, SANDBOX_TIMEOUT_SECONDS
from src.utils.sandbox import safe_execute_sql, SandboxError


def introspect_schema(db_path: str = DB_PATH) -> str:
    """
    自省数据库结构，生成给 LLM 看的 schema 描述文本。
    包含：表名、字段名、字段类型、外键关系、每个表的样例数据（前 2 行），
    帮助 LLM 理解字段的实际取值范围（例如 gender 是 'Male'/'Female' 而不是 0/1）。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    schema_lines = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns_info = cursor.fetchall()
        col_descriptions = [f"{col[1]} ({col[2]})" for col in columns_info]

        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()
        fk_descriptions = [f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks]

        cursor.execute(f"SELECT * FROM {table} LIMIT 2")
        sample_rows = cursor.fetchall()

        block = f"表名: {table}\n  字段: {', '.join(col_descriptions)}"
        if fk_descriptions:
            block += f"\n  外键: {', '.join(fk_descriptions)}"
        if sample_rows:
            block += f"\n  样例数据: {sample_rows}"
        schema_lines.append(block)

    conn.close()
    return "\n\n".join(schema_lines)


def generate_sql_from_nl(natural_language_query: str, schema_description: str, llm_client) -> str:
    """调用 LLM，将自然语言问题转换为一条只读 SELECT 语句"""
    system_prompt = f"""你是一个 SQL 生成引擎。根据用户的自然语言问题和下面的数据库 schema，生成一条 SQLite 兼容的 SELECT 语句。

数据库 schema:
{schema_description}

规则：
- 只输出 SQL 语句本身，不要任何解释、注释或 markdown 代码块标记
- 只能生成 SELECT 语句，禁止任何写操作
- 涉及多表时使用 JOIN，并通过外键关系正确关联
- 如果问题涉及分组统计，使用 GROUP BY + 聚合函数
- 限制返回行数不超过 1000 行（除非用户明确要求更少/更多）
"""
    response = llm_client.chat.completions.create(
        model=llm_client.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language_query},
        ],
        temperature=0,
        max_tokens=500,
    )
    raw_sql = response.choices[0].message.content.strip()
    # 防御性清理：去掉 LLM 可能附带的 markdown 代码块标记
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    return raw_sql


def query_database(natural_language_query: str, llm_client, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    工具主入口：自然语言 -> SQL -> 沙箱安全执行 -> 结构化结果

    返回:
        {
            "success": bool,
            "sql": 生成的 SQL,
            "columns": [...],
            "rows": [...],
            "dataframe": pd.DataFrame (供后续统计/ML工具直接消费),
            "error": 出错信息（如果失败）
        }
    """
    schema_description = introspect_schema(db_path)

    try:
        sql = generate_sql_from_nl(natural_language_query, schema_description, llm_client)
    except Exception as e:
        return {"success": False, "error": f"SQL 生成失败: {e}"}

    try:
        result = safe_execute_sql(db_path, sql, timeout_seconds=SANDBOX_TIMEOUT_SECONDS)
    except SandboxError as e:
        return {"success": False, "sql": sql, "error": str(e)}

    df = pd.DataFrame(result["rows"], columns=result["columns"])

    return {
        "success": True,
        "sql": sql,
        "columns": result["columns"],
        "rows": result["rows"],
        "row_count": len(result["rows"]),
        "truncated": result["truncated"],
        "dataframe": df,
    }


if __name__ == "__main__":
    # 独立测试：直接用一条手写 SQL 验证 introspect + sandbox 链路是否打通
    # （不依赖真实 LLM API key，方便离线冒烟测试）
    print("=== Schema 自省结果 ===")
    print(introspect_schema())

    print("\n=== 沙箱执行测试（手写 SQL，跳过 LLM 生成环节） ===")
    from src.utils.sandbox import safe_execute_sql
    test_sql = """
        SELECT p.has_churned, AVG(p.account_balance) as avg_balance, COUNT(*) as n
        FROM customer_profile p
        GROUP BY p.has_churned
    """
    result = safe_execute_sql(DB_PATH, test_sql)
    print(result)

    print("\n=== 沙箱拦截危险 SQL 测试 ===")
    try:
        safe_execute_sql(DB_PATH, "DROP TABLE customer_profile")
    except SandboxError as e:
        print(f"成功拦截: {e}")
