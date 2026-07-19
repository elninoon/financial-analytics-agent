"""
安全沙箱模块
职责：LLM 生成的内容（SQL / Python 代码片段）在进入执行前必须先过这里的校验，
杜绝 DROP TABLE、os.system、eval 等危险操作。

设计原则：白名单优先于黑名单——SQL 只允许 SELECT，Python 沙箱只暴露必要的安全内置函数。
"""
import re
import signal
import sqlite3
import contextlib
from typing import Any, Dict


class SandboxError(Exception):
    """沙箱拒绝执行时抛出"""
    pass


# ==================== SQL 安全校验 ====================

_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|PRAGMA|REPLACE|EXEC)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    """
    校验 LLM 生成的 SQL：
    1. 只允许单条 SELECT 语句
    2. 禁止任何写操作/DDL/PRAGMA
    3. 禁止多语句拼接（防止 '; DROP TABLE ...' 注入）
    """
    cleaned = sql.strip().rstrip(";")

    if ";" in cleaned:
        raise SandboxError("检测到多条 SQL 语句拼接，出于安全考虑已拒绝执行。")

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise SandboxError("只允许执行 SELECT 查询，拒绝执行非只读操作。")

    if _FORBIDDEN_SQL_KEYWORDS.search(cleaned):
        raise SandboxError("SQL 中包含被禁止的关键字（写操作/DDL），拒绝执行。")

    return cleaned


@contextlib.contextmanager
def _timeout(seconds: int):
    """基于 SIGALRM 的超时保护（仅在类 Unix 系统可用）"""
    def _handler(signum, frame):
        raise SandboxError(f"执行超时（超过 {seconds} 秒），已强制中止。")

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def safe_execute_sql(db_path: str, sql: str, timeout_seconds: int = 15, max_rows: int = 5000) -> Dict[str, Any]:
    """
    在只读连接上安全执行 SQL，并限制返回行数，防止意外拉取超大结果集耗尽内存。
    """
    validated_sql = validate_sql(sql)

    # 以只读模式打开连接（uri=True + mode=ro），双重保险：即便校验遗漏，数据库层也拒绝写操作
    uri_path = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri_path, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        with _timeout(timeout_seconds):
            cursor = conn.cursor()
            cursor.execute(validated_sql)
            rows = cursor.fetchmany(max_rows)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return {
            "columns": columns,
            "rows": [dict(row) for row in rows],
            "truncated": len(rows) == max_rows,
        }
    except sqlite3.Error as e:
        raise SandboxError(f"SQL 执行错误: {e}")
    finally:
        conn.close()


# ==================== Python 动态代码沙箱（预留给未来扩展的自定义分析代码） ====================

_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "sorted": sorted, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "float": float, "int": int, "str": str, "bool": bool,
}


def safe_exec_python(code: str, local_vars: Dict[str, Any], timeout_seconds: int = 15) -> Dict[str, Any]:
    """
    受限环境下执行一小段 Python 代码（仅暴露白名单内置函数 + 调用方显式传入的变量，
    例如 pandas DataFrame）。禁止 import、文件系统访问、网络访问。
    """
    forbidden_patterns = [r"\bimport\b", r"\b__\w+__\b", r"\bopen\b", r"\bexec\b", r"\beval\b",
                           r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\bsocket\b"]
    for pattern in forbidden_patterns:
        if re.search(pattern, code):
            raise SandboxError(f"检测到被禁止的代码模式 ({pattern})，拒绝执行。")

    restricted_globals = {"__builtins__": _SAFE_BUILTINS}
    try:
        with _timeout(timeout_seconds):
            exec(code, restricted_globals, local_vars)
    except SandboxError:
        raise
    except Exception as e:
        raise SandboxError(f"代码执行出错: {e}")
    return local_vars
