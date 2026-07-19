"""
全局配置文件
所有密钥优先从环境变量读取，避免硬编码泄露到 Git 仓库。

密钥的正确配置方式（推荐）：
  1. 复制项目根目录的 .env.example 为 .env
  2. 在 .env 里填入真实密钥（.env 已被 .gitignore 排除，不会被提交到 Git）
  3. 本文件通过 python-dotenv 自动加载 .env 中的环境变量

⚠️ 不要把真实密钥直接硬编码在本文件中——本文件（config.py）本身是会被提交到 Git 仓库的。
"""
import os
from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件（如果存在）；找不到也不会报错，此时纯粹依赖系统环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ==================== LLM API 配置 ====================
# 兼容 OpenAI SDK 的任意大模型：OpenAI / Qwen(DashScope) / DeepSeek / Moonshot 等
# 只需替换 BASE_URL 和 MODEL_NAME 即可切换供应商

LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")

# 示例（任选其一）：
# OpenAI:    base_url="https://api.openai.com/v1",        model="gpt-4o"
# Qwen:      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus"
# DeepSeek:  base_url="https://api.deepseek.com/v1",       model="deepseek-chat"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

# 控制生成的随机性，Agent 决策阶段建议调低以保证工具调用稳定
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2000

# ==================== 数据库配置 ====================
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finance.db")

# ==================== Agent 行为配置 ====================
MAX_AGENT_STEPS = 8          # ReAct 循环最大步数，防止死循环/无限调用浪费 token
SANDBOX_TIMEOUT_SECONDS = 15  # 动态代码执行沙箱超时时间

# ==================== 输出配置 ====================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
