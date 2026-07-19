"""
Agent 核心大脑：控制 ReAct (Reasoning -> Acting -> Observation) 循环

工作流程：
  用户问题 -> LLM 决策（调用哪个工具/直接回答）-> 执行工具（走沙箱）
  -> 把结果作为 Observation 喂回 LLM -> 重复，直到 LLM 决定生成报告或直接回答
  -> 全过程记录 execution_trace，供 report_tool 渲染成 PDF

这是全系统唯一"允许 LLM 自由发挥"的地方——但发挥的范围被 TOOL_SCHEMAS 严格约束，
LLM 只能在预定义的 4 个工具之间做选择，不能执行任意代码。
"""
import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME,
                            LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_AGENT_STEPS, DB_PATH)
from src.prompts import TOOL_SCHEMAS, build_system_prompt, REACT_LOOP_INSTRUCTION
from src.tools.db_tool import query_database, introspect_schema
from src.tools.stat_tool import run_statistical_test
from src.tools.ml_tool import train_ml_model
from src.tools.report_tool import generate_report


class LLMClient:
    """对 OpenAI SDK 的一层薄封装，方便统一切换供应商（OpenAI/Qwen/DeepSeek 等）"""

    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL,
                 model_name: str = LLM_MODEL_NAME):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.chat = self.client.chat  # 让 db_tool.py 里 llm_client.chat.completions.create 可直接用

    def decide_next_action(self, messages: List[Dict[str, Any]]) -> Any:
        """调用带工具的 chat completion，让 LLM 决定下一步动作"""
        return self.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )


class FinancialAnalyticsAgent:
    """
    金融数据分析 Agent 主类。

    self.dataframe: 上一次 query_database 返回的数据，作为后续统计/ML工具的默认数据源
                     （模拟真实分析流程：先取数，再在取到的数据上做分析，而不是每个工具各自查库）
    self.execution_trace: 完整的工具调用轨迹，用于最终生成报告
    """

    def __init__(self, llm_client: LLMClient = None, db_path: str = DB_PATH):
        self.llm_client = llm_client or LLMClient()
        self.db_path = db_path
        self.dataframe = None
        self.execution_trace: List[Dict[str, Any]] = []

    # -------------------- 工具分发 --------------------

    def _dispatch_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "query_database":
            result = query_database(
                natural_language_query=tool_args["natural_language_query"],
                llm_client=self.llm_client,
                db_path=self.db_path,
            )
            if result.get("success"):
                self.dataframe = result["dataframe"]
            # DataFrame 对象不能被 json.dumps，剔除后再喂回给 LLM
            return {k: v for k, v in result.items() if k != "dataframe"}

        elif tool_name == "run_statistical_test":
            if self.dataframe is None:
                return {"success": False, "error": "尚未查询任何数据，请先调用 query_database 获取数据。"}
            return run_statistical_test(
                df=self.dataframe,
                test_type=tool_args["test_type"],
                value_column=tool_args["value_column"],
                group_column=tool_args.get("group_column"),
            )

        elif tool_name == "train_ml_model":
            if self.dataframe is None:
                return {"success": False, "error": "尚未查询任何数据，请先调用 query_database 获取数据。"}
            return train_ml_model(
                df=self.dataframe,
                target_column=tool_args["target_column"],
                feature_columns=tool_args["feature_columns"],
            )

        elif tool_name == "generate_report":
            return generate_report(
                title=tool_args["title"],
                summary=tool_args["summary"],
                execution_trace=self.execution_trace,
            )

        return {"success": False, "error": f"未知工具: {tool_name}"}

    # -------------------- 主循环 --------------------

    def run(self, user_question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        执行完整的 ReAct 循环，直到 LLM 生成报告 / 直接给出文字回答 / 达到最大步数。
        """
        schema_description = introspect_schema(self.db_path)
        system_prompt = build_system_prompt(schema_description)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        self.dataframe = None
        self.execution_trace = []
        final_answer = None

        for step in range(1, MAX_AGENT_STEPS + 1):
            response = self.llm_client.decide_next_action(messages)
            choice = response.choices[0].message

            # 情况 1：LLM 决定不再调用工具，直接给出文字回答 -> 结束循环
            if not choice.tool_calls:
                final_answer = choice.content
                if verbose:
                    print(f"\n[Step {step}] Agent 直接回答: {final_answer}")
                break

            messages.append(choice.model_dump(exclude_none=True))

            # 情况 2：LLM 请求调用工具（本设计中每轮只处理一个工具调用，保持轨迹清晰可控）
            #
            # 重要修复：某些模型即使 tool_choice="auto" 也可能一次返回多个 tool_calls。
            # 但我们只会执行 choice.tool_calls[0] 这一个，如果把完整的 tool_calls 列表
            # 原样存入历史，API 会报错："assistant message with tool_calls must be
            # followed by tool messages responding to EACH tool_call_id"（数量对不上）。
            # 因此这里必须把刚存入历史的助手消息裁剪成只含"我们实际执行的那一个" tool_call，
            # 确保「声明的工具调用数」与「后续工具响应数」严格一致。
            tool_call = choice.tool_calls[0]
            messages[-1]["tool_calls"] = [messages[-1]["tool_calls"][0]]

            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            if verbose:
                print(f"\n[Step {step}] Action: {tool_name}({tool_args})")

            tool_result = self._dispatch_tool(tool_name, tool_args)
            self.execution_trace.append({"tool": tool_name, "input": tool_args, "result": tool_result})

            if verbose:
                preview = json.dumps(tool_result, ensure_ascii=False, default=str)[:300]
                print(f"[Step {step}] Observation: {preview}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })

            # generate_report 是终止信号：报告即最终交付物
            if tool_name == "generate_report" and tool_result.get("success"):
                final_answer = f"分析报告已生成: {tool_result.get('pdf_path')}"
                break
        else:
            final_answer = "已达到最大执行步数，Agent 提前终止。以下是已收集到的分析轨迹。"

        return {
            "final_answer": final_answer,
            "execution_trace": self.execution_trace,
            "steps_used": len(self.execution_trace),
        }
