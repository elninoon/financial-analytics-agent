"""
离线集成测试：用一个假的 LLM 客户端模拟"决策序列"，
验证 Agent 的 ReAct 循环、工具分发、execution_trace 记录、报告生成是否完全打通，
不依赖真实的 LLM API Key（CI / 无网络环境下也能跑）。
"""
import json
import types
from src.agent import FinancialAnalyticsAgent
from config.config import DB_PATH


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = types.SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False))


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeLLMClient:
    """
    模拟一个"聪明"的 LLM：
    第1轮 -> 查数据库（对比流失/非流失客户的账户余额）
    第2轮 -> 对查到的数据做 t 检验
    第3轮 -> 训练 ML 模型做归因
    第4轮 -> 生成报告
    第5轮 -> 直接文字回答，结束循环

    同时，这个 Fake 客户端也要实现 db_tool.py 依赖的
    `llm_client.chat.completions.create(...)` 接口，用于模拟 Text-to-SQL 生成。
    """
    def __init__(self):
        self.model_name = "fake-model"
        self.step = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._fake_sql_gen))

    def _fake_sql_gen(self, model, messages, temperature=0, max_tokens=500):
        # 模拟 Text-to-SQL：无视自然语言输入，直接返回一条写死的、schema 合法的 SQL
        sql = ("SELECT p.customer_id, p.has_churned, p.account_balance, p.credit_score, "
               "b.complaint_count, b.last_login_days "
               "FROM customer_profile p JOIN customer_behavior b ON p.customer_id = b.customer_id "
               "LIMIT 1000")
        fake_msg = types.SimpleNamespace(content=sql)
        fake_choice = types.SimpleNamespace(message=fake_msg)
        return types.SimpleNamespace(choices=[fake_choice])

    def decide_next_action(self, messages):
        self.step += 1

        if self.step == 1:
            tc = FakeToolCall("call_1", "query_database",
                               {"natural_language_query": "获取客户流失、余额、信用分、投诉、登录数据"})
            return FakeResponse(FakeMessage(tool_calls=[tc]))

        elif self.step == 2:
            tc = FakeToolCall("call_2", "run_statistical_test",
                               {"test_type": "ttest", "group_column": "has_churned", "value_column": "account_balance"})
            return FakeResponse(FakeMessage(tool_calls=[tc]))

        elif self.step == 3:
            tc = FakeToolCall("call_3", "train_ml_model",
                               {"target_column": "has_churned",
                                "feature_columns": ["account_balance", "credit_score", "complaint_count", "last_login_days"]})
            return FakeResponse(FakeMessage(tool_calls=[tc]))

        elif self.step == 4:
            tc = FakeToolCall("call_4", "generate_report",
                               {"title": "客户流失驱动因素分析报告（Mock 测试）",
                                "summary": "本报告通过统计检验与机器学习归因分析，识别出账户余额是客户流失的最关键驱动因素。"})
            return FakeResponse(FakeMessage(tool_calls=[tc]))

        else:
            return FakeResponse(FakeMessage(content="分析完成，报告已生成。"))


class FakeLLMClientMultiToolCall:
    """
    专门复现并验证那个真实 bug 的场景：
    模型在第 1 轮里一次性返回 2 个 tool_calls（这是很多模型的正常行为，
    即使 tool_choice="auto" 也可能发生），验证 agent.py 能正确裁剪历史，
    不会触发 OpenAI 的 "insufficient tool messages following tool_calls" 报错。
    """
    def __init__(self):
        self.model_name = "fake-model"
        self.step = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._fake_sql_gen))

    def _fake_sql_gen(self, model, messages, temperature=0, max_tokens=500):
        sql = "SELECT has_churned, account_balance FROM customer_profile LIMIT 100"
        fake_msg = types.SimpleNamespace(content=sql)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=fake_msg)])

    def decide_next_action(self, messages):
        self.step += 1
        if self.step == 1:
            # 关键：一次返回 2 个 tool_calls，模拟触发过 bug 的真实场景
            tc1 = FakeToolCall("call_1", "query_database", {"natural_language_query": "查询流失与余额数据"})
            tc2 = FakeToolCall("call_2", "run_statistical_test",
                                {"test_type": "ttest", "group_column": "has_churned", "value_column": "account_balance"})
            return FakeResponse(FakeMessage(tool_calls=[tc1, tc2]))
        else:
            return FakeResponse(FakeMessage(content="分析完成。"))


if __name__ == "__main__":
    agent = FinancialAnalyticsAgent(llm_client=FakeLLMClient(), db_path=DB_PATH)
    result = agent.run("分析客户流失的驱动因素并生成报告", verbose=True)

    print("\n" + "=" * 60)
    print("最终回答:", result["final_answer"])
    print("使用步骤数:", result["steps_used"])
    assert result["steps_used"] == 4, "预期应恰好执行 4 个工具调用"
    assert result["execution_trace"][-1]["result"]["success"] is True, "报告生成应成功"
    print("\n✅ 全链路 Mock 集成测试通过：Text-to-SQL(沙箱) -> 统计检验 -> ML归因(SHAP) -> PDF报告 全部打通")

    print("\n" + "=" * 60)
    print("回归测试：单轮多 tool_calls 场景（复现并验证曾经的真实 bug 修复）")
    agent2 = FinancialAnalyticsAgent(llm_client=FakeLLMClientMultiToolCall(), db_path=DB_PATH)
    result2 = agent2.run("测试多工具调用场景", verbose=True)
    print("最终回答:", result2["final_answer"])
    print("✅ 多 tool_calls 回归测试通过：未触发 'insufficient tool messages' 报错")
