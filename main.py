"""
系统总入口

用法:
    export LLM_API_KEY="your-key"
    export LLM_BASE_URL="https://api.openai.com/v1"      # 或 Qwen/DeepSeek 等兼容端点
    export LLM_MODEL_NAME="gpt-4o"
    python main.py "分析客户流失的主要驱动因素，并检验流失客户和非流失客户的账户余额是否有显著差异"
"""
import sys
from src.agent import FinancialAnalyticsAgent


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "分析客户流失的主要驱动因素，并检验流失客户和非流失客户的账户余额是否有显著差异，最后生成一份报告"

    print(f"用户问题: {question}\n{'='*60}")

    agent = FinancialAnalyticsAgent()
    result = agent.run(question)

    print(f"\n{'='*60}")
    print(f"最终回答: {result['final_answer']}")
    print(f"共使用 {result['steps_used']} 个工具调用步骤")


if __name__ == "__main__":
    main()
