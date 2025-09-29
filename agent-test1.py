import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType

# --- 1. 配置 API Key ---
# ⚠️ 注意：请在运行代码前，将以下行中的 YOUR_OPENAI_KEY 替换为您自己的密钥，
#    并将 SERPER_API_KEY 替换为您用于网络搜索的密钥（如果您要使用搜索工具）。


# 错误
os.environ["OPENAI_API_KEY"] = "sk-5b5804c01b9f4bfa927f5a142fd2c5d0" 
os.environ["SERPAPI_API_KEY"] = "sk-5b5804c01b9f4bfa927f5a142fd2c5d0" 

# 建议在运行环境中设置环境变量，而不是直接写在代码中。
# 如果您在终端中设置了环境变量（例如 export OPENAI_API_KEY="..."），则无需修改上方代码。
if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPAPI_API_KEY"):
    print("----------------------------------------------------------------------")
    print("⚠️ 错误：请先设置 OPENAI_API_KEY 和 SERPER_API_KEY 环境变量，或在代码中取消注释并填入您的密钥。")
    print("----------------------------------------------------------------------")
    exit()

# --- 2. 初始化 LLM（Agent 的大脑）---
# temperature=0 确保结果更稳定和确定
llm = OpenAI(temperature=0) 

# --- 3. 定义工具 (Tools) ---
# 'llm-math' 用于计算，'serpapi' 用于网络搜索
print("正在加载工具...")
tools = load_tools(["llm-math", "serpapi"], llm=llm)

# --- 4. 创建 Agent 执行器 ---
# AgentType.ZERO_SHOT_REACT_DESCRIPTION 使用 ReAct (Reasoning and Acting) 框架
# verbose=True 会打印 Agent 的思考过程
print("正在初始化 Agent...")
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True 
)

# --- 5. 运行 Agent ---

print("\n" + "="*50)
print(" Agent 运行测试：场景一（纯计算）")
print("="*50)
prompt_math = "5的3.5次方是多少？结果保留两位小数。"
print(f"**用户问题：{prompt_math}**")

try:
    response_math = agent.run(prompt_math)
    print(f"\n✅ 最终答案：{response_math}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")


print("\n" + "="*50)
print(" Agent 运行测试：场景二（搜索+计算）")
print("="*50)
prompt_search_and_math = "现在是哪一年？这一年如果乘以1.5是多少？"
print(f"**用户问题：{prompt_search_and_math}**")

try:
    response_search_and_math = agent.run(prompt_search_and_math)
    print(f"\n✅ 最终答案：{response_search_and_math}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")