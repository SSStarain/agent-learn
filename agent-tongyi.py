import os
# 导入 LangChain 中用于 DashScope 的 Tongyi LLM 类
from langchain_community.llms import Tongyi 
# 导入 Agent 相关的模块和工具加载器
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType

# --- 1. 配置 API Key 和检查 ---
# ⚠️ 请确保 DASHSCOPE_API_KEY 和 SERPER_API_KEY 已经设置为环境变量。

os.environ["DASHSCOPE_API_KEY"] = "sk-5b5804c01b9f4bfa927f5a142fd2c5d0" 
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_KEY" # 仍然需要用于网络搜索工具

if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("SERPER_API_KEY"):
    print("----------------------------------------------------------------------")
    print("⚠️ 错误：请先设置 DASHSCOPE_API_KEY 和 SERPER_API_KEY 环境变量。")
    print("----------------------------------------------------------------------")
    exit()

# --- 2. 初始化 LLM（Agent 的大脑）---
# 使用 Tongyi 类，指定模型为 qwen-turbo（或其他您想用的通义模型）
DASHSCOPE_LLM_MODEL = "qwen-turbo" 

print(f"正在初始化 DashScope LLM: {DASHSCOPE_LLM_MODEL}...")
llm = Tongyi(
    model=DASHSCOPE_LLM_MODEL, 
    temperature=0,
    # DashScope 的 API Key 将自动从 DASHSCOPE_API_KEY 环境变量读取
) 

# --- 3. 定义工具 (Tools) ---
# 'llm-math' 用于计算，'serpapi' 用于网络搜索
print("正在加载工具...")
# load_tools 的使用方式保持不变，它会使用上面定义的 llm 实例
tools = load_tools(["llm-math", "serpapi"], llm=llm)

# --- 4. 创建 Agent 执行器 ---
print("正在初始化 Agent...")
# AgentType.ZERO_SHOT_REACT_DESCRIPTION 使用 ReAct 框架
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