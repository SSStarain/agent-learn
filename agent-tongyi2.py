import os
# 导入 LangChain 中用于 DashScope 的 Tongyi LLM 类
from langchain_community.llms import Tongyi 
# 导入 Agent 相关的模块和工具加载器
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType


os.environ["DASHSCOPE_API_KEY"] = "sk-5b5804c01b9f4bfa927f5a142fd2c5d0" 

# --- 1. 配置 API Key 和检查 ---
# ⚠️ 请确保 DASHSCOPE_API_KEY 已经设置为环境变量。
if not os.getenv("DASHSCOPE_API_KEY"):
    print("----------------------------------------------------------------------")
    print("⚠️ 错误：请先设置 DASHSCOPE_API_KEY 环境变量。")
    print("----------------------------------------------------------------------")
    exit()

# --- 2. 初始化 LLM（Agent 的大脑）---
# 使用 Tongyi 类，指定模型为 qwen-turbo
DASHSCOPE_LLM_MODEL = "qwen-turbo" 

print(f"正在初始化 DashScope LLM: {DASHSCOPE_LLM_MODEL}...")
llm = Tongyi(
    model=DASHSCOPE_LLM_MODEL, 
    temperature=0,
    # DashScope 的 API Key 将自动从 DASHSCOPE_API_KEY 环境变量读取
) 

# --- 3. 定义工具 (Tools) ---
# 仅加载 'llm-math'（计算器）工具，移除了 'serpapi'
print("正在加载工具：llm-math (计算器)")
tools = load_tools(["llm-math"], llm=llm)

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
print(" Agent 运行测试：任务一（纯数学计算）")
print("="*50)
prompt_math = "5的3.5次方是多少？结果保留两位小数。"
print(f"**用户问题：{prompt_math}**")

try:
    response_math = agent.run(prompt_math)
    print(f"\n✅ 最终答案：{response_math}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")


print("\n" + "="*50)
print(" Agent 运行测试：任务二（非搜索型知识问答）")
print("="*50)
# Agent 现在无法访问实时信息，只能依赖 LLM 内部的知识
prompt_knowledge = "李白是什么朝代的诗人？请用两句话描述他的风格。"
print(f"**用户问题：{prompt_knowledge}**")

try:
    # 对于这个知识问题，Agent 将不会使用任何工具，而是直接由 LLM 回答
    response_knowledge = agent.run(prompt_knowledge)
    print(f"\n✅ 最终答案：{response_knowledge}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")