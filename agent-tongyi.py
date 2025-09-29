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
# 为数学计算创建带计算器工具的 agent
print("正在加载工具：llm-math (计算器)")
math_tools = load_tools(["llm-math"], llm=llm)

# --- 4. 创建不同类型的 Agent 执行器 ---
print("正在初始化数学计算 Agent...")
# 为数学计算创建带工具的 agent
math_agent = initialize_agent(
    math_tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

print("知识问答将直接使用 LLM，无需创建 Agent...")
# 对于知识问答，直接使用 LLM 而不需要 agent

# --- 5. 运行不同类型的 Agent ---

print("\n" + "="*50)
print(" Agent 运行测试：任务一（纯数学计算）")
print("="*50)
prompt_math = "5的3.5次方是多少？结果保留两位小数。"
print(f"**用户问题：{prompt_math}**")

try:
    # 使用数学计算 agent，它会使用计算器工具
    response_math = math_agent.run(prompt_math)
    print(f"\n✅ 最终答案：{response_math}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")


print("\n" + "="*50)
print(" Agent 运行测试：任务二（非搜索型知识问答）")
print("="*50)
# 知识问答 agent 没有工具，直接由 LLM 回答
prompt_knowledge = "李白是什么朝代的诗人？请用两句话描述他的风格。"
print(f"**用户问题：{prompt_knowledge}**")

try:
    # 直接使用 LLM 回答知识问题，不会尝试使用任何工具
    response_knowledge = llm.invoke(prompt_knowledge)
    print(f"\n✅ 最终答案：{response_knowledge}")
except Exception as e:
    print(f"\n❌ 运行失败：{e}")