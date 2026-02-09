import os
from dotenv import load_dotenv

# Compatibility layer for 2026 LangChain
try:
    from langchain_classic.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_react_agent

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tools import tools

load_dotenv()

# --- FIX 1: STRICTOR PROMPT TEMPLATE ---
# We force the 'Action:' keyword and provide a clear negative constraint.
REACT_PROMPT = """You are a helpful AI agent that can use tools.

You have access to the following tools:
{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the user

CRITICAL RULE: Never provide a Thought without an immediate Action or Final Answer. 
If you forget the 'Action:' or 'Final Answer:' line, the system will fail.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=["input", "tool_names", "tools", "agent_scratchpad"],
    template=REACT_PROMPT
)

# --- FIX 2: STABLE MODEL CONFIG ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    version="v1",          # Force stable v1 endpoint
    temperature=0.1,       # Lower temperature reduces "hallucinated" formatting
    max_retries=6,
    timeout=180
)

agent = create_react_agent(llm, tools, prompt)

# --- FIX 3: CUSTOM ERROR HANDLER ---
# Instead of True, we give a string that tells the agent how to fix itself.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Error: You missed the 'Action:' keyword after your Thought. Please re-format your response using: Action: [tool_name] followed by Action Input: [input].",
    max_iterations=12,    # Increased to allow for self-correction loops
    max_execution_time=300 # 5 minutes max
)

def run_upsc_task(user_input):
    print(f"\n🚀 Starting Task: {user_input}\n")
    try:
        result = agent_executor.invoke({"input": user_input})
        print("\n🎯 FINAL MOCK TEST:\n", result["output"])
    except Exception as e:
        print(f"\n❌ Critical Failure: {e}")

if __name__ == "__main__":
    # Optimized Task String for February 2026
    batch_task = (
        "Find exactly 20 major environmental news stories from India reported between Feb 1 and 8 Feb , 2026. "
        "Search sources like PIB.gov.in and DownToEarth. "
        "Then, create 15 high-quality UPSC Prelims MCQs based on these stories. "
        "Ensure each MCQ has at least 3 statements and 4 options, a correct answer, and a detailed explanation."
    )
    
    run_upsc_task(batch_task)