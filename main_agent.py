import os
import time
from dotenv import load_dotenv

# Use the standard LangChain imports for 2026
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tools import tools

load_dotenv()

# --- THE FIX: Aligning the Reasoning Logic with your Tools ---
# Look closely at the "Action:" line below
REACT_PROMPT = """You are a UPSC Environmental Research Agent. 
Today is February 9, 2026. 

YOUR OBJECTIVE:
1. Identify major 2026 news stories using 'Web_Search'.
2. For each major story, find the UPSC 'Mindset' using 'UPSC_PYQ_Search'.
3. Verify if the topic matches the 'Syllabus_Mapper'.

You have access to the following tools:
{tools}

Use the following format strictly:
Question: {input}
Thought: I need to find the latest news from Jan-Feb 2026 first.
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have the news and the examiner's pattern.
Final Answer: [Your complete 20 stories + 5 MCQs here]

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=REACT_PROMPT
)

# Using Gemini 2.0 Flash for speed and high reasoning capability
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1
)

class UPSCResearchExecutor(AgentExecutor):
    def _take_next_step(self, *args, **kwargs):
        # Increased delay for Free Tier stability
        time.sleep(10) 
        return super()._take_next_step(*args, **kwargs)

agent = create_react_agent(llm, tools, prompt)

agent_executor = UPSCResearchExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15  # Balanced for 20 stories
)

def run_upsc_task(user_input):
    print(f"\n🚀 Starting Agentic Reasoning Task...\n")
    try:
        result = agent_executor.invoke({"input": user_input})
        print("\n🎯 FINAL REPORT GENERATED:\n", result["output"])
    except Exception as e:
        if "Final Answer" in str(e):
             # Handle cases where the LLM finishes but misses the tag
             print("⚠️ Task completed. Check logs above for the Final Answer.")
        else:
             print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    task = (
        "Retrieve 20 environmental news items from Jan-Feb 2026. "
        "Then, generate 5 high-quality MCQs based on those stories, "
        "ensuring each MCQ follows a past UPSC question pattern."
    )
    run_upsc_task(task)