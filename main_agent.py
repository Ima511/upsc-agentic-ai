import os
import time
from dotenv import load_dotenv

try:
    from langchain_classic.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_react_agent

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tools import tools

load_dotenv()

# --- THE FIX: We must include {tools} and {tool_names} inside the string ---
REACT_PROMPT = """You are a UPSC Environmental Research Agent. 
Today is February 9, 2026. 

STRICT DATA SOURCE RULE:
You are ONLY allowed to use information from these verified portals:
1. Govt: pib.gov.in, moef.gov.in, newsonair.gov.in, wii.gov.in, missionlife-moefcc.nic.in, ndap.niti.gov.in, cpreecenvis.nic.in
2. Knowledge: indiaenvironmentportal.org.in, downtoearth.org.in, indiawaterportal.org, indiabiodiversity.org, india.mongabay.com, prakati.in, academy.wwfindia.org
3. NGOs/Research: cseindia.org, atree.org, greenpeace.org, tarumitra.org

You have access to the following tools:
{tools}

Use the following format strictly:
Question: the input question you must answer
Thought: I need to use the site: operator to search only the allowed portals.
Action: the action to take, must be one of [{tool_names}]
Action Input: the search query with site: restrictions
Observation: the result of the action
... (this Thought/Action/Observation can repeat)
Thought: I now know the final answer
Final Answer: The final response

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

# --- THE FIX: input_variables must match the placeholders in REACT_PROMPT ---
prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=REACT_PROMPT
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    max_retries=6
)

class UPSCResearchExecutor(AgentExecutor):
    def _take_next_step(self, *args, **kwargs):
        # 8-second delay to ensure we stay under the 10 RPM limit
        time.sleep(8) 
        return super()._take_next_step(*args, **kwargs)

# This line was failing because 'tools' wasn't being passed correctly to the prompt
agent = create_react_agent(llm, tools, prompt)

agent_executor = UPSCResearchExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=25 
)

def run_upsc_task(user_input):
    print(f"🚀 Launching Agent...")
    try:
        result = agent_executor.invoke({"input": user_input})
        print("\n🎯 FINAL RESULT:\n", result["output"])
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Combined prompt to ensure the agent knows exactly where to look
    task = (
        "Find 20 major environmental news stories from India (Jan-Feb 2026) "
        "using ONLY the portals listed in your instructions. "
        "Then create 5 UPSC Prelims MCQs with 3-statement format."
    )
    run_upsc_task(task)