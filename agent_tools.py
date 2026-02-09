import os
from dotenv import load_dotenv
from langchain.tools import Tool, tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tavily import TavilyClient

# Load API Keys
load_dotenv(dotenv_path=r"E:\one drive\OneDrive\Desktop\upsc_agentic_ai\.env")

# --- 1. GLOBAL SETUP ---
# Standardize the embedding model used across all databases
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- 2. SEARCH TOOL (Optimized to fix 400-char limit) ---
def tavily_search(query: str) -> str:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # We move the site list here. Tavily handles this internally, 
    # so it doesn't count against your 400-character query limit.
    trusted_portals = [
        "pib.gov.in", "moef.gov.in", "newsonair.gov.in", "wii.gov.in",
        "downtoearth.org.in", "india.mongabay.com", "cseindia.org",
        "insightsonindia.com", "indiaenvironmentportal.org.in"
    ]
    
    results = tavily.search(
        query=query, 
        max_results=5, 
        include_domains=trusted_portals,
        search_depth="advanced"
    )
    return str(results)

search_tool = Tool(
    name="Web_Search",
    func=tavily_search,
    description="Fetches current 2026 news. Keep the query short (e.g., 'Project Dolphin news 2026')."
)

# --- 3. RETRIEVAL TOOL (The 'Reasoning' Memory) ---
def query_upsc_db(query):
    # Use the global embeddings defined above for consistency
    db = Chroma(persist_directory="./upsc_db", embedding_function=embeddings)
    docs = db.similarity_search(query, k=3)
    return "\n---\n".join([
        f"Past UPSC Pattern (Year {d.metadata.get('year','N/A')}): {d.page_content}" 
        for d in docs
    ])

upsc_retrieval_tool = Tool(
    name="UPSC_PYQ_Search",
    func=query_upsc_db,
    description="Use this to find how UPSC framed questions on this topic in the past 10 years."
)

# --- 4. SYLLABUS TOOL (The 'Map') ---
def get_syllabus_context(query):
    db = Chroma(persist_directory="./syllabus_db", embedding_function=embeddings)
    docs = db.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

syllabus_tool = Tool(
    name="Syllabus_Mapper",
    func=get_syllabus_context,
    description="Checks if a topic is relevant to GS Paper 1, 2, or 3."
)

# Final tools list for main_agent.py
tools = [search_tool, upsc_retrieval_tool, syllabus_tool]