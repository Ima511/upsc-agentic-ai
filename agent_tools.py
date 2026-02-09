from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"E:\one drive\OneDrive\Desktop\upsc_agentic_ai\.env")


# 1. Setup the Search Tool (The "Hands" for the Internet)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(query: str) -> str:
    results = tavily.search(query=query, max_results=3)
    return str(results)

search_tool = Tool(
    name="Web_Search",
    func=tavily_search,
    description="Fetches current news from the internet (The Hindu, Indian Express, etc.)"
)

# 2. Setup the Retrieval Tool (The 'Hands' for your PDF Memory)
def query_upsc_db(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory="./upsc_db", embedding_function=embeddings)
    docs = db.similarity_search(query, k=3)
    return "\n---\n".join([f"Source {d.metadata.get('year','N/A')}: {d.page_content}" for d in docs])

upsc_retrieval_tool = Tool(
    name="UPSC_PYQ_Search",
    func=query_upsc_db,
    description="Find how UPSC has asked questions on a topic in the past."
)

# 3. Setup the Syllabus Tool (The 'Hands' for the Map)
def get_syllabus_context(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory="./syllabus_db", embedding_function=embeddings)
    docs = db.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

syllabus_tool = Tool(
    name="Syllabus_Mapper",
    func=get_syllabus_context,
    description="Checks whether a topic is in the UPSC syllabus."
)

tools = [search_tool, upsc_retrieval_tool, syllabus_tool]
