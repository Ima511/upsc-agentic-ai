import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def test_search():
    # 1. Load the same embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Connect to your existing PYQ database
    db = Chroma(persist_directory="./upsc_db", embedding_function=embeddings)
    
    # 3. Ask a test question
    query = "What questions have been asked about the Preamble of the Constitution?"
    
    print(f"\n🔍 Searching for: {query}...")
    
    # Search for the top 3 most relevant snippets
    docs = db.similarity_search(query, k=3)
    
    print("\n--- RESULTS FOUND ---")
    for i, doc in enumerate(docs):
        year = doc.metadata.get("year", "Unknown")
        source = doc.metadata.get("source", "Unknown")
        print(f"\nMatch {i+1} (Year: {year} | Source: {source}):")
        print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
        print("-" * 30)

if __name__ == "__main__":
    test_search()