import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_data():
    # 1. Initialize Gemini Embeddings
    # Use the stable 2026 model name
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Process Syllabus
    print("⏳ Building Syllabus Memory...")
    syllabus_path = "data/upsc_syllabus.pdf"
    if os.path.exists(syllabus_path):
        loader = PyPDFLoader(syllabus_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        syllabus_docs = loader.load_and_split(splitter)
        Chroma.from_documents(syllabus_docs, embeddings, persist_directory="./syllabus_db")
        print("✅ Syllabus DB Created.")

    # 3. Process PYQs with Year Metadata
    print("⏳ Reading PYQ PDFs...")
    pyq_final_docs = []
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ Error: 'data' folder not found!")
        return

    for file in os.listdir("data"):
        if file.endswith(".pdf") and "syllabus" not in file.lower():
            year_match = re.search(r'(\d{4})', file)
            year = year_match.group(1) if year_match else "Unknown"
            
            print(f"  -> Reading {file} (Year: {year})")
            loader = PyPDFLoader(f"data/{file}")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = loader.load_and_split(splitter)
            
            for chunk in chunks:
                chunk.metadata["year"] = year
                chunk.metadata["source"] = file
            
            pyq_final_docs.extend(chunks)
    
    # 4. MICRO-BATCH Ingestion (The Fix for 429)
    if pyq_final_docs:
        print(f"🚀 Starting Micro-Batch Ingestion of {len(pyq_final_docs)} chunks...")
        
        # Open (or create) the database
        vector_db = Chroma(
            embedding_function=embeddings, 
            persist_directory="./upsc_db"
        )
        
        # Batch size of 10 is very safe for the free tier
        batch_size = 10 
        for i in range(0, len(pyq_final_docs), batch_size):
            batch = pyq_final_docs[i : i + batch_size]
            
            # Retry logic for individual batches
            success = False
            for attempt in range(3):
                try:
                    vector_db.add_documents(batch)
                    print(f"✅ [{i + len(batch)}/{len(pyq_final_docs)}] Chunks added.")
                    success = True
                    # Mandatory pause to avoid hitting the 100-requests-per-minute limit
                    time.sleep(12) 
                    break 
                except Exception as e:
                    wait_time = 30 * (attempt + 1)
                    print(f"⚠️ Batch {i} failed. Rate limit hit? Waiting {wait_time}s... (Attempt {attempt+1}/3)")
                    time.sleep(wait_time)
            
            if not success:
                print(f"❌ Could not upload batch starting at {i}. Skipping to save progress.")

        print(f"🎉 Process Complete! Check the './upsc_db' folder.")
    else:
        print("❌ Error: No PYQ PDFs found in /data folder!")

if __name__ == "__main__":
    ingest_data()