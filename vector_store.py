import os
from dotenv import load_dotenv
# Corrected imports for modern LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

class UPSCVectorDB:
    def __init__(self, persist_directory="./upsc_db"):
        # We specify the model to ensure consistency
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.persist_directory = persist_directory
        self.vector_db = None

    def create_and_store(self, question_list, year_metadata):
        """Converts raw strings into searchable Vector Documents."""
        docs = [
            Document(
                page_content=q, 
                metadata={"year": year_metadata}
            ) for q in question_list
        ]
        
        # This creates the DB and saves it to your folder
        self.vector_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Successfully stored {len(docs)} questions in {self.persist_directory}")

    def get_retriever(self):
        """Loads the DB from disk if it exists and returns a retriever."""
        if not self.vector_db:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        return self.vector_db.as_retriever(search_kwargs={"k": 3})