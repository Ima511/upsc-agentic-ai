from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

class UPSCSyllabusManager:
    def __init__(self, persist_dir="./syllabus_db"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.persist_dir = persist_dir

    def check_relevance(self, topic):
        db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        # Check if the topic exists in syllabus
        results = db.similarity_search_with_relevance_scores(topic, k=1)
        if results and results[0][1] > 0.7:
            return True, results[0][0].page_content
        return False, None