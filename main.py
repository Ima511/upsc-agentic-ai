from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import UPSCVectorDB
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def generate_mock_question(current_news_topic):
    db = UPSCVectorDB()
    retriever = db.get_retriever()
    
    # 1. Get historical patterns
    past_questions = retriever.invoke(current_news_topic)
    context = "\n".join([d.page_content for d in past_questions])
    
    # 2. Construct Prompt
    prompt = f"""
    You are a UPSC Paper Setter. 
    CURRENT NEWS: {current_news_topic}
    HISTORICAL PATTERNS: {context}
    
    TASK: Create a fresh UPSC Prelims MCQ based on the current news, 
    matching the difficulty and 'tricky' style of the historical questions provided.
    Include 4 options and the correct answer with an explanation.
    """
    
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    # Example Run
    topic = "New digital currency regulations in India 2026"
    print(generate_mock_question(topic))