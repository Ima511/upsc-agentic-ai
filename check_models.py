import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)
    print("🔍 Listing available models for your API key:\n")
    try:
        for m in genai.list_models():
            # Check if the model supports embedding content
            if 'embedContent' in m.supported_generation_methods:
                print(f"✅ Found Embedding Model: {m.name}")
    except Exception as e:
        print(f"❌ API Error: {e}")