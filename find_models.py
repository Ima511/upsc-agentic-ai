import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

    print("🔍 Fetching models compatible with your API key...\n")
    try:
        # List all available models
        for m in genai.list_models():
            # Check if the model supports 'generateContent'
            if 'generateContent' in m.supported_generation_methods:
                # We strip the 'models/' prefix for LangChain compatibility
                model_name = m.name.replace('models/', '')
                print(f"✅ Compatible: {model_name}")
                print(f"   Description: {m.description}\n")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")