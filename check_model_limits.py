import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def diagnose_models():
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found.")
        return

    genai.configure(api_key=api_key)
    
    print("🔍 Testing your API key against all available models...\n")
    
    # Get all models that support generation
    models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    status_report = []

    for model_info in models:
        m_name = model_info.name.replace('models/', '')
        print(f"🧪 Testing {m_name}...", end=" ", flush=True)
        
        try:
            # Send a tiny, low-token request to check availability
            model = genai.GenerativeModel(m_name)
            response = model.generate_content("Hi", generation_config={"max_output_tokens": 1})
            
            print("✅ WORKING")
            status_report.append({"model": m_name, "status": "Available", "exhausted": False})
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print("⚠️ EXHAUSTED (Quota Limit Reached)")
                status_report.append({"model": m_name, "status": "Exhausted", "exhausted": True})
            elif "403" in err_msg:
                print("🚫 PERMISSION DENIED (Not in your tier)")
                status_report.append({"model": m_name, "status": "Forbidden", "exhausted": False})
            else:
                print(f"❌ ERROR: {err_msg[:50]}...")
                status_report.append({"model": m_name, "status": "Other Error", "exhausted": False})

    print("\n--- 📊 FINAL SUMMARY ---")
    for item in status_report:
        icon = "❌" if item['exhausted'] else "✅"
        print(f"{icon} {item['model']}: {item['status']}")

if __name__ == "__main__":
    diagnose_models()