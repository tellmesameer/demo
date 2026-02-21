import google.generativeai as genai

genai.configure(api_key="XXXXXXXXXXXXXXXXX")
for m in genai.list_models():
    if "gemini" in m.name.lower():
        print(m.name)
