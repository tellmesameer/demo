import google.generativeai as genai

genai.configure(api_key="AIzaSyC0sP2W8NRPMrGxCDjZq7x8yQvZL-t5H-U")
for m in genai.list_models():
    if "gemini" in m.name.lower():
        print(m.name)
