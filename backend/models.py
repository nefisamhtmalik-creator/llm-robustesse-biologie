import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

AVAILABLE_MODELS = {
    "gemini"      : "Gemini (Google)",
    "deepseek"    : "DeepSeek",
    "flan-t5-small": "FLAN-T5 (local)",
}

def generate_response(question: str, model_name: str = "gemini") -> str:
    if model_name == "gemini":
        return _ask_gemini(question)
    elif model_name == "deepseek":
        return _ask_deepseek(question)
    else:
        return _ask_flant5(question)

def _ask_gemini(question: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Réponds en français à cette question de biologie de façon concise (3-4 phrases maximum) : {question}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur Gemini : {str(e)}"

def _ask_deepseek(question: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en biologie. Réponds en français de façon concise (3-4 phrases maximum)."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur DeepSeek : {str(e)}"

def _ask_flant5(question: str) -> str:
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-small"
        ).to(device)
        prompt = f"Answer in French this biology question: {question}"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Erreur FLAN-T5 : {str(e)}"

def load_model(model_name: str = "gemini"):
    return None, None