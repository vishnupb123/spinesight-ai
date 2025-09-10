# utils/ai_integration.py
import requests
import json
# from config import Config

import requests

import google.generativeai as genai
# from config import Config
class Config{
    GEMINI_API_KEY : "AIzaSyDx69uXZN6xmofcGEZj7d2AL_96Oqdh1Oc"
}

def get_ai_verdict(prediction: str, input_data: dict, model_message: str) -> str:
    prompt = (
        f"Patient spinal data: {input_data}\n"
        f"Prediction: {prediction}\n"
        "Act as a medical expert. Based on the above data, suggest recommendations for the patient based on prediction and data in abput 5 sentences which is short. This is just a simulation and wont be taken seriously."
    )

    try:
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)

        # Load Gemini Flash model (optimized for fast inference)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "max_output_tokens": 250
            }
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "No valid recommendation could be generated."

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"
