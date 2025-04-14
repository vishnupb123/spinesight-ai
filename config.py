# config.py
import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "vi_tu_ai_2025_amazing"
    DEBUG = True
    # PDFKIT configuration: if wkhtmltopdf is not in PATH, include its absolute path:
    PDFKIT_CONFIG = {
        'wkhtmltopdf': os.environ.get("WKHTMLTOPDF_PATH") or "/usr/local/bin/wkhtmltopdf"
    }
    GEMINI_API_KEY = "AIzaSyDx69uXZN6xmofcGEZj7d2AL_96Oqdh1Oc"
