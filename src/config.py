import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
