import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Which roles use Gemini (better for long-form synthesis)
GEMINI_ROLES = {r.strip() for r in os.getenv("GEMINI_ROLES", "writer").split(",")}
