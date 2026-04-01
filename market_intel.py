import os
from google import genai
from google.genai import types


def get_market_intelligence(topic: str, brand_name: str) -> dict:
    """Uses Gemini 3 Flash with Google Search Grounding to fetch live market intel."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)

    prompt = f"Using latest Google Search data, summarize the most recent trending news and competitor movements regarding '{topic}'. Relate this back as actionable advice for the brand '{brand_name}'."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
            ),
        )
        return {"status": "success", "intelligence": response.text}
    except Exception as e:
        return {"status": "error", "message": f"Intelligence gathering failed: {e}"}
