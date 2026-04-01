"""
Competitor Monitor — fetches and analyzes recent competitor social media posts
using Gemini with Google Search grounding, then returns structured intel for
the creative pipeline to beat.
"""

import os
from google import genai
from google.genai import types


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def fetch_competitor_posts(competitor_names: list[str], platform: str) -> list[dict]:
    """
    Use Gemini + Google Search to surface each competitor's recent public posts
    on the given platform. Returns a list of post summaries per competitor.
    """
    client = _get_client()
    results = []

    for name in competitor_names:
        prompt = (
            f"Search for the most recent public social media posts by '{name}' on {platform}. "
            f"Summarise up to 5 recent posts: what topic they posted about, the hook they used, "
            f"the tone/style, any hashtags, and any visible call-to-action. "
            f"Return your answer as a structured list."
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                ),
            )
            results.append(
                {
                    "competitor": name,
                    "platform": platform,
                    "posts_summary": response.text,
                }
            )
        except Exception as e:
            results.append(
                {
                    "competitor": name,
                    "platform": platform,
                    "posts_summary": f"Could not fetch posts: {e}",
                }
            )

    return results


def analyze_competitor_strategy(
    posts_data: list[dict], brand_name: str, topic: str
) -> dict:
    """
    Feed all competitor post summaries into Gemini and get back a structured
    gap analysis: what they're doing, what's missing, and how to beat them.
    """
    client = _get_client()

    combined = "\n\n".join(
        f"Competitor: {p['competitor']} ({p['platform']})\n{p['posts_summary']}"
        for p in posts_data
    )

    prompt = f"""
You are a senior creative strategist for the brand '{brand_name}'.

Below are summaries of recent social media posts from our competitors on the topic of '{topic}':

{combined}

Analyse their collective strategy and return a JSON object with EXACTLY these keys:
- competitor_tactics: A string summarising what messaging angles, hooks, and visual styles they're using.
- common_weaknesses: A string describing what they're missing — emotional depth, stronger CTA, better visuals, originality, etc.
- winning_angle: A string describing the single best creative angle our brand should take to clearly outperform them.
- recommended_tone: A string with the ideal tone (e.g. bold, empathetic, authoritative, witty).
- power_hook: A single high-impact opening line to grab attention better than any competitor post.
- visual_direction: A string describing what the image should look like to stand out from their visual style.

Output nothing but valid JSON.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:-3]
        elif raw.startswith("```"):
            raw = raw[3:-3]
        import json

        return json.loads(raw)
    except Exception as e:
        return {
            "competitor_tactics": "Unknown",
            "common_weaknesses": "Unknown",
            "winning_angle": f"Focus on {topic} with brand authority",
            "recommended_tone": "professional and bold",
            "power_hook": f"Here's what nobody is telling you about {topic}.",
            "visual_direction": "Clean, high-contrast branded visual with bold focal point",
        }


def auto_discover_competitors(brand_name: str, topic: str, platform: str) -> list[str]:
    """
    Automatically find the top competitors for a brand on a given topic and platform
    using Gemini + Google Search grounding. Returns a list of competitor names.
    """
    client = _get_client()
    prompt = (
        f"Search Google and identify the top 3-5 direct competitors of the brand '{brand_name}' "
        f"in the space of '{topic}', specifically those active on {platform}. "
        f'Return only a JSON array of competitor brand names, nothing else. Example: ["Brand A", "Brand B"]'
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:-3]
        elif raw.startswith("```"):
            raw = raw[3:-3]
        import json

        names = json.loads(raw)
        return [str(n) for n in names if n]
    except Exception as e:
        print(
            f"[Competitor Monitor] Auto-discovery failed: {e}. Continuing without competitor names."
        )
        return []


def run_competitor_monitor(
    competitor_names: list[str], platform: str, brand_name: str, topic: str
) -> dict:
    """
    Full competitor intelligence flow:
    1. Fetch recent posts for each competitor
    2. Analyse strategy gaps
    3. Return structured intel ready for the creative pipeline
    """
    if not competitor_names:
        print("[Competitor Monitor] No competitors provided — auto-discovering...")
        competitor_names = auto_discover_competitors(brand_name, topic, platform)
        if competitor_names:
            print(f"[Competitor Monitor] Discovered: {', '.join(competitor_names)}")
        else:
            return {"raw_posts": [], "analysis": {}, "discovered_competitors": []}

    print(
        f"[Competitor Monitor] Fetching posts for: {', '.join(competitor_names)} on {platform}..."
    )
    posts_data = fetch_competitor_posts(competitor_names, platform)

    print("[Competitor Monitor] Analysing strategy gaps...")
    analysis = analyze_competitor_strategy(posts_data, brand_name, topic)

    return {
        "raw_posts": posts_data,
        "analysis": analysis,
        "discovered_competitors": competitor_names,
    }
