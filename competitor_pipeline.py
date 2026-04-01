"""
Competitor Creative Pipeline — generates social media content that beats
competitors, brands it, and publishes (or schedules) it automatically.
"""

import os
import json
from pathlib import Path

from google import genai
from pipeline import generate_base_image, PipelineError, PLATFORM_SPECS
from process_image import process_image
from competitor_monitor import run_competitor_monitor
from social_publisher import SocialPublisher


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def generate_rival_content(inputs: dict, competitor_intel: dict) -> dict:
    """
    Generate a post specifically crafted to outperform competitors.
    Uses the gap analysis from competitor_monitor to inform every creative decision.
    """
    client = _get_client()
    analysis = competitor_intel.get("analysis", {})

    prompt = f"""
You are a world-class social media creative director for the brand '{inputs["brand_name"]}'.

Your competitors are doing this:
{analysis.get("competitor_tactics", "")}

Their weaknesses:
{analysis.get("common_weaknesses", "")}

Your winning angle:
{analysis.get("winning_angle", "")}

Use this power hook to open the post:
"{analysis.get("power_hook", "")}"

Now create a {inputs["platform"]} post on the topic: '{inputs["topic"]}'
- Tone: {analysis.get("recommended_tone", inputs.get("tone", "bold"))}
- Target audience: {inputs.get("target_audience", "general")}
- CTA: {inputs.get("cta", "Learn more")}
- Post type: {inputs.get("post_type", "post")}

Platform rules:
- LinkedIn: Professional, structured (hook → value → CTA), stats/insights welcome, medium-long.
- Instagram: Short, punchy, emoji-friendly, hashtag-heavy, strong visual CTA.
- Facebook: Conversational, community feel, slightly longer than Instagram.

Return a JSON object with EXACTLY these keys:
- caption: Full post text including formatting, emojis, and the power hook as the first line.
- hashtags: A list of 5-10 relevant hashtag strings.
- image_prompt: A highly descriptive image generation prompt based on this visual direction: "{analysis.get("visual_direction", "bold branded visual")}". Do NOT include any text in the image prompt.
- competitor_edge: A one-sentence explanation of why this post beats the competition.

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
        return json.loads(raw)
    except Exception as e:
        raise PipelineError(f"Rival content generation failed: {e}")


def run_rival_pipeline(input_data: dict) -> dict:
    """
    Full rival pipeline:
    1. Monitor competitor posts
    2. Analyse gaps
    3. Generate superior creative
    4. Brand the image
    5. Publish or schedule to social media

    Expected input_data keys:
        brand_name, topic, platform, post_type, tone, target_audience,
        cta, competitor_names (list[str]),
        publish (bool, default False),
        schedule_time (str, "now" or ISO datetime),
        token (str), actor_id (str)
    """
    platform = input_data.get("platform", "instagram")
    competitor_names = input_data.get("competitor_names", [])

    if not competitor_names:
        raise PipelineError("At least one competitor name is required.")

    # Step 1 — Competitor intelligence
    print("[1] Running competitor monitor...")
    competitor_intel = run_competitor_monitor(
        competitor_names=competitor_names,
        platform=platform,
        brand_name=input_data["brand_name"],
        topic=input_data["topic"],
    )

    # Step 2 — Generate rival-beating content
    print("[2] Generating rival-beating creative...")
    content = generate_rival_content(input_data, competitor_intel)

    # Step 3 — Generate base image
    print("[3] Generating base image...")
    client = _get_client()
    base_img_path = generate_base_image(
        content.get("image_prompt", "bold branded abstract visual"), client
    )

    # Step 4 — Brand the image
    print("[4] Branding image...")
    plat_key = str(platform).lower()
    dimensions = PLATFORM_SPECS.get(plat_key, {"width": 1080, "height": 1080})
    if (
        plat_key == "instagram"
        and str(input_data.get("post_type", "")).lower() == "reel"
    ):
        dimensions = PLATFORM_SPECS["instagram reel cover"]

    final_output_path = Path("static/_rival_final.png")
    try:
        img_meta = process_image(
            input_path=base_img_path,
            output_path=final_output_path,
            width=dimensions["width"],
            height=dimensions["height"],
            resize_mode="cover",
            logo_variant="auto",
            logo_position="auto",
            logo_scale=15.0,
            logo_margin=20,
            logo_opacity=0.85,
            logo_blend_mode="normal",
            background_color="white",
            output_format="png",
            quality=95,
        )
    except Exception as e:
        raise PipelineError(f"Image branding failed: {e}")

    base_img_path.unlink(missing_ok=True)

    full_caption = (
        content.get("caption", "") + "\n\n" + " ".join(content.get("hashtags", []))
    )

    # Step 5 — Publish or schedule
    publish_result = {"status": "pending_approval"}
    if input_data.get("publish", False):
        print("[5] Publishing to social media...")
        token = input_data.get("token", "")
        actor_id = input_data.get("actor_id", "")
        schedule_time = input_data.get("schedule_time", "now")

        if schedule_time == "now":
            try:
                if "linkedin" in plat_key:
                    publish_result = SocialPublisher.post_to_linkedin(
                        full_caption, str(final_output_path), token, actor_id
                    )
                elif "facebook" in plat_key:
                    publish_result = SocialPublisher.post_to_facebook(
                        full_caption, str(final_output_path), token, actor_id
                    )
                elif "instagram" in plat_key:
                    publish_result = SocialPublisher.post_to_instagram(
                        full_caption, "http://example.com/mock.png", token, actor_id
                    )
            except Exception as e:
                publish_result = {"status": "error", "message": str(e)}
        else:
            from database import add_post

            creds = {"token": token, "actor_id": actor_id}
            post_id = add_post(
                platform, full_caption, str(final_output_path), schedule_time, creds
            )
            publish_result = {
                "status": "scheduled",
                "post_id": post_id,
                "schedule_time": schedule_time,
            }

    return {
        "platform": platform,
        "caption": content.get("caption"),
        "hashtags": content.get("hashtags"),
        "competitor_edge": content.get("competitor_edge"),
        "competitor_analysis": competitor_intel.get("analysis"),
        "image": {
            "final_processed": "/static/" + final_output_path.name,
            "dimensions": img_meta.get("final_size"),
            "format": "PNG",
        },
        "publish_result": publish_result,
    }
