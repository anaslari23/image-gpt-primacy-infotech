import os
import sys
import json
import argparse
from pathlib import Path
import base64
import requests
from bs4 import BeautifulSoup
from google import genai
from pydantic import BaseModel, Field

# Import the existing image processing logic securely
try:
    from process_image import process_image
except ImportError:
    print("Error: process_image.py not found or has import errors.")
    exit(1)

# Platform specific configurations
PLATFORM_SPECS = {
    "instagram post": {"width": 1080, "height": 1080},
    "instagram reel cover": {"width": 1080, "height": 1920},
    "linkedin": {"width": 1080, "height": 1080},
    "facebook": {"width": 1080, "height": 1080},
    "facebook link thumbnail": {"width": 1200, "height": 630},
    "linkedin link thumbnail": {"width": 1200, "height": 627},
    "twitter post": {"width": 1200, "height": 675},
    "pinterest pin": {"width": 1080, "height": 1440},
    "youtube thumbnail": {"width": 1280, "height": 720},
    "tiktok cover": {"width": 1080, "height": 1920},
}


class PipelineError(Exception):
    pass


def synthesize_brand_intel(url: str, brand_name: str, client: genai.Client) -> dict:
    """Step 1: Scrape website and extract brand intelligence."""
    # Attempt simple scrape
    text_content = ""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract main textual content
            text_content = soup.get_text(separator=" ", strip=True)[
                :3000
            ]  # Limit tokens
    except Exception as e:
        print(
            f"Warning: Failed to scrape {url}. Relying on brand name only. Error: {e}"
        )

    prompt = f"""
    Analyze the following brand context and text from their website ({url}) to extract "Brand Intelligence":
    Brand Name: {brand_name}
    Website content snippet:
    {text_content}
    
    Return a JSON object with EXACTLY these keys:
    - brand_voice: A string describing the tone (e.g., professional, bold, friendly).
    - keywords: A list of 4-6 strings with frequent services/keywords.
    - hashtags: A list of 3-5 standard hashtags to use.
    
    Output nothing but valid JSON.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        # Parse output ensuring JSON format
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Warning: Brand Intel synthesis failed: {e}")
        return {
            "brand_voice": "professional and modern",
            "keywords": [brand_name],
            "hashtags": [f"#{brand_name.replace(' ', '')}"],
        }


def generate_social_content(
    inputs: dict,
    brand_intel: dict,
    market_intel: str,
    client: genai.Client,
    competitor_intel: dict = None,
) -> dict:
    """Step 2: Generate platform-specific content, rival-aware by default."""

    competitor_block = ""
    if competitor_intel and competitor_intel.get("analysis"):
        a = competitor_intel["analysis"]
        competitor_block = f"""
    COMPETITOR INTELLIGENCE (use this to craft a post that clearly beats the competition):
    - What competitors are doing: {a.get("competitor_tactics", "")}
    - Their weaknesses to exploit: {a.get("common_weaknesses", "")}
    - Your winning angle: {a.get("winning_angle", "")}
    - Recommended tone: {a.get("recommended_tone", "")}
    - Power hook (use this as the opening line): {a.get("power_hook", "")}
    - Visual direction: {a.get("visual_direction", "")}
    """

    prompt = f"""
    Generate a social media post based on the following:
    Topic: {inputs.get("topic")}
    Platform: {inputs.get("platform")}
    Post Type: {inputs.get("post_type")}
    Tone: {inputs.get("tone")} (Brand voice: {brand_intel.get("brand_voice")})
    Target Audience: {inputs.get("target_audience")}
    Brand Name: {inputs.get("brand_name")}
    Call to Action (CTA): {inputs.get("cta")}
    Brand Keywords: {", ".join(brand_intel.get("keywords", []))}
    Brand Hashtags: {" ".join(brand_intel.get("hashtags", []))}

    CRITICAL MARKET INTELLIGENCE (Integrate these live insights naturally into the hook or value proposition):
    {market_intel}
    {competitor_block}
    Platform specific rules:
    - LinkedIn: Professional tone, structured formatting (hook -> value -> CTA), include insights/stats. Medium to long.
    - Instagram: Engaging, short-form, hook-driven, emojis allowed, strong CTA, hashtag optimized.
    - Facebook: Conversational tone, slightly longer than Instagram, community-focused.

    Return a JSON object with EXACTLY these keys:
    - caption: The full post text including formatting and emojis.
    - hashtags: A list of strings containing the hashtags to append.
    - image_prompt: A highly descriptive prompt for an AI image generator to create an accompanying image based on the topic. Do NOT include text in the image prompt, just visuals.
    - competitor_edge: A one-sentence explanation of why this post beats the competition (empty string if no competitor data).

    Output nothing but valid JSON.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())
    except Exception as e:
        raise PipelineError(f"Social content generation failed: {e}")


def generate_base_image(image_prompt: str, client: genai.Client, aspect_ratio: str = "1:1") -> Path:
    """Step 3: Generate the base contextual image."""
    base_img_path = Path("static/_tmp_base_image.png")
    base_img_path.parent.mkdir(exist_ok=True)

    print(f"-> Image Prompt ({aspect_ratio}): {image_prompt}")
    try:
        # Note: If Imagen is not available via standard SDK permissions,
        # we will fallback to a generated placeholder to ensure pipeline continues.
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=image_prompt,
            config=dict(
                number_of_images=1, aspect_ratio=aspect_ratio, output_mime_type="image/png"
            ),
        )
        if response.generated_images:
            with open(base_img_path, "wb") as f:
                f.write(response.generated_images[0].image.image_bytes)
            return base_img_path
        else:
            raise PipelineError("No images returned from API.")
    except Exception as e:
        print(
            f"Warning: Native Image generation failed or unauthenticated ({e}). Yielding a fallback placeholder."
        )
        # Fallback to an external random generator for structural testing
        res = requests.get("https://picsum.photos/1080/1080")
        if res.status_code == 200:
            with open(base_img_path, "wb") as f:
                f.write(res.content)
        return base_img_path


def run_pipeline(input_data: dict) -> dict:

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError(
            "GEMINI_API_KEY environment variable not set. Please set it to run the AI engine."
        )

    client = genai.Client(api_key=api_key)

    print("[1] Extracting Brand Intelligence...")
    brand_intel = synthesize_brand_intel(
        input_data.get("website_url"), input_data.get("brand_name"), client
    )

    print("[1.5] Fetching Market Intelligence via Search Grounding...")
    try:
        from market_intel import get_market_intelligence

        intel_res = get_market_intelligence(
            input_data.get("topic"), input_data.get("brand_name")
        )
        market_intel_txt = intel_res.get("intelligence", "")
    except Exception as e:
        print(f"Market Intel Warning: {e}")
        market_intel_txt = ""

    print("[1.7] Running Rival Mode — Auto-discovering & analysing competitors...")
    competitor_intel = {}
    try:
        from competitor_monitor import run_competitor_monitor

        competitor_intel = run_competitor_monitor(
            competitor_names=input_data.get("competitor_names", []),
            platform=input_data.get("platform", "instagram"),
            brand_name=input_data.get("brand_name", ""),
            topic=input_data.get("topic", ""),
        )
        discovered = competitor_intel.get("discovered_competitors", [])
        if discovered:
            print(f"[Rival Mode] Competing against: {', '.join(discovered)}")
    except Exception as e:
        print(f"Rival Mode Warning: {e}")

    print("[2] Generating Rival-Beating Social Content...")
    content = generate_social_content(
        input_data, brand_intel, market_intel_txt, client, competitor_intel
    )

    # Determine target dimensions based on platform
    plat_key = str(input_data.get("platform")).lower()
    dimensions = PLATFORM_SPECS.get(plat_key, {"width": 1080, "height": 1080})
    if plat_key == "instagram" and str(input_data.get("post_type")).lower() == "reel":
        dimensions = PLATFORM_SPECS["instagram reel cover"]

    w = dimensions["width"]
    h = dimensions["height"]
    ai_aspect_ratio = "1:1"
    if w > h:
        ai_aspect_ratio = "16:9" if w / h >= 1.5 else "4:3"
    elif h > w:
        ai_aspect_ratio = "9:16" if h / w >= 1.5 else "3:4"

    print(f"[3] Generating Base Media at {ai_aspect_ratio} natively...")
    base_img_path = generate_base_image(
        content.get("image_prompt", "Corporate background image"), client, aspect_ratio=ai_aspect_ratio
    )

    print("[4] Executing Image Branding Pipeline...")

    final_output_path = Path("static/_final_processed.png")

    # Critical: Call the existing `process_image` untouched
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
        raise PipelineError(f"Image branding engine failed: {e}")

    # Cleanup base image
    base_img_path.unlink(missing_ok=True)

    print("[5 & 6] Assembling Post & Approval Email...")
    subject = f"Approval required: {input_data.get('platform').capitalize()} Post - {input_data.get('topic')[:30]}..."
    body = (
        f"Hi CEO,\n\n"
        f"Please review the following post scheduled for {input_data.get('platform')}.\n\n"
        f"CAPTION:\n{content.get('caption')}\n\n"
        f"HASHTAGS:\n{' '.join(content.get('hashtags'))}\n\n"
        f"IMAGE GENERATED:\n"
        f"Dimensions: {img_meta.get('final_size')}\n"
        f"Review Link: {final_output_path.absolute()}\n\n"
        f"Please reply with APPROVE or REJECT.\n\n"
        f"Thanks,\nMarketing Automation System"
    )

    print("[7] Generating Output Payload...")

    output = {
        "platform": input_data.get("platform"),
        "caption": content.get("caption"),
        "hashtags": content.get("hashtags"),
        "rival_mode": {
            "active": True,
            "discovered_competitors": competitor_intel.get(
                "discovered_competitors", []
            ),
            "competitor_edge": content.get("competitor_edge", ""),
            "analysis": competitor_intel.get("analysis", {}),
        },
        "image": {
            "final_processed": "/static/" + final_output_path.name,
            "dimensions": img_meta.get("final_size"),
            "format": "PNG",
        },
        "status": "Pending Approval",
        "approval_email": {"subject": subject, "body": body},
    }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marketing Content Pipeline")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSON file"
    )
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input: {e}")
        exit(1)

    try:
        result = run_pipeline(data)
        print("\n================ FINAL JSON PAYLOAD ================\n")
        print(json.dumps(result, indent=2))
    except PipelineError as e:
        print(f"\nPipeline Error: {e}", file=sys.stderr)
        exit(1)
