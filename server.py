#!/usr/bin/env python3
"""
FastAPI server for the Image Processing Assistant.
Serves the frontend and exposes a /process endpoint.
"""

import io
import uuid
import base64
from pathlib import Path
from typing import Annotated

import asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from process_image import process_image, LOGO_PATHS, _parse_color

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Image Processing Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from database import init_db, get_due_posts, update_post_status, add_post
from social_publisher import SocialPublisher
from market_intel import get_market_intelligence

async def publish_worker():
    while True:
        try:
            posts = get_due_posts()
            for p in posts:
                print(f"[Scheduler] Triggering scheduled post {p['id']} for {p['platform']}")
                platform = str(p['platform']).lower()
                res = {}
                try:
                    if 'linkedin' in platform:
                        res = SocialPublisher.post_to_linkedin(
                            p['caption'], p['image_path'],
                            p['credentials'].get("token", ""),
                            p['credentials'].get("actor_id", "")
                        )
                    elif 'facebook' in platform:
                        res = SocialPublisher.post_to_facebook(
                            p['caption'], p['image_path'],
                            p['credentials'].get("token", ""),
                            p['credentials'].get("actor_id", "")
                        )
                    elif 'instagram' in platform:
                        res = SocialPublisher.post_to_instagram(
                            p['caption'], p['credentials'].get("public_image_url", "http://example.com/img.png"),
                            p['credentials'].get("token", ""),
                            p['credentials'].get("actor_id", "")
                        )
                    print(f"Post {p['id']} result: {res}")
                    update_post_status(p['id'], 'Published' if res.get('status') == 'success' else 'Failed')
                except Exception as e:
                    print(f"Post {p['id']} completely failed: {e}")
                    update_post_status(p['id'], 'Failed')
        except Exception as e:
             pass
        await asyncio.sleep(20) # Poll every 20 seconds for demo

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(publish_worker())

BASE_DIR  = Path(__file__).parent
STATIC    = BASE_DIR / "static"
STATIC.mkdir(exist_ok=True)

LOGO_DIR  = BASE_DIR / "logo"

app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/logo", StaticFiles(directory=LOGO_DIR), name="logo")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "static" / "index.html").read_text()

from pipeline import run_pipeline, PipelineError
from competitor_pipeline import run_rival_pipeline

class CampaignRequest(BaseModel):
    topic: str
    platform: str
    tone: str
    post_type: str
    target_audience: str
    brand_name: str
    website_url: str
    cta: str
    schedule_time: str

@app.post("/campaign")
async def generate_campaign(req: CampaignRequest):
    try:
        # Convert Pydantic object to dict securely
        result = run_pipeline(req.dict())
        return result
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error. See console.")

class RivalRequest(BaseModel):
    brand_name: str
    topic: str
    platform: str
    post_type: str
    tone: str
    target_audience: str
    cta: str
    competitor_names: list[str]
    publish: bool = False
    schedule_time: str = "now"
    token: str = ""
    actor_id: str = ""

@app.post("/rival")
async def rival_campaign(req: RivalRequest):
    """
    Monitor competitor posts, analyse gaps, generate a superior creative,
    brand it, and optionally publish/schedule it to social media.
    """
    try:
        result = run_rival_pipeline(req.dict())
        return result
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error. See console.")


class PublishRequest(BaseModel):
    platform: str
    caption: str
    image_path: str
    schedule_time: str
    token: str
    actor_id: str

@app.post("/publish")
async def publish_post(req: PublishRequest):
    # If schedule_time is 'now', publish immediately. 
    # Otherwise, add to DB.
    creds = {"token": req.token, "actor_id": req.actor_id}
    
    if req.schedule_time == "now":
        platform = req.platform.lower()
        try:
             res = {}
             if 'linkedin' in platform:
                 res = SocialPublisher.post_to_linkedin(req.caption, req.image_path, req.token, req.actor_id)
             elif 'facebook' in platform:
                 res = SocialPublisher.post_to_facebook(req.caption, req.image_path, req.token, req.actor_id)
             elif 'instagram' in platform:
                 res = SocialPublisher.post_to_instagram(req.caption, "http://example.com/mock.png", req.token, req.actor_id)
             return res
        except Exception as e:
             raise HTTPException(500, str(e))
    else:
        post_id = add_post(req.platform, req.caption, req.image_path, req.schedule_time, creds)
        return {"status": "success", "message": f"Post scheduled with ID {post_id}"}

class ApprovalRequest(BaseModel):
    platform: str
    caption: str
    image_path: str
    schedule_time: str
    ceo_email: str
    token: str
    actor_id: str

@app.post("/request_approval")
async def request_approval(req: ApprovalRequest):
    import datetime
    db_schedule = req.schedule_time
    if req.schedule_time == "now":
        db_schedule = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
    creds = {"token": req.token, "actor_id": req.actor_id}
    post_id = add_post(req.platform, req.caption, req.image_path, db_schedule, creds, status="Pending Approval")
    
    approval_link = f"http://127.0.0.1:8000/approve/{post_id}"
    
    import smtplib
    from email.message import EmailMessage
    from email.utils import make_msgid
    import os
    
    # Configure your Real Email Credentials Here!
    sender_email = os.environ.get("SMTP_EMAIL", "anaslari4work@gmail.com") 
    sender_pass = os.environ.get("SMTP_PASSWORD", "qfbe uikz xhjc pshj")
    
    msg = EmailMessage()
    msg["Subject"] = "Urgent: CEO Approval Required for Marketing Post"
    msg["From"] = sender_email
    msg["To"] = req.ceo_email
    
    approval_link = f"http://127.0.0.1:8000/approve/{post_id}"
    rejection_link = f"http://127.0.0.1:8000/reject/{post_id}"
    
    image_cid = make_msgid()
    
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f9fafb; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: white; border-radius: 12px; padding: 30px; border: 1px solid #e5e7eb;">
          <h2 style="color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 15px; margin-top:0;">Approval Required: {req.platform} Post</h2>
          
          <h3 style="color: #4b5563; margin-bottom: 5px;">Generated Caption:</h3>
          <p style="white-space: pre-wrap; color: #1f2937; background: #f3f4f6; padding: 15px; border-radius: 8px;">{req.caption}</p>
          
          <h3 style="color: #4b5563; margin-bottom: 5px;">Media Preview:</h3>
          <div style="text-align: center; margin: 20px 0;">
            <img src="cid:{image_cid[1:-1]}" style="max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb;" alt="Post Media (Download pending...)" />
          </div>
          
          <div style="margin-top: 30px; text-align: center;">
            <a href="{approval_link}" style="display: inline-block; background-color: #10B981; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; margin-right: 15px;">✓ APPROVE &amp; PUBLISH</a>
            <a href="{rejection_link}" style="display: inline-block; background-color: #ef4444; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">✗ REJECT POST</a>
          </div>
        </div>
      </body>
    </html>
    """
    
    msg.set_content("Please enable HTML viewing format to interact with the content.")
    msg.add_alternative(html_content, subtype='html')
    
    # Securely embed the generated image inside the email payload directly
    try:
        img_full_path = req.image_path.lstrip('/')
        if os.path.exists(img_full_path):
            with open(img_full_path, 'rb') as f:
                img_data = f.read()
            msg.get_payload()[1].add_related(img_data, 'image', 'png', cid=image_cid)
    except Exception as img_err:
        print(f"Warning: Could not embed image natively: -> {img_err}")
    
    email_status = ""
    try:
        if sender_pass == "put_your_gmail_app_password_here":
             print(f"\n[EMAIL MOCK - CREDENTIALS MISSING] Configured Email:\n{msg.get_content()}\n")
             email_status = "Simulated: Please update sender_email and sender_pass in server.py"
        else:
             server = smtplib.SMTP('smtp.gmail.com', 587)
             server.starttls()
             server.login(sender_email, sender_pass)
             server.send_message(msg)
             server.quit()
             email_status = "Successfully sent via Gmail SMTP"
             print(f"-> Real HTML approval email sent to {req.ceo_email}")
    except Exception as e:
        print(f"Error sending real email: {e}")
        email_status = f"SMTP Error failed to send: {e}"
        
    return {"status": "success", "message": f"Approval requested! {email_status}", "email_status": email_status}

@app.get("/approve/{post_id}", response_class=HTMLResponse)
async def approve_post(post_id: int):
    update_post_status(post_id, "Approved")
    return f'''
    <html>
      <body style='font-family: sans-serif; text-align: center; padding: 50px; background-color: #f9fafb;'>
        <h1 style='color: #10B981;'>Post Approved!</h1>
        <p style='color: #4b5563;'>Post ID {post_id} has been securely marked as approved. Our AI engine will publish it directly to the platform.</p>
        <script>setTimeout(()=>window.close(), 3000);</script>
      </body>
    </html>
    '''

@app.get("/reject/{post_id}", response_class=HTMLResponse)
async def reject_post(post_id: int):
    update_post_status(post_id, "Rejected")
    return f'''
    <html>
      <body style='font-family: sans-serif; text-align: center; padding: 50px; background-color: #f9fafb;'>
        <h1 style='color: #ef4444;'>Post Rejected.</h1>
        <p style='color: #4b5563;'>Post ID {post_id} has been cancelled securely. It will not be published.</p>
        <script>setTimeout(()=>window.close(), 3000);</script>
      </body>
    </html>
    '''

class IntelRequest(BaseModel):
    topic: str
    brand_name: str

@app.post("/intel")
async def fetch_intel(req: IntelRequest):
    return get_market_intelligence(req.topic, req.brand_name)


@app.post("/process")
async def process(
    file:           UploadFile = File(...),
    width:          int        = Form(1080),
    height:         int        = Form(1080),
    resize_mode:    str        = Form("cover"),
    logo_variant:   str        = Form("auto"),
    logo_position:  str        = Form("auto"),
    logo_scale:     float      = Form(15.0),
    logo_margin:    int        = Form(20),
    logo_opacity:   float      = Form(0.85),
    logo_blend:     str        = Form("normal"),
    background_color: str      = Form("white"),
    output_format:  str        = Form("png"),
    quality:        int        = Form(95),
):
    # Validate
    if width  < 1 or width  > 8000: raise HTTPException(400, "width must be 1–8000")
    if height < 1 or height > 8000: raise HTTPException(400, "height must be 1–8000")
    if not 0.0 <= logo_opacity <= 1.0: raise HTTPException(400, "opacity must be 0–1")
    if not 1 <= quality <= 100:        raise HTTPException(400, "quality must be 1–100")

    allowed = {"png","jpeg","webp"}
    if output_format not in allowed:
        raise HTTPException(400, f"format must be one of {allowed}")

    # Save upload to temp path
    suffix    = Path(file.filename or "img.jpg").suffix or ".jpg"
    tmp_in    = STATIC / f"_in_{uuid.uuid4().hex}{suffix}"
    tmp_out   = STATIC / f"_out_{uuid.uuid4().hex}.{output_format}"

    try:
        tmp_in.write_bytes(await file.read())

        meta = process_image(
            input_path       = tmp_in,
            output_path      = tmp_out,
            width            = width,
            height           = height,
            resize_mode      = resize_mode,
            logo_variant     = logo_variant,
            logo_position    = logo_position,
            logo_scale       = logo_scale,
            logo_margin      = logo_margin,
            logo_opacity     = logo_opacity,
            logo_blend_mode  = logo_blend,
            background_color = background_color,
            output_format    = output_format,
            quality          = quality,
        )

        # Return image as base64 so the browser can show it inline
        img_bytes = tmp_out.read_bytes()
        mime_map  = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
        b64       = base64.b64encode(img_bytes).decode()

        return JSONResponse({
            "image":    f"data:{mime_map[output_format]};base64,{b64}",
            "filename": tmp_out.name,
            "meta":     meta,
        })

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    finally:
        tmp_in.unlink(missing_ok=True)
        # tmp_out kept until download; client downloads via /download/<name>


@app.get("/download/{filename}")
async def download(filename: str):
    # Sanitise — no path traversal
    name = Path(filename).name
    path = STATIC / name
    if not path.exists() or not name.startswith("_out_"):
        raise HTTPException(404, "File not found")
    from fastapi.responses import FileResponse
    ext = path.suffix.lstrip(".")
    media = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=f"processed.{ext}")
