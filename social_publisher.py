import requests
import json
import os

class SocialPublisher:
    """Handles direct posting to social media platforms via their APIs."""

    @staticmethod
    def post_to_linkedin(text: str, image_path: str, access_token: str, author_id: str) -> dict:
        """
        Publishes a post to LinkedIn. Requires an access token and an author urn (e.g., urn:li:person:12345).
        Note: For full implementation, image_path requires an upload step first (Asset API).
        This is structurally ready for the final API hooks.
        """
        print(f"Executing LinkedIn Post for {author_id}...")
        if not access_token or not author_id:
            raise ValueError("Missing LinkedIn access_token or author_id")
            
        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json"
        }
        
        payload = {
            "author": author_id,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE" # Update to IMAGE once asset upload is integrated
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code in (200, 201):
            return {"status": "success", "platform": "LinkedIn", "data": response.json()}
        else:
            return {"status": "error", "platform": "LinkedIn", "message": response.text}

    @staticmethod
    def post_to_facebook(text: str, image_path: str, access_token: str, page_id: str) -> dict:
        """Publishes an image directly to a Facebook Page."""
        print(f"Executing Facebook Post for Page {page_id}...")
        if not access_token or not page_id:
            raise ValueError("Missing Facebook access_token or page_id")
            
        url = f"https://graph.facebook.com/v19.0/{page_id}/photos"
        payload = {
            "message": text,
            "access_token": access_token
        }
        
        # In a real environment, you attach the file as multipart/form-data
        try:
             with open(image_path, "rb") as image_file:
                 files = {"source": image_file}
                 response = requests.post(url, data=payload, files=files)
        except Exception as e:
             return {"status": "error", "message": f"File read error: {e}"}
             
        if response.status_code == 200:
            return {"status": "success", "platform": "Facebook", "data": response.json()}
        else:
            return {"status": "error", "platform": "Facebook", "message": response.text}

    @staticmethod
    def post_to_instagram(text: str, image_url: str, access_token: str, ig_user_id: str) -> dict:
        """
        Publishes to Instagram using the Facebook Graph API.
        Requires public image_url, access_token, and Instagram Business Account ID.
        """
        print(f"Executing Instagram Post for User {ig_user_id}...")
        if not access_token or not ig_user_id:
            raise ValueError("Missing Instagram access_token or ig_user_id")

        # Step 1: Create Container
        container_url = f"https://graph.facebook.com/v19.0/{ig_user_id}/media"
        c_payload = {
            "image_url": image_url, 
            "caption": text,
            "access_token": access_token
        }
        c_res = requests.post(container_url, data=c_payload)
        c_data = c_res.json()
        
        if 'id' not in c_data:
            return {"status": "error", "step": "container_creation", "message": c_data}
            
        container_id = c_data['id']
        
        # Step 2: Publish Container
        publish_url = f"https://graph.facebook.com/v19.0/{ig_user_id}/media_publish"
        p_payload = {
            "creation_id": container_id,
            "access_token": access_token
        }
        p_res = requests.post(publish_url, data=p_payload)
        
        if p_res.status_code == 200:
            return {"status": "success", "platform": "Instagram", "data": p_res.json()}
        else:
            return {"status": "error", "platform": "Instagram", "message": p_res.text}
