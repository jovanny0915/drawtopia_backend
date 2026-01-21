from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import os
import httpx
import uuid
from dotenv import load_dotenv
from rate_limiter import limiter

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_API_URL = "https://api.resend.com/emails"
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

router = APIRouter()

class EmailSchema(BaseModel):
    to: str
    subject: str
    body: str

@router.post("/send-email")
async def send_email(email: EmailSchema):
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "from": "noreply@email.drawtopia.ai",  # Use your verified domain email
        "to": email.to,
        "subject": email.subject,
        "html": email.body
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(RESEND_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    return {"message": "Email sent successfully"}


@router.post("/emails/welcome")
@limiter.limit("10/minute")
async def send_welcome_email_endpoint(request: Request):
    """Send welcome email to new user"""
    import main  # Import here to avoid circular import
    from email_service import send_welcome
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        
        if not to_email:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: to_email"
            )
        
        # Validate email format
        to_email = to_email.strip().lower()
        if "@" not in to_email or "." not in to_email.split("@")[1]:
            raise HTTPException(
                status_code=400,
                detail="Invalid email address format"
            )
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send welcome email and check result
        result = await send_welcome(
            to_email=to_email,
            customer_name=customer_name
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send welcome email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send welcome email: {error_msg}"
            )
        
        main.logger.info(f"✅ Welcome email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Welcome email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending welcome email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/parental-consent")
@limiter.limit("10/minute")
async def send_parental_consent_email_endpoint(request: Request):
    """Send parental consent verification email"""
    import main  # Import here to avoid circular import
    from email_service import send_parental_consent
    
    try:
        body = await request.json()
        
        parent_email = body.get("parent_email")
        parent_name = body.get("parent_name")
        child_name = body.get("child_name")
        
        if not all([parent_email, parent_name, child_name]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: parent_email, parent_name, child_name"
            )
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate consent link (expires in 48 hours)
        consent_token = str(uuid.uuid4())
        consent_link = f"{FRONTEND_URL}/consent/verify?token={consent_token}"
        
        # Send the email
        await send_parental_consent(
            to_email=parent_email,
            parent_name=parent_name,
            child_name=child_name,
            consent_link=consent_link
        )
        
        main.logger.info(f"✅ Parental consent email sent to {parent_email}")
        return {"success": True, "message": "Parental consent email sent"}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending parental consent email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/gift-notification")
@limiter.limit("10/minute")
async def send_gift_notification_email_endpoint(request: Request):
    """Send gift notification email"""
    import main  # Import here to avoid circular import
    from email_service import send_gift_notification
    
    try:
        body = await request.json()
        
        recipient_email = body.get("recipient_email")
        recipient_name = body.get("recipient_name")
        giver_name = body.get("giver_name")
        occasion = body.get("occasion")
        gift_message = body.get("gift_message", "")
        
        if not all([recipient_email, recipient_name, giver_name, occasion]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: recipient_email, recipient_name, giver_name, occasion"
            )
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send the email (Note: scheduled sending not supported without queue)
        await send_gift_notification(
            to_email=recipient_email,
            recipient_name=recipient_name,
            giver_name=giver_name,
            occasion=occasion,
            gift_message=gift_message
        )
        
        main.logger.info(f"✅ Gift notification email sent to {recipient_email}")
        return {"success": True, "message": "Gift notification email sent"}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending gift notification email: {e}")
        raise HTTPException(status_code=500, detail=str(e))
