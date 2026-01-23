from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import httpx
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from rate_limiter import limiter
import resend

load_dotenv()

# Email Configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_API_URL = "https://api.resend.com/emails"
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@email.drawtopia.ai")
FROM_NAME = os.getenv("FROM_NAME", "Drawtopia")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Email service enabled check
EMAIL_ENABLED = bool(RESEND_API_KEY)
FROM_EMAIL_FORMATTED = f"{FROM_NAME} <{FROM_EMAIL}>"
resend.api_key = RESEND_API_KEY
# Templates directory (parent directory of apis folder)
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

logger = logging.getLogger(__name__)

router = APIRouter()

# Helper function to check if email service is enabled
def is_email_enabled() -> bool:
    """Check if email service is enabled"""
    return EMAIL_ENABLED

# Helper function to load email templates
def _load_template(template_name: str) -> str:
    """Load HTML template from file"""
    template_path = TEMPLATES_DIR / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}")
        raise

# Helper function to send email via Resend API
async def _send_email(
    to_email: str,
    template_id: str,
    template_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send an email using Resend API
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        template_id: Resend template ID
        template_data: Template data
    
    Returns:
        Dict with 'success' boolean and 'id' or 'error'
    """
    if not EMAIL_ENABLED:
        logger.warning("Email service not enabled, skipping email send")
        return {"success": False, "error": "Email service not configured"}
    
    if not RESEND_API_KEY:
        logger.error("Resend API key is missing")
        return {"success": False, "error": "Resend API key not configured"}
    
    try:
        payload = {
            "from": FROM_EMAIL_FORMATTED,
            "to": to_email,
            "template": {
                "id": template_id,
                "variables": template_data
            }
        }
        
        response = resend.Emails.send(payload)

        logger.info(f"✅ Email sent successfully to {to_email} (ID: {response})")
        
        if response.status_code in [200, 201]:
            try:
                response_data = response.json()
                email_id = response_data.get("id", f"resend_{datetime.now().timestamp()}")
                logger.info(f"✅ Email sent successfully to {to_email} (ID: {email_id})")
                return {"success": True, "id": email_id}
            except Exception:
                email_id = f"resend_{datetime.now().timestamp()}"
                logger.info(f"✅ Email sent successfully to {to_email} (ID: {email_id})")
                return {"success": True, "id": email_id}
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("message", response.text)
            except:
                error_message = response.text
            logger.error(f"❌ Resend API error sending email to {to_email}: {response.status_code} - {error_message}")
            return {"success": False, "error": f"Resend API error ({response.status_code}): {error_message}"}
            
    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP error sending email to {to_email}: {e}")
        return {"success": False, "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Failed to send email to {to_email}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/emails/welcome")
@limiter.limit("10/minute")
async def send_welcome_email_endpoint(request: Request):
    """Send welcome email to new user"""
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        name = customer_name or "there"
        subject = "🎉 Welcome to Drawtopia - Let's Create Something Amazing!"
        
        # Send welcome email
        result = await _send_email(to_email, template_id="welcome-notification", template_data={"name": name, "frontend_url": FRONTEND_URL, "current_year": datetime.now().year})
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send welcome email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send welcome email: {error_msg}"
            )
        
        logger.info(f"✅ Welcome email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Welcome email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending welcome email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/parental-consent")
@limiter.limit("10/minute")
async def send_parental_consent_email_endpoint(request: Request):
    """Send parental consent verification email"""
    # Email sending logic moved directly into API send_parental_consent
    
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate consent link (expires in 48 hours)
        consent_token = str(uuid.uuid4())
        consent_link = f"{FRONTEND_URL}/consent/verify?token={consent_token}"
        
        # Send the email
        result = await _send_email(parent_email, template_id="parental-consent-request", template_data={"parent_name": parent_name, "child_name": child_name, "consent_link": consent_link, "current_year": str(datetime.now().year)})
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send parental consent email to {parent_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send parental consent email: {error_msg}"
            )
        
        logger.info(f"✅ Parental consent email sent to {parent_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Parental consent email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending parental consent email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/gift-notification")
@limiter.limit("10/minute")
async def send_gift_notification_email_endpoint(request: Request):
    """Send gift notification email"""
    # Email sending logic moved directly into API send_gift_notification
    
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        delivery_method = body.get("delivery_method", "immediate_email")
        
        delivery_info = ""
        if delivery_method == 'immediate_email':
            delivery_info = "Your story will be ready to read very soon! We'll send you another email when it's complete, usually within 1-2 hours."
        elif delivery_method == 'scheduled_delivery':
            delivery_info = "Your story will be delivered soon! Keep an eye on your email."
        else:
            delivery_info = f"{giver_name} is asking a grown-up in your life to help create your story. Ask them to check their email for the creation link."
        
        # Send the email
        result = await _send_email(recipient_email, template_id="gift-notification", template_data={"recipient_name": recipient_name, "giver_name": giver_name, "occasion": occasion, "gift_message": gift_message, "delivery_info": delivery_info, "current_year": str(datetime.now().year)})
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send gift notification email to {recipient_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send gift notification email: {error_msg}"
            )
        
        logger.info(f"✅ Gift notification email sent to {recipient_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Gift notification email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending gift notification email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/payment-success")
@limiter.limit("10/minute")
async def send_payment_success_email_endpoint(request: Request):
    """Send payment success confirmation email"""
    # Email sending logic moved directly into API send_payment_success
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        plan_type = body.get("plan_type", "monthly")
        amount = body.get("amount")
        next_billing_date = body.get("next_billing_date")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        amount_display = amount or ("$9.99" if plan_type == "monthly" else "$99.99")
        
        next_billing_row = f'<tr><td style="color: #718096; padding: 8px 0;">Next billing</td><td style="color: #1a1a2e; text-align: right;">{next_billing_date}</td></tr>' if next_billing_date else ''
          
        # Send payment success email
        result = await _send_email(
            to_email, 
            template_id="premium-subscription-confirmation", 
            template_data={
                "name": name, 
                "plan_display": plan_display, 
                "amount_display": amount_display, 
                "next_billing_row": next_billing_row, 
                "frontend_url": FRONTEND_URL, 
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send payment success email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send payment success email: {error_msg}"
            )
        
        logger.info(f"✅ Payment success email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Payment success email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending payment success email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/payment-failed")
@limiter.limit("10/minute")
async def send_payment_failed_email_endpoint(request: Request):
    """Send payment failure notification email"""
    # Email sending logic moved directly into API send_payment_failed
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        plan_type = body.get("plan_type", "monthly")
        amount = body.get("amount")
        retry_url = body.get("retry_url")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        amount_display = amount or ("$9.99" if plan_type == "monthly" else "$99.99")
        update_url = retry_url or f"{FRONTEND_URL}/account"

        # Send payment failed email
        result = await _send_email(
            to_email, template_id="payment-failed-notice", 
            template_data={
                "name": name, 
                "plan_display": plan_display,
                "amount_display": amount_display,
                "update_url": update_url,
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send payment failed email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send payment failed email: {error_msg}"
            )
        
        logger.info(f"✅ Payment failed email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Payment failed email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending payment failed email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/subscription-cancelled")
@limiter.limit("10/minute")
async def send_subscription_cancelled_email_endpoint(request: Request):
    """Send subscription cancellation confirmation email"""
    # Email sending logic moved directly into API send_subscription_cancelled
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        plan_type = body.get("plan_type", "monthly")
        access_until = body.get("access_until")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        access_info = f"Your premium access will remain active until <strong>{access_until}</strong>." if access_until else "Your premium access has been deactivated."
        
        # Send subscription cancelled email
        result = await _send_email(
            to_email, 
            template_id="subscription-cancellation-notice", 
            template_data={
                "name": name,
                "plan_display": plan_display,
                "access_info": access_info,
                "FRONTEND_URL": FRONTEND_URL,
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send subscription cancelled email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send subscription cancelled email: {error_msg}"
            )
        
        logger.info(f"✅ Subscription cancelled email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Subscription cancelled email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending subscription cancelled email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/subscription-renewal-reminder")
@limiter.limit("10/minute")
async def send_subscription_renewal_reminder_email_endpoint(request: Request):
    """Send subscription renewal reminder email"""
    # Email sending logic moved directly into API send_subscription_renewal_reminder
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        plan_type = body.get("plan_type", "monthly")
        renewal_amount = body.get("renewal_amount")
        renewal_date = body.get("renewal_date")
        manage_link = body.get("manage_link")
        cancel_link = body.get("cancel_link")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Parse renewal_date if it's a string
        renewal_date_obj = None
        if renewal_date:
            if isinstance(renewal_date, str):
                try:
                    renewal_date_obj = datetime.fromisoformat(renewal_date.replace('Z', '+00:00'))
                except:
                    renewal_date_obj = datetime.fromisoformat(renewal_date)
            elif isinstance(renewal_date, datetime):
                renewal_date_obj = renewal_date
        
        # Generate email content
        renewal_date_final = renewal_date_obj or datetime.utcnow()
        
        # Send subscription renewal reminder email
        result = await _send_email(
            to_email, 
            template_id="",
            template_data={
                "customer_name": customer_name,
                "plan_type": plan_type,
                "renewal_date": renewal_date,
                "renewal_amount": renewal_amount,
                "renewal_date_final": renewal_date_final,
                "manage_link": manage_link,
                "cancel_link": cancel_link,
                "FRONTEND_URL": FRONTEND_URL,
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send subscription renewal reminder email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send subscription renewal reminder email: {error_msg}"
            )
        
        logger.info(f"✅ Subscription renewal reminder email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Subscription renewal reminder email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending subscription renewal reminder email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/receipt")
@limiter.limit("10/minute")
async def send_receipt_email_endpoint(request: Request):
    """Send receipt email for purchase"""
    # Email sending logic moved directly into API send_receipt
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        customer_name = body.get("customer_name")
        transaction_id = body.get("transaction_id")
        items = body.get("items", [])
        subtotal = body.get("subtotal")
        tax = body.get("tax", 0)
        total = body.get("total")
        transaction_date = body.get("transaction_date")
        
        if not to_email:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: to_email"
            )
        
        if not transaction_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: transaction_id"
            )
        
        # Validate email format
        to_email = to_email.strip().lower()
        if "@" not in to_email or "." not in to_email.split("@")[1]:
            raise HTTPException(
                status_code=400,
                detail="Invalid email address format"
            )
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Parse transaction_date if it's a string
        transaction_date_obj = None
        if transaction_date:
            if isinstance(transaction_date, str):
                try:
                    transaction_date_obj = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
                except:
                    transaction_date_obj = datetime.fromisoformat(transaction_date)
            elif isinstance(transaction_date, datetime):
                transaction_date_obj = transaction_date
        else:
            transaction_date_obj = datetime.utcnow()
        
        # Generate email content
        customer_name_final = customer_name or "Customer"
        transaction_date_final = transaction_date_obj
        subtotal_final = subtotal or 0
        total_final = total or subtotal_final or 0
        
        items_html = ""
        items_text = ""
        for item in items:
            items_html += f"""
                <tr>
                    <td style="color: #4a5568; padding: 8px 0;">{item['name']}</td>
                    <td style="color: #1a1a2e; text-align: right;">${item['amount']:.2f}</td>
                </tr>"""
            items_text += f"- {item['name']}: ${item['amount']:.2f}\n"
        
        # Send receipt email
        result = await _send_email(
            to_email,
            template_id="order-receipt-confirmation",
            template_data={
                "customer_name": customer_name,
                "transaction_id": transaction_id,
                "transaction_date": transaction_date_final.strftime('%B %d, %Y'),
                "items": items,
                "subtotal": subtotal_final,
                "tax": tax,
                "total": str(total_final),
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send receipt email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send receipt email: {error_msg}"
            )
        
        logger.info(f"✅ Receipt email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Receipt email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending receipt email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/book-completion")
@limiter.limit("10/minute")
async def send_book_completion_email_endpoint(request: Request):
    """Send book completion notification email"""
    # Email sending logic moved directly into API send_book_completion
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        parent_name = body.get("parent_name")
        child_name = body.get("child_name")
        character_name = body.get("character_name")
        character_type = body.get("character_type")
        book_title = body.get("book_title")
        special_ability = body.get("special_ability")
        book_format = body.get("book_format", "story_adventure")
        preview_link = body.get("preview_link")
        download_link = body.get("download_link")
        story_world = body.get("story_world")
        adventure_type = body.get("adventure_type")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        if book_format == 'interactive_search':
            format_description = "an 8-scene Where's Waldo-style adventure in the Enchanted Forest"
            format_details = f"""
                <li>Format: Interactive Search (Where's Waldo style)</li>
                <li>Scenes: 4 magical locations</li>
                <li>Character: {character_name} ({character_type})</li>
                <li>Special Ability: {special_ability}</li>
                <li>Reading Time: ~15-20 minutes</li>"""
        else:
            format_description = f"a 5-page adventure where {character_name} the {character_type} uses their special power"
            format_details = f"""
                <li>Format: Story Adventure (5-page narrative)</li>
                <li>Pages: 5 beautifully illustrated pages</li>
                <li>Character: {character_name} ({character_type})</li>
                <li>Special Ability: {special_ability}</li>
                <li>World: {story_world or 'Magical World'}</li>
                <li>Adventure Type: {adventure_type or 'Epic Quest'}</li>
                <li>Reading Time: ~10 minutes</li>"""
        
        # Send book completion email
        result = await _send_email(
            to_email,
            template_id="story-creation-notification",
            template_data={
                "child_name": child_name,
                "parent_name": parent_name,
                "book_title": book_title,
                "format_description": format_description,
                "format_details": format_details,
                "preview_link": preview_link,
                "download_link": download_link,
                "current_year": str(datetime.now().year)
            }
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send book completion email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send book completion email: {error_msg}"
            )
        
        logger.info(f"✅ Book completion email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Book completion email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending book completion email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/gift-delivery")
@limiter.limit("10/minute")
async def send_gift_delivery_email_endpoint(request: Request):
    """Send gift delivery email"""
    # Email sending logic moved directly into API send_gift_delivery
    
    try:
        body = await request.json()
        
        to_email = body.get("to_email")
        recipient_name = body.get("recipient_name")
        giver_name = body.get("giver_name")
        character_name = body.get("character_name")
        character_type = body.get("character_type")
        book_title = body.get("book_title")
        special_ability = body.get("special_ability")
        gift_message = body.get("gift_message", "")
        story_link = body.get("story_link")
        download_link = body.get("download_link")
        book_format = body.get("book_format", "story_adventure")
        
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
        
        if not is_email_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Generate email content
        subject = f"Your gift has arrived! Open '{book_title}' now 🎁📖"
        format_info = "4-scene Where's Waldo-style adventure" if book_format == 'interactive_search' else "5-page magical adventure"
        
        html_content = _load_template("gift_delivery.html").format(
            recipient_name=recipient_name,
            giver_name=giver_name,
            character_name=character_name,
            character_type=character_type,
            book_title=book_title,
            special_ability=special_ability,
            format_info=format_info,
            story_link=story_link,
            download_link=download_link,
            gift_message=gift_message,
            current_year=datetime.now().year
        )
        
        text_content = f"""
Your gift has arrived! Open '{book_title}' now 🎁📖

Hi {recipient_name},

Your gift is here! 🎉✨

{giver_name} created a special personalized storybook just for you called:

📖 "{book_title}"

Featuring {character_name}, a {character_type} with the special ability to {special_ability}!

It's a {format_info} where you'll have an amazing adventure!

🎬 Open Your Gift: {story_link}
📥 Download PDF: {download_link}

From {giver_name}: "{gift_message}"

This is your special copy! You can read it anytime, share it with friends, or download it to keep forever. 💫

Happy reading!

Love your gift? Send a thank you to {giver_name}!

© {datetime.now().year} Drawtopia
"""
        
        # Send gift delivery email
        result = await _send_email(to_email, subject, html_content, text_content)
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            logger.error(f"❌ Failed to send gift delivery email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send gift delivery email: {error_msg}"
            )
        
        logger.info(f"✅ Gift delivery email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Gift delivery email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending gift delivery email: {e}")
        raise HTTPException(status_code=500, detail=str(e))
