from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
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


@router.post("/emails/payment-success")
@limiter.limit("10/minute")
async def send_payment_success_email_endpoint(request: Request):
    """Send payment success confirmation email"""
    import main  # Import here to avoid circular import
    from email_service import send_payment_success
    
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
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send payment success email
        result = await send_payment_success(
            to_email=to_email,
            customer_name=customer_name,
            plan_type=plan_type,
            amount=amount,
            next_billing_date=next_billing_date
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send payment success email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send payment success email: {error_msg}"
            )
        
        main.logger.info(f"✅ Payment success email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Payment success email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending payment success email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/payment-failed")
@limiter.limit("10/minute")
async def send_payment_failed_email_endpoint(request: Request):
    """Send payment failure notification email"""
    import main  # Import here to avoid circular import
    from email_service import send_payment_failed
    
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
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send payment failed email
        result = await send_payment_failed(
            to_email=to_email,
            customer_name=customer_name,
            plan_type=plan_type,
            amount=amount,
            retry_url=retry_url
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send payment failed email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send payment failed email: {error_msg}"
            )
        
        main.logger.info(f"✅ Payment failed email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Payment failed email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending payment failed email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/subscription-cancelled")
@limiter.limit("10/minute")
async def send_subscription_cancelled_email_endpoint(request: Request):
    """Send subscription cancellation confirmation email"""
    import main  # Import here to avoid circular import
    from email_service import send_subscription_cancelled
    
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
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send subscription cancelled email
        result = await send_subscription_cancelled(
            to_email=to_email,
            customer_name=customer_name,
            plan_type=plan_type,
            access_until=access_until
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send subscription cancelled email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send subscription cancelled email: {error_msg}"
            )
        
        main.logger.info(f"✅ Subscription cancelled email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Subscription cancelled email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending subscription cancelled email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/subscription-renewal-reminder")
@limiter.limit("10/minute")
async def send_subscription_renewal_reminder_email_endpoint(request: Request):
    """Send subscription renewal reminder email"""
    import main  # Import here to avoid circular import
    from email_service import send_subscription_renewal_reminder
    
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
        
        if not main.email_service.is_enabled():
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
        
        # Send subscription renewal reminder email
        result = await send_subscription_renewal_reminder(
            to_email=to_email,
            customer_name=customer_name,
            plan_type=plan_type,
            renewal_amount=renewal_amount,
            renewal_date=renewal_date_obj or datetime.utcnow(),
            manage_link=manage_link or f"{FRONTEND_URL}/account",
            cancel_link=cancel_link or f"{FRONTEND_URL}/account"
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send subscription renewal reminder email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send subscription renewal reminder email: {error_msg}"
            )
        
        main.logger.info(f"✅ Subscription renewal reminder email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Subscription renewal reminder email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending subscription renewal reminder email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/receipt")
@limiter.limit("10/minute")
async def send_receipt_email_endpoint(request: Request):
    """Send receipt email for purchase"""
    import main  # Import here to avoid circular import
    from email_service import send_receipt
    
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
        
        if not main.email_service.is_enabled():
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
        
        # Send receipt email
        result = await send_receipt(
            to_email=to_email,
            customer_name=customer_name or "Customer",
            transaction_id=transaction_id,
            items=items,
            subtotal=subtotal or 0,
            tax=tax,
            total=total or subtotal or 0,
            transaction_date=transaction_date_obj
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send receipt email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send receipt email: {error_msg}"
            )
        
        main.logger.info(f"✅ Receipt email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Receipt email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending receipt email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/book-completion")
@limiter.limit("10/minute")
async def send_book_completion_email_endpoint(request: Request):
    """Send book completion notification email"""
    import main  # Import here to avoid circular import
    from email_service import send_book_completion
    
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
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send book completion email
        result = await send_book_completion(
            to_email=to_email,
            parent_name=parent_name,
            child_name=child_name,
            character_name=character_name,
            character_type=character_type,
            book_title=book_title,
            special_ability=special_ability,
            book_format=book_format,
            preview_link=preview_link,
            download_link=download_link,
            story_world=story_world,
            adventure_type=adventure_type
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send book completion email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send book completion email: {error_msg}"
            )
        
        main.logger.info(f"✅ Book completion email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Book completion email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending book completion email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/gift-delivery")
@limiter.limit("10/minute")
async def send_gift_delivery_email_endpoint(request: Request):
    """Send gift delivery email"""
    import main  # Import here to avoid circular import
    from email_service import send_gift_delivery
    
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
        
        if not main.email_service.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Email service not available"
            )
        
        # Send gift delivery email
        result = await send_gift_delivery(
            to_email=to_email,
            recipient_name=recipient_name,
            giver_name=giver_name,
            character_name=character_name,
            character_type=character_type,
            book_title=book_title,
            special_ability=special_ability,
            gift_message=gift_message,
            story_link=story_link,
            download_link=download_link,
            book_format=book_format
        )
        
        # Check if email was sent successfully
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error sending email")
            main.logger.error(f"❌ Failed to send gift delivery email to {to_email}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send gift delivery email: {error_msg}"
            )
        
        main.logger.info(f"✅ Gift delivery email sent to {to_email} (ID: {result.get('id', 'N/A')})")
        return {"success": True, "message": "Gift delivery email sent", "email_id": result.get("id")}
            
    except HTTPException:
        raise
    except Exception as e:
        main.logger.error(f"Error sending gift delivery email: {e}")
        raise HTTPException(status_code=500, detail=str(e))
