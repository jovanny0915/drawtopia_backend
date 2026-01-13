"""
Email Service for Drawtopia
Uses Gmail SMTP for sending transactional emails (no domain verification needed)

Supported email types:
- Welcome email on registration
- Parental consent verification
- Book completion notification
- Payment success confirmation
- Payment failure notification
- Subscription cancellation confirmation
- Subscription activation confirmation
- Subscription renewal reminders
- Gift notification emails
- Gift delivery emails

Setup:
1. Enable 2-Step Verification in your Google Account
2. Go to Google Account → Security → App passwords
3. Create an App Password for "Mail"
4. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to .env
"""

import os
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Gmail SMTP Configuration
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_SMTP_SERVER = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587  # TLS port

# General Configuration
FROM_EMAIL = os.getenv("FROM_EMAIL", GMAIL_ADDRESS)
FROM_NAME = os.getenv("FROM_NAME", "Drawtopia")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Initialize email service
if GMAIL_ADDRESS and GMAIL_APP_PASSWORD:
    logger.info("✅ Email service (Gmail SMTP) initialized successfully")
    EMAIL_ENABLED = True
else:
    EMAIL_ENABLED = False
    logger.warning("⚠️ Gmail SMTP not configured. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env")


class EmailService:
    """Email service for sending transactional emails via Gmail SMTP"""
    
    def __init__(self):
        self.enabled = EMAIL_ENABLED
        self.from_email = f"{FROM_NAME} <{FROM_EMAIL}>"
    
    def is_enabled(self) -> bool:
        """Check if email service is enabled"""
        return self.enabled
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an email using Gmail SMTP
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML body of the email
            text_content: Plain text body (optional, for email clients that don't support HTML)
        
        Returns:
            Dict with 'success' boolean and 'id' or 'error'
        """
        if not self.enabled:
            logger.warning("Email service not enabled, skipping email send")
            return {"success": False, "error": "Email service not configured"}
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.from_email
            message["To"] = to_email
            
            # Add plain text part (for email clients that don't support HTML)
            if text_content:
                part1 = MIMEText(text_content, "plain")
                message.attach(part1)
            
            # Add HTML part
            part2 = MIMEText(html_content, "html")
            message.attach(part2)
            
            # Create secure SSL/TLS context
            context = ssl.create_default_context()
            
            # Connect to Gmail SMTP server
            with smtplib.SMTP(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
                server.sendmail(GMAIL_ADDRESS, to_email, message.as_string())
            
            logger.info(f"✅ Email sent successfully to {to_email}")
            return {"success": True, "id": f"gmail_{datetime.now().timestamp()}"}
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ Gmail authentication failed: {e}")
            return {"success": False, "error": "Gmail authentication failed. Check your App Password."}
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error sending email to {to_email}: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"❌ Failed to send email to {to_email}: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== EMAIL TEMPLATES ====================
    
    async def send_payment_success_email(
        self,
        to_email: str,
        customer_name: Optional[str] = None,
        plan_type: str = "monthly",
        amount: Optional[str] = None,
        next_billing_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send payment success confirmation email
        
        Args:
            to_email: Customer email address
            customer_name: Customer name (optional)
            plan_type: Subscription plan type (monthly/yearly)
            amount: Payment amount (e.g., "$9.99")
            next_billing_date: Next billing date
        """
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        amount_display = amount or ("$9.99" if plan_type == "monthly" else "$99.99")
        
        subject = "🎉 Payment Successful - Welcome to Drawtopia Premium!"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Creative AI Companion</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: #10b981; width: 60px; height: 60px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 30px;">✓</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Payment Successful!</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Thank you for subscribing to Drawtopia Premium! Your payment has been processed successfully.
            </p>
            
            <!-- Payment Details Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">Payment Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Plan</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{plan_display}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Amount</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{amount_display}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Status</td>
                        <td style="color: #10b981; text-align: right; font-weight: 600;">✓ Paid</td>
                    </tr>
                    {f'<tr><td style="color: #718096; padding: 8px 0;">Next billing</td><td style="color: #1a1a2e; text-align: right;">{next_billing_date}</td></tr>' if next_billing_date else ''}
                </table>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Your premium features are now active! You have unlimited access to:
            </p>
            
            <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px;">
                <li>Unlimited AI image generations</li>
                <li>Priority processing</li>
                <li>Advanced story creation tools</li>
                <li>Premium templates and styles</li>
            </ul>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{FRONTEND_URL}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    Start Creating
                </a>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Questions? Just reply to this email and we'll help you out!
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
            <p style="margin: 8px 0 0 0;">
                <a href="{FRONTEND_URL}/account" style="color: #667eea; text-decoration: none;">Manage Subscription</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Payment Successful - Welcome to Drawtopia Premium!

Hi {name},

Thank you for subscribing to Drawtopia Premium! Your payment has been processed successfully.

Payment Details:
- Plan: {plan_display}
- Amount: {amount_display}
- Status: Paid
{f'- Next billing: {next_billing_date}' if next_billing_date else ''}

Your premium features are now active! You have unlimited access to:
• Unlimited AI image generations
• Priority processing
• Advanced story creation tools
• Premium templates and styles

Start creating: {FRONTEND_URL}

Questions? Just reply to this email!

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_payment_failed_email(
        self,
        to_email: str,
        customer_name: Optional[str] = None,
        plan_type: str = "monthly",
        amount: Optional[str] = None,
        retry_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send payment failure notification email
        
        Args:
            to_email: Customer email address
            customer_name: Customer name (optional)
            plan_type: Subscription plan type (monthly/yearly)
            amount: Payment amount that failed
            retry_url: URL to retry payment
        """
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        amount_display = amount or ("$9.99" if plan_type == "monthly" else "$99.99")
        update_url = retry_url or f"{FRONTEND_URL}/account"
        
        subject = "⚠️ Payment Failed - Action Required"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Creative AI Companion</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: #ef4444; width: 60px; height: 60px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 30px;">!</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Payment Failed</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                We were unable to process your payment for your Drawtopia subscription. This could be due to:
            </p>
            
            <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin-bottom: 20px;">
                <li>Insufficient funds</li>
                <li>Expired card</li>
                <li>Card declined by your bank</li>
            </ul>
            
            <!-- Payment Details Box -->
            <div style="background: #fef2f2; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #ef4444;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">Payment Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Plan</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{plan_display}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Amount</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{amount_display}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Status</td>
                        <td style="color: #ef4444; text-align: right; font-weight: 600;">✗ Failed</td>
                    </tr>
                </table>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                <strong>To keep your premium access</strong>, please update your payment method within the next 7 days.
            </p>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{update_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    Update Payment Method
                </a>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Need help? Reply to this email and we'll assist you right away.
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Payment Failed - Action Required

Hi {name},

We were unable to process your payment for your Drawtopia subscription.

This could be due to:
• Insufficient funds
• Expired card
• Card declined by your bank

Payment Details:
- Plan: {plan_display}
- Amount: {amount_display}
- Status: Failed

To keep your premium access, please update your payment method within the next 7 days.

Update payment: {update_url}

Need help? Reply to this email!

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_subscription_cancelled_email(
        self,
        to_email: str,
        customer_name: Optional[str] = None,
        plan_type: str = "monthly",
        access_until: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send subscription cancellation confirmation email
        
        Args:
            to_email: Customer email address
            customer_name: Customer name (optional)
            plan_type: Subscription plan type that was cancelled
            access_until: Date until which access remains
        """
        name = customer_name or "there"
        plan_display = "Monthly" if plan_type == "monthly" else "Yearly"
        access_info = f"Your premium access will remain active until <strong>{access_until}</strong>." if access_until else "Your premium access has been deactivated."
        
        subject = "Your Drawtopia Subscription Has Been Cancelled"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Creative AI Companion</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Subscription Cancelled</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                We're sorry to see you go! Your <strong>{plan_display}</strong> subscription to Drawtopia has been cancelled.
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                {access_info}
            </p>
            
            <!-- Info Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #718096;">
                <h3 style="color: #1a1a2e; margin: 0 0 12px 0; font-size: 16px;">What happens next?</h3>
                <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li>You can continue using free features</li>
                    <li>Your created content remains accessible</li>
                    <li>You can resubscribe anytime</li>
                </ul>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                We'd love to have you back! If you change your mind, you can resubscribe at any time.
            </p>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{FRONTEND_URL}/pricing" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    Resubscribe
                </a>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Was this a mistake? Reply to this email and we'll help sort it out.
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
            <p style="margin: 8px 0 0 0;">
                We hope to see you again soon! 💜
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Your Drawtopia Subscription Has Been Cancelled

Hi {name},

We're sorry to see you go! Your {plan_display} subscription to Drawtopia has been cancelled.

{access_info.replace('<strong>', '').replace('</strong>', '')}

What happens next?
• You can continue using free features
• Your created content remains accessible
• You can resubscribe anytime

We'd love to have you back! If you change your mind, resubscribe at:
{FRONTEND_URL}/pricing

Was this a mistake? Reply to this email and we'll help!

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_subscription_activated_email(
        self,
        to_email: str,
        customer_name: Optional[str] = None,
        plan_type: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Send subscription activation confirmation email (for new subscriptions)
        
        Args:
            to_email: Customer email address
            customer_name: Customer name (optional)
            plan_type: Subscription plan type
        """
        # This uses the same template as payment success for new subscriptions
        return await self.send_payment_success_email(
            to_email=to_email,
            customer_name=customer_name,
            plan_type=plan_type
        )
    
    async def send_parental_consent_email(
        self,
        to_email: str,
        parent_name: str,
        child_name: str,
        consent_link: str
    ) -> Dict[str, Any]:
        """
        Send parental consent verification email (COPPA compliance)
        
        Args:
            to_email: Parent email address
            parent_name: Parent's name
            child_name: Child's name
            consent_link: Link to consent verification (48-hour expiration)
        """
        subject = f"Verify your account on Drawtopia — Help {child_name} create magical stories"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Where Drawings Come to Life</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0;">Parental Consent Required</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {parent_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Welcome to Drawtopia! 🎨✨
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                {child_name}'s caregiver has started setting up an account on Drawtopia, 
                a platform that transforms children's drawings into personalized storybooks.
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                To complete the setup, we need you to verify that you consent to collect 
                {child_name}'s information. This is required by law (COPPA compliance) and 
                helps us keep their data safe.
            </p>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{consent_link}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    Verify Consent
                </a>
            </div>
            
            <p style="color: #718096; font-size: 14px; margin-bottom: 20px;">
                Or copy this link: <a href="{consent_link}" style="color: #667eea; word-break: break-all;">{consent_link}</a>
            </p>
            
            <div style="background: #fef3c7; border-radius: 12px; padding: 16px; margin: 24px 0; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; margin: 0; font-size: 14px;">
                    ⏰ This link expires in 48 hours
                </p>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Questions? Reply to this email or contact hello@drawtopia.ai
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Verify your account on Drawtopia — Help {child_name} create magical stories

Hi {parent_name},

Welcome to Drawtopia! 🎨✨

{child_name}'s caregiver has started setting up an account on Drawtopia, a platform that transforms children's drawings into personalized storybooks.

To complete the setup, we need you to verify that you consent to collect {child_name}'s information. This is required by law (COPPA compliance) and helps us keep their data safe.

Verify Consent: {consent_link}

⏰ This link expires in 48 hours

Questions? Reply to this email or contact hello@drawtopia.ai

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_welcome_email(
        self,
        to_email: str,
        customer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send welcome email on successful registration/first login
        
        Args:
            to_email: User email address
            customer_name: User name (optional)
        """
        name = customer_name or "there"
        
        subject = "🎉 Welcome to Drawtopia - Let's Create Something Amazing!"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Creative AI Companion</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 70px; height: 70px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 35px;">🎉</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Welcome to Drawtopia!</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                We're thrilled to have you join our creative community! Your account has been created successfully and you're ready to start creating amazing AI-powered artwork.
            </p>
            
            <!-- Features Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">What you can do with Drawtopia</h3>
                <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li>Transform your ideas into stunning AI artwork</li>
                    <li>Create illustrated stories with AI assistance</li>
                    <li>Explore different artistic styles and templates</li>
                    <li>Save and share your creative masterpieces</li>
                </ul>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Ready to unleash your creativity? Click the button below to start your journey!
            </p>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{FRONTEND_URL}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    Start Creating
                </a>
            </div>
            
            <!-- Pro Tip Box -->
            <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 12px; padding: 20px; margin: 24px 0; text-align: center;">
                <p style="color: #667eea; margin: 0; font-size: 14px;">
                    💡 <strong>Pro Tip:</strong> Check out our <a href="{FRONTEND_URL}/pricing" style="color: #764ba2; text-decoration: underline;">Premium plans</a> for unlimited generations and exclusive features!
                </p>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Have questions? Just reply to this email – we're here to help!
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
            <p style="margin: 8px 0 0 0;">
                Made with 💜 for creative minds everywhere
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Welcome to Drawtopia!

Hi {name},

We're thrilled to have you join our creative community! Your account has been created successfully and you're ready to start creating amazing AI-powered artwork.

What you can do with Drawtopia:
• Transform your ideas into stunning AI artwork
• Create illustrated stories with AI assistance
• Explore different artistic styles and templates
• Save and share your creative masterpieces

Ready to unleash your creativity? Visit: {FRONTEND_URL}

Pro Tip: Check out our Premium plans for unlimited generations and exclusive features!
{FRONTEND_URL}/pricing

Have questions? Just reply to this email!

© {datetime.now().year} Drawtopia
Made with 💜 for creative minds everywhere
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)


    async def send_book_completion_email(
        self,
        to_email: str,
        parent_name: str,
        child_name: str,
        character_name: str,
        character_type: str,
        book_title: str,
        special_ability: str,
        book_format: str,  # 'interactive_search' or 'story_adventure'
        preview_link: str,
        download_link: str,
        story_world: Optional[str] = None,
        adventure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send book completion notification (format-specific)
        
        Args:
            to_email: Parent email address
            parent_name: Parent's name
            child_name: Child's name
            character_name: Generated character name
            character_type: Person/Animal/Magical Creature
            book_title: Story title
            special_ability: Character's special ability
            book_format: 'interactive_search' or 'story_adventure'
            preview_link: Link to preview/read the book (30-day expiration)
            download_link: Link to download PDF
            story_world: For story_adventure - Forest/Space/Underwater
            adventure_type: For story_adventure - Treasure Hunt/Helping Friend
        """
        if book_format == 'interactive_search':
            subject = f"{character_name}'s Enchanted Forest Adventure is ready! 🎮"
            format_description = "an 8-scene Where's Waldo-style adventure in the Enchanted Forest"
            format_details = f"""
                <li>Format: Interactive Search (Where's Waldo style)</li>
                <li>Scenes: 4 magical locations</li>
                <li>Character: {character_name} ({character_type})</li>
                <li>Special Ability: {special_ability}</li>
                <li>Reading Time: ~15-20 minutes</li>
"""
        else:
            subject = f"{character_name}'s Magical Adventure is here! 📖✨"
            format_description = f"a 5-page adventure where {character_name} the {character_type} uses their special power"
            format_details = f"""
                <li>Format: Story Adventure (5-page narrative)</li>
                <li>Pages: 5 beautifully illustrated pages</li>
                <li>Character: {character_name} ({character_type})</li>
                <li>Special Ability: {special_ability}</li>
                <li>World: {story_world or 'Magical World'}</li>
                <li>Adventure Type: {adventure_type or 'Epic Quest'}</li>
                <li>Reading Time: ~10 minutes</li>
"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Story is Ready!</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: #10b981; width: 60px; height: 60px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 30px;">🎉</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Great news! {character_name}'s story is ready!</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {parent_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                {child_name} created "<strong>{book_title}</strong>" — {format_description}.
            </p>
            
            <!-- CTA Buttons -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{preview_link}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block; margin-bottom: 12px;">
                    📖 Start Reading
                </a>
                <br>
                <a href="{download_link}" style="background: #8B4CDF; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    💾 Download PDF
                </a>
            </div>
            
            <!-- Story Details Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">Story Details</h3>
                <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    {format_details}
                </ul>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                {child_name} will love seeing their drawing come to life! You can also print the PDF as a keepsake.
            </p>
            
            <div style="background: #fef3c7; border-radius: 12px; padding: 16px; margin: 24px 0; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; margin: 0; font-size: 14px;">
                    💡 This book is available for 30 days. Download it now to keep forever!
                </p>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Happy reading! 📚
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
{character_name}'s story is ready!

Hi {parent_name},

Great news! 🎉 {character_name}'s story is ready!

{child_name} created "{book_title}" — {format_description}.

📖 Start Reading: {preview_link}
💾 Download PDF: {download_link}

Story Details:
- Character: {character_name} ({character_type})
- Special Ability: {special_ability}
- Reading Time: ~{'15-20' if book_format == 'interactive_search' else '10'} minutes

{child_name} will love seeing their drawing come to life!

💡 This book is available for 30 days. Download it now to keep forever!

Happy reading! 📚

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_receipt_email(
        self,
        to_email: str,
        customer_name: str,
        transaction_id: str,
        items: List[Dict[str, Any]],
        subtotal: float,
        tax: float,
        total: float,
        transaction_date: datetime
    ) -> Dict[str, Any]:
        """
        Send receipt email for purchase
        
        Args:
            to_email: Customer email address
            customer_name: Customer name
            transaction_id: Stripe transaction ID
            items: List of purchased items [{'name': str, 'amount': float}]
            subtotal: Subtotal amount
            tax: Tax amount
            total: Total amount
            transaction_date: Date of transaction
        """
        subject = f"Receipt for your Drawtopia purchase (Order #{transaction_id[:8]})"
        
        items_html = ""
        items_text = ""
        for item in items:
            items_html += f"""
                <tr>
                    <td style="color: #4a5568; padding: 8px 0;">{item['name']}</td>
                    <td style="color: #1a1a2e; text-align: right;">${item['amount']:.2f}</td>
                </tr>
"""
            items_text += f"- {item['name']}: ${item['amount']:.2f}\n"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Receipt</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0;">Thank you for your purchase!</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {customer_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Here's your receipt for your recent Drawtopia purchase.
            </p>
            
            <!-- Receipt Details Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">Order Details</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 16px;">
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Order ID</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{transaction_id}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Date</td>
                        <td style="color: #1a1a2e; text-align: right;">{transaction_date.strftime('%B %d, %Y')}</td>
                    </tr>
                </table>
                
                <h3 style="color: #1a1a2e; margin: 16px 0 12px 0; font-size: 16px;">Items</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    {items_html}
                    <tr style="border-top: 2px solid #e2e8f0;">
                        <td style="color: #718096; padding: 8px 0;">Subtotal</td>
                        <td style="color: #1a1a2e; text-align: right;">${subtotal:.2f}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Tax</td>
                        <td style="color: #1a1a2e; text-align: right;">${tax:.2f}</td>
                    </tr>
                    <tr style="border-top: 2px solid #e2e8f0;">
                        <td style="color: #1a1a2e; padding: 8px 0; font-weight: 600; font-size: 18px;">Total</td>
                        <td style="color: #667eea; text-align: right; font-weight: 600; font-size: 18px;">${total:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Need help? Reply to this email or contact hello@drawtopia.ai
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Receipt for your Drawtopia purchase (Order #{transaction_id[:8]})

Hi {customer_name},

Thank you for your purchase! Here's your receipt.

Order Details:
- Order ID: {transaction_id}
- Date: {transaction_date.strftime('%B %d, %Y')}

Items:
{items_text}
Subtotal: ${subtotal:.2f}
Tax: ${tax:.2f}
Total: ${total:.2f}

Need help? Reply to this email or contact hello@drawtopia.ai

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_subscription_renewal_reminder_email(
        self,
        to_email: str,
        customer_name: str,
        plan_type: str,
        renewal_amount: float,
        renewal_date: datetime,
        manage_link: str,
        cancel_link: str
    ) -> Dict[str, Any]:
        """
        Send subscription renewal reminder (7 days before renewal)
        
        Args:
            to_email: Customer email address
            customer_name: Customer name
            plan_type: Subscription plan type
            renewal_amount: Amount to be charged
            renewal_date: Date of renewal
            manage_link: Link to manage subscription
            cancel_link: Link to cancel subscription
        """
        subject = f"Your Drawtopia subscription renews on {renewal_date.strftime('%B %d')}"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎨 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Subscription Renewal Reminder</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0;">Your subscription renews soon</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {customer_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                This is a friendly reminder that your <strong>{plan_type}</strong> subscription to Drawtopia 
                will automatically renew on <strong>{renewal_date.strftime('%B %d, %Y')}</strong>.
            </p>
            
            <!-- Renewal Details Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">Renewal Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Plan</td>
                        <td style="color: #1a1a2e; text-align: right; font-weight: 600;">{plan_type}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Renewal Date</td>
                        <td style="color: #1a1a2e; text-align: right;">{renewal_date.strftime('%B %d, %Y')}</td>
                    </tr>
                    <tr>
                        <td style="color: #718096; padding: 8px 0;">Amount</td>
                        <td style="color: #667eea; text-align: right; font-weight: 600; font-size: 18px;">${renewal_amount:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                No action is needed! Your subscription will renew automatically and you'll continue to enjoy all premium features.
            </p>
            
            <!-- CTA Buttons -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{manage_link}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block; margin-bottom: 12px;">
                    Manage Subscription
                </a>
                <br>
                <a href="{cancel_link}" style="color: #718096; text-decoration: underline; font-size: 14px;">
                    Cancel Subscription
                </a>
            </div>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Questions? Reply to this email and we'll help!
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
Your Drawtopia subscription renews on {renewal_date.strftime('%B %d')}

Hi {customer_name},

This is a friendly reminder that your {plan_type} subscription to Drawtopia will automatically renew on {renewal_date.strftime('%B %d, %Y')}.

Renewal Details:
- Plan: {plan_type}
- Renewal Date: {renewal_date.strftime('%B %d, %Y')}
- Amount: ${renewal_amount:.2f}

No action is needed! Your subscription will renew automatically.

Manage Subscription: {manage_link}
Cancel Subscription: {cancel_link}

Questions? Reply to this email!

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_gift_notification_email(
        self,
        to_email: str,
        recipient_name: str,
        giver_name: str,
        occasion: str,
        gift_message: str,
        delivery_method: str = 'immediate_email'
    ) -> Dict[str, Any]:
        """
        Send gift notification email (recipient is notified of incoming gift)
        
        Args:
            to_email: Gift recipient email
            recipient_name: Recipient's name
            giver_name: Gift giver's name
            occasion: Occasion (Birthday, First Day of School, etc.)
            gift_message: Personal message from giver
            delivery_method: 'immediate_email', 'scheduled_delivery', or 'send_creation_link'
        """
        subject = "You've been sent a gift on Drawtopia! 🎁✨"
        
        delivery_info = ""
        if delivery_method == 'immediate_email':
            delivery_info = "Your story will be ready to read very soon! We'll send you another email when it's complete, usually within 1-2 hours."
        elif delivery_method == 'scheduled_delivery':
            delivery_info = "Your story will be delivered soon! Keep an eye on your email."
        else:
            delivery_info = f"{giver_name} is asking a grown-up in your life to help create your story. Ask them to check their email for the creation link."
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎁 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">You've Received a Gift!</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%); width: 70px; height: 70px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 35px;">🎁</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">You're about to receive a very special gift! 🎉</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {recipient_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                <strong>{giver_name}</strong> is creating a personalized storybook just for you!
            </p>
            
            <!-- Gift Details Box -->
            <div style="background: #fef3c7; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #f59e0b;">
                <h3 style="color: #1a1a2e; margin: 0 0 16px 0; font-size: 16px;">About Your Gift</h3>
                <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li><strong>Occasion:</strong> {occasion}</li>
                    <li><strong>Message from {giver_name}:</strong> "{gift_message}"</li>
                    <li><strong>Status:</strong> Being created with your character...</li>
                </ul>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                {delivery_info}
            </p>
            
            <!-- How It Works Box -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #667eea;">
                <h3 style="color: #1a1a2e; margin: 0 0 12px 0; font-size: 16px;">How It Works</h3>
                <ol style="color: #4a5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li>Your grown-up creates your character</li>
                    <li>We generate a magical story featuring YOU</li>
                    <li>You read your personalized adventure!</li>
                </ol>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px; text-align: center;">
                We can't wait for you to meet your character! 🌟
            </p>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Questions? Reply to this email or contact hello@drawtopia.ai
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_content = f"""
You've been sent a gift on Drawtopia! 🎁✨

Hi {recipient_name},

You're about to receive a very special gift! 🎉

{giver_name} is creating a personalized storybook just for you!

About Your Gift:
- Occasion: {occasion}
- Message from {giver_name}: "{gift_message}"
- Status: Being created with your character...

{delivery_info}

How It Works:
1. Your grown-up creates your character
2. We generate a magical story featuring YOU
3. You read your personalized adventure!

We can't wait for you to meet your character! 🌟

Questions? Reply to this email or contact hello@drawtopia.ai

© {datetime.now().year} Drawtopia
"""
        
        return await self.send_email(to_email, subject, html_content, text_content)
    
    async def send_gift_delivery_email(
        self,
        to_email: str,
        recipient_name: str,
        giver_name: str,
        character_name: str,
        character_type: str,
        book_title: str,
        special_ability: str,
        gift_message: str,
        story_link: str,
        download_link: str,
        book_format: str = 'story_adventure'
    ) -> Dict[str, Any]:
        """
        Send gift delivery email (final story delivered to recipient)
        
        Args:
            to_email: Gift recipient email
            recipient_name: Recipient's name
            giver_name: Gift giver's name
            character_name: Character name
            character_type: Person/Animal/Magical Creature
            book_title: Story title
            special_ability: Character's special ability
            gift_message: Personal message from giver
            story_link: Link to read the story
            download_link: Link to download PDF
            book_format: 'interactive_search' or 'story_adventure'
        """
        subject = f"Your gift has arrived! Open '{book_title}' now 🎁📖"
        
        format_info = "4-scene Where's Waldo-style adventure" if book_format == 'interactive_search' else "5-page magical adventure"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7fa;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0; padding: 40px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 28px;">🎁 Drawtopia</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 8px;">Your Gift Has Arrived!</p>
        </div>
        
        <!-- Content -->
        <div style="background: white; padding: 40px; border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%); width: 70px; height: 70px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                    <span style="font-size: 35px;">🎁</span>
                </div>
            </div>
            
            <h2 style="color: #1a1a2e; margin: 0 0 20px 0; text-align: center;">Your gift is here! 🎉✨</h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Hi {recipient_name},
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                <strong>{giver_name}</strong> created a special personalized storybook just for you called:
            </p>
            
            <h3 style="color: #667eea; text-align: center; font-size: 22px; margin: 24px 0;">
                📖 "{book_title}"
            </h3>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                Featuring the character YOU created: <strong>{character_name}</strong>, a {character_type} 
                with the special ability to {special_ability}!
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                It's a {format_info} where you'll have an amazing adventure!
            </p>
            
            <!-- CTA Buttons -->
            <div style="text-align: center; margin: 32px 0;">
                <a href="{story_link}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block; margin-bottom: 12px;">
                    🎬 Open Your Gift
                </a>
                <br>
                <a href="{download_link}" style="background: #8B4CDF; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block;">
                    📥 Download PDF
                </a>
            </div>
            
            <!-- Gift Message Box -->
            <div style="background: #fef3c7; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; margin: 0;">
                    <strong>From {giver_name}:</strong><br>
                    "{gift_message}"
                </p>
            </div>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px;">
                This is your special copy! Only you can access it. You can read it anytime, share it with friends, or download it to keep forever. 💫
            </p>
            
            <p style="color: #4a5568; line-height: 1.6; margin-bottom: 20px; text-align: center;">
                Happy reading!
            </p>
            
            <p style="color: #718096; font-size: 14px; text-align: center; margin-top: 32px;">
                Love your gift? Send a thank you to {giver_name}!
            </p>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 24px; color: #718096; font-size: 12px;">
            <p style="margin: 0;">© {datetime.now().year} Drawtopia. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
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
        
        return await self.send_email(to_email, subject, html_content, text_content)


# Create a singleton instance
email_service = EmailService()


# Convenience functions for direct use
async def send_payment_success(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send payment success email"""
    return await email_service.send_payment_success_email(to_email, **kwargs)


async def send_payment_failed(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send payment failed email"""
    return await email_service.send_payment_failed_email(to_email, **kwargs)


async def send_subscription_cancelled(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send subscription cancelled email"""
    return await email_service.send_subscription_cancelled_email(to_email, **kwargs)


async def send_subscription_activated(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send subscription activated email"""
    return await email_service.send_subscription_activated_email(to_email, **kwargs)


async def send_welcome(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send welcome email on registration"""
    return await email_service.send_welcome_email(to_email, **kwargs)


async def send_parental_consent(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send parental consent verification email"""
    return await email_service.send_parental_consent_email(to_email, **kwargs)


async def send_book_completion(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send book completion notification"""
    return await email_service.send_book_completion_email(to_email, **kwargs)


async def send_receipt(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send receipt email"""
    return await email_service.send_receipt_email(to_email, **kwargs)


async def send_subscription_renewal_reminder(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send subscription renewal reminder email"""
    return await email_service.send_subscription_renewal_reminder_email(to_email, **kwargs)


async def send_gift_notification(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send gift notification email"""
    return await email_service.send_gift_notification_email(to_email, **kwargs)


async def send_gift_delivery(to_email: str, **kwargs) -> Dict[str, Any]:
    """Send gift delivery email"""
    return await email_service.send_gift_delivery_email(to_email, **kwargs)

