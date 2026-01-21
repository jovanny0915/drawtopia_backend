"""
Email Service for Drawtopia
Uses Resend API for sending transactional emails

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
1. Sign up at https://resend.com
2. Get your API key from the dashboard
3. Verify your domain (e.g., email.drawtopia.ai)
4. Add RESEND_API_KEY and FROM_EMAIL to .env
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Resend API Configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_API_URL = "https://api.resend.com/emails"

# General Configuration
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@email.drawtopia.ai")
FROM_NAME = os.getenv("FROM_NAME", "Drawtopia")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Initialize email service
if RESEND_API_KEY:
    logger.info("✅ Email service (Resend API) initialized successfully")
    EMAIL_ENABLED = True
else:
    EMAIL_ENABLED = False
    logger.warning("⚠️ Resend API not configured. Set RESEND_API_KEY in .env")


class EmailService:
    """Email service for sending transactional emails via Resend API"""
    
    def __init__(self):
        self.enabled = EMAIL_ENABLED
        self.from_email = f"{FROM_NAME} <{FROM_EMAIL}>"
        self.api_key = RESEND_API_KEY
        self.api_url = RESEND_API_URL
        # Get templates directory path
        self.templates_dir = Path(__file__).parent / "templates"
    
    def is_enabled(self) -> bool:
        """Check if email service is enabled"""
        return self.enabled
    
    def _load_template(self, template_name: str) -> str:
        """Load HTML template from file"""
        template_path = self.templates_dir / template_name
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an email using Resend API
        
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
        
        if not self.api_key:
            logger.error("Resend API key is missing")
            return {"success": False, "error": "Resend API key not configured"}
        
        try:
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload
            payload = {
                "from": self.from_email,
                "to": to_email,
                "subject": subject,
                "html": html_content
            }
            
            # Add text content if provided
            if text_content:
                payload["text"] = text_content
            
            logger.debug(f"Sending email to {to_email} via Resend API (from: {self.from_email})")
            
            # Send email via Resend API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
            
            # Check response (Resend API returns 200 or 201 for success)
            if response.status_code in [200, 201]:
                try:
                    response_data = response.json()
                    email_id = response_data.get("id", f"resend_{datetime.now().timestamp()}")
                    logger.info(f"✅ Email sent successfully to {to_email} (ID: {email_id})")
                    return {"success": True, "id": email_id}
                except Exception as json_error:
                    # If response is not JSON, still consider it successful if status is 200/201
                    logger.warning(f"Could not parse JSON response, but status is {response.status_code}: {json_error}")
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
                return {"success": False, "error": f"Resend API error ({response.status_code}): {error_message})"}
            
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP error sending email to {to_email}: {e}")
            return {"success": False, "error": f"HTTP error: {str(e)}"}
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
        
        # Prepare template variables
        next_billing_row = f'<tr><td style="color: #718096; padding: 8px 0;">Next billing</td><td style="color: #1a1a2e; text-align: right;">{next_billing_date}</td></tr>' if next_billing_date else ''
        
        # Load and render template
        html_content = self._load_template("payment_success.html").format(
            name=name,
            plan_display=plan_display,
            amount_display=amount_display,
            next_billing_row=next_billing_row,
            frontend_url=FRONTEND_URL,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("payment_failed.html").format(
            name=name,
            plan_display=plan_display,
            amount_display=amount_display,
            update_url=update_url,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("subscription_cancelled.html").format(
            name=name,
            plan_display=plan_display,
            access_info=access_info,
            frontend_url=FRONTEND_URL,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("parental_consent.html").format(
            parent_name=parent_name,
            child_name=child_name,
            consent_link=consent_link,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("welcome.html").format(
            name=name,
            frontend_url=FRONTEND_URL,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("book_completion.html").format(
            character_name=character_name,
            parent_name=parent_name,
            child_name=child_name,
            book_title=book_title,
            format_description=format_description,
            preview_link=preview_link,
            download_link=download_link,
            format_details=format_details,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("receipt.html").format(
            customer_name=customer_name,
            transaction_id=transaction_id,
            transaction_date=transaction_date.strftime('%B %d, %Y'),
            items_html=items_html,
            subtotal=f"{subtotal:.2f}",
            tax=f"{tax:.2f}",
            total=f"{total:.2f}",
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("subscription_renewal_reminder.html").format(
            customer_name=customer_name,
            plan_type=plan_type,
            renewal_date=renewal_date.strftime('%B %d, %Y'),
            renewal_amount=f"{renewal_amount:.2f}",
            manage_link=manage_link,
            cancel_link=cancel_link,
            current_year=datetime.now().year
        )
        
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
        
        # Load and render template
        html_content = self._load_template("gift_notification.html").format(
            recipient_name=recipient_name,
            giver_name=giver_name,
            occasion=occasion,
            gift_message=gift_message,
            delivery_info=delivery_info,
            current_year=datetime.now().year
        )
        
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
        book_format: str = 'story_adventure',
        dashboard_link: Optional[str] = None
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
            dashboard_link: Link to dashboard notifications (optional)
        """
        subject = f"Your gift has arrived! Open '{book_title}' now 🎁📖"
        
        format_info = "4-scene Where's Waldo-style adventure" if book_format == 'interactive_search' else "5-page magical adventure"
        
        # Default dashboard link if not provided
        if not dashboard_link:
            dashboard_link = f"{FRONTEND_URL}/dashboard"
        
        # Load and render template
        html_content = self._load_template("gift_delivery.html").format(
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
            dashboard_link=dashboard_link,
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
🔔 View on Dashboard: {dashboard_link}

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

