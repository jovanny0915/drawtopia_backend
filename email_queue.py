"""
Email Queue Manager
Handles asynchronous email sending using Supabase as queue backend
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from supabase import Client
from email_service import email_service
import json

logger = logging.getLogger(__name__)


class EmailQueueManager:
    """Manages email queue using Supabase as the backend"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.is_processing = False
    
    def queue_email(
        self,
        email_type: str,
        to_email: str,
        email_data: Dict[str, Any],
        priority: int = 5,
        scheduled_for: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Queue an email for asynchronous sending
        
        Args:
            email_type: Type of email (welcome, parental_consent, book_completion, etc.)
            to_email: Recipient email address
            email_data: Data needed for email template
            priority: Priority (1-10, 1 is highest)
            scheduled_for: Schedule email for future delivery (optional)
        
        Returns:
            Created email job record
        """
        try:
            email_job = {
                "email_type": email_type,
                "to_email": to_email,
                "email_data": email_data,
                "status": "pending",
                "priority": priority,
                "retry_count": 0,
                "max_retries": 5,
                "scheduled_for": scheduled_for.isoformat() if scheduled_for else None,
                "created_at": datetime.now().isoformat()
            }
            
            result = self.supabase.table("email_queue").insert(email_job).execute()
            
            if result.data and len(result.data) > 0:
                job = result.data[0]
                logger.info(f"✅ Queued {email_type} email to {to_email} (Job ID: {job.get('id')})")
                return job
            else:
                logger.error(f"❌ Failed to queue email: No data returned")
                return {"success": False, "error": "Failed to create email job"}
                
        except Exception as e:
            logger.error(f"❌ Error queuing email: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_email_queue(self, batch_size: int = 10):
        """
        Process pending emails from the queue
        
        Args:
            batch_size: Number of emails to process in one batch
        """
        if self.is_processing:
            logger.debug("Email queue processor already running")
            return
        
        self.is_processing = True
        
        try:
            # Get pending emails (not scheduled or scheduled_for <= now)
            now = datetime.now().isoformat()
            
            result = self.supabase.table("email_queue") \
                .select("*") \
                .eq("status", "pending") \
                .or_(f"scheduled_for.is.null,scheduled_for.lte.{now}") \
                .order("priority", desc=False) \
                .order("created_at", desc=False) \
                .limit(batch_size) \
                .execute()
            
            if not result.data:
                logger.debug("No pending emails in queue")
                return
            
            logger.info(f"📬 Processing {len(result.data)} pending emails")
            
            for email_job in result.data:
                await self._process_single_email(email_job)
                
        except Exception as e:
            logger.error(f"❌ Error processing email queue: {e}")
        finally:
            self.is_processing = False
    
    async def _process_single_email(self, email_job: Dict[str, Any]):
        """Process a single email job"""
        job_id = email_job["id"]
        email_type = email_job["email_type"]
        to_email = email_job["to_email"]
        email_data = email_job["email_data"]
        
        try:
            # Mark as processing
            self.supabase.table("email_queue") \
                .update({"status": "processing", "updated_at": datetime.now().isoformat()}) \
                .eq("id", job_id) \
                .execute()
            
            # Send email based on type
            result = await self._send_email_by_type(email_type, to_email, email_data)
            
            if result.get("success"):
                # Mark as completed
                self.supabase.table("email_queue") \
                    .update({
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "result": result,
                        "updated_at": datetime.now().isoformat()
                    }) \
                    .eq("id", job_id) \
                    .execute()
                
                logger.info(f"✅ Successfully sent {email_type} email to {to_email}")
            else:
                # Handle failure with retry logic
                await self._handle_email_failure(email_job, result.get("error", "Unknown error"))
                
        except Exception as e:
            logger.error(f"❌ Error processing email job {job_id}: {e}")
            await self._handle_email_failure(email_job, str(e))
    
    async def _send_email_by_type(
        self,
        email_type: str,
        to_email: str,
        email_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email based on type"""
        try:
            if email_type == "welcome":
                return await email_service.send_welcome_email(to_email, **email_data)
            
            elif email_type == "parental_consent":
                return await email_service.send_parental_consent_email(to_email, **email_data)
            
            elif email_type == "book_completion":
                return await email_service.send_book_completion_email(to_email, **email_data)
            
            elif email_type == "payment_success":
                return await email_service.send_payment_success_email(to_email, **email_data)
            
            elif email_type == "payment_failed":
                return await email_service.send_payment_failed_email(to_email, **email_data)
            
            elif email_type == "receipt":
                return await email_service.send_receipt_email(to_email, **email_data)
            
            elif email_type == "subscription_cancelled":
                return await email_service.send_subscription_cancelled_email(to_email, **email_data)
            
            elif email_type == "subscription_renewal_reminder":
                return await email_service.send_subscription_renewal_reminder_email(to_email, **email_data)
            
            elif email_type == "gift_notification":
                return await email_service.send_gift_notification_email(to_email, **email_data)
            
            elif email_type == "gift_delivery":
                return await email_service.send_gift_delivery_email(to_email, **email_data)
            
            else:
                logger.error(f"Unknown email type: {email_type}")
                return {"success": False, "error": f"Unknown email type: {email_type}"}
                
        except Exception as e:
            logger.error(f"Error sending {email_type} email: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_email_failure(self, email_job: Dict[str, Any], error_message: str):
        """Handle email sending failure with retry logic"""
        job_id = email_job["id"]
        retry_count = email_job["retry_count"] + 1
        max_retries = email_job["max_retries"]
        
        if retry_count < max_retries:
            # Calculate exponential backoff delay
            delay_minutes = 2 ** retry_count  # 2, 4, 8, 16, 32 minutes
            retry_at = datetime.now() + timedelta(minutes=delay_minutes)
            
            self.supabase.table("email_queue") \
                .update({
                    "status": "pending",
                    "retry_count": retry_count,
                    "scheduled_for": retry_at.isoformat(),
                    "last_error": error_message,
                    "updated_at": datetime.now().isoformat()
                }) \
                .eq("id", job_id) \
                .execute()
            
            logger.warning(f"⚠️ Email job {job_id} failed (attempt {retry_count}/{max_retries}). Retrying in {delay_minutes} minutes.")
        else:
            # Max retries reached, mark as failed
            self.supabase.table("email_queue") \
                .update({
                    "status": "failed",
                    "retry_count": retry_count,
                    "last_error": error_message,
                    "failed_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }) \
                .eq("id", job_id) \
                .execute()
            
            logger.error(f"❌ Email job {job_id} failed permanently after {max_retries} attempts: {error_message}")
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get email queue statistics"""
        try:
            stats = {}
            
            # Count by status
            for status in ["pending", "processing", "completed", "failed"]:
                result = self.supabase.table("email_queue") \
                    .select("id", count="exact") \
                    .eq("status", status) \
                    .execute()
                stats[status] = result.count if hasattr(result, 'count') else 0
            
            return stats
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
    
    async def cleanup_old_emails(self, days_old: int = 30):
        """
        Clean up completed/failed emails older than specified days
        
        Args:
            days_old: Delete emails older than this many days
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            result = self.supabase.table("email_queue") \
                .delete() \
                .in_("status", ["completed", "failed"]) \
                .lt("created_at", cutoff_date) \
                .execute()
            
            count = len(result.data) if result.data else 0
            logger.info(f"🧹 Cleaned up {count} old email records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old emails: {e}")


# Helper functions for common email operations
async def queue_welcome_email(
    supabase: Client,
    to_email: str,
    customer_name: str
) -> Dict[str, Any]:
    """Queue welcome email"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="welcome",
        to_email=to_email,
        email_data={"customer_name": customer_name},
        priority=3
    )


async def queue_parental_consent_email(
    supabase: Client,
    to_email: str,
    parent_name: str,
    child_name: str,
    consent_link: str
) -> Dict[str, Any]:
    """Queue parental consent verification email"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="parental_consent",
        to_email=to_email,
        email_data={
            "parent_name": parent_name,
            "child_name": child_name,
            "consent_link": consent_link
        },
        priority=1  # High priority for compliance
    )


async def queue_book_completion_email(
    supabase: Client,
    to_email: str,
    **kwargs
) -> Dict[str, Any]:
    """Queue book completion notification"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="book_completion",
        to_email=to_email,
        email_data=kwargs,
        priority=2
    )


async def queue_payment_success_email(
    supabase: Client,
    to_email: str,
    **kwargs
) -> Dict[str, Any]:
    """Queue payment success email"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="payment_success",
        to_email=to_email,
        email_data=kwargs,
        priority=1  # High priority for payment confirmations
    )


async def queue_gift_notification_email(
    supabase: Client,
    to_email: str,
    scheduled_for: Optional[datetime] = None,
    **kwargs
) -> Dict[str, Any]:
    """Queue gift notification email (with optional scheduling)"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="gift_notification",
        to_email=to_email,
        email_data=kwargs,
        priority=2,
        scheduled_for=scheduled_for
    )


async def queue_gift_delivery_email(
    supabase: Client,
    to_email: str,
    **kwargs
) -> Dict[str, Any]:
    """Queue gift delivery email"""
    queue_manager = EmailQueueManager(supabase)
    return queue_manager.queue_email(
        email_type="gift_delivery",
        to_email=to_email,
        email_data=kwargs,
        priority=2
    )

