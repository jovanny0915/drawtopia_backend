from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_API_URL = "https://api.resend.com/emails"

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
