#!/usr/bin/env python3
"""
Test script for Stripe Payment Intents integration
Tests backend API endpoints and verifies responses
"""

import os
import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")

# Test data
TEST_USER_ID = "test-user-123"
TEST_USER_EMAIL = "test@example.com"
TEST_GIFT_ID = "test-gift-456"

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60 + "\n")

def test_create_payment_intent():
    """Test creating a payment intent"""
    print_section("Test 1: Create Payment Intent")
    
    url = f"{API_BASE_URL}/api/stripe/create-payment-intent"
    payload = {
        "purchase_type": "gift",
        "user_id": TEST_USER_ID,
        "user_email": TEST_USER_EMAIL,
        "gift_id": TEST_GIFT_ID
    }
    
    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Verify response structure
            assert data.get("success") == True, "Success should be True"
            assert "client_secret" in data, "client_secret missing"
            assert "payment_intent_id" in data, "payment_intent_id missing"
            assert "amount" in data, "amount missing"
            
            print("\n✅ Test PASSED: Payment intent created successfully")
            return data
        else:
            print(f"Error Response: {response.text}")
            print("\n❌ Test FAILED: Non-200 status code")
            return None
            
    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")
        return None

def test_invalid_purchase_type():
    """Test with invalid purchase type"""
    print_section("Test 2: Invalid Purchase Type")
    
    url = f"{API_BASE_URL}/api/stripe/create-payment-intent"
    payload = {
        "purchase_type": "invalid_type",  # Invalid
        "user_id": TEST_USER_ID,
        "user_email": TEST_USER_EMAIL
    }
    
    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            print("\n✅ Test PASSED: Correctly rejected invalid purchase type")
        else:
            print("\n❌ Test FAILED: Should return 400 for invalid purchase type")
            
    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")

def test_missing_stripe_key():
    """Test behavior when Stripe is not configured"""
    print_section("Test 3: Missing Stripe Configuration")
    
    # This test only makes sense if we can temporarily unset the key
    # In production, you'd mock this
    print("⏭️  Skipping - requires environment manipulation")

def test_webhook_endpoint():
    """Test webhook endpoint accessibility"""
    print_section("Test 4: Webhook Endpoint")
    
    url = f"{API_BASE_URL}/api/stripe/webhook"
    
    # We can't fully test webhooks without Stripe CLI or actual events
    # But we can check if the endpoint exists
    print(f"POST {url}")
    
    try:
        # Send empty payload (will fail signature check but shows endpoint exists)
        response = requests.post(url, data=b"", headers={"stripe-signature": "test"})
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code in [400, 401, 403]:
            print("✅ Webhook endpoint exists (rejected invalid signature as expected)")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_checkout_session():
    """Test creating checkout session (for comparison)"""
    print_section("Test 5: Create Checkout Session (Legacy)")
    
    url = f"{API_BASE_URL}/api/stripe/create-onetime-checkout"
    payload = {
        "purchase_type": "gift",
        "user_id": TEST_USER_ID,
        "user_email": TEST_USER_EMAIL,
        "gift_id": TEST_GIFT_ID,
        "success_url": "http://localhost:5173/success",
        "cancel_url": "http://localhost:5173/cancel"
    }
    
    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print("\n✅ Test PASSED: Checkout session created successfully")
        else:
            print(f"Response: {response.text}")
            print("\n⚠️  Test may have failed (check response)")
            
    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "🎯 " * 20)
    print(" Stripe Payment Intents Integration Tests")
    print("🎯 " * 20)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run tests
    test_create_payment_intent()
    test_invalid_purchase_type()
    test_missing_stripe_key()
    test_webhook_endpoint()
    test_checkout_session()
    
    print_section("Test Summary")
    print("✅ All basic tests completed")
    print("\nNote: For full integration testing, use Stripe test cards in the UI:")
    print("  - Success: 4242 4242 4242 4242")
    print("  - Decline: 4000 0000 0000 0002")
    print("  - 3D Secure: 4000 0027 6000 3184")
    print("\nFor webhook testing, use Stripe CLI:")
    print("  stripe listen --forward-to localhost:8000/api/stripe/webhook")
    print("  stripe trigger payment_intent.succeeded")

if __name__ == "__main__":
    run_all_tests()
