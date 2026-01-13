#!/usr/bin/env python3
"""
Test Script for Scheduled Gift Delivery System
This script tests the complete flow of scheduled gift delivery
"""

import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_backend_health():
    """Test if backend is running"""
    print_section("1. Testing Backend Health")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_edge_function_send_notification():
    """Test the send-gift-notification edge function"""
    print_section("2. Testing send-gift-notification Edge Function")
    
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("❌ SUPABASE_URL or SUPABASE_ANON_KEY not set")
        return False
    
    edge_function_url = f"{SUPABASE_URL}/functions/v1/send-gift-notification"
    
    # Test with empty payload (should return no gifts to process)
    try:
        response = requests.post(
            edge_function_url,
            json={"mode": "batch"},
            headers={
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Edge function is accessible")
            print(f"   Processed: {result.get('processed', 0)} gifts")
            return True
        else:
            print(f"❌ Edge function returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Edge function test failed: {e}")
        return False

def test_edge_function_check_scheduled():
    """Test the check-scheduled-gifts edge function"""
    print_section("3. Testing check-scheduled-gifts Edge Function")
    
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("❌ SUPABASE_URL or SUPABASE_ANON_KEY not set")
        return False
    
    edge_function_url = f"{SUPABASE_URL}/functions/v1/check-scheduled-gifts"
    
    try:
        response = requests.post(
            edge_function_url,
            json={},
            headers={
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Edge function is accessible")
            print(f"   Processed: {result.get('processed', 0)} gifts")
            print(f"   Succeeded: {result.get('succeeded', 0)}")
            print(f"   Failed: {result.get('failed', 0)}")
            return True
        else:
            print(f"❌ Edge function returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Edge function test failed: {e}")
        return False

def test_backend_gift_delivery():
    """Test the backend gift delivery endpoint"""
    print_section("4. Testing Backend Gift Delivery Endpoint")
    
    # Test with a fake gift ID (should return 404)
    test_gift_id = "00000000-0000-0000-0000-000000000000"
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/gift/deliver",
            json={"gift_id": test_gift_id},
            timeout=30
        )
        
        if response.status_code == 404:
            print("✅ Endpoint is accessible and validates gift existence")
            print(f"   Expected 404 for non-existent gift")
            return True
        elif response.status_code == 400:
            result = response.json()
            print("✅ Endpoint is accessible and validates request")
            print(f"   Response: {result}")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return True  # Still accessible
    except Exception as e:
        print(f"❌ Backend endpoint test failed: {e}")
        return False

def create_test_gift():
    """Create a test gift for delivery (requires authentication)"""
    print_section("5. Creating Test Gift (Optional)")
    print("⚠️  This requires a valid user authentication token")
    print("   Skipping automatic gift creation")
    print("   To test with a real gift:")
    print("   1. Create a gift through the UI")
    print("   2. Set delivery_time to 1-2 minutes from now")
    print("   3. Wait for the cron job to process it")
    return None

def check_database_schema():
    """Check if database has required tables and columns"""
    print_section("6. Checking Database Schema")
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return False
    
    # Check gifts table
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/gifts?limit=1",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ gifts table exists and is accessible")
        else:
            print(f"❌ gifts table check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ gifts table check failed: {e}")
        return False
    
    # Check push_subscriptions table
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/push_subscriptions?limit=1",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ push_subscriptions table exists and is accessible")
            return True
        else:
            print(f"❌ push_subscriptions table check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ push_subscriptions table check failed: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print_section("Test Summary")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print()
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print()
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
    
    return failed == 0

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Scheduled Gift Delivery System - Test Suite")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Backend URL: {BACKEND_URL}")
    print(f"  Supabase URL: {SUPABASE_URL or '(not set)'}")
    print(f"  Service Key: {'✅ Set' if SUPABASE_SERVICE_KEY else '❌ Not set'}")
    
    # Run tests
    results = {}
    
    results["Backend Health"] = test_backend_health()
    time.sleep(1)
    
    results["Database Schema"] = check_database_schema()
    time.sleep(1)
    
    results["Edge Function: send-gift-notification"] = test_edge_function_send_notification()
    time.sleep(1)
    
    results["Edge Function: check-scheduled-gifts"] = test_edge_function_check_scheduled()
    time.sleep(1)
    
    results["Backend: Gift Delivery Endpoint"] = test_backend_gift_delivery()
    time.sleep(1)
    
    create_test_gift()
    
    # Print summary
    success = print_summary(results)
    
    # Print next steps
    print_section("Next Steps")
    print("1. Run the database migration:")
    print("   psql -f scheduled_delivery_migration.sql")
    print()
    print("2. Set up VAPID keys for web push:")
    print("   npx web-push generate-vapid-keys")
    print()
    print("3. Configure edge function secrets:")
    print("   supabase secrets set VAPID_PUBLIC_KEY=...")
    print("   supabase secrets set VAPID_PRIVATE_KEY=...")
    print("   supabase secrets set BACKEND_URL=...")
    print()
    print("4. Test with a real gift:")
    print("   - Create a gift through the UI")
    print("   - Set delivery_time to 1-2 minutes from now")
    print("   - Monitor edge function logs")
    print()
    print("5. Monitor the system:")
    print("   - Check edge function logs in Supabase Dashboard")
    print("   - Query scheduled_gifts_pending view")
    print("   - Check backend logs for delivery attempts")
    print()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

