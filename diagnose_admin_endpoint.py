"""
Diagnostic script to check admin endpoint configuration
Run this to verify the admin routes are properly registered
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("ADMIN ENDPOINT DIAGNOSTIC")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')[:30]}...")
print(f"   SUPABASE_ANON_KEY: {'SET' if os.getenv('SUPABASE_ANON_KEY') else 'NOT SET'}")
print(f"   SUPABASE_SERVICE_KEY: {'SET' if os.getenv('SUPABASE_SERVICE_KEY') else 'NOT SET'}")
print(f"   ALLOWED_ORIGINS: {os.getenv('ALLOWED_ORIGINS', '*')}")
print(f"   ALLOWED_HOSTS: {os.getenv('ALLOWED_HOSTS', '*')}")
print(f"   ENVIRONMENT: {os.getenv('ENVIRONMENT', 'development')}")

# Check if we can import the admin module
print("\n2. Checking Admin Module:")
try:
    from apis.admin import router as admin_router
    print(f"   ✅ Admin router imported successfully")
    print(f"   Routes in admin router:")
    for route in admin_router.routes:
        print(f"      - {route.methods} {route.path}")
except Exception as e:
    print(f"   ❌ Failed to import admin router: {e}")

# Check Supabase connection
print("\n3. Checking Supabase Connection:")
try:
    from supabase import create_client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    
    key_to_use = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
    
    if SUPABASE_URL and key_to_use:
        supabase = create_client(SUPABASE_URL, key_to_use)
        
        # Try to fetch templates
        response = supabase.table("book_templates").select("*").limit(1).execute()
        print(f"   ✅ Supabase connection successful")
        print(f"   Sample query returned {len(response.data)} row(s)")
    else:
        print(f"   ❌ Missing SUPABASE_URL or key")
except Exception as e:
    print(f"   ❌ Supabase connection failed: {e}")

# Check main app configuration
print("\n4. Checking Main App Configuration:")
try:
    from main import app
    print(f"   ✅ Main app imported successfully")
    
    # List all routes
    admin_routes = [route for route in app.routes if '/admin/' in str(route.path)]
    
    if admin_routes:
        print(f"   ✅ Found {len(admin_routes)} admin route(s):")
        for route in admin_routes:
            print(f"      - {route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}")
    else:
        print(f"   ⚠️  No admin routes found in app!")
        print(f"   Total routes in app: {len(app.routes)}")
        
except Exception as e:
    print(f"   ❌ Failed to import main app: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf the endpoint works locally but not in deployment:")
print("1. Check ALLOWED_ORIGINS includes your frontend URL")
print("2. Check ALLOWED_HOSTS includes your backend domain")
print("3. Verify all environment variables are set in deployment")
print("4. Check deployment logs for errors")
print("5. Verify the admin router is included with prefix='/api'")
