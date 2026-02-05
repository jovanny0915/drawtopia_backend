"""
Test script for the /api/books/{id}/state PATCH endpoint

This script tests the update_story_state endpoint to ensure it properly handles:
1. Valid "generating" state updates
2. Valid "completed" state updates with optional fields
3. Invalid state values (should return 400)
4. Missing story IDs (should return 404)

Usage:
    python tests/test_update_story_state.py
"""
import requests
import json

# Configuration
API_BASE = "http://localhost:8000"  # Change this to your backend URL
TEST_STORY_ID = "your-story-id-here"  # Replace with an actual story ID from your database

def test_update_to_generating():
    """Test updating story state to 'generating'"""
    print("\n=== Test 1: Update state to 'generating' ===")
    url = f"{API_BASE}/api/books/{TEST_STORY_ID}/state"
    payload = {"state": "generating"}
    
    response = requests.patch(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ SUCCESS: Story state updated to 'generating'")
        return True
    else:
        print(f"❌ FAILED: Expected 200, got {response.status_code}")
        return False

def test_update_to_completed():
    """Test updating story state to 'completed' with optional fields"""
    print("\n=== Test 2: Update state to 'completed' with data ===")
    url = f"{API_BASE}/api/books/{TEST_STORY_ID}/state"
    payload = {
        "state": "completed",
        "story_content": json.dumps({"pages": ["Page 1", "Page 2"]}),
        "scene_images": ["image1.jpg", "image2.jpg"],
        "audio_urls": ["audio1.mp3", "audio2.mp3"],
        "dedication_text": "For my child",
        "dedication_image": "dedication.jpg",
        "story_cover": "cover.jpg"
    }
    
    response = requests.patch(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")  # Truncate long response
    
    if response.status_code == 200:
        print("✅ SUCCESS: Story state updated to 'completed' with data")
        return True
    else:
        print(f"❌ FAILED: Expected 200, got {response.status_code}")
        return False

def test_invalid_state():
    """Test updating story state with invalid value (should return 400)"""
    print("\n=== Test 3: Update with invalid state value ===")
    url = f"{API_BASE}/api/books/{TEST_STORY_ID}/state"
    payload = {"state": "invalid_state"}
    
    response = requests.patch(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 400:
        print("✅ SUCCESS: Correctly rejected invalid state")
        return True
    else:
        print(f"❌ FAILED: Expected 400, got {response.status_code}")
        return False

def test_missing_story():
    """Test updating non-existent story (should return 404)"""
    print("\n=== Test 4: Update non-existent story ===")
    url = f"{API_BASE}/api/books/nonexistent-id-12345/state"
    payload = {"state": "generating"}
    
    response = requests.patch(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 404:
        print("✅ SUCCESS: Correctly returned 404 for missing story")
        return True
    else:
        print(f"❌ FAILED: Expected 404, got {response.status_code}")
        return False

def test_empty_body():
    """Test updating with empty body (should use default 'generating')"""
    print("\n=== Test 5: Update with empty body ===")
    url = f"{API_BASE}/api/books/{TEST_STORY_ID}/state"
    payload = {}
    
    response = requests.patch(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ SUCCESS: Empty body accepted, defaulted to 'generating'")
        return True
    else:
        print(f"❌ FAILED: Expected 200, got {response.status_code}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing /api/books/{id}/state PATCH endpoint")
    print("=" * 60)
    
    if TEST_STORY_ID == "your-story-id-here":
        print("\n⚠️  WARNING: Please set TEST_STORY_ID to an actual story ID from your database")
        print("You can find a story ID by querying your 'stories' table in Supabase")
        exit(1)
    
    results = []
    results.append(("Update to generating", test_update_to_generating()))
    results.append(("Update to completed", test_update_to_completed()))
    results.append(("Invalid state", test_invalid_state()))
    results.append(("Missing story", test_missing_story()))
    results.append(("Empty body", test_empty_body()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
