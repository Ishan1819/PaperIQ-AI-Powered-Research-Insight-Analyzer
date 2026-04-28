"""
Quick test script to verify authentication system is working correctly.
Run this after setting up the database and environment variables.
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_signup():
    """Test user signup"""
    print("\n=== Testing Signup ===")
    response = requests.post(
        f"{BASE_URL}/auth",
        json={
            "email": "test@example.com",
            "password": "test123456"
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.cookies

def test_login(cookies=None):
    """Test user login"""
    print("\n=== Testing Login ===")
    response = requests.post(
        f"{BASE_URL}/auth",
        json={
            "email": "test@example.com",
            "password": "test123456"
        },
        cookies=cookies
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.cookies

def test_get_me(cookies):
    """Test getting current user"""
    print("\n=== Testing /me ===")
    response = requests.get(
        f"{BASE_URL}/me",
        cookies=cookies
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_save_chat(cookies):
    """Test saving chat message"""
    print("\n=== Testing Save Chat ===")
    response = requests.post(
        f"{BASE_URL}/chat/save",
        json={
            "user_message": "What is machine learning?",
            "ai_response": "Machine learning is a subset of artificial intelligence..."
        },
        cookies=cookies
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_get_history(cookies):
    """Test getting chat history"""
    print("\n=== Testing Get Chat History ===")
    response = requests.get(
        f"{BASE_URL}/chat/history",
        cookies=cookies
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_logout(cookies):
    """Test logout"""
    print("\n=== Testing Logout ===")
    response = requests.post(
        f"{BASE_URL}/logout",
        cookies=cookies
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_unauthorized():
    """Test accessing protected route without auth"""
    print("\n=== Testing Unauthorized Access ===")
    response = requests.get(f"{BASE_URL}/me")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("🔍 Starting Authentication System Tests...")
    print(f"Testing against: {BASE_URL}")
    
    try:
        # Test signup or login
        cookies = test_signup()
        
        # If user already exists, try login
        # cookies = test_login()
        
        # Test authenticated endpoints
        test_get_me(cookies)
        test_save_chat(cookies)
        test_get_history(cookies)
        
        # Test logout
        test_logout(cookies)
        
        # Test unauthorized access
        test_unauthorized()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nMake sure:")
        print("1. Flask app is running (python app.py)")
        print("2. PostgreSQL database is configured")
        print("3. Environment variables are set")
