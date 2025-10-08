"""
Test script for ThesisBE Backend API
Tests authentication and disease history endpoints
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5004"
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
}

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def test_health_check():
    print_section("1. HEALTH CHECK")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_register():
    print_section("2. REGISTER NEW USER")
    response = requests.post(
        f"{BASE_URL}/api/auth/register",
        json=TEST_USER
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 201:
        return data['data']['token']
    elif response.status_code == 400 and "already" in data.get('message', '').lower():
        print("\nUser already exists, trying login...")
        return None
    else:
        print("Registration failed!")
        return None

def test_login():
    print_section("3. LOGIN")
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 200:
        return data['data']['token']
    else:
        print("Login failed!")
        return None

def test_get_profile(token):
    print_section("4. GET USER PROFILE")
    response = requests.get(
        f"{BASE_URL}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_save_disease_history(token):
    print_section("5. SAVE DISEASE HISTORY")
    
    # Test data
    disease_data = {
        "diseaseName": "Tomato - Late blight",
        "diseaseNameRaw": "Tomato___Late_blight",
        "confidence": 0.9852,
        "userQuery": "What is this disease on my tomato plant?",
        "aiResponse": "Based on the image analysis, your plant has Late blight disease...",
        "imageName": "tomato_test.jpg",
        "processingTime": 1500
    }
    
    response = requests.post(
        f"{BASE_URL}/api/disease/save",
        json=disease_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 201:
        return data['data']['id']
    return None

def test_get_disease_history(token):
    print_section("6. GET DISEASE HISTORY")
    response = requests.get(
        f"{BASE_URL}/api/disease/history?limit=10&page=1",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    return response.status_code == 200

def test_get_disease_stats(token):
    print_section("7. GET DISEASE STATISTICS")
    response = requests.get(
        f"{BASE_URL}/api/disease/stats",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_get_disease_detail(token, disease_id):
    print_section("8. GET DISEASE DETAIL")
    response = requests.get(
        f"{BASE_URL}/api/disease/{disease_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    print("=" * 60)
    print("THESIS BACKEND API - TEST SUITE")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Test User: {TEST_USER['email']}")
    
    try:
        # Test 1: Health check
        if not test_health_check():
            print("\n❌ Health check failed! Is the server running?")
            return
        
        time.sleep(1)
        
        # Test 2 & 3: Register or Login
        token = test_register()
        if not token:
            token = test_login()
        
        if not token:
            print("\n❌ Authentication failed!")
            return
        
        print(f"\n✅ JWT Token obtained: {token[:20]}...")
        time.sleep(1)
        
        # Test 4: Get profile
        test_get_profile(token)
        time.sleep(1)
        
        # Test 5: Save disease history
        disease_id = test_save_disease_history(token)
        time.sleep(1)
        
        # Test 6: Get disease history
        test_get_disease_history(token)
        time.sleep(1)
        
        # Test 7: Get statistics
        test_get_disease_stats(token)
        time.sleep(1)
        
        # Test 8: Get disease detail (if we have an ID)
        if disease_id:
            test_get_disease_detail(token, disease_id)
        
        print_section("✅ ALL TESTS COMPLETED")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend server!")
        print("Make sure the server is running: npm run dev")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    main()
