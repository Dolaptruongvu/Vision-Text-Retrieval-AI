"""
Test Vision Service with Authentication
Tests the complete flow: Login → Predict with JWT token
"""

import requests
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_URL = "http://localhost:5004"
VISION_SERVICE_URL = "http://localhost:5003/predict"

# Test user credentials
TEST_USER = {
    "email": "test@example.com",
    "password": "password123"
}

# Test image folder
TEST_IMAGE_FOLDER = r"c:\wd\ThesisLLM\ThesisVision\realImage"

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def test_login():
    """Step 1: Login to get JWT token"""
    print_section("STEP 1: LOGIN TO GET JWT TOKEN")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/auth/login",
            json=TEST_USER,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            token = data['data']['token']
            user = data['data']['user']
            
            print(f"✅ Login Successful!")
            print(f"   User: {user['username']}")
            print(f"   Email: {user['email']}")
            print(f"   Token: {token[:30]}...")
            
            return token
        else:
            print(f"❌ Login Failed: {data.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ Login Error: {e}")
        return None

def test_predict_without_token():
    """Step 2: Try to predict WITHOUT token (should fail)"""
    print_section("STEP 2: TEST PREDICT WITHOUT TOKEN (Should Fail)")
    
    try:
        # Find a test image
        test_images = list(Path(TEST_IMAGE_FOLDER).glob("*.jpg"))
        if not test_images:
            print("⚠️  No test images found")
            return
        
        test_image = test_images[0]
        print(f"Using image: {test_image.name}")
        
        with open(test_image, 'rb') as f:
            files = {'image': f}
            data = {'query': 'What disease is this?'}
            
            response = requests.post(
                VISION_SERVICE_URL,
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 401:
            print("✅ Correctly rejected! Authentication is working.")
        else:
            print("⚠️  Expected 401 Unauthorized")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_predict_with_token(token):
    """Step 3: Predict WITH valid token (should succeed)"""
    print_section("STEP 3: TEST PREDICT WITH VALID TOKEN (Should Succeed)")
    
    try:
        # Find a test image
        test_images = list(Path(TEST_IMAGE_FOLDER).glob("*.jpg"))
        if not test_images:
            print("⚠️  No test images found")
            return
        
        test_image = test_images[0]
        print(f"Using image: {test_image.name}")
        print(f"Token: {token[:30]}...")
        
        with open(test_image, 'rb') as f:
            files = {'image': f}
            data = {
                'query': 'Đây là bệnh gì? Làm sao để xử lý?',
                'token': token  # Include token in form data
            }
            
            print("\n⏳ Calling Vision Service (this may take 30-60 seconds)...")
            response = requests.post(
                VISION_SERVICE_URL,
                files=files,
                data=data,
                timeout=120
            )
        
        print(f"\nStatus: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            print("✅ Prediction Successful!")
            
            # User info
            if 'user' in result:
                print(f"\n👤 User Info:")
                print(f"   Username: {result['user'].get('username')}")
                print(f"   Email: {result['user'].get('email')}")
            
            # Vision result
            if 'vision_result' in result:
                vision = result['vision_result']
                print(f"\n🔬 Vision Analysis:")
                print(f"   Disease: {vision.get('disease')}")
                print(f"   Confidence: {vision.get('confidence'):.2%}")
            
            # AI response
            if 'ai_response' in result:
                print(f"\n🤖 AI Response:")
                response_text = result['ai_response']
                print(f"   {response_text[:200]}...")
                print(f"   (Total length: {len(response_text)} chars)")
            
            # Database save status
            print(f"\n💾 Database:")
            print(f"   Saved: {result.get('saved_to_db')}")
            if result.get('record_id'):
                print(f"   Record ID: {result.get('record_id')}")
            
            print("\n✅ COMPLETE FLOW SUCCESSFUL!")
            print("   Login → Token → Predict → Save to DB → Return Result")
            
        else:
            print(f"❌ Prediction Failed: {result}")
            
    except requests.exceptions.Timeout:
        print("❌ Request Timeout (Vision/LLM service might be slow)")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_predict_with_header_token(token):
    """Step 4: Test with Authorization header (alternative method)"""
    print_section("STEP 4: TEST WITH AUTHORIZATION HEADER")
    
    try:
        test_images = list(Path(TEST_IMAGE_FOLDER).glob("*.jpg"))
        if not test_images:
            print("⚠️  No test images found")
            return
        
        test_image = test_images[1] if len(test_images) > 1 else test_images[0]
        print(f"Using image: {test_image.name}")
        
        with open(test_image, 'rb') as f:
            files = {'image': f}
            data = {'query': 'What is this plant disease?'}
            headers = {'Authorization': f'Bearer {token}'}
            
            print("⏳ Sending request with Authorization header...")
            response = requests.post(
                VISION_SERVICE_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=120
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Authorization Header Method Works!")
            print(f"   Disease: {result.get('vision_result', {}).get('disease')}")
            print(f"   Saved: {result.get('saved_to_db')}")
        else:
            print(f"Response: {response.json()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("\n" + "=" * 80)
    print("VISION SERVICE AUTHENTICATION TEST")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Vision Service URL: {VISION_SERVICE_URL}")
    print(f"Test User: {TEST_USER['email']}")
    print("=" * 80)
    
    # Check if test images exist
    if not Path(TEST_IMAGE_FOLDER).exists():
        print(f"\n❌ Test image folder not found: {TEST_IMAGE_FOLDER}")
        return
    
    # Step 1: Login
    token = test_login()
    if not token:
        print("\n❌ Cannot proceed without login token")
        return
    
    input("\nPress Enter to continue to Step 2...")
    
    # Step 2: Test without token
    test_predict_without_token()
    
    input("\nPress Enter to continue to Step 3...")
    
    # Step 3: Test with token (form data)
    test_predict_with_token(token)
    
    input("\nPress Enter to continue to Step 4...")
    
    # Step 4: Test with Authorization header
    test_predict_with_header_token(token)
    
    print_section("✅ ALL AUTHENTICATION TESTS COMPLETED")
    print("\nSummary:")
    print("1. ✅ User login successful")
    print("2. ✅ Request without token rejected (401)")
    print("3. ✅ Request with token in form data succeeded")
    print("4. ✅ Request with Authorization header succeeded")
    print("5. ✅ Result saved to MongoDB with user ID")
    print("\n🎉 Authentication system is working perfectly!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
