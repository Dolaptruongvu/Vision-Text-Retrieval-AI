# Test API Client - Test Vision Service (which calls LLM Service internally)
# Client only needs to talk to Vision Service, Vision Service handles LLM communication

import os
import requests
import json
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_URL = "http://localhost:5004"
# NEW ARCHITECTURE: Frontend/Test → Backend API → Vision Service → LLM Service
PREDICT_URL = f"{BACKEND_URL}/api/disease/predict"

# Test user credentials (must be registered first)
TEST_USER = {
    "email": "test@example.com",
    "password": "password123"
}

# Test image folder - CHANGE THIS TO YOUR IMAGE FOLDER
TEST_IMAGE_FOLDER = r"c:\wd\ThesisLLM\ThesisVision\realImage"

# Global token storage
JWT_TOKEN = None

# Test cases: (image_filename, user_question)
TEST_CASES = [
    ("late_blight_tomato_unusual-symptomsx1200.jpg", "Đây là bệnh gì?"),
    ("bacterial-spot-of-pepper-pepper-1560240277.jpg", "Dựa vào ảnh cây trồng của tôi cho tôi biết về bệnh này"),
    (None, "Làm thế nào để phòng trừ bệnh đào ôn?"),  # Text-only query
    ("late_blight_tomato_unusual-symptomsx1200.jpg", None),  # Image-only (will auto-generate query)
    (None, "Tóm tắt các câu hỏi tôi vừa hỏi bạn")
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def login_user():
    """Login to get JWT token"""
    global JWT_TOKEN
    
    print("\n" + "=" * 80)
    print("AUTHENTICATING USER")
    print("=" * 80)
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/auth/login",
            json=TEST_USER,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            JWT_TOKEN = data['data']['token']
            user = data['data']['user']
            
            print(f"✅ Login Successful!")
            print(f"   User: {user['username']}")
            print(f"   Email: {user['email']}")
            print(f"   Token: {JWT_TOKEN[:30]}...")
            print("=" * 80)
            return True
        else:
            print(f"❌ Login Failed: {response.json().get('message')}")
            print("💡 Make sure user is registered in backend")
            return False
            
    except Exception as e:
        print(f"❌ Login Error: {e}")
        print("💡 Make sure backend is running on port 5004")
        return False

def get_monthly_stats():
    """Get monthly disease statistics from Backend API"""
    global JWT_TOKEN
    
    print("\n" + "=" * 80)
    print("GETTING MONTHLY DISEASE STATISTICS")
    print("=" * 80)
    
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/disease/monthly-stats",
            headers={'Authorization': f'Bearer {JWT_TOKEN}'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Statistics Retrieved!")
            
            # Period
            period = data['data']['period']
            print(f"\n📅 Period: {period['monthName']} {period['year']}")
            
            # Summary
            summary = data['data']['summary']
            print(f"\n📈 Summary:")
            print(f"   Total Detections: {summary['totalDetections']}")
            print(f"   Trend: {summary['trend']}")
            
            # Top diseases
            print(f"\n🏆 Top 5 Diseases:")
            top_diseases = data['data']['topDiseases']
            
            if not top_diseases:
                print("   (No data for this month)")
            else:
                for i, disease in enumerate(top_diseases, 1):
                    print(f"   {i}. {disease['diseaseName']}: {disease['percentage']}% ({disease['count']} cases)")
            
            print("=" * 80)
            return data
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return None

def test_service_health(service_name, url):
    """Check if service is healthy"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"[{service_name}] Service is healthy")
            return True
        else:
            print(f"[{service_name}] Service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[{service_name}] Service not available: {e}")
        return False

def call_vision_service(image_path=None, user_query=None, token=None):
    """
    Call Backend API to predict disease (which forwards to Vision Service)
    
    NEW ARCHITECTURE: Test Client → Backend API → Vision Service → LLM Service
    
    Args:
        image_path: Path to image file (optional)
        user_query: User question text (optional)
        token: JWT token for authentication (required)
        Note: At least one must be provided
    
    Returns:
        {
            "success": bool,
            "data": {
                "vision_result": {
                    "disease": str,
                    "confidence": float
                },
                "ai_response": str,
                "saved_to_db": bool,
                "record_id": str
            },
            "error": str (optional)
        }
    """
    try:
        files = {}
        data = {}
        
        # Prepare image if provided
        if image_path and os.path.exists(image_path):
            print(f"\n[CLIENT] Sending image: {os.path.basename(image_path)}")
            files['image'] = (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
        
        # Prepare query if provided
        if user_query:
            print(f"[CLIENT] Sending query: '{user_query}'")
            data['query'] = user_query
        
        # Add JWT token for authentication
        headers = {}
        if token:
            print(f"[CLIENT] Using JWT Token: {token[:20]}...")
            headers['Authorization'] = f'Bearer {token}'
        
        # Make request to BACKEND API (not Vision Service directly)
        print(f"[CLIENT] Calling Backend API...")
        response = requests.post(
            PREDICT_URL,
            files=files if files else None,
            data=data if data else None,
            headers=headers,
            timeout=200  # Long timeout since Backend → Vision Service → LLM Service
        )
        
        # Close file handles
        for f in files.values():
            if hasattr(f[1], 'close'):
                f[1].close()
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                # Extract data from Backend response structure
                data = result.get('data', {})
                
                # Display vision result if available
                if data.get('vision_result'):
                    vision = data['vision_result']
                    print(f"[CLIENT] Vision predicted: {vision['disease']} (confidence: {vision['confidence']:.4f})")
                
                # Display AI response
                if data.get('ai_response'):
                    print(f"[CLIENT] AI response received (length: {len(data['ai_response'])} chars)")
                
                # Display save status
                if data.get('saved_to_db'):
                    print(f"[CLIENT] Saved to database (ID: {data.get('record_id')})")
                
                return result
            else:
                print(f"[CLIENT] Error: {result.get('error')}")
                return result
        else:
            print(f"[CLIENT] HTTP Error {response.status_code}: {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"[CLIENT] Exception: {e}")
        return {"success": False, "error": str(e)}

def process_test_case(image_filename, user_question):
    """
    Process a complete test case
    
    Args:
        image_filename: Image file name (None for text-only query)
        user_question: User question text (None to auto-generate from image)
    """
    print("\n" + "=" * 80)
    print(f"TEST CASE:")
    if image_filename:
        print(f"  IMAGE: {image_filename}")
    if user_question:
        print(f"  QUERY: {user_question}")
    if not image_filename and not user_question:
        print("  [ERROR] At least image or query must be provided")
        return
    print("=" * 80)
    
    # Prepare image path
    image_path = None
    if image_filename:
        image_path = os.path.join(TEST_IMAGE_FOLDER, image_filename)
        if not os.path.exists(image_path):
            print(f"\n[ERROR] Image not found: {image_path}")
            return
    
    # Call Backend API (which forwards to Vision Service → LLM) with JWT token
    result = call_vision_service(image_path, user_question, JWT_TOKEN)
    
    if result.get('success'):
        print("\n" + "-" * 80)
        print("FINAL RESULT:")
        print("-" * 80)
        
        # Extract data from Backend response
        data = result.get('data', {})
        
        # Show vision result
        if data.get('vision_result'):
            vision = data['vision_result']
            print(f"Vision Analysis:")
            print(f"  Disease: {vision['disease']}")
            print(f"  Confidence: {vision['confidence']:.2%}")
            print()
        
        # Show AI response
        if data.get('ai_response'):
            print(f"AI Response:")
            print(data['ai_response'])
        
        # Show database status
        if data.get('saved_to_db'):
            print(f"\n✅ Saved to database (ID: {data.get('record_id')})")
        
        print("-" * 80)
    else:
        print(f"\n[ERROR] Request failed: {result.get('error')}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("TESTING BACKEND → VISION → LLM SERVICES INTEGRATION")
    print("NEW ARCHITECTURE: Client → Backend API → Vision Service → LLM Service")
    print("=" * 80)
    
    # Step 1: Login to get JWT token
    if not login_user():
        print("\n[ERROR] Authentication failed. Cannot proceed.")
        print("💡 Make sure:")
        print("   1. Backend is running: cd ThesisApp/ThesisBE && npm run dev")
        print("   2. User is registered: python test_api.py")
        return
    
    # Step 2: Check services health
    print("\n[HEALTH CHECK] Checking services...")
    backend_healthy = test_service_health("Backend API", f"{BACKEND_URL}/health")
    vision_healthy = test_service_health("Vision Service", "http://localhost:5003/health")
    llm_healthy = test_service_health("LLM Service", "http://localhost:5002/health")
    
    if not backend_healthy:
        print("\n[ERROR] Backend API is not healthy. Cannot proceed.")
        return
    
    if not vision_healthy:
        print("\n[WARNING] Vision Service is not healthy. Image-based tests will fail.")
    
    if not llm_healthy:
        print("\n[ERROR] LLM Service is not healthy. Cannot proceed.")
        return
    
    # Step 3: Get monthly statistics
    print("\n[INFO] Getting monthly disease statistics...")
    get_monthly_stats()
    
    # Step 4: Check test image folder
    if not os.path.exists(TEST_IMAGE_FOLDER):
        print(f"\n[WARNING] Test image folder not found: {TEST_IMAGE_FOLDER}")
        print("Image-based tests will fail.")
    
    # Step 5: Run test cases with authentication
    print("\n[INFO] Running test cases with JWT authentication...")
    for i, (image_filename, user_question) in enumerate(TEST_CASES, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# TEST CASE {i}/{len(TEST_CASES)}")
        print(f"{'#' * 80}")
        
        process_test_case(image_filename, user_question)
        
        # Wait between tests
        if i < len(TEST_CASES):
            print("\n[INFO] Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # Step 6: Get updated monthly statistics after tests
    print("\n\n" + "=" * 80)
    print("GETTING UPDATED MONTHLY STATISTICS AFTER TESTS")
    print("=" * 80)
    get_monthly_stats()
    
    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
