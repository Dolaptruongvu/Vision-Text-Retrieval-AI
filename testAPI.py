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
VISION_SERVICE_URL = "http://localhost:5003/predict"

# Test image folder - CHANGE THIS TO YOUR IMAGE FOLDER
TEST_IMAGE_FOLDER = r"c:\wd\ThesisLLM\ThesisVision\realImage"

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

def call_vision_service(image_path=None, user_query=None):
    """
    Call Vision Service (which internally calls LLM Service)
    
    Args:
        image_path: Path to image file (optional)
        user_query: User question text (optional)
        Note: At least one must be provided
    
    Returns:
        {
            "success": bool,
            "vision_result": {
                "disease": str,
                "confidence": float
            },
            "ai_response": str,
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
        
        # Make request
        print(f"[CLIENT] Calling Vision Service...")
        response = requests.post(
            VISION_SERVICE_URL,
            files=files if files else None,
            data=data if data else None,
            timeout=200  # Long timeout since Vision Service calls LLM Service
        )
        
        # Close file handles
        for f in files.values():
            if hasattr(f[1], 'close'):
                f[1].close()
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                # Display vision result if available
                if result.get('vision_result'):
                    vision = result['vision_result']
                    print(f"[CLIENT] Vision predicted: {vision['disease']} (confidence: {vision['confidence']:.4f})")
                
                # Display AI response
                if result.get('ai_response'):
                    print(f"[CLIENT] AI response received (length: {len(result['ai_response'])} chars)")
                
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
    
    # Call Vision Service (which handles LLM internally)
    result = call_vision_service(image_path, user_question)
    
    if result.get('success'):
        print("\n" + "-" * 80)
        print("FINAL RESULT:")
        print("-" * 80)
        
        # Show vision result
        if result.get('vision_result'):
            vision = result['vision_result']
            print(f"Vision Analysis:")
            print(f"  Disease: {vision['disease']}")
            print(f"  Confidence: {vision['confidence']:.2%}")
            print()
        
        # Show AI response
        if result.get('ai_response'):
            print(f"AI Response:")
            print(result['ai_response'])
        
        print("-" * 80)
    else:
        print(f"\n[ERROR] Request failed: {result.get('error')}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("TESTING VISION + LLM SERVICES INTEGRATION")
    print("=" * 80)
    
    # Check services health
    print("\n[HEALTH CHECK] Checking services...")
    vision_healthy = test_service_health("Vision Service", "http://localhost:5003/health")
    llm_healthy = test_service_health("LLM Service", "http://localhost:5002/health")
    
    if not vision_healthy:
        print("\n[WARNING] Vision Service is not healthy. Image-based tests will fail.")
    
    if not llm_healthy:
        print("\n[ERROR] LLM Service is not healthy. Cannot proceed.")
        return
    
    # Check test image folder
    if not os.path.exists(TEST_IMAGE_FOLDER):
        print(f"\n[WARNING] Test image folder not found: {TEST_IMAGE_FOLDER}")
        print("Image-based tests will fail.")
    
    # Run test cases
    print("\n[INFO] Running test cases...")
    for i, (image_filename, user_question) in enumerate(TEST_CASES, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# TEST CASE {i}/{len(TEST_CASES)}")
        print(f"{'#' * 80}")
        
        process_test_case(image_filename, user_question)
        
        # Wait between tests
        if i < len(TEST_CASES):
            print("\n[INFO] Waiting 2 seconds before next test...")
            time.sleep(2)
    
    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
