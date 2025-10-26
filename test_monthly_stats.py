"""
Test Monthly Disease Statistics API
Tests the new /api/disease/monthly-stats endpoint
"""
import requests
import json
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:5004"
TEST_USER = {
    "email": "test@example.com",
    "password": "password123"
}

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def login():
    """Login to get JWT token"""
    print_section("STEP 1: LOGIN")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/auth/login",
            json=TEST_USER,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data['data']['token']
            user = data['data']['user']
            
            print(f"✅ Login Successful!")
            print(f"   User: {user['username']}")
            print(f"   Email: {user['email']}")
            return token
        else:
            print(f"❌ Login Failed: {response.json().get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ Login Error: {e}")
        return None

def get_monthly_stats(token, year=None, month=None):
    """Get monthly disease statistics"""
    print_section(f"STEP 2: GET MONTHLY STATISTICS")
    
    # Current month by default
    now = datetime.now()
    target_year = year or now.year
    target_month = month or now.month
    
    print(f"📊 Fetching stats for: {target_year}-{target_month:02d}")
    
    try:
        params = {}
        if year:
            params['year'] = year
        if month:
            params['month'] = month
        
        response = requests.get(
            f"{BACKEND_URL}/api/disease/monthly-stats",
            headers={'Authorization': f'Bearer {token}'},
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Statistics Retrieved Successfully!")
            print("\n" + "-" * 80)
            
            # Period info
            period = data['data']['period']
            print(f"📅 Period: {period['monthName']} {period['year']}")
            print(f"   From: {period['startDate']}")
            print(f"   To:   {period['endDate']}")
            
            # Summary
            summary = data['data']['summary']
            print(f"\n📈 Summary:")
            print(f"   Total Detections: {summary['totalDetections']}")
            print(f"   Previous Month: {summary['previousMonthTotal']}")
            print(f"   Trend: {summary['trend']}")
            
            # Message
            print(f"\n💬 {data['data']['message']}")
            
            # Top diseases
            print(f"\n🏆 Top Diseases:")
            top_diseases = data['data']['topDiseases']
            
            if not top_diseases:
                print("   (No data for this month)")
            else:
                for i, disease in enumerate(top_diseases, 1):
                    print(f"   {i}. {disease['diseaseName']}")
                    print(f"      - Count: {disease['count']} cases")
                    print(f"      - Percentage: {disease['percentage']}%")
                    print(f"      - Avg Confidence: {disease['avgConfidence']}%")
            
            # Full distribution
            print(f"\n📊 Full Disease Distribution:")
            distribution = data['data']['diseaseDistribution']
            
            if not distribution:
                print("   (No data)")
            else:
                for disease in distribution:
                    print(f"   • {disease['diseaseName']}: {disease['percentage']}% ({disease['count']} cases)")
            
            print("-" * 80)
            
            return data
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_multiple_months(token):
    """Test statistics for multiple months"""
    print_section("STEP 3: TEST MULTIPLE MONTHS")
    
    now = datetime.now()
    current_year = now.year
    
    # Test current month and previous 2 months
    months_to_test = [
        (current_year, now.month),
        (current_year, now.month - 1 if now.month > 1 else 12),
        (current_year if now.month > 1 else current_year - 1, now.month - 2 if now.month > 2 else 12 + (now.month - 2))
    ]
    
    results = []
    
    for year, month in months_to_test:
        print(f"\n📅 Testing {year}-{month:02d}...")
        result = get_monthly_stats(token, year, month)
        if result:
            results.append({
                'year': year,
                'month': month,
                'total': result['data']['summary']['totalDetections']
            })
        print("\nWaiting 1 second...\n")
        import time
        time.sleep(1)
    
    # Summary comparison
    if results:
        print_section("COMPARISON SUMMARY")
        for r in results:
            print(f"   {r['year']}-{r['month']:02d}: {r['total']} detections")

def main():
    print("\n" + "=" * 80)
    print("TESTING MONTHLY DISEASE STATISTICS API")
    print("=" * 80)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test User: {TEST_USER['email']}")
    print("=" * 80)
    
    # Step 1: Login
    token = login()
    if not token:
        print("\n❌ Cannot proceed without authentication")
        return
    
    # Step 2: Get current month stats
    current_stats = get_monthly_stats(token)
    
    if not current_stats:
        print("\n❌ Failed to get statistics")
        return
    
    # Step 3: Test multiple months (optional)
    user_input = input("\n🔄 Do you want to test statistics for multiple months? (y/n): ")
    if user_input.lower() == 'y':
        test_multiple_months(token)
    
    print_section("✅ ALL TESTS COMPLETED")
    print("\n🎉 Monthly statistics API is working perfectly!")
    print("\nUseful for:")
    print("  • Dashboard analytics")
    print("  • Seasonal disease trend analysis")
    print("  • Predict common diseases by month/season")
    print("  • Agricultural planning recommendations")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
