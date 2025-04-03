import requests
import time

class Crawl4AiTester:
    def __init__(self, base_url: str = "http://localhost:11235", api_token: str = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

    def submit_and_wait(self, request_data: dict, timeout: int = 300) -> dict:
        # Submit crawl job
        response = requests.post(f"{self.base_url}/crawl", json=request_data, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"❌ Failed to submit task: {response.status_code} - {response.text}")
        
        task_id = response.json().get("task_id")
        print(f"📌 Task ID: {task_id}")

        # Poll until completion
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"⏰ Task {task_id} timed out after {timeout} seconds")

            result = requests.get(f"{self.base_url}/task/{task_id}", headers=self.headers)
            data = result.json()
            status = data.get("status")

            print(f"⏳ Status: {status}")
            if status == "completed":
                return data
            elif status == "failed":
                raise Exception(f"❌ Task failed: {data}")
            elif status == "error":
                raise Exception(f"⚠️ Server error: {data}")
            
            time.sleep(2)

def test_deployment():
    api_token = "mysupersecrettoken"
    tester = Crawl4AiTester(api_token=api_token)

    request = {
        "urls": "https://vnexpress.net/an-do-song-co-nguy-co-nhiem-san-la-gan-4868061.html",
        "priority": 10
    }

    result = tester.submit_and_wait(request)
    print("✅ Crawl completed successfully!")
    markdown = result["result"]["markdown"]
    print("📄 Markdown preview:\n", markdown[:30000])

if __name__ == "__main__":
    test_deployment()
