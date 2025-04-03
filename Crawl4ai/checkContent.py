import asyncio
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig,
    CacheMode
)
from dotenv import load_dotenv
load_dotenv()

async def run_scrape(url: str):
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector="#single_post_content",  # CSS selector để lấy nội dung chính xác
        verbose=True
    )

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        # Không dùng async for vì arun() trả về CrawlResultContainer, không phải async iterable
        result = await crawler.arun(url, config=config)

        print(f"\nURL: {result.url}")
        if result.success:
            print("\n=== Extracted Markdown Content ===")
            print(result.markdown)
        else:
            print("Failed to crawl:", result.error_message)

if __name__ == "__main__":
    test_url = "https://geopard.tech/blog/how-to-control-crop-diseases-with-smart-agriculture/"
    asyncio.run(run_scrape(test_url))
