import os
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=".content",
    )

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        result = await crawler.arun(
            url="https://nongnghiep.vn/muong-khuong-chuyen-doi-co-cau-cay-trong-de-but-pha-d745249.html",
            config=config
        )

        if result.success:
            print("--- Markdown content from '.content' section ---\n")
            print(result.markdown[:10000])
        else:
            print("Crawl failed:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
