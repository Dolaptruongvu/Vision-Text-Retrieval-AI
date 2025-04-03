import os
import asyncio
import json
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig,
    CacheMode, LLMConfig
)
# Keep original imports
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy # Keep even if implicitly used
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.chunking_strategy import OverlappingWindowChunking
from crawl4ai.extraction_strategy import LLMExtractionStrategy
load_dotenv()

# Keep original schema definition
class RagBlock(BaseModel):
    tags: list[str]
    content: list[str]

async def main():

    # Keep original API key loading
    gemini_token = os.getenv("GEMINI_API_KEY")
    if not gemini_token:
        raise ValueError("Missing GEMINI_API_KEY in .env file.")


    # Keep original chunking strategy
    chunking_strategy = OverlappingWindowChunking(window_size=800, overlap=200)


    # Keep original LLM extraction strategy setup
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash-exp", # Keep original model
            api_token=gemini_token
        ),
        extraction_type="schema",
        schema=RagBlock.model_json_schema(),
        instruction = (
            "Tách nội dung bài viết thành các phần theo từng chủ đề rõ ràng để phục vụ hệ thống RAG. "
            "Trả về một danh sách các object JSON theo mẫu:\n"
            "{ \"tags\": [\"chủ đề cụ thể\"], \"content\": [\"nội dung markdown\"] }\n"
            "Không trả về chuỗi đơn hoặc nội dung ngoài schema. Không trả về quảng cáo, link hoặc javascript, điều hướng, bình luận, chuyên mục khác, liên hệ, giá vàng/ thị trường, các title chuyên ngành khác."
            "Tất cả tags và nội dung phải bằng tiếng Việt."
            "Quan trọng nhất là đừng trả về chuỗi JSON dạng string, chỉ trả về list object JSON chuẩn có schema như tôi đã nhắc như trên."
        ),
        apply_chunking=True,
        chunk_token_threshold=2048,
        input_format="markdown",
        chunking_strategy=chunking_strategy,
        extra_args={"temperature": 0.2, "max_tokens": 2048}
    )

    # Keep original deep crawl configuration components
    filter_chain = FilterChain([
        URLPatternFilter(patterns=["*/tin-tuc/tin-tuc-nong-nghiep/*.html"])
    ])


    keywords = [
        "benh", "sau", "nam", "virus", "vi-khuan",
        "nguyen-nhan", "ly-do", "tai-sao",
        "trieu-chung", "dau-hieu", "hien-tuong", "bieu-hien",
        "cach-chua", "dieu-tri", "giai-phap", "khac-phuc",
        "phong-ngua", "phong-tranh", "han-che", "kiem-soat",
        "xu-ly", "diet-tru", "phun-thuoc",
        "bao-ve", "an-toan",
        "gay-hai", "tac-hai", "anh-huong",
        "bi",
        "vang-la", "xoan-la", "rung-la", "chay-la", "dom-la",
        "thoi-re", "thoi-than", "thoi-nhun", "thoi-trai",
        "heo-ru", "heo-xanh", "chet-nhanh", "chet-cham",
        "kho-canh", "kho-qua", "thui-choi",
        "nut-qua", "nut-than", "bien-dang",
        "lep-hat", "rung-hoa", "rung-qua",
        "u-su", "sung",
        "phan-trang", "suong-mai", "kham",
        "soc",
        "con-trung", "sau-hai",
        "bo-tri", "ray-xanh", "sau-buom", "sau-cuon-la",
        "sau-duc-than", "sau-duc-qua",
        "nhen-do", "rep-sap", "ruoi-vang", "oc-sen",
        "tuyen-trung",
        "bo-canh-cung", "bo-xit",
        "thuoc-tru-sau", "thuoc-tru-benh", "thuoc-bao-ve-thuc-vat",
        "thuoc-nam", "thuoc-vi-khuan",
        "sinh-hoc", "huu-co",
        "phan-bon",
        "sau-benh", "dich-hai",
        "benh-cay", "benh-hai", "benh-nong-nghiep",
        "mua-vu", "thoi-tiet",
        "dat-trong", "gia-the", "dinh-duong",
        "giong", "khang-benh",
        "cham-soc", "ky-thuat", "meo"
    ]


    scorer = KeywordRelevanceScorer(
        keywords=keywords,
        weight=1
    )


    strategy = BestFirstCrawlingStrategy(
        max_depth=1,
        include_external=False,
        url_scorer=scorer,
        max_pages=10, # Keep original max_pages
        filter_chain=filter_chain
    )


    # Configure CrawlerRunConfig
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS, # Keep original cache mode
        css_selector=".news",  # Keep original selector (!! Please verify this selector for article content !!)
        verbose=True,
        word_count_threshold=50,
        stream=True # Changed from False to True to enable streaming results
    )

    # Initialize crawler
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        start_url = "https://bachnong.vn/tin-tuc/tin-tuc-nong-nghiep" # Keep original start URL
        print(f"Starting prioritized deep crawl from: {start_url} with streaming enabled.")

        # --- Process results using async for loop (due to stream=True) ---
        print("\n--- Extracting RAG blocks and appending to file ---")
        pages_processed_this_run = 0
        output_file = "./rawData/rag_blocks_milvus.jsonl" # Keep original output file

        # Open file in append mode ('a')
        with open(output_file, "a", encoding="utf-8") as f:
            # Use async for to iterate over results as they become available
            async for result in await crawler.arun(start_url, config=config):
                pages_processed_this_run += 1
                # Keep original print statements for title and URL
                title = result.metadata.get("title", "Khong co title") # Use original default value
                print(f"Title: {title}")
                print(f"URL: {result.url}")

                # Keep original processing logic for each result
                if result.success:
                    if result.extracted_content:
                        try:
                            # Basic cleaning for potential markdown code fences around JSON
                            content_str = result.extracted_content.strip()
                            if content_str.startswith("```json"): content_str = content_str[7:]
                            if content_str.endswith("```"): content_str = content_str[:-3]
                            content_str = content_str.strip()

                            # LLM might return a string containing JSON or a direct list
                            blocks = []
                            if isinstance(content_str, str): blocks = json.loads(content_str)
                            elif isinstance(content_str, list): blocks = content_str
                            else:
                                print(f"Unexpected extracted content type: {type(content_str)}. Skipping.")
                                continue

                            if not isinstance(blocks, list):
                                print(f"LLM did not return a list of blocks. Skipping.")
                                continue

                        except json.JSONDecodeError as e:
                            print("JSON decode failed:", e)
                            print(f"Raw content causing error: {result.extracted_content[:500]}...") # Print raw content on error
                            continue # Skip this result if JSON is invalid

                        # Keep original print statement for raw LLM output
                        print("Raw extracted content from LLM:")
                        print(blocks)
                        print("\n--- Extracted RAG-ready Markdown ---\n")
                        # Keep original block processing logic
                        for block in blocks:
                            try:
                                parsed = RagBlock(**block)
                                markdown_text = "\n".join(parsed.content).strip()
                                if not markdown_text: continue # Skip empty blocks

                                entry = {
                                    "id": str(uuid.uuid4()), # Keep original ID generation
                                    "content": markdown_text,
                                    # Keep original metadata structure for the entry
                                    "metadata": {"tags": parsed.tags, "title": title, "url": result.url}
                                }
                                # Write entry to the already opened file
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                # Keep original print statement for the processed block
                                print(f"### {' / '.join(parsed.tags)}\n{markdown_text}\n")
                            except Exception as e:
                                # Keep original error handling for invalid blocks
                                print("Invalid block:", block)
                                print("Error:", e)
                    else:
                         # Keep original message when no content is extracted
                        print("Crawled successfully, but no extracted content available.")
                else:
                    # Keep original message for crawl failure
                    print(f"Crawl failed: {result.error_message}")
                print("-" * 80) # Keep original separator
        # --- End async for loop ---

        print(f"\n--- Run Summary ---")
        print(f"Total pages processed in this run: {pages_processed_this_run}")
        # Note: Total blocks written is harder to track precisely outside the loop in streaming mode without extra variables

        # Keep original token usage display
        print("\n--- Token Usage ---")
        llm_strategy.show_usage()

# Keep original main execution block
if __name__ == "__main__":
    asyncio.run(main())