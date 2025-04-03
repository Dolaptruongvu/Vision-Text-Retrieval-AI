import os
import asyncio
import json
import uuid
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel
from crawl4ai.chunking_strategy import OverlappingWindowChunking
load_dotenv()

class RagBlock(BaseModel):
    tags: list[str]
    content: list[str]

async def main():
    gemini_token = os.getenv("GEMINI_API_KEY")
    if not gemini_token:
        raise ValueError("Missing GEMINI_API_KEY in .env file.")
    chunking_strategy = OverlappingWindowChunking(window_size=800, overlap=200)
    # Cấu hình LLM strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash-exp",
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

    config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        css_selector=".detail-content", # Focus on this
    )

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        result = await crawler.arun(
            url="https://bachnong.vn/tin-tuc/tin-tuc-nong-nghiep/top-9-benh-cay-trong-thuong-gap-phai-va-cach-khac-phuc.html",
            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return
        print("Raw result")
        print(result)
        # Parse extracted content JSON
        try:
            blocks = json.loads(result.extracted_content)
        except json.JSONDecodeError as e:
            print("JSON decode failed:", e)
            return
        print("Raw from LLM")
        print(blocks)
        print("\n✅ --- Extracted RAG-ready Markdown ---\n")
        with open("./data/rag_blocks_milvus.jsonl", "a", encoding="utf-8") as f:
            for block in blocks:
                try:
                    parsed = RagBlock(**block)
                    markdown_text = "\n".join(parsed.content)
                    entry = {
                        "id": str(uuid.uuid4()),
                        "content": markdown_text,
                        "metadata": {"tags": parsed.tags}
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    print(f"### {' / '.join(parsed.tags)}\n{markdown_text}\n")
                except Exception as e:
                    print(f"Invalid block: {block}")
                    print("Error:", e)
        print("\n📊 --- Token Usage ---")
        llm_strategy.show_usage()

if __name__ == "__main__":
    asyncio.run(main())
