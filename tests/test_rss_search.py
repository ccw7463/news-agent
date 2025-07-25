import asyncio
import sys
import os

# RSS 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'news-agent', 'src'))

from src.modules.mcp_servers.rss import GoogleRSSTools

async def test_search_news():
    """search_news 함수를 테스트하는 간단한 함수"""
    
    # 테스트할 검색어들
    test_queries = [
        "인공지능",
        "코딩",
        "파이썬"
    ]
    
    print("🔍 RSS 뉴스 검색 테스트 시작...\n")
    
    async with GoogleRSSTools(language="ko", region="KR") as rss_tools:
        for query in test_queries:
            print(f"📰 검색어: '{query}'")
            print("-" * 50)
            
            try:
                # search_news 함수 호출
                results = await rss_tools.search_news(
                    query=query,
                    max_results=3,  # 최대 3개 결과
                    max_length=1000  # 최대 1000자
                )
                
                print(f"✅ 검색 결과: {len(results)}개 기사 발견\n")
                
                # 결과 출력
                for i, article in enumerate(results, 1):
                    print(f"📄 기사 {i}:")
                    print(f"   제목: {article.get('article_title', 'N/A')}")
                    print(f"   URL: {article.get('article_url', 'N/A')}")
                    print(f"   이미지: {article.get('article_image_url', 'N/A')}")
                    print(f"   발행일: {article.get('article_published', 'N/A')}")
                    
                    # 내용 미리보기 (처음 200자)
                    content = article.get('article_content', '')
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"   내용 미리보기: {preview}")
                    
                    print()
                
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}\n")
            
            print("=" * 60)
            print()

if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(test_search_news()) 