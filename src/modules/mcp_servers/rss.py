
import feedparser
import requests
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import logging
from bs4 import BeautifulSoup
import re
import json
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RSSItem:
    """
    Data class representing an RSS item with title, link, and publication date.
    
    Attributes:
        title (str): The title of the RSS item
        link (str): The URL link to the RSS item
        published (Optional[datetime]): The publication date of the RSS item
        agency (Optional[str]): The news agency/source name
    """
    title: str
    link: str
    published: Optional[datetime] = None
    agency: Optional[str] = None

@dataclass
class RSSFeed:
    """
    Data class representing an RSS feed with metadata and items.
    
    Attributes:
        title (str): The title of the RSS feed
        link (str): The URL link to the RSS feed
        language (Optional[str]): The language of the RSS feed
        last_updated (Optional[datetime]): The last update time of the RSS feed
        items (List[RSSItem]): List of RSS items in the feed
    """
    title: str
    link: str
    language: Optional[str] = None
    last_updated: Optional[datetime] = None
    items: List[RSSItem] = None

    def __post_init__(self):
        """Initialize items list if not provided."""
        if self.items is None:
            self.items = []

class GoogleRSSTools:
    """
    Tool class for processing Google RSS feeds with support for multiple languages and regions.
    
    This class provides functionality to search Google News RSS feeds, extract article content,
    and handle Google News redirects to get actual article URLs.
    
    Attributes:
        session (aiohttp.ClientSession): HTTP session for making requests
        language (str): Language code for RSS feeds (e.g., 'en', 'ja', 'zh', 'ko')
        region (str): Region code for RSS feeds (e.g., 'US', 'JP', 'CN', 'KR')
        headers (Dict[str, str]): HTTP headers for requests
        timeout (int): Timeout in seconds for HTTP requests (default: 30)
    """
    
    def __init__(self, language: str = "ko", region: str = "KR", timeout: int = 10):
        """
        Initialize GoogleRSSTools with language and region settings.
        
        Args:
            language (str): Language code for RSS feeds
            region (str): Region code for RSS feeds
            timeout (int): Timeout in seconds for HTTP requests (default: 10)
        """
        self.session = None
        self.language = language
        self.region = region
        self.timeout = timeout
        
        # User-Agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def __aenter__(self):
        """
        Async context manager entry point.
        
        Returns:
            GoogleRSSTools: Self instance with initialized session
        """
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=timeout_config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.
        
        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any
        """
        if self.session:
            await self.session.close()
    
    async def search_news(self, query: str, max_results: int = 5, max_length: int = 5000) -> List[Dict[str, Any]]:
        """
        Search for news articles and extract their content in one operation.
        
        This method performs RSS search and content extraction in a single call,
        returning comprehensive article information including title, URL, content,
        and publication date.
        
        Args:
            query (str): Search query for news articles
            max_results (int): Maximum number of results to return (default: 5)
            max_length (int): Maximum length of article content in characters (default: 5000)
            
        Returns:
            List[Dict[str, Any]]: List of article information dictionaries containing:
                - article_title: Title of the article
                - article_url: URL of the article
                - article_image_url: URL of the main article image
                - article_published: Publication date
                - article_agency: News agency/source name
                - article_content: Extracted content of the article
                - user_query: Original search query
        """
        # Get all RSS items and process until we have enough successful results
        rss_items = await self._get_news_list(query)
        # logger.info(f"[Tool : search_news] 💡 Found {len(rss_items)} items for query '{query}'")
        
        # Create tasks for parallel processing
        tasks = []
        for item in rss_items[:max_results * 2]:  # Process up to 2x max_results to account for failures
            task = asyncio.create_task(
                self._process_single_article(item, max_length, query)
            )
            tasks.append(task)
        
        # Execute all tasks in parallel and collect results
        results = []
        processed_count = 0
        
        # Process results as they complete using asyncio.as_completed
        for completed_task in asyncio.as_completed(tasks):
            try:
                article_data = await completed_task
                if article_data:
                    results.append(article_data)
                    if len(results) >= max_results:
                        # Cancel remaining tasks if we have enough results
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
            except asyncio.TimeoutError:
                logger.warning("Timeout processing article")
                continue
            except Exception as e:
                logger.warning(f"Failed to process article: {str(e)}")
                continue
            
            processed_count += 1
        
        # logger.info(f"[Tool : search_news] ✅ Successfully processed {len(results)} out of {processed_count} attempted articles")
        return results
    
    async def _process_single_article(self, item: RSSItem, max_length: int, query: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to process a single article.
        
        Args:
            item (RSSItem): RSS item to process
            max_length (int): Maximum content length
            query (str): Original search query
            
        Returns:
            Optional[Dict[str, Any]]: Processed article data or None (if failed)
        """
        try:
            # Extract actual URL content and image with timeout
            article_data = await asyncio.wait_for(
                self._get_actual_url_content_and_image(rss_item=item, max_length=max_length),
                timeout=self.timeout
            )
            article_data['user_query'] = query
            return article_data
        except asyncio.TimeoutError:
            logger.warning(f"Timeout processing article: {item.title}")
            return None
        except Exception as e:
            logger.warning(f"Failed to process article '{item.title}': {str(e)}")
            return None
    
    async def _process_single_topic_article(self, item: RSSItem, max_length: int, topic: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to process a single topic article.
        
        Args:
            item (RSSItem): RSS item to process
            max_length (int): Maximum content length
            topic (str): Original topic category
            
        Returns:
            Optional[Dict[str, Any]]: Processed article data or None (if failed)
        """
        try:
            # Extract actual URL content and image with timeout
            article_data = await asyncio.wait_for(
                self._get_actual_url_content_and_image(rss_item=item, max_length=max_length),
                timeout=self.timeout
            )
            article_data['topic'] = topic
            return article_data
        except asyncio.TimeoutError:
            logger.warning(f"Timeout processing article: {item.title}")
            return None
        except Exception as e:
            logger.warning(f"Failed to process article '{item.title}': {str(e)}")
            return None
        
    async def search_specific_topic_news(self, topic: str, max_results: int = 5, max_length: int = 5000) -> List[Dict[str, Any]]:
        """
        Search for news articles from specific topics and extract their content.
        
        This method retrieves news from predefined topic categories such as top stories,
        world news, business, technology, etc.
        
        Args:
            topic (str): Topic category to search for. Supported topics:
                - "top": Top stories
                - "world": World news
                - "business": Business news
                - "technology": Technology news
                - "entertainment": Entertainment news
                - "sports": Sports news
                - "science": Science news
                - "health": Health news
            max_results (int): Maximum number of results to return (default: 5)
            max_length (int): Maximum length of article content in characters (default: 5000)
            
        Returns:
            List[Dict[str, Any]]: List of article information dictionaries containing:
                - article_title: Title of the article
                - article_url: URL of the article
                - article_image_url: URL of the main article image
                - article_published: Publication date
                - article_agency: News agency/source name
                - article_content: Extracted content of the article
                - topic: Original topic category
        Raises:
            ValueError: If the specified topic is not supported
        """
        # Get all RSS items for the given topic and process until we have enough successful results
        rss_items = await self._get_specific_topic_news_list(topic)
        # logger.info(f"[Tool : search_specific_topic_news] 💡 Found {len(rss_items)} items for topic '{topic}'")
        
        # Create tasks for parallel processing
        tasks = []
        for item in rss_items[:max_results * 2]:  # Process up to 2x max_results to account for failures
            task = asyncio.create_task(
                self._process_single_topic_article(item, max_length, topic)
            )
            tasks.append(task)
        
        # Execute all tasks in parallel and collect results
        results = []
        processed_count = 0
        
        # Process results as they complete using asyncio.as_completed
        for completed_task in asyncio.as_completed(tasks):
            try:
                article_data = await completed_task
                if article_data:
                    results.append(article_data)
                    if len(results) >= max_results:
                        # Cancel remaining tasks if we have enough results
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
            except asyncio.TimeoutError:
                logger.warning("Timeout processing article")
                continue
            except Exception as e:
                logger.warning(f"Failed to process article: {str(e)}")
                continue
            
            processed_count += 1
        
        # logger.info(f"[Tool : search_specific_topic_news] ✅ Successfully processed {len(results)} out of {processed_count} attempted articles for topic '{topic}'")
        return results
    
    async def _get_news_list(self, query: str) -> List[RSSItem]:
        """
        Perform Google News RSS search for a given query.
        
        Args:
            query (str): Search query for news articles
            
        Returns:
            List[RSSItem]: List of RSS items matching the search query
        """
        # Construct Google News RSS URL with language and region settings
        encoded_query = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}"
        
        try:
            feed = await self._fetch_rss_feed(rss_url)
            return feed.items
        except Exception as e:
            logger.error(f"Google News RSS search failed: {str(e)}")
            return []
    
    async def _get_specific_topic_news_list(self, topic: str = "top") -> List[RSSItem]:
        """
        Get news articles from specific topics from Google News.
        
        Args:
            topic (str): Topic category to retrieve news from (default: "top")
            
        Returns:
            List[RSSItem]: List of RSS items from the specified topic
            
        Raises:
            ValueError: If the specified topic is not supported
        """
        topic_urls = {
            "top": f"https://news.google.com/rss?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "world": f"https://news.google.com/rss/headlines/section/topic/WORLD?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "business": f"https://news.google.com/rss/headlines/section/topic/BUSINESS?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "technology": f"https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "entertainment": f"https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "sports": f"https://news.google.com/rss/headlines/section/topic/SPORTS?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "science": f"https://news.google.com/rss/headlines/section/topic/SCIENCE?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}",
            "health": f"https://news.google.com/rss/headlines/section/topic/HEALTH?hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}"
        }
        
        if topic not in topic_urls:
            raise ValueError(f"Unsupported topic: {topic}. Supported topics: {list(topic_urls.keys())}")

        try:
            feed = await self._fetch_rss_feed(topic_urls[topic])
            return feed.items
        except Exception as e:
            logger.error(f"Failed to get Google News topic: {str(e)}")
            return []

    async def _fetch_rss_feed(self, feed_url: str) -> RSSFeed:
        """
        Fetch and parse an RSS feed from the given URL.
        
        Args:
            feed_url (str): URL of the RSS feed to fetch
            
        Returns:
            RSSFeed: Parsed RSS feed object containing metadata and items
            
        Raises:
            Exception: If the RSS feed cannot be fetched or parsed
        """
        try:
            if self.session:
                async with self.session.get(feed_url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {feed_url}")
                    content = await response.text()
            else:
                response = requests.get(feed_url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                content = response.text
            
            # Parse with feedparser
            parsed = feedparser.parse(content)
            
            """
            # for debugging
            logger.info(f"Parsed RSS feed keys: {list(parsed.keys())}")
            logger.info(f"Feed info: {parsed.feed}")
            logger.info(f"Entries count: {len(parsed.entries) if hasattr(parsed, 'entries') else 'No entries'}")
            logger.info(f"Bozo: {parsed.bozo}")
            if parsed.bozo:
                logger.error(f"Feed parsing errors: {parsed.bozo_exception}")
            """
            
            # Extract feed metadata
            feed_info = parsed.feed
            feed = RSSFeed(
                title=feed_info.get('title', ''),
                link=feed_info.get('link', ''),
                language=feed_info.get('language'),
                last_updated=self._parse_date(feed_info.get('updated', ''))
            )
            
            # Parse items
            for entry in parsed.entries:
                title = self._clean_text(entry.get('title', ''))
                
                # extract news agency from title (title - news agency)
                agency = ""
                if " - " in title:
                    parts = title.split(" - ", 1)
                    if len(parts) == 2:
                        title = parts[0].strip()
                        agency = parts[1].strip()
                
                item = RSSItem(
                    title=title,
                    link=entry.get('link', ''),
                    published=self._parse_date(entry.get('published', '')),
                    agency=agency
                )
                feed.items.append(item)
            
            # logger.info(f"Retrieved {len(feed.items)} items from RSS feed '{feed.title}'.")
            return feed
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed: {feed_url} - {str(e)}")
            raise


    async def _get_actual_url_content_and_image(self, rss_item: RSSItem, max_length: int = 5000) -> Dict[str, Any]:
        """
        Extract actual article content from an RSS item.
        
        This method handles Google News redirects to get the actual article URL
        and then extracts the full article content from that URL.
        
        Args:
            rss_item (RSSItem): RSS item containing article information
            max_length (int): Maximum length of article content in characters
            
        Returns:
            Dict[str, Any]: Dictionary containing article information:
                - article_title: Title of the article
                - article_url: Actual URL of the article (after redirect resolution)
                - article_image_url: URL of the main article image
                - article_published: Publication date
                - article_agency: News agency/source name
                - article_content: Extracted content of the article
        """
        # Handle Google News redirect to get actual article URL
        article_url = await self._extract_actual_url(rss_item.link)
        
        # Extract actual article content
        article_data = await self._extract_actual_article_content(article_url, max_length)
        
        # Extract image from the actual article URL
        article_image_url = await self._extract_image_from_html(article_url)
        
        return {
            'article_title': rss_item.title,
            'article_url': article_url,
            'article_image_url': article_image_url,
            'article_published': rss_item.published,
            "article_agency": rss_item.agency,
            'article_content': article_data.get('article_content', '')
        }

    async def _extract_actual_url(self, google_news_url: str) -> str:
        """
        Extract the actual article URL from a Google News URL.
        
        Google News URLs are redirects that need to be resolved to get the actual
        article URL. This method handles the multi-step process of extracting
        the real URL from Google News redirects.
        
        Args:
            google_news_url (str): Google News RSS link that needs to be resolved
            
        Returns:
            str: Actual article URL after resolving Google News redirect
        """
        try:
            if not self.session:
                resp = requests.get(google_news_url, headers=self.headers, timeout=self.timeout)
            else:
                async with self.session.get(google_news_url) as resp:
                    if resp.status != 200:
                        return google_news_url
                    resp = await resp.text()
                    resp = type('Response', (), {'text': resp})()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            c_wiz_element = soup.select_one('c-wiz[data-p]')
            
            if not c_wiz_element:
                logger.error("c-wiz[data-p] element not found")
                return google_news_url
            
            data_p = c_wiz_element.get('data-p')
            if not data_p:
                logger.error("data-p attribute not found")
                return google_news_url
    
            obj = json.loads(data_p.replace('%.@.', '["garturlreq",'))
            payload = {
                'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
            }        
            headers = {
                'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'user-agent': self.headers['User-Agent'],
            }
            
            url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
            
            if not self.session:
                response = requests.post(url, headers=headers, data=payload, timeout=self.timeout)
                response_text = response.text
            else:
                async with self.session.post(url, headers=headers, data=payload) as response:
                    if response.status != 200:
                        return google_news_url
                    response_text = await response.text()
            
            # Step 5: Extract actual URL from response
            array_string = json.loads(response_text.replace(")]}'", ""))[0][2]
            article_url = json.loads(array_string)[1]
            return article_url
            
        except Exception as e:
            logger.error(f"Failed to resolve Google News redirect: {str(e)}")
            return google_news_url
    
    async def _extract_actual_article_content(self, url: str, max_length: int = 5000) -> Dict[str, Any]:
        """
        Extract article content from the actual article URL.
        
        This method uses LangChain's AsyncHtmlLoader and Html2TextTransformer
        to extract and clean article content from web pages.
        
        Args:
            url (str): Actual article URL to extract content from
            max_length (int): Maximum length of article content in characters
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - article_content: Extracted and cleaned article content
        """
        try:
            # Create a custom loader with timeout
            loader = AsyncHtmlLoader(url)
            # Set timeout for the loader
            docs = await asyncio.wait_for(loader.aload(), timeout=self.timeout)
            
            if not docs:
                return {'article_content': ''}
        
            html2text = Html2TextTransformer()
            docs = html2text.transform_documents(docs, metadata_type="html")
            
            # Separate title and body content
            full_content = docs[0].page_content
            
            # Clean article text
            article_content = self._clean_text(full_content)
            
            # Limit text length
            if len(article_content) > max_length:
                article_content = article_content[:max_length] + "..."
            
            return {'article_content': article_content}
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout extracting article content from {url}")
            return {'article_content': ''}
        except Exception as e:
            logger.error(f"Failed to extract article content from {url}: {str(e)}")
            return {'article_content': ''}

    async def _extract_image_from_html(self, url: str) -> str:
        """
        Extract the main image URL from an article's HTML page.
        
        This method looks for images in the following order:
        1. Open Graph og:image meta tag
        2. Twitter Card twitter:image meta tag
        3. Schema.org image markup
        4. First large image in the article body
        
        Args:
            url (str): URL of the article to extract image from
            
        Returns:
            str: URL of the main article image, or empty string if not found
        """
        try:
            if not self.session:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                html_content = response.text
            else:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        return ""
                    html_content = await response.text()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 1. Try Open Graph og:image
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
            
            # 2. Try Twitter Card twitter:image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                return twitter_image['content']
            
            # 3. Try Schema.org image
            schema_image = soup.find('meta', attrs={'itemprop': 'image'})
            if schema_image and schema_image.get('content'):
                return schema_image['content']
            
            # 4. Look for JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Check for image in various schema formats
                        image_url = self._extract_image_from_json_ld(data)
                        if image_url:
                            return image_url
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # 5. Find the first large image in the article body
            # Look for images in common article containers
            article_selectors = [
                'article', '.article', '.post', '.entry', '.content',
                '[role="main"]', '.main-content', '.story-body'
            ]
            
            for selector in article_selectors:
                article_container = soup.select_one(selector)
                if article_container:
                    # Find images with reasonable size (likely to be main image)
                    images = article_container.find_all('img')
                    for img in images:
                        src = img.get('src') or img.get('data-src')
                        if src:
                            # Check if it's a relative URL and make it absolute
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                from urllib.parse import urljoin
                                src = urljoin(url, src)
                            elif not src.startswith('http'):
                                from urllib.parse import urljoin
                                src = urljoin(url, src)
                            
                            # Skip small images, icons, and ads
                            if self._is_valid_article_image(img, src):
                                return src
            
            # 6. Fallback: find any large image on the page
            all_images = soup.find_all('img')
            for img in all_images:
                src = img.get('src') or img.get('data-src')
                if src and self._is_valid_article_image(img, src):
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        from urllib.parse import urljoin
                        src = urljoin(url, src)
                    elif not src.startswith('http'):
                        from urllib.parse import urljoin
                        src = urljoin(url, src)
                    return src
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to extract image from {url}: {str(e)}")
            return ""
    
    def _extract_image_from_json_ld(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract image URL from JSON-LD structured data.
        
        Args:
            data (Dict[str, Any]): JSON-LD data object
            
        Returns:
            Optional[str]: Image URL if found, None otherwise
        """
        # Check for image property
        if 'image' in data:
            image_data = data['image']
            if isinstance(image_data, str):
                return image_data
            elif isinstance(image_data, dict) and 'url' in image_data:
                return image_data['url']
            elif isinstance(image_data, list) and len(image_data) > 0:
                first_image = image_data[0]
                if isinstance(first_image, str):
                    return first_image
                elif isinstance(first_image, dict) and 'url' in first_image:
                    return first_image['url']
        
        # Check for mainEntity (common in Article schema)
        if 'mainEntity' in data:
            main_entity = data['mainEntity']
            if isinstance(main_entity, dict):
                return self._extract_image_from_json_ld(main_entity)
        
        # Check for @graph (multiple entities)
        if '@graph' in data:
            graph = data['@graph']
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict):
                        image_url = self._extract_image_from_json_ld(item)
                        if image_url:
                            return image_url
        
        return None
    
    def _is_valid_article_image(self, img_tag, src: str) -> bool:
        """
        Check if an image is likely to be a valid article image.
        
        Args:
            img_tag: BeautifulSoup img tag
            src (str): Image source URL
            
        Returns:
            bool: True if image is likely to be a valid article image
        """
        # Skip small images, icons, and ads
        width = img_tag.get('width')
        height = img_tag.get('height')
        
        if width and height:
            try:
                w, h = int(width), int(height)
                if w < 200 or h < 200:  # Too small to be main image
                    return False
            except ValueError:
                pass
        
        # Skip common ad/icon patterns
        skip_patterns = [
            'ad', 'ads', 'banner', 'icon', 'logo', 'avatar', 'thumbnail',
            'sponsor', 'promo', 'button', 'social', 'share', 'facebook',
            'twitter', 'instagram', 'youtube', 'play', 'pause', 'close'
        ]
        
        src_lower = src.lower()
        for pattern in skip_patterns:
            if pattern in src_lower:
                return False
        
        # Skip data URIs and very short URLs
        if src.startswith('data:') or len(src) < 10:
            return False
        
        return True

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse various date formats into datetime objects.
        
        This method attempts to parse date strings using multiple formats,
        including RSS standard formats and common variations.
        
        Args:
            date_str (str): Date string to parse
            
        Returns:
            Optional[datetime]: Parsed datetime object or None if parsing fails
        """
        if not date_str:
            return None
        
        # Let feedparser attempt automatic parsing
        try:
            parsed_date = feedparser._parse_date(date_str)
            if parsed_date:
                return datetime.fromtimestamp(parsed_date, tz=timezone.utc)
        except:
            pass
        
        # Manual parsing attempts with common date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
         
    def _clean_text(self, text: str) -> str:
        """
        Remove HTML tags and clean text content.
        
        This method performs comprehensive text cleaning including:
        - HTML tag removal
        - HTML entity decoding
        - Whitespace normalization
        - Special character filtering
        
        Args:
            text (str): Raw text content to clean
            
        Returns:
            str: Cleaned text content
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        text = text.strip()
        
        # Handle HTML entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#39;', "'", text)
        
        # Remove unnecessary characters
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'[^\w\s\-.,!?;:()가-힣]', '', text)
        
        # Clean up consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def to_dict(self, obj) -> Dict[str, Any]:
        """
        Convert RSSFeed or RSSItem objects to dictionaries.
        
        This method handles the conversion of dataclass objects to dictionaries,
        including proper datetime serialization.
        
        Args:
            obj: RSSFeed or RSSItem object to convert
            
        Returns:
            Dict[str, Any]: Dictionary representation of the object
        """
        if isinstance(obj, (RSSFeed, RSSItem)):
            result = asdict(obj)
            # Convert datetime objects to strings
            for key, value in result.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
            return result
        return obj
