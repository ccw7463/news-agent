import asyncio
from dotenv import load_dotenv
import os
import pytz
import json
from datetime import datetime
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from typing import List, Any, Dict
from src.modules.implementations.logger_manager import LoggerManager

load_dotenv()

# Initialize time
kst = pytz.timezone('Asia/Seoul')
time_now = datetime.now(kst)
time_now_str = time_now.strftime("%Y%m%d_%H%M%S")

# Initialize LoggerManager
logger_manager = LoggerManager()
logger = logger_manager.setup_logger(
    name="news_scraper", 
    log_file=f"./log/{time_now_str}_news_scraper.log",
    console_output=True
)

class NewsAgentState(MessagesState):
    articles: List[Dict[str, Any]]
    
async def main():
    
    # Header
    logger.info("=" * 80)
    logger.info("🚀 AI News Search with LangGraph & FastMCP")
    logger.info("Powered by Google RSS and OpenAI GPT-4o-mini")
    logger.info("=" * 80)
    
    # Initialize model
    model = init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", ""))
    
    # Initialize MCP client
    logger.info("📡 Connecting to Google RSS FastMCP server...")
    
    client = MultiServerMCPClient(
        {
            "google-rss-mcp": {
                "command": "python",
                "args": ["./src/modules/mcp_servers/server.py"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    
    logger.info("✅ FastMCP server connected successfully!")
    
    # Display available tools
    logger.info("🔧 Available Google News RSS FastMCP Tools:")
    for tool in tools:
        logger.info(f"   • {tool.name}: {tool.description}")
    
    # Build LangGraph
    logger.info("🔨 Building LangGraph workflow...")
    
    def call_model(state: NewsAgentState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    def summary_node(state: NewsAgentState):
        
        # Extract article data from ToolMessage
        articles = []
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        articles.extend(data)
                except Exception as e:
                    logger.error(f"❌ ToolMessage JSON decode error: {e}")

        # Extract user's question
        user_query = None
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if not user_query:
            user_query = "No user question provided. Please summarize the article content in 5 sentences or less based on the article title."

        # Generate summary
        logger.info(f"Generating AI summaries for {len(articles)} articles...")
        summary_success_count = 0
        for idx, article in enumerate(articles):
            content = article.get("article_content", "")
            title = article.get("article_title", "")
            prompt = [
                SystemMessage(content="You are a news summarization expert. Please summarize the article content in 5 sentences or less based on the user's question and article title."),
                HumanMessage(content=f"Question: {user_query}\nArticle Title: {title}\nArticle Content: {content}\nSummary:")
            ]
            try:
                summary = model.invoke(prompt).content.strip()
                article["summary"] = summary
                logger.info(f"   - Article {idx+1}: {title[:60]}...")
                summary_success_count += 1
            except Exception as e:
                logger.error(f"❌ Summary generation failed for article {idx+1}: {e}")
                article["summary"] = "Summary generation failed"
        state["articles"] = articles
        logger.info(f"✅ AI summary generation completed: {summary_success_count}/{len(articles)} articles")
        return state

    def final_answer_node(state: NewsAgentState):
        # Get summary list
        articles = state.get("articles", [])
        summaries = [f"- {a.get('summary', '')}" for a in articles if a.get('summary', '') and a.get('summary', '') != "Summary generation failed"]
        summary_text = "\n".join(summaries)

        # Extract user's question
        user_query = None
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if not user_query:
            user_query = "No user question provided."

        # Prompt (customize as desired)
        prompt = [
            SystemMessage(content="Below are summaries of multiple news articles. Please provide a comprehensive summary based on the user's question."),
            HumanMessage(content=f"Question: {user_query}\nArticle Summaries:\n{summary_text}\nFinal Summary:")
        ]
        response = model.invoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(NewsAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summary_node", summary_node)
    builder.add_node("final_answer_node", final_answer_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model",tools_condition)
    builder.add_edge("tools", "summary_node")
    builder.add_edge("summary_node", "final_answer_node")
    graph = builder.compile()
    logger.info("✅ LangGraph workflow built successfully!")
    
    # Test 1: What news topics are available?
    logger.info("")
    logger.info("-" * 50)
    logger.info("🔍 QUESTION 1: What news topics are available?")
    try:
        # Find get_available_topics tool
        topics_tool = None
        for tool in tools:
            if tool.name == "get_available_topics":
                topics_tool = tool
                break
        
        if topics_tool:
            topics_result = await topics_tool.ainvoke({})
            
            topics_result = json.loads(topics_result)
            logger.info(f"Available topics: {', '.join(topics_result)}")
        else:
            logger.error("❌ get_available_topics tool not found")
    except Exception as e:
        logger.error(f"❌ Error getting topics: {str(e)}")
    
    # Test 2: Get 3 recent news from a specific topic
    logger.info("")
    logger.info("-" * 50)
    logger.info("🔍 QUESTION 2: Get 3 recent technology news articles")
    
    try:
        # Find search_specific_topic_news tool
        topic_news_tool = None
        for tool in tools:
            if tool.name == "search_specific_topic_news":
                topic_news_tool = tool
                break
        
        if topic_news_tool:
            topic_result = await topic_news_tool.ainvoke({
                "topic": "technology",
                "max_results": 3,
                "max_length": 2000,
                "timeout": 10
            })
            topic_result = json.loads(topic_result)
            
            logger.info(f"Found {len(topic_result)} technology articles:")
            for i, article in enumerate(topic_result):
                if isinstance(article, dict):
                    title = article.get('article_title', 'No title')
                    url = article.get('article_url', 'No URL')
                    published = article.get('article_published', 'No date')
                    logger.info(f"   {i+1}. {title}")
                    logger.info(f"      📅 {published} | 🔗 {url}")
                else:
                    logger.warning(f"   {i+1}. Invalid article format: {type(article)}")
        else:
            logger.error("❌ search_specific_topic_news tool not found")
    except Exception as e:
        logger.error(f"❌ Error getting topic news: {str(e)}")
    
    # Test 3: Get 6 latest AI-related news articles using LangGraph
    logger.info("")
    logger.info("-" * 50)
    logger.info("🔍 QUESTION 3: Get 6 latest AI-related news articles with AI summary")
    
    question = "Find 6 latest AI-related news articles"
    logger.info("Running LangGraph workflow...")
    
    try:
        response = await graph.ainvoke({"messages": question})
        
        # Display results
        messages = response["messages"]
        if messages and hasattr(messages[-1], 'content') and messages[-1].content:
            result_content = messages[-1].content
            logger.info("")
            logger.info("-" * 50)
            logger.info("📋 AI News Summary:")
            logger.info(result_content)
        else:
            logger.error("❌ No response found")
            
    except Exception as e:
        logger.error(f"❌ Error during search: {str(e)}")
    logger.info("-" * 50)
    
    # Footer
    logger.info("")
    logger.info("🎉 All three questions completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
