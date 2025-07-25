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
from langgraph.graph import StateGraph, MessagesState, START, END
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
    console_output=False
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

    builder = StateGraph(NewsAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summary_node", summary_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model",tools_condition)
    builder.add_edge("tools", "summary_node")
    builder.add_edge("summary_node", END)
    graph = builder.compile()
    logger.info("✅ LangGraph workflow built successfully!")
    
    # Get 6 latest AI-related news articles using LangGraph
    logger.info("")
    logger.info("-" * 50)
    logger.info("🔍 QUESTION : 최신 AI 뉴스 6가지 정도 알려주세요.")
    
    question = "최신 AI 뉴스 16가지 정도 알려주세요."
    logger.info("Running LangGraph workflow...")
    
    try:
        response = await graph.ainvoke({"messages": question})            
    except Exception as e:
        logger.error(f"❌ Error during search: {str(e)}")
    messages = response["articles"]
    for msg in messages:
        logger.info(msg)
    logger.info("-" * 50)
    
    # Footer
    logger.info("")
    logger.info("🎉 All three questions completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
