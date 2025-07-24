import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import time
from src.modules.implementations.logger_manager import LoggerManager


### TODO 파일이름 날짜 추가필요
### TODO 그래프 구체화 필요.
### TODO 출력양식 수정필요

load_dotenv()

# LoggerManager 초기화
logger_manager = LoggerManager()
logger = logger_manager.setup_logger(
    name="news_scraper", 
    log_file="./log/news_scraper.log",
    console_output=False
)

async def main():
    
    # Header
    logger.info("🤖 AI News Search with LangGraph & FastMCP")
    logger.info("Powered by Google RSS and OpenAI GPT-4o-mini")
    
    # Initialize model
    model = init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", ""))
    
    # Initialize MCP client
    logger.info("⛓️‍💥 Connecting to Google RSS FastMCP server...")
    
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
        logger.info(f"  - {tool.name}: {tool.description}")
    
    # Build LangGraph
    logger.info("⚙️ Building LangGraph workflow...")
    
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    
    logger.info("✅ LangGraph workflow built successfully!")
    
    # Execute search with more specific examples
    question = "what's the latest news about AI?"
    
    logger.info(f"🔍 Searching for: {question}")
    
    logger.info("🚀 Running LangGraph workflow...")
    
    try:
        response = await graph.ainvoke({"messages": question})
        
        logger.info("✅ Search completed successfully!")
        
        # Display results
        messages = response["messages"]
        if messages and hasattr(messages[-1], 'content') and messages[-1].content:
            
            # Create a beautiful result display
            result_content = messages[-1].content
            
            # Display as general content
            logger.info("AI News Summary:")
            logger.info(result_content)
        else:
            logger.error("❌ No response found")
            
    except Exception as e:
        logger.error(f"❌ Error during search: {str(e)}")
    
    # Footer
    logger.info("✨ All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
