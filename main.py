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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

load_dotenv()

# Initialize console
console = Console()

# Initialize time
kst = pytz.timezone('Asia/Seoul')
time_now = datetime.now(kst)
time_now_str = time_now.strftime("%Y%m%d_%H%M%S")

class NewsAgentState(MessagesState):
    articles: List[Dict[str, Any]]
    
async def main():
    
    # Header
    console.print(Panel.fit(
        "[bold blue]🚀 AI News Search with LangGraph & FastMCP[/bold blue]\n"
        "[dim]Powered by Google RSS and OpenAI GPT-4o-mini[/dim]\n"
        f"[dim]Started at: {time_now_str}[/dim]",
        border_style="blue"
    ))
    
    # Initialize model
    console.print(Panel(
        "[bold yellow]🤖 Initializing OpenAI GPT-4o-mini model...[/bold yellow]",
        border_style="yellow"
    ))
    model = init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", ""))
    console.print(Panel(
        "[bold green]✅ Model initialized successfully![/bold green]",
        border_style="green"
    ))
    
    # Initialize MCP client
    console.print(Panel(
        "[bold yellow]📡 Connecting to Google RSS FastMCP server...[/bold yellow]",
        border_style="yellow"
    ))
    
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
    
    console.print(Panel(
        "[bold green]✅ FastMCP server connected successfully![/bold green]",
        border_style="green"
    ))
    
    # Display available tools
    tools_table = Table(title="🔧 Available Google News RSS FastMCP Tools", 
                        show_header=True, header_style="bold white")
    tools_table.add_column("Tool Name", style="cyan", no_wrap=True)
    tools_table.add_column("Description", style="white")
    
    for tool in tools:
        tools_table.add_row(tool.name, tool.description)
    
    console.print(tools_table)
    
    # Build LangGraph
    console.print(Panel(
        "[bold yellow]🔨 Building LangGraph workflow...[/bold yellow]",
        border_style="yellow"
    ))
    
    def call_model(state: NewsAgentState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    async def summary_node(state: NewsAgentState):
        # Extract article data from ToolMessage
        articles = []
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        articles.extend(data)
                except Exception as e:
                    console.print(f"[red]ToolMessage JSON decode error: {e}[/red]")

        # Extract user's question
        user_query = None
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if not user_query:
            user_query = "No user question provided. Please summarize the article content in 5 sentences or less based on the article title."

        # Generate summary with progress
        console.print(Panel(
            f"[bold yellow]🤖 Generating AI summaries for {len(articles)} articles...[/bold yellow]",
            border_style="yellow"
        ))
        
        # Create tasks for parallel processing
        async def process_single_summary(article, idx):
            content = article.get("article_content", "")
            title = article.get("article_title", "")
            prompt = [
                SystemMessage(content="You are a news summarization expert. Please summarize the article content in 5 sentences or less based on the user's question and article title."),
                HumanMessage(content=f"Question: {user_query}\nArticle Title: {title}\nArticle Content: {content}\nSummary:")
            ]
            try:
                summary = await model.ainvoke(prompt)
                article["summary"] = summary.content.strip()
                console.print(f"[green]   ✓ Article {idx+1}: {title[:60]}...[/green]")
                return True
            except Exception as e:
                console.print(f"[red]   ✗ Summary generation failed for article {idx+1}: {e}[/red]")
                article["summary"] = "Summary generation failed"
                return False
        
        # Execute all summary tasks in parallel
        tasks = []
        for idx, article in enumerate(articles):
            task = asyncio.create_task(process_single_summary(article, idx))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful summaries
        summary_success_count = sum(1 for result in results if result is True)
        
        state["articles"] = articles
        console.print(Panel(
            f"[bold green]✅ AI summary generation completed: {summary_success_count}/{len(articles)} articles[/bold green]",
            border_style="green"
        ))
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
    builder.add_edge("final_answer_node", END)
    graph = builder.compile()
    
    console.print(Panel(
        "[bold green]✅ LangGraph workflow built successfully![/bold green]",
        border_style="green"
    ))
    
    question = "Find 16 latest AI-related news articles"
    console.print(Panel(
        f"[bold pink1]🔍 QUESTION: {question}[/bold pink1]",
        border_style="pink1",
        padding=(1, 2)
    ))
    
    console.print(Panel(
        "[bold yellow]🚀 Running LangGraph workflow...[/bold yellow]",
        border_style="yellow"
    ))
    
    try:
        response = await graph.ainvoke({"messages": question})
        
        # Display results
        messages = response["messages"]
        if messages and hasattr(messages[-1], 'content') and messages[-1].content:
            result_content = messages[-1].content
            
            # Create a beautiful result display
            console.print(Panel(
                result_content,
                title="[bold magenta]📋 AI News Summary[/bold magenta]",
                border_style="magenta",
                padding=(1, 2)
            ))
        else:
            console.print(Panel(
                "[bold red]❌ No response found[/bold red]",
                border_style="red"
            ))
            
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Error during search: {str(e)}[/bold red]",
            border_style="red"
        ))
    
    # Footer
    console.print(Panel(
        "[dim]🎉 All questions completed successfully![/dim]",
        border_style="dim"
    ))

if __name__ == "__main__":
    asyncio.run(main())
