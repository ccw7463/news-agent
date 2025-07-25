import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

load_dotenv()

console = Console()

async def main():
    
    # Header
    console.print(Panel.fit(
        "[bold blue]🤖 AI News Search with LangGraph & FastMCP[/bold blue]\n"
        "[dim]Powered by Google RSS and OpenAI GPT-4o-mini[/dim]",
        border_style="blue"
    ))
    
    # Initialize model
    model = init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", ""))
    
    # Initialize MCP client
    console.print(Panel(
        "[bold yellow]⛓️‍💥 Connecting to Google RSS FastMCP server...[/bold yellow]",
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
    tools_table.add_column("Tool Name", style="white", no_wrap=True)
    tools_table.add_column("Description", style="white")
    
    for tool in tools:
        tools_table.add_row(tool.name, tool.description)
    
    console.print(tools_table)
    
    # Build LangGraph
    console.print(Panel(
        "[bold yellow]⚙️ Building LangGraph workflow...[/bold yellow]",
        border_style="yellow"
    ))
    
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}
    
    def logger_node(state: MessagesState):
        # messages 리스트의 각 요소를 구분해서 출력
        console.print(Panel(
            "[bold white]📋 State Messages 분석[/bold white]",
            border_style="green"
        ))
        
        messages = state["messages"]
        for i, message in enumerate(messages):
            console.print(Panel(
                f"[bold cyan]Message {i+1}:[/bold cyan]\n"
                f"[bold yellow]Type:[/bold yellow] {type(message).__name__}\n"
                f"[bold yellow]Content:[/bold yellow] {message.content}\n"
                f"[bold yellow]Additional Info:[/bold yellow] {getattr(message, 'tool_calls', 'N/A') if hasattr(message, 'tool_calls') else 'N/A'}",
                border_style="blue",
                padding=(0, 1)
            ))

        return {"messages": state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("logger", logger_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "logger")
    builder.add_edge("logger", "call_model")
    graph = builder.compile()
    
    console.print(Panel(
        "[bold green]✅ LangGraph workflow built successfully![/bold green]",
        border_style="green"
    ))
    
    # Execute search with more specific examples
    question = "what's the latest news about AI?"
    
    console.print(Panel(
        f"[bold pink1]🔍 {question}[/bold pink1]",
        border_style="pink1",
        padding=(1, 2)
    ))
    
    console.print(Panel(
        "[bold yellow]🚀 Running LangGraph workflow...[/bold yellow]",
        border_style="yellow"
    ))
    
    try:
        response = await graph.ainvoke({"messages": question})
        
        console.print(Panel(
            "[bold green]✅ Search completed successfully![/bold green]",
            border_style="green"
        ))
        
        # Display results
        messages = response["messages"]
        if messages and hasattr(messages[-1], 'content') and messages[-1].content:
            
            # Create a beautiful result display
            result_content = messages[-1].content
            
            # Display as general content
            console.print(Panel(
                result_content,
                title=f"[bold magenta]AI News Summary[/bold magenta]",
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
        "[dim]✨ All tests completed successfully![/dim]",
        border_style="dim"
    ))

if __name__ == "__main__":
    asyncio.run(main())
