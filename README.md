# news-agent

## Overview

https://github.com/user-attachments/assets/effd531e-7296-4f4c-8e98-7230542b6e41

This project is a simple news agent built using the [google-rss-mcp](https://github.com/ccw7463/google-rss-mcp) server in combination with LangGraph.

The google-rss-mcp is a custom MCP (Model Context Protocol) server that I developed for fetching and processing Google News RSS feeds. This news-agent project demonstrates how to integrate such a custom MCP server with LangGraph workflows.

Key Features:
- Real-time news collection through Google News RSS feeds
- Workflow-based news processing using LangGraph
- News summarization and analysis using OpenAI GPT-4o-mini
- Modularized server architecture through FastMCP protocol

This project is the result of testing and implementing an efficient method for processing and analyzing news data by combining MCP (Model Context Protocol) servers with LangGraph.

## Project Structure

```
news-agent/
├── main.py                          # Main execution file
├── pyproject.toml                   # Poetry dependency management
├── .env                             # Environment variables
├── log/                             # Log files directory
│   └── *.log                        # Execution log files
└── src/                             # Source code
    └── modules/                     # Modules directory
        ├── implementations/         # Implementations
        │   └── logger_manager.py    # Logger manager
        ├── interfaces/              # Interfaces
        │   └── logger_manager.py    # Logger interface
        └── mcp_servers/             # MCP servers
            ├── server.py            # MCP server main
            └── rss.py               # RSS processing logic
```

## Getting Started

### 1. Clone the Repository

First, clone the project from GitHub:

```bash
# Clone the repository
git clone https://github.com/ccw7463/news-agent.git

# Navigate to the project directory
cd news-agent
```

### 2. Poetry Setup

First, install Poetry based on your operating system:

```bash
# Standard installation
curl -sSL https://install.python-poetry.org | python3 -

# For Python 3.11 specific installation (recommended for latest libraries)
curl -sSL https://install.python-poetry.org | python3.11 -
```

#### Environment Variables Setup

After installing Poetry, you need to add it to your PATH:

```bash
# Add Poetry to PATH (Linux/macOS)
export PATH="$HOME/.local/bin:$PATH"

# For permanent setup, add to your shell configuration file
# For bash users:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# For zsh users:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### After Installation
```bash
# Install dependencies
poetry install

# Verify installation
poetry --version
```

### 3. Environment Variables

Create a `.env` file and set up the required API keys:

```bash
# Create .env file and Add the following content to .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Execution

Run the application using Poetry:

```bash
poetry run python main.py
```
