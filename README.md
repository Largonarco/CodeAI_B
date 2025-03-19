# CodeAI

CodeAI is a multi-agent system designed to assist developers with various tasks, including code review, documentation research, and answering general development questions.

## Features

- **Documentation Research Agent**: Fetches and structures documentation for libraries, frameworks, and APIs.
- **PR Review Agent**: Analyzes GitHub pull requests to identify bugs, security issues, and performance improvements.
- **General Assistant Agent**: Handles general development questions and tasks.
- **Intelligent Supervisor**: Routes requests to the appropriate specialized agent based on the user's query.

## Architecture

CodeAI uses a LangGraph-based workflow system with MongoDB for state persistence and Firebase for authentication. The system integrates with OpenAI's Chat models to power the agents.

### Components

- **Supervisor**: Analyzes user requests and routes them to the appropriate agent
- **Documentation Research Agent**: Searches for, extracts, and structures documentation for libraries and APIs
- **PR Review Agent**: Analyzes GitHub pull requests, providing detailed code review feedback
- **General Agent**: Handles general queries that don't fit into the other specialized agents

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/codeai.git
cd codeai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=your_mongodb_uri
```

4. Set up Firebase credentials in a `credentials.txt` file in the project root.

## Usage

1. Start the server:

```bash
uvicorn server:app --reload
```

2. Make requests to the API:

```bash
curl -X POST \
  http://localhost:8000/agent/run \
  -H 'Authorization: Bearer YOUR_FIREBASE_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "request": "Document how to use pagination in the Mongoose library",
    "github_token": "optional_github_token"
  }'
```

## API Endpoints

- `POST /agent/run`: Run the agent with the specified request
- `GET /health`: Health check endpoint

## Required Dependencies

- `langchain`, `langgraph`: For agent architecture
- `openai`: For LLM access
- `pymongo`: For MongoDB interaction
- `firebase-admin`: For authentication
- `fastapi`: For API server
- `trafilatura`: For web scraping
- `pydantic`: For data validation
- `github`: For GitHub integration
- `dotenv`: For environment variable management

## Authentication

The system uses Firebase Authentication. To make requests, you need to include a valid Firebase JWT token in the Authorization header.

## State Persistence

Conversation state is persisted in MongoDB, allowing for continuity across different API calls.

## Agent Capabilities

### Documentation Research Agent

This agent:

1. Analyzes the requested functionality to generate search terms
2. Searches for documentation using DuckDuckGo
3. Extracts content from documentation websites
4. Structures and synthesizes documentation into a usable format

### PR Review Agent

This agent:

1. Fetches files from a GitHub pull request
2. Retrieves the full content of modified files
3. Analyzes code for bugs, security issues, and performance problems
4. Provides detailed feedback for each file

### Example Use Cases

- "Document how to implement authentication with Passport.js"
- "Review my PR at https://github.com/user/repo/pull/123"
- "What's the best way to handle form validation in React?"

## License

[Your License Here]

## Contributing

[Your Contribution Guidelines Here]
