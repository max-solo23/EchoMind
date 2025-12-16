# EchoMind

**EchoMind** is a personal AI chatbot application that represents you as a configurable persona in conversational interactions. It uses your profile (configured via YAML) to answer questions about your skills, experience, projects, and contact information in a natural, authentic way.

## What Does It Do?

- **Persona-Based Chat**: The AI acts as you, based on your persona configuration
- **Tool Calling**: Captures interested users' contact info and logs unknown questions
- **Quality Control**: Optional response evaluation to maintain professional, on-brand replies
- **Dual Interface**:
  - Gradio UI for interactive local testing
  - FastAPI REST API for integration with portfolio websites
- **Multi-Provider Support**: Works with OpenAI, Gemini, DeepSeek, Grok, and Ollama
- **Streaming Responses**: Real-time token streaming via Server-Sent Events (SSE)

## Setup Instructions

```bash
# clone repository
git clone <repository-url>
cd <project-folder>

# create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# start Gradio UI (local development)
python app.py

# OR start FastAPI server (production API)
python -m api.main
```

The Gradio interface runs at `http://localhost:7860`
The FastAPI server runs at `http://localhost:8000` with docs at `/docs`

## Environment Variables

Create your own .env file in the root directory of the project. This file is required to store your personal API keys and tokens securely. Do not commit it to version control.

Pushover credentials are optional and only needed if you want to enable notification features.

Example `.env` content:

```
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-5.2-nano
# For OpenAI-compatible providers (DeepSeek/Grok/Ollama), set:
# LLM_BASE_URL=http://localhost:11434/v1

# Gemini example:
# LLM_PROVIDER=gemini
# LLM_API_KEY=your_gemini_api_key
# LLM_MODEL=gemini-1.5-flash

# Notification Service (Optional)
PUSHOVER_USER=your_pushover_user_key
PUSHOVER_TOKEN=your_pushover_token

# Response Quality Evaluation (Optional)
USE_EVALUATOR=false

# API Configuration (Required for FastAPI)
API_KEY=your_secret_api_key_here
ALLOWED_ORIGINS=http://localhost:3000,https://your-portfolio.com
```
Replace the placeholder values with your actual credentials before running the app.

**For FastAPI:** `API_KEY` and `ALLOWED_ORIGINS` are required.

## Profile Configuration

Create your persona profile:

1. Rename `persona.yaml.example` to `persona.yaml`
2. Fill in your personal information (name, skills, projects, contacts, etc.)
3. The AI will use this to represent you authentically

## API Usage

### Starting the API

```bash
python -m api.main
```

Server runs at `http://localhost:8000`

### Endpoints

**POST /api/v1/chat** - Chat with AI persona (requires API key)

Query Parameters:
- `stream` (optional): Set to `true` to enable streaming response via SSE

**Standard Response (default):**

Request:
```json
{
  "message": "What are your main skills?",
  "history": []
}
```

Headers:
- `X-API-Key`: Your API key
- `Content-Type`: application/json

Response:
```json
{
  "reply": "I specialize in..."
}
```

**Streaming Response:**

Request: `POST /api/v1/chat?stream=true`

Same JSON body as above. Response is Server-Sent Events (SSE) stream:

```
data: {"delta": "I", "metadata": null}

data: {"delta": " specialize", "metadata": null}

data: {"delta": " in...", "metadata": null}

data: {"delta": null, "metadata": {"done": true}}

```

Event metadata may include tool call status:
```
data: {"delta": null, "metadata": {"tool_call": "record_user_details", "status": "executing"}}
data: {"delta": null, "metadata": {"tool_call": "record_user_details", "status": "success"}}
```

Note: Streaming mode bypasses the evaluator for real-time responses.

**GET /health** - Health check (no auth required)

**GET /docs** - Interactive API documentation
