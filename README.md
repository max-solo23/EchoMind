---
title: EchoMind
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---
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
PUSHOVER_USER=youruser
PUSHOVER_TOKEN=yourtoken
OPENAI_API_KEY=yourkey
OPENAI_MODEL=gpt-5-nano
USE_EVALUATOR=false
API_KEY=your_secret_api_key
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

**GET /health** - Health check (no auth required)

**GET /docs** - Interactive API documentation
