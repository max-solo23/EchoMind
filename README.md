# EchoMind

Personal AI chatbot that embodies a configurable persona for portfolio websites and conversational interfaces.

## Features

- **Persona System** - Define personality, background, and expertise via YAML configuration
- **Multi-Provider LLM Support** - OpenAI, Gemini, and OpenAI-compatible APIs (DeepSeek, Grok, etc.)
- **REST API** - FastAPI server with streaming support (SSE)
- **Gradio UI** - Interactive demo interface
- **Context-Aware Caching** - Intelligent response caching with TTL to reduce API costs
- **Quality Evaluation** - Optional response quality control via evaluator agent
- **Tool Calling** - Capture user contact info, log unknown questions
- **Push Notifications** - Alerts via Pushover when users engage

## Quick Start

```bash
# Clone and setup
git clone https://github.com/max-solo23/EchoMind.git
cd EchoMind

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run
python app.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
LLM_PROVIDER=openai          # openai, gemini, or openai-compatible
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini

# Optional - OpenAI-compatible providers
LLM_BASE_URL=https://api.deepseek.com/v1

# Optional - Quality evaluation
USE_EVALUATOR=false

# Optional - Push notifications
PUSHOVER_TOKEN=your-token
PUSHOVER_USER=your-user-key

# Required for API server
API_KEY=your-api-key
ALLOWED_ORIGINS=https://yoursite.com,http://localhost:3000

# Optional - Database (enables caching & logging)
POSTGRES_URL=postgresql+asyncpg://user:pass@host:5432/echomind
```

### Persona Configuration

Create `persona.yaml` in the project root:

```yaml
name: "Alex Chen"
title: "Full Stack Developer"
background: |
  5 years of experience building web applications.
  Passionate about clean code and user experience.
expertise:
  - Python
  - React
  - PostgreSQL
  - Cloud deployment
personality: |
  Friendly and approachable. Explains complex topics simply.
  Enthusiastic about helping others learn.
```

## API Endpoints

### Chat

```bash
# Non-streaming
curl -X POST https://your-api/api/v1/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about yourself", "history": []}'

# Streaming (SSE)
curl -X POST "https://your-api/api/v1/chat?stream=true" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about yourself", "history": []}'
```

### Admin

```bash
# Cache stats
GET /api/v1/admin/cache/stats

# List cache entries
GET /api/v1/admin/cache/entries?page=1&limit=20

# Clear expired cache
POST /api/v1/admin/cache/cleanup

# Health check
GET /api/v1/admin/health
```

## Caching System

EchoMind includes intelligent response caching:

- **Context-Aware Keys** - Same question after different responses creates different cache entries
- **TTL Expiration** - Knowledge queries: 30 days, Conversational: 24 hours
- **Denylist Filtering** - Short acknowledgements ("ok", "thanks") are not cached in continuations
- **Similarity Matching** - TF-IDF with 90% threshold for fuzzy question matching
- **Answer Variations** - Up to 3 variations per question with rotation

## Database Setup

Optional PostgreSQL for caching and conversation logging:

```bash
# Run migrations
alembic upgrade head
```

## Project Structure

```
EchoMind/
├── app.py                 # Gradio interface entry point
├── Chat.py                # Main chat orchestration
├── Me.py                  # Persona loader
├── Tools.py               # LLM function calling
├── EvaluatorAgent.py      # Response quality control
├── Evaluation.py          # Evaluation models
├── PushOver.py            # Push notification service
├── config.py              # Configuration management
├── database.py            # Database connection setup
├── api/
│   ├── main.py            # FastAPI application
│   ├── dependencies.py    # Dependency injection
│   ├── routes/
│   │   ├── chat.py        # Chat endpoints
│   │   ├── admin.py       # Admin endpoints
│   │   └── health.py      # Health check endpoint
│   └── middleware/
│       ├── auth.py        # API key authentication
│       ├── cors.py        # CORS configuration
│       └── rate_limit.py  # Rate limiting (10/hour)
├── services/
│   ├── cache_service.py   # Context-aware caching
│   ├── conversation_logger.py
│   └── similarity_service.py
├── repositories/
│   ├── cache_repo.py      # Cache database operations
│   └── conversation_repo.py
├── core/
│   ├── interfaces.py      # Protocol definitions
│   └── llm/
│       ├── factory.py     # Provider factory
│       ├── provider.py    # Base provider
│       ├── types.py       # Type definitions
│       └── providers/
│           ├── openai_compatible.py
│           └── gemini.py
├── models/
│   ├── models.py          # SQLAlchemy models
│   ├── requests.py        # Pydantic request models
│   └── responses.py       # Pydantic response models
├── alembic/               # Database migrations
└── tests/                 # Test suite
```

## Development

```bash
# Run API server (development)
uvicorn api.main:app --reload --port 8000

# Run Gradio interface
python app.py

# Run migrations
alembic upgrade head

# Create new migration
alembic revision -m "description"
```

## License

MIT
