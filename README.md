# EchoMind

Personal AI chatbot that embodies a configurable persona for portfolio websites and conversational interfaces.

## Features

- **Persona System** - Define personality, background, and expertise via YAML configuration
- **Multi-Provider LLM Support** - OpenAI, Gemini, and OpenAI-compatible APIs (DeepSeek, Grok, etc.)
- **REST API** - FastAPI server with streaming support (SSE)
- **Dynamic Rate Limiting** - Configurable rate limits (toggle on/off via admin API without restart)
- **Message Validation** - Filters gibberish and invalid input with user-friendly error messages
- **Context-Aware Caching** - Intelligent response caching with TTL to reduce API costs
- **Conversation Logging** - PostgreSQL-backed session and conversation tracking
- **Graceful Error Handling** - User-friendly messages for LLM errors (rate limits, timeouts, etc.)
- **Tool Calling** - Capture user contact info, log unknown questions
- **Push Notifications** - Alerts via Pushover when users engage
- **Admin Dashboard** - Endpoints for cache management, session control, and rate limit configuration

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

# Run API server
uvicorn api.main:app --port 8000
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
LLM_PROVIDER=openai          # openai, gemini, or openai-compatible
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-5.2            # or gpt-5-mini, gpt-5-nano

# Optional - default reasoning effort (none..max); the admin panel overrides it at runtime
LLM_REASONING_EFFORT=none

# Optional - OpenAI-compatible providers
LLM_BASE_URL=https://api.deepseek.com/v1

# Optional - Push notifications
PUSHOVER_TOKEN=your-token
PUSHOVER_USER=your-user-key

# Required for API server
API_KEY=your-api-key
ALLOWED_ORIGINS=https://yoursite.com,http://localhost:3000

# Rate Limiting (configurable via admin API)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_HOUR=10

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
# Health check
GET /health

# Cache management
GET /api/v1/admin/cache/stats
GET /api/v1/admin/cache/entries?page=1&limit=20&sort_by=created_at
POST /api/v1/admin/cache/cleanup

# Session management
GET /api/v1/admin/sessions?page=1&limit=20
DELETE /api/v1/admin/sessions/{session_id}
DELETE /api/v1/admin/sessions

# Model reasoning effort and answer cache switch
GET /api/v1/admin/settings
PUT /api/v1/admin/settings -d '{"reasoning_effort": "low", "cache_enabled": true}'

# Rate limiting (dynamic configuration)
GET /api/v1/admin/rate-limit
POST /api/v1/admin/rate-limit \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "rate_per_hour": 15}'
```

## Rate Limiting

Dynamic, runtime-configurable rate limiting via admin API:

- **Default Limit** - 10 requests/hour per IP address
- **Dynamic Configuration** - Toggle on/off or change rate without server restart
- **Admin Control** - `POST /api/v1/admin/rate-limit` to update settings
- **Shared and Persistent** - Stored in the `app_settings` table, so every instance applies a change within 30 s and it survives restarts
- **User-Friendly Errors** - 429 responses include retry-after information

**Example: Disable rate limiting**
```bash
curl -X POST https://your-api/api/v1/admin/rate-limit \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

## Message Validation

EchoMind filters invalid and gibberish input:

- **Length Check** - Rejects messages < 3 characters
- **Alphabetic Ratio** - Requires ≥30% alphabetic characters (prevents keyboard mashing)
- **Multi-language Support** - Recognizes Latin, Cyrillic, and accented characters
- **User-Friendly Errors** - Returns 400 Bad Request with clear error messages

## Error Handling

Graceful handling of LLM API errors with user-friendly messages:

- **Rate Limit Errors** - "I'm experiencing high demand right now. Please try again in a moment."
- **Timeout Errors** - "I'm taking longer than expected to respond. Please try again."
- **Connection Errors** - "I'm having trouble connecting to my AI service. Please try again shortly."
- **Generic API Errors** - Fallback messages with proper error logging

## Caching System

EchoMind includes an answer cache, switched on and off from the admin panel (off by default;
when off, a chat request does no cache work at all):

- **Exact Keys** - SHA256 of persona hash + previous assistant reply + normalized question (case and punctuation ignored)
- **Similar Questions** - First-turn questions are also matched by character n-gram TF-IDF (works for English and Italian, tolerates typos) plus a content-word check, so "C# projects" never matches "AI projects". Threshold 0.60 has precision 1.00 on the labeled pairs in `tests/data/`
- **Three Variations** - A matched question goes to the LLM until three answers are stored, then the cache serves them in rotation
- **Follow-ups** - Messages with history use only the exact context-bound key
- **TTL Expiration** - Knowledge queries: 30 days, Conversational: 24 hours
- **Denylist Filtering** - Short acknowledgements ("ok", "thanks") are not cached in continuations

## Database Setup

Optional PostgreSQL for caching and conversation logging:

```bash
# Run migrations
alembic upgrade head
```

## Project Structure

```
EchoMind/
├── config.py                  # Configuration management
├── core/
│   ├── chat.py                # Main chat orchestration
│   ├── persona.py             # Persona loader
│   ├── interfaces.py          # Protocol definitions
│   └── llm/
│       ├── factory.py         # Provider factory
│       ├── provider.py        # Base provider
│       ├── types.py           # Type definitions
│       └── providers/
│           ├── openai_compatible.py
│           └── gemini.py
├── services/
│   ├── push_over.py           # Push notification service
│   ├── cache_service.py       # Context-aware caching
│   ├── conversation_logger.py
│   └── similarity_service.py
├── repositories/
│   ├── connection.py          # Database connection setup
│   ├── cache_repo.py          # Cache database operations
│   └── conversation_repo.py
├── tools/
│   └── llm_tools.py           # LLM function calling
├── models/
│   ├── models.py              # SQLAlchemy models
│   ├── requests.py            # Pydantic request models
│   └── responses.py           # Pydantic response models
├── api/
│   ├── main.py                # FastAPI application
│   ├── dependencies.py        # Dependency injection
│   ├── routes/
│   │   ├── chat.py            # Chat endpoints
│   │   ├── admin.py           # Admin endpoints
│   │   └── health.py          # Health check endpoint
│   └── middleware/
│       ├── auth.py            # API key authentication
│       ├── cors.py            # CORS configuration
│       └── rate_limit.py      # Rate limiting
├── alembic/                   # Database migrations
└── tests/                     # Test suite (70%+ coverage)
```

## Development

```bash
# Run API server (development)
uvicorn api.main:app --reload --port 8000

# Run tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run migrations
alembic upgrade head

# Create new migration
alembic revision -m "description"

# Latency benchmark with a zero-latency fake LLM (no database or API key needed)
python scripts/bench_chat.py          # in-process: request overhead + event-loop lag probe
python scripts/bench_chat.py --http   # real uvicorn over TCP: time-to-first-byte
```

## Deployment

EchoMind is designed for deployment on Fly.io with PostgreSQL:

```bash
# Deploy to Fly.io
fly deploy

# View logs
fly logs

# Check status
fly status

# Open app
fly open
```

**Environment Setup:**
- Set all environment variables via `fly secrets set KEY=value`
- Attach PostgreSQL database via Fly.io Postgres
- Configure `ALLOWED_ORIGINS` with your frontend domain
- Set `PORT` to 8080 (Fly.io default)

**Production Checklist:**
- ✅ Database migrations applied (`alembic upgrade head`)
- ✅ API key configured and secure
- ✅ Rate limiting enabled
- ✅ CORS origins restricted to your domain
- ✅ PostgreSQL database attached
- ✅ Environment secrets set (not committed to git)

## License

MIT
