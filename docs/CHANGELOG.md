# Changelog

All notable changes to EchoMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Released]

## [2.1.0] - 2025-12-30

### Added
- **Delete session endpoint**: `DELETE /api/v1/admin/sessions/{session_id}` to delete a single session and its conversations
- **Clear all sessions endpoint**: `DELETE /api/v1/admin/sessions` to delete all sessions

---

## [2.0.0] - 2025-12-30

### Added
- **Context-aware cache keys**: Cache key now includes last assistant message to prevent cross-conversation contamination
- **TTL-based cache expiration**: Knowledge cache (30 days), Conversational cache (24 hours)
- **Role-aware denylist**: Short inputs like "ok", "thanks" are not cached in continuations
- **New admin endpoint**: `POST /api/v1/admin/cache/cleanup` to delete expired entries
- **Cache type classification**: Entries categorized as `knowledge` or `conversational`

### Changed
- **CachedAnswer model**: Added `cache_key`, `context_preview`, `cache_type`, `expires_at` fields
- **Cache stats response**: Now includes `knowledge_entries`, `conversational_entries`, `expired_entries`
- **Cache entry responses**: All endpoints return new fields (`cache_key`, `cache_type`, `expires_at`, `context_preview`)
- **Cache sort options**: Added `expires_at` and `cache_type` sort fields

### Fixed
- **Context-insensitive cache hits**: Short inputs like "ok" no longer return cached responses from unrelated conversations

### Migration
- `b3dc51b73835_add_context_aware_cache_with_ttl.py` - Run `alembic upgrade head`

---

## [1.1.0] - 2025-12-29

### Added
- Admin endpoints for session and cache management
- Database integration with Alembic migrations
- Conversation logging with session tracking

---

## [1.0.0] - 2025-12-28

### Added
- Initial release
- Gradio chat interface
- FastAPI REST API with streaming support
- Rate limiting (10 requests/hour)
- API key authentication
- LLM provider abstraction (OpenAI, Gemini)
- Persona system via YAML configuration
- Optional evaluator agent for response quality
- Tool calling support (record_user_details, record_unknown_question)
- PushOver notifications
