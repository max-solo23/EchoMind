# Changelog

All notable changes to EchoMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Released]

## [2.3.1] - 2025-12-30

### Fixed
- **Clear all sessions endpoint**: Fixed foreign key constraint violation when deleting all sessions
  - Now deletes conversations before sessions to avoid constraint errors
  - Prevents `IntegrityError: violates foreign key constraint "conversations_session_id_fkey"`

---

## [2.3.0] - 2025-12-30

### Added
- **Dynamic rate limiting**: Admin endpoints to control rate limiting without server restart
  - `GET /api/v1/admin/rate-limit` - Get current rate limit settings
  - `POST /api/v1/admin/rate-limit` - Update settings (enable/disable, change rate)
- **RateLimitState**: Thread-safe global state manager for rate limiting configuration
- **Message validation**: Filter gibberish and invalid input with `InvalidMessageError`
  - Rejects messages < 3 characters
  - Rejects keyboard mashing (< 30% alphabetic characters)
  - Returns 400 Bad Request with user-friendly error message
- **LLM error handling**: Graceful handling of OpenAI API errors
  - Rate limit errors: "I'm experiencing high demand right now..."
  - Timeout errors: "I'm taking longer than expected..."
  - Connection errors: "I'm having trouble connecting..."
  - Generic API errors with fallback messages
- **Enhanced logging**: Request details in chat endpoint
  - Session ID, message length, history presence, client IP, streaming mode

### Changed
- **Rate limiting**: Now dynamically configurable at runtime (no restart required)
- **Rate limit default**: Changed from 15/hour to 10/hour per IP
- **Port configuration**: Respects `PORT` environment variable for Fly.io compatibility
- **Evaluation logging**: Changed from `print()` to `logger.debug()` for cleaner logs
- **Error responses**: User-friendly messages for all LLM error types
- **Rate limit error message**: Improved clarity ("You've sent too many messages...")

### Fixed
- **Environment documentation**: Clarified rate limiting configuration in `.env.example`
- **Streaming error handling**: Specific error codes for different OpenAI API failures

### Technical Notes
- Rate limiting can be disabled via admin API: `POST /api/v1/admin/rate-limit {"enabled": false}`
- Message validation supports multiple languages (Latin, Cyrillic, accented characters)
- All LLM errors are logged with full context (`exc_info=True`)
- Rate limit state uses threading.Lock for thread safety

---

## [2.2.0] - 2025-12-30

### Removed
- **Rate limit metadata from responses**: Removed `rate_limit` field from `ChatResponse` model (frontend no longer displays this information)
- **RateLimitInfo model**: Removed unused Pydantic model from `models/responses.py`

### Changed
- **Simplified response format**: Non-streaming chat responses now only return `{"reply": "..."}` without rate limit metadata (~150 bytes smaller per response)

### Technical Notes
- Rate limiting functionality is unchanged - only the response metadata was removed
- Frontend already ignores this metadata as of FRONTEND-SIMPLIFICATION.md
- Backwards compatible change - no API contract modifications needed

---

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
- FastAPI REST API with streaming support
- Rate limiting (10 requests/hour)
- API key authentication
- LLM provider abstraction (OpenAI, Gemini)
- Persona system via YAML configuration
- Tool calling support (record_user_details, record_unknown_question)
- PushOver notifications
