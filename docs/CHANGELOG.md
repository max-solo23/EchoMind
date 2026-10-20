# Changelog

All notable changes to EchoMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Released]

## [Unreleased]

### Added
- **OpenAI Responses API provider** (2026-10-20): `LLM_PROVIDER=openai` without a base URL now uses `/v1/responses` through `OpenAIResponsesProvider`, which converts the Chat Completions history and stream events so `Chat` is unchanged. Needed because `gpt-6-luna` rejects function tools on `/v1/chat/completions` unless `reasoning_effort="none"`. Compatible providers behind `LLM_BASE_URL` stay on Chat Completions.
- **Admin runtime settings** (2026-10-20): `app_settings` table (migration `c41e9a7d2f60`) and `GET/PUT /api/v1/admin/settings` for `reasoning_effort` (none..max) and `cache_enabled`, re-read at most every 30 s per machine. `LLM_REASONING_EFFORT` sets the default. The Portfolio admin shows them in a "Model" card.
- **Token usage logging** (2026-10-20): every LLM call logs prompt, cached, completion and reasoning tokens.
- **Similar-question cache** (2026-10-20): first-turn questions are matched by an in-memory character n-gram TF-IDF index (threshold 0.60) plus a strict content-word check; precision 1.00 on the labeled pairs in `tests/data/` (recall 0.67 tuning, 0.70 held-out). An entry is served only after its own exact question stored three LLM answers, which then rotate; similar questions read entries but never write into them (cache poisoning). Off by default; when off, a chat request does no cache work.

### Removed
- **`tfidf_vector` column** (migration `d7a3b5e19c42`): the per-request vectors were written but never read. `persona_hash` was added to scope similar-question matches to the current persona.
- **`api/middleware/rate_limit_state.py`**: rate-limit settings now live in `app_settings`.

### Changed
- **Chat response flow**: Consolidated cache-write and response logging paths while preserving non-streaming responses and the SSE event contract.
- **Cache repository mapping**: Consolidated repeated cache-record mapping while retaining each public method's existing fields and query behavior.
- **Cache matching**: Exact keys are normalized (case and punctuation ignored) and still bound to persona and previous reply; follow-ups never use similar-question matching. A cache hit returns at once and its logging and rotation run in the background (hit 5-10 ms vs ~1.3 s LLM in the end-to-end check).
- **Cache eligibility**: Low-signal question inputs like `?` and `ok?` are skipped instead of being cached.
- **Streaming first byte**: the SSE kickstart now leaves before the cache lookup, so the client's EventSource opens immediately instead of after a database round trip (local benchmark with a simulated 20 ms database: time-to-first-byte 31 ms -> 1.8 ms).
- **Conversation logging off the critical path**: cache and conversation writes run as a background task after the reply is delivered and are drained on shutdown; a failed write is logged instead of turning an already generated reply into a 500 (reply latency with a simulated database: 62 ms -> 31 ms).
- **Faster cold start**: scikit-learn, scipy and pandas load on the first cache write instead of at import, cutting `import api.main` from 1.94 s to 1.07 s. Matters on Fly with `min_machines_running = 0`.

### Fixed
- **Tool calls blocked the event loop**: the synchronous Pushover notification inside a tool call stalled every other request for the duration of the HTTP call (measured 298 ms event-loop lag). Tool handlers now run in a worker thread via `asyncio.to_thread` (lag 11 ms, the timer floor).
- **Chat reliability**: Cache write failures no longer fail a successfully logged chat response.
- **Production outage after switching to `gpt-6-luna`** (2026-10-20): its default reasoning effort is not `none`, so every chat with tools got HTTP 400; fixed by `LLM_REASONING_EFFORT` and then the Responses API provider.
- **App logs missing in production** (2026-10-20): uvicorn configures only its own loggers, so every `logger.info` (token usage, cache hits) was dropped; `api/main.py` now calls `logging.basicConfig`.
- **Cache never used similar questions or more than one variation** (2026-10-20): vectors were refit per request and never read, stopword-only questions ("Who are you?") raised and were never cached, and `add_variation` was unreachable after the first write.
- **Rate-limit admin changes** (2026-10-20) applied to one Fly machine only and were lost when it slept; they are now stored in `app_settings`.

---

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
