"""
Rate limiting middleware using SlowAPI.

Provides IP-based rate limiting for API endpoints to prevent abuse.
Configured to allow 15 requests per hour per IP address by default.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from fastapi.responses import JSONResponse


def rate_limit_exceeded_handler(request: Request, exc) -> JSONResponse:
    """
    Custom handler for 429 Too Many Requests responses.

    Returns a JSON response with error details and Retry-After header
    indicating when the client can retry the request.

    Args:
        request: The FastAPI request object
        exc: The RateLimitExceeded exception

    Returns:
        JSONResponse with 429 status code and rate limit details
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "You have exceeded the request limit. Please try again later.",
            "retry_after": "3600"  # seconds until reset (1 hour)
        },
        headers={"Retry-After": "3600"}
    )


# Initialize the rate limiter with IP-based tracking
# get_remote_address automatically handles X-Forwarded-For headers
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[]  # No default limits, apply per-route
)
