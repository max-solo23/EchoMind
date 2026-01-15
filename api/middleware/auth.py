from fastapi import Header, HTTPException, status

from api.dependencies import get_config


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Verify API key from X-API-Key header."""
    config = get_config()

    if not config.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )

    if x_api_key != config.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
