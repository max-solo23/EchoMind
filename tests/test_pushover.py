"""Tests for PushOver - mobile notification service."""

from unittest.mock import patch

from PushOver import PushOver


class TestPushOver:
    """Test PushOver notification service."""

    def test_init_stores_credentials(self):
        """
        Verify token and user are stored.

        Why: Push notifications require valid API credentials.
        If not stored correctly, all notifications fail silently.
        """
        pushover = PushOver(token="test_token", user="test_user")

        assert pushover.token == "test_token"
        assert pushover.user == "test_user"

    @patch("PushOver.requests.post")
    def test_push_sends_to_api(self, mock_post):
        """
        Verify push sends correct data to Pushover API.

        Why: Core functionality - must send correct token, user,
        and message to API endpoint for notifications to work.
        """
        pushover = PushOver(token="api_token", user="user_key")

        pushover.push("Test notification")

        mock_post.assert_called_once_with(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": "api_token",
                "user": "user_key",
                "message": "Test notification",
            },
        )
