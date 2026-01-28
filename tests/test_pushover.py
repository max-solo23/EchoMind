from unittest.mock import patch

from services.push_over import PushOver


class TestPushOver:
    def test_init_stores_credentials(self):
        pushover = PushOver(token="test_token", user="test_user")

        assert pushover.token == "test_token"
        assert pushover.user == "test_user"

    @patch("services.push_over.requests.post")
    def test_push_sends_to_api(self, mock_post):
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
