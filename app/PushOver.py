import requests


class PushOver:
    def __init__(self, token, user):
        self.token = token
        self.user = user

    def push(self, text):
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.token,
                "user": self.user,
                "message": text,
            }
        )
