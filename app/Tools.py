import json
from typing import Optional


class Tools:
    def __init__(self, message_app: Optional[object] = None):
        self.message_app = message_app
        self.record_user_details_json = {
            "name": "record_user_details",
            "description": "Use this tool to record that a user interested in being in touch and provided and email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address of this user"
                    },
                    "name": {
                        "type": "string",
                        "description": "The user's name, if they provided it"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any additional information about the conversation that's worth recording to the given context"
                    }
                },
                "required": ["email"],
                "additionalProperties": False
            }
        }

        self.record_unknown_question_json = {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that couldn't be answered"
                    },
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }

        self.tools = [
            {"type": "function", "function": self.record_user_details_json},
            {"type": "function", "function": self.record_unknown_question_json}
        ]

    def record_user_details(self, email, name="Name not provided", notes="not provided"):
        if self.message_app:
            self.message_app.push(f"Recording {name} with email {email} and notes {notes}")
        else:
            print(f"[tools] record_user_details -> {name=} {email=} {notes=}", flush=True)
        return {"recorded": "ok"}

    def record_unknown_question(self, question):
        if self.message_app:
            self.message_app.push(f"Recording {question}")
        else:
            print(f"[tools] record_unknown_question -> {question=}", flush=True)
        return {"recorded": "ok"}

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = getattr(self, tool_name, None)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results


