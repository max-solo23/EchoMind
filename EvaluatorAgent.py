from Evaluation import Evaluation
from core.interfaces import LLMProvider


class EvaluatorAgent:
    def __init__(
            self, 
            persona, 
            llm: LLMProvider, 
            model
        ):
        self.name = persona.name
        self.summary = persona.summary
        self.llm = llm
        self.model = model

        self.evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
        You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's \
        latest response is acceptable quality. \
        The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
        The Agent has been instructed to be professional and engaging, as if talking to a potential client or future \
        employer who came across the website. \
        The Agent has been provided with context on {self.name} in the form of their summary and LinkedIn details. \
        Here's the information:"

        self.evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n"
        self.evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether \
        the response is acceptable and your feedback."

    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(self, reply, message, history) -> Evaluation:
        messages = [
            {"role": "system", "content": self.evaluator_system_prompt}
                   ] + [
            {"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}
        ]
        response = self.llm.chat.completions.parse(model=self.model, messages=messages, response_format=Evaluation)

        return response.choices[0].message.parsed
