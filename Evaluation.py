from pydantic import BaseModel


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str
