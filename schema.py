from pydantic import BaseModel, Field


class Message(BaseModel):
    prompt: str = Field(min_length=1)


class GeneratedTextResponse(BaseModel):
    sujet: str = Field(min_length=1)
    explication: str = Field(min_length=1)
    utilisation: list[str] = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str