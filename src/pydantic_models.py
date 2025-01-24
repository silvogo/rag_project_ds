from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class ModelName(str, Enum):
    GPT4_O_MINI = "gpt-4o-mini"


class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
