from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class ModelName(str, Enum):
    GPT4_O_MINI = "gpt-4o-mini"


class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None,
                                      description="Optional session ID")
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

    # defines what json structure of the model should apear in Swagger's example request
    class Config:
        json_schema_extra = {
            "example":{
                "question": "What is the nps score?",
                "session_id": None,
                "model": "gpt-4o-mini"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime