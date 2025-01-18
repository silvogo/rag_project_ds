from pydantic import BaseModel, Field

class ModelName:
    GPT4_O_MINI = 'gpt-4o-mini'

class QueryInput(BaseModel):
    question: str
    session_id: str= Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
