from pydantic import BaseModel, Field
from typing import Optional, Dict, Union, List
from datetime import datetime


class ChatRequest(BaseModel):
    provider: str = Field(..., example="openai")
    prompt: str = Field(..., example="What is the capital of France?")
    settings: Optional[Dict[str, Union[str, float, int, bool]]] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    response: Union[str, Dict]  # In case of DeepSeek or error messages

class Prompt(BaseModel):
    promptName: str
    promptDescription: str
    promptCategory: str

class ChatRecord(BaseModel):
    model: str
    prompt: str
    response: str
    session_id: Optional[str] = None
    timestamp: datetime

class ChatHistoryResponse(BaseModel):
    history: List[ChatRecord]

class ModelSettings(BaseModel):
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    web_search: Optional[bool] = False
    custom_instructions: Optional[str] = None

class ModelComparisonRequest(BaseModel):
    prompt: str
    models: List[str]
    settings: Dict[str, ModelSettings]  # Settings per model
    files: Optional[List[str]] = []     # Future use

class ComparisonResult(BaseModel):
    provider: str
    response: str
    tokens: Optional[int] = None
    latency: int  # in milliseconds

class ModelComparisonResponse(BaseModel):
    comparisons: List[ComparisonResult]