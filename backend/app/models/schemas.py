from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    description: str

class ClassifyRequest(BaseModel):
    description: str
