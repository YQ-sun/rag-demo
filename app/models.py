from pydantic import BaseModel

class AddDocRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int