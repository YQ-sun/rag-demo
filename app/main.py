from fastapi import FastAPI,UploadFile
from .models import AddDocRequest, QueryRequest
from .rag import add_document, retrieve_similar_documents
from .ingest import ingestfile

app = FastAPI()

@app.post("/add")
async def add_doc(request: AddDocRequest):
    result = add_document(request.text)
    return result

@app.post("/query")
async def query_docs(request: QueryRequest):
    similar_texts = retrieve_similar_documents(request.query, top_k=request.top_k)
    return {"query": request.query, "docs": similar_texts}

@app.post("/ingestfile")
async def ingest_file(file: UploadFile):
    result = ingestfile(file)
    return result