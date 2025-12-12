from fastapi import UploadFile, HTTPException
from typing import List
import pdfplumber
from .db import get_db
from .rag import get_embedding

CHUNK_SIZE = 500  # Number of characters per chunk
OVERLAP_SIZE = 50  # Number of overlapping characters between chunks

def chunk_text(text:str) -> List[str]:
    words= text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP_SIZE
    return chunks

def ingestfile(file:UploadFile):
    #upload pdf file
    if not file.filename.endswith(('.txt','.pdf')):
        raise HTTPException(status_code=400,detail="仅支持txt和pdf文件")
    
    text=""
    if file.filename.endswith('.txt'):
        text=file.file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text+=page.extract_text() + "\n"

    chunks=chunk_text(text)
    table=get_db()
    
    for chunk in chunks:
        embedding=get_embedding(chunk)
        table.add([{"id":table.count()+1,"text":chunk,"embedding":embedding}])
    
    return {"status":"ok","chunks":len(chunks)}

