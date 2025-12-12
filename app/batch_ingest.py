import os
from app.db import get_db
from app.rag import get_embedding
import pdfplumber

DOCS_PATH = "data/docs"

CHUNK_SIZE = 500  # 每个文本块的大小
CHUNK_OVERLAP = 50  # 文本块之间的重叠部分

def chunk_text(text:str):
    """将文本分割成较小的块"""
    words= text.split()
    chunks = []
    start = 0
    words_length = len(words)

    while start < words_length:
        end = min(start + CHUNK_SIZE, words_length)
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def load_file(file_path: str):
    """加载文件内容"""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None
    
def batch_ingest():
    table= get_db()
    files=[f for f in os.listdir(DOCS_PATH) if f.endswith((".pdf", ".txt"))][:10]
    total_chunks=0
    for idx,file_name in enumerate(files,start=1):
        file_path=os.path.join(DOCS_PATH, file_name)
        text= load_file(file_path)
        if not text:
            continue
        chunks= chunk_text(text)
        for chunk in chunks:
            embedding= get_embedding(chunk)
            doc_id= table.count_rows() + 1
            table.add([{
                "id": doc_id,
                "text": chunk,
                "embedding": embedding
            }])
            total_chunks += 1
        print(f"已处理文件 {idx}/{len(files)}: {file_name}, 分块数: {len(chunks)}")
    print(f"批量导入完成，总分块数: {total_chunks}")

if __name__ == "__main__":
    batch_ingest()
    