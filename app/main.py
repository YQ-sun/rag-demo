from fastapi import BackgroundTasks, FastAPI,UploadFile
from .models import AddDocRequest, QueryRequest
from .rag import add_document
from .retriever import embedding_retriever, bm25_retriever, build_bm25_index
from .ingest import ingestfile
from .batch_ingest import batch_ingest
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.post("/add")
async def add_doc(request: AddDocRequest):
    result = add_document(request.text)
    return result

@app.post("/query")
async def query_docs(request: QueryRequest):
    vector_hits = embedding_retriever(request.query, top_k=request.top_k)
    bm25_hits = bm25_retriever(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "retrievers": {
            "vector": vector_hits,
            "bm25": bm25_hits
        }
    }

@app.post("/bm25/reindex")
async def reindex():
    build_bm25_index()
    return {"status": "ok", "message": "BM25 index rebuilt successfully."}

@app.get("/", response_class=HTMLResponse)
def index_page():
    html = """
    <html>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>RAG Retriever Demo</h2>
        <form onsubmit="sendQuery(event)">
            <input id="q" type="text" placeholder="Enter query..." style="width:100%; padding:8px"/>
            <input id="k" type="number" placeholder="top_k" value="3" min="1" style="width:100%; padding:8px; margin-top:5px"/>
            <button type="submit" style="margin-top:10px">Search</button>
        </form>
        <h3>Results</h3>
        <pre id="result" style="background:#f2f2f2; padding:10px"></pre>
    </body>

    <script>
        async function sendQuery(e){
            e.preventDefault();
            const q = document.getElementById('q').value;
            const k = parseInt(document.getElementById('k').value);
            const resultBox = document.getElementById('result');

            const res = await fetch("/query", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({"query": q, "top_k": k})
            });

            const data = await res.json();
            resultBox.textContent = JSON.stringify(data, null, 2);
        }
    </script>
    </html>
    """
    return html


@app.post("/ingestfile")
async def ingest_file(file: UploadFile):
    result = ingestfile(file)
    return result
