from fastapi import BackgroundTasks, FastAPI,UploadFile
from .models import AddDocRequest, QueryRequest
from .rag import add_document
from .retriever import embedding_retriever, bm25_retriever, build_bm25_index
from .ingest import ingestfile
from .batch_ingest import batch_ingest
from fastapi.responses import HTMLResponse
from .chat import chat

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

@app.post("/chat")
async def chat_api(request: QueryRequest):
    #return chat(request.query, use_history=True, top_k=request.top_k)
    conversation_id=getattr(request, 'conversation_id', None)
    return chat(request.query, conversation_id=request.conversation_id, top_k=request.top_k)
   
@app.get("/", response_class=HTMLResponse)
def index_page():
    html = """
    <html>
    <head>
        <title>RAG Multi-turn Chat</title>
        <style>
            body { font-family:sans-serif; max-width:600px; margin:40px auto; }
            #chat { border:1px solid #ccc; padding:10px; height:400px; overflow-y:auto; background:#f9f9f9; }
            #userInput { width:100%; padding:8px; }
            button { padding:6px 12px; margin-top:5px; }
        </style>
    </head>
    <body>
        <h2>RAG Multi-turn Chat Demo</h2>
        <div id="chat"></div>
        <input id="userInput" type="text" placeholder="Ask something..." />
        <button onclick="sendMessage()">Send</button>

        <script>
            // 前端不必传 conversation_id 和 top_k
            async function sendMessage(){
                const input = document.getElementById("userInput");
                const chatDiv = document.getElementById("chat");
                const userMsg = input.value;
                if(!userMsg) return;

                chatDiv.innerHTML += "<b>You:</b> " + userMsg + "<br/>";
                input.value = "";

                const res = await fetch("/chat", {
                    method:"POST",
                    headers:{"Content-Type":"application/json"},
                    body: JSON.stringify({"query": userMsg})  // 只发送 query
                });

                const data = await res.json();
                chatDiv.innerHTML += "<b>Bot:</b> " + data.answer + "<br/><hr/>";
                chatDiv.scrollTop = chatDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return html

@app.post("/ingestfile")
async def ingest_file(file: UploadFile):
    result = ingestfile(file)
    return result
