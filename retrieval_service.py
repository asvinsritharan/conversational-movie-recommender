import os
os.environ["OMP_NUM_THREADS"] = "4"       # FAISS threading, isolated to this process
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from retriever import HybridRetriever

app = FastAPI()
print("loading retriever (FAISS + BM25 + embedder)...")
retriever = HybridRetriever()             # embedder can go back to MPS here — no MLX in this process
print("retrieval service ready")

class Query(BaseModel):
    query: str
    k: int = 3

@app.post("/search")
def search(q: Query):
    hits = retriever.search(q.query, k=q.k)
    return {"results": [{"item_id": h["item_id"], "title": h["title"]} for h in hits]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)