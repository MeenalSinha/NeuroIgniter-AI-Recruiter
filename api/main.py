from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neuroigniter.retrieval.hybrid import HybridRetriever

app = FastAPI(title="NeuroIgniter API", version="2.0")
retriever = None

# On startup, load candidates and initialize retriever
@app.on_event("startup")
def startup_event():
    global retriever
    # Find candidates.jsonl
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    jsonl_path = os.path.join(base_dir, 'India_runs_data_and_ai_challenge', 'candidates.jsonl')
    
    candidates = []
    if os.path.exists(jsonl_path):
        # Only load a subset for memory constraints during demo
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5000: # limit to 5k for fast startup
                    break
                try:
                    c = json.loads(line)
                    candidates.append(c)
                except:
                    pass
        print(f"Loaded {len(candidates)} candidates.")
        retriever = HybridRetriever(candidates)
    else:
        print(f"Dataset not found at {jsonl_path}")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class FeedbackRequest(BaseModel):
    candidate_id: str
    is_positive: bool

@app.get("/")
def read_root():
    return {"status": "ok", "message": "NeuroIgniter API V2 Running", "retriever_ready": retriever is not None}

@app.post("/search")
def search_candidates(request: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    results = retriever.search(request.query, top_k=request.top_k)
    # Return minimal fields for the response
    out = []
    for r in results:
        out.append({
            "candidate_id": r.get("candidate_id"),
            "score": r.get("_hybrid_score"),
            "name": r.get("profile", {}).get("name", "Unknown")
        })
    return {"query": request.query, "results": out}

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    # Log feedback to file
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(out_dir, exist_ok=True)
    fb_path = os.path.join(out_dir, 'feedback.jsonl')
    with open(fb_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"candidate_id": request.candidate_id, "is_positive": request.is_positive}) + "\n")
    return {"status": "success", "logged": request.dict()}
