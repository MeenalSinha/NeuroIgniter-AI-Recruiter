from typing import List, Dict, Any
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    BM25Okapi = None
    SentenceTransformer = None

class HybridRetriever:
    def __init__(self, candidates: List[Dict[str, Any]]):
        self.candidates = candidates
        self.bm25 = None
        self.model = None
        self.corpus_embeddings = None
        
        if BM25Okapi and SentenceTransformer:
            self._prepare_bm25()
            self._prepare_dense()
            
    def _prepare_bm25(self):
        corpus = [
            " ".join(c.get('skills', []) + [e.get('title', '') for e in c.get('experience', [])])
            for c in self.candidates
        ]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def _prepare_dense(self):
        # Load a small fast model for dense embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus = [
            " ".join(c.get('skills', []) + [e.get('title', '') for e in c.get('experience', [])])
            for c in self.candidates
        ]
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.model:
            # Fallback if libraries not installed
            return self.candidates[:top_k]
            
        # BM25 Lexical Score
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Dense Semantic Score
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()
        
        # Hybrid Score (Simple linear combination)
        # Normalize BM25 scores to [0, 1] for combination
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / bm25_max for s in bm25_scores]
        
        hybrid_scores = [0.4 * bm25_norm[i] + 0.6 * dense_scores[i] for i in range(len(self.candidates))]
        
        # Rank
        ranked_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
        
        results = []
        for idx in ranked_indices[:top_k]:
            c = self.candidates[idx].copy()
            c['_hybrid_score'] = float(hybrid_scores[idx])
            results.append(c)
            
        return results
