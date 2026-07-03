"""
NeuroIgniter Skill Ontology
===========================
Canonical skill names → set of aliases/synonyms.
Built from real-world variation in job postings and CVs.

This is the semantic layer that costs zero compute:
  "SBERT" → sentence transformers
  "ANN" → vector search
  "IR" → information retrieval
"""

import os

# ── CANONICAL → ALIASES ───────────────────────────────────────────────────────
# Keys are canonical names that match our scoring taxonomy.
# Values are frozensets of all equivalent strings (lowercased).

SKILL_ALIASES: dict[str, frozenset[str]] = {

    # ── Vector Search & Retrieval ──────────────────────────────────────────
    "faiss": frozenset({
        "faiss", "facebook ai similarity search", "faiss index",
        "faiss vector search", "ivf index", "pq index", "flat index",
    }),
    "vector search": frozenset({
        "vector search", "similarity search", "semantic search",
        "dense retrieval", "ann search", "approximate nearest neighbor",
        "ann", "hnsw", "approximate nearest neighbor search",
        "knn search", "k-nearest neighbor search", "nearest neighbor search",
        "vector retrieval", "dense vector search", "embedding search",
        "vector similarity", "cosine similarity search",
    }),
    "weaviate": frozenset({
        "weaviate", "weaviate vector db", "weaviate vector database",
    }),
    "pinecone": frozenset({
        "pinecone", "pinecone db", "pinecone vector db",
    }),
    "qdrant": frozenset({
        "qdrant", "qdrant db", "qdrant vector database",
    }),
    "milvus": frozenset({
        "milvus", "milvus vector db",
    }),
    "elasticsearch": frozenset({
        "elasticsearch", "elastic search", "elastic", "es",
        "elasticsearch bm25", "elasticsearch knn",
    }),
    "opensearch": frozenset({
        "opensearch", "open search", "amazon opensearch",
        "aws opensearch",
    }),
    "chromadb": frozenset({
        "chromadb", "chroma", "chroma db", "chroma vector store",
    }),
    "pgvector": frozenset({
        "pgvector", "pg vector", "postgres vector", "postgresql vector",
    }),
    "redis vector": frozenset({
        "redis vector", "redis vss", "redis ai",
    }),

    # ── Embeddings ─────────────────────────────────────────────────────────
    "embeddings": frozenset({
        "embeddings", "text embeddings", "vector embeddings", "dense embeddings",
        "embedding model", "embedding vector", "sentence embeddings",
        "document embeddings", "word embeddings", "contextual embeddings",
    }),
    "sentence transformers": frozenset({
        "sentence transformers", "sentence-transformers", "sbert",
        "sentence bert", "sentence transformer", "bi-encoder",
        "dual encoder", "siamese bert", "all-minilm",
        "paraphrase-mpnet", "e5", "bge", "bge-m3",
        "nomic-embed", "openai embeddings", "text-embedding-ada",
        "cohere embeddings", "jina embeddings",
    }),

    # ── Retrieval Systems ──────────────────────────────────────────────────
    "information retrieval": frozenset({
        "information retrieval", "ir", "search relevance", "search quality",
        "document retrieval", "passage retrieval", "text retrieval",
        "retrieval system", "search engineering", "search infrastructure",
    }),
    "bm25": frozenset({
        "bm25", "bm 25", "okapi bm25", "sparse retrieval",
        "tf-idf retrieval", "tfidf retrieval", "inverted index",
        "keyword search", "lexical search",
    }),
    "hybrid search": frozenset({
        "hybrid search", "hybrid retrieval", "sparse-dense hybrid",
        "dense-sparse hybrid", "dense + sparse", "reciprocal rank fusion",
        "rrf", "rank fusion", "late interaction",
    }),
    "retrieval": frozenset({
        "retrieval", "retrieve", "document ranking", "passage ranking",
        "search pipeline", "query pipeline", "search system",
    }),
    "ranking": frozenset({
        "ranking", "rank", "reranking", "re-ranking", "result ranking",
        "search ranking", "candidate ranking", "document ranking",
    }),
    "reranking": frozenset({
        "reranking", "re-ranking", "cross-encoder", "cross encoder",
        "pointwise reranker", "listwise reranker", "pairwise reranker",
        "cohere rerank", "colbert", "colbert v2", "late interaction model",
    }),
    "rag": frozenset({
        "rag", "retrieval augmented generation", "retrieval-augmented generation",
        "rag pipeline", "rag system", "rag architecture",
    }),
    "learning to rank": frozenset({
        "learning to rank", "ltr", "lambdamart", "ranknet", "listnet",
        "lambdarank", "rankboost", "gbrt ranking",
    }),

    # ── LLMs ──────────────────────────────────────────────────────────────
    "llm": frozenset({
        "llm", "large language model", "large language models",
        "foundation model", "language model", "language models",
        "gpt", "gpt-4", "gpt-3.5", "claude", "gemini", "llama",
        "mistral", "falcon", "palm", "generative ai",
    }),
    "fine-tuning": frozenset({
        "fine-tuning", "finetuning", "fine tuning", "instruction tuning",
        "rlhf", "dpo", "sft", "supervised fine-tuning", "model adaptation",
        "model fine-tuning",
    }),
    "lora": frozenset({
        "lora", "low-rank adaptation", "qlora", "q-lora",
        "peft", "parameter efficient fine tuning",
        "adapter tuning", "soft prompt tuning", "prefix tuning",
    }),
    "hugging face": frozenset({
        "hugging face", "huggingface", "hf", "transformers library",
        "transformers", "datasets library", "accelerate", "peft library",
        "trl", "huggingface hub", "model hub",
    }),

    # ── ML Frameworks ─────────────────────────────────────────────────────
    "pytorch": frozenset({
        "pytorch", "torch", "libtorch", "pytorch lightning",
        "lightning", "pytorch-lightning",
    }),
    "tensorflow": frozenset({
        "tensorflow", "tf", "keras", "tf.keras", "tensorflow 2",
    }),
    "sklearn": frozenset({
        "sklearn", "scikit-learn", "scikit learn",
    }),
    "xgboost": frozenset({
        "xgboost", "xgb",
    }),
    "lightgbm": frozenset({
        "lightgbm", "lgbm", "light gbm",
    }),
    "catboost": frozenset({
        "catboost", "cat boost", "yandex catboost",
    }),
    "gradient boosting": frozenset({
        # Generic/ambiguous terms that don't specify which GBM implementation —
        # mapped to a neutral parent concept rather than guessing a specific
        # library. A candidate writing "GBM" could mean XGBoost, LightGBM, or
        # CatBoost; crediting them with one specific tool would be a false claim.
        "gradient boosting", "gbm", "gradient boosted trees", "boosted trees",
    }),

    # ── NLP ───────────────────────────────────────────────────────────────
    "nlp": frozenset({
        "nlp", "natural language processing", "natural language understanding",
        "nlu", "computational linguistics", "text mining",
        "text analytics", "language processing",
    }),
    "bert": frozenset({
        "bert", "roberta", "albert", "electra", "deberta",
        "distilbert", "bert-base", "bert-large", "multilingual bert",
    }),

    # ── Python ─────────────────────────────────────────────────────────────
    "python": frozenset({
        "python", "python 3", "python3", "cpython", "py",
        "python programming", "python development",
    }),

    # ── MLOps & Serving ───────────────────────────────────────────────────
    "mlflow": frozenset({
        "mlflow", "ml flow", "experiment tracking",
    }),
    "wandb": frozenset({
        "wandb", "weights & biases", "weights and biases",
        "w&b", "experiment tracking",
    }),
    "model serving": frozenset({
        "model serving", "model deployment", "inference serving",
        "triton", "triton inference server", "torchserve", "bentoml",
        "seldon", "kfserving", "mlserver",
    }),
    "mlops": frozenset({
        "mlops", "ml ops", "ml operations", "mlops platform",
        "ml platform", "model registry", "model monitoring",
        "feature store", "data versioning",
    }),
    "kubeflow": frozenset({
        "kubeflow", "kube flow", "kubeflow pipelines",
    }),

    # ── Evaluation Metrics ────────────────────────────────────────────────
    "ndcg": frozenset({
        "ndcg", "normalized discounted cumulative gain",
        "ndcg@10", "ndcg@100", "ranking evaluation", "ranking metrics",
    }),
    "mrr": frozenset({
        "mrr", "mean reciprocal rank", "reciprocal rank",
    }),
    "map": frozenset({
        "map", "mean average precision", "average precision",
    }),
    "a/b testing": frozenset({
        "a/b testing", "ab testing", "a-b testing", "online evaluation",
        "controlled experiment", "randomized experiment",
        "split testing", "online a/b test",
    }),

    # ── Infrastructure ─────────────────────────────────────────────────────
    "distributed systems": frozenset({
        "distributed systems", "distributed computing",
        "large-scale systems", "scale", "horizontal scaling",
    }),
    "kafka": frozenset({
        "kafka", "apache kafka", "kafka streams", "event streaming",
    }),

    # ── Emerging ──────────────────────────────────────────────────────────
    "llamaindex": frozenset({
        "llamaindex", "llama index", "llama-index",
    }),
    "langchain": frozenset({
        "langchain", "lang chain",
    }),
    "semantic search": frozenset({
        "semantic search", "neural search", "dense search",
        "meaning-based search", "conceptual search",
    }),
    "prompt engineering": frozenset({
        "prompt engineering", "prompt design", "system prompt",
        "few-shot prompting", "chain of thought", "cot",
        "prompt optimization",
    }),
}

# ── REVERSE INDEX: alias → canonical ────────────────────────────────────────
ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias] = canonical
    ALIAS_TO_CANONICAL[canonical] = canonical  # Self-maps too


def normalize_skill(skill_name: str) -> str:
    """Normalize a skill name to its canonical form."""
    normalized = skill_name.lower().strip()
    return ALIAS_TO_CANONICAL.get(normalized, normalized)


def skills_are_equivalent(skill_a: str, skill_b: str) -> bool:
    """Check if two skill names refer to the same underlying skill."""
    return normalize_skill(skill_a) == normalize_skill(skill_b)


# ── SKILL CLUSTERS: related skill groups for coherence scoring ───────────────
SKILL_CLUSTERS = {
    "vector_retrieval": {
        "faiss", "vector search", "weaviate", "pinecone", "qdrant",
        "milvus", "elasticsearch", "opensearch", "chromadb",
        "pgvector", "redis vector", "hybrid search",
    },
    "embeddings_encoding": {
        "embeddings", "sentence transformers", "bert", "hugging face",
    },
    "ranking_systems": {
        "ranking", "reranking", "learning to rank", "bm25",
        "information retrieval", "ndcg", "mrr", "map",
    },
    "llm_ecosystem": {
        "llm", "fine-tuning", "lora", "hugging face", "rag",
        "prompt engineering", "llamaindex", "langchain",
    },
    "ml_core": {
        "python", "pytorch", "tensorflow", "sklearn",
        "xgboost", "lightgbm", "gradient boosting", "mlflow", "wandb",
    },
    "production_ml": {
        "mlops", "model serving", "kubeflow", "a/b testing",
        "distributed systems",
    },
    "nlp_core": {
        "nlp", "bert", "sentence transformers", "hugging face",
    },
}


def get_skill_cluster(canonical_skill: str) -> list[str]:
    """Return the cluster names this skill belongs to."""
    clusters = []
    for cluster_name, members in SKILL_CLUSTERS.items():
        if canonical_skill in members:
            clusters.append(cluster_name)
    return clusters


def compute_cluster_coherence(canonical_skills: set[str]) -> float:
    """
    Score how coherent/specialized the candidate's skills are.
    A candidate with FAISS + Weaviate + Elasticsearch + Qdrant + BM25
    is demonstrably an IR specialist, vs someone with one of those skills.

    Anti-gaming constraint: a candidate who lists 5 vector databases
    (FAISS, Weaviate, Pinecone, Qdrant, Milvus) but nothing else IR-related
    knows "many vector DB names," not necessarily deep IR expertise.
    Full coherence (1.0) requires depth in the primary cluster AND
    presence in at least one adjacent cluster — real specialists span
    retrieval + ranking + evaluation, not just a tool inventory.

    Returns 0.0-1.0.
    """
    if not canonical_skills:
        return 0.0

    cluster_counts: dict[str, int] = {}
    for skill in canonical_skills:
        for cluster in get_skill_cluster(skill):
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    if not cluster_counts:
        return 0.0

    # Find the most populated cluster
    max_count = max(cluster_counts.values())
    best_cluster = max(cluster_counts, key=cluster_counts.get)
    cluster_size = len(SKILL_CLUSTERS[best_cluster])

    # Base coherence = fraction of best cluster covered
    base_coherence = min(max_count / max(cluster_size * 0.4, 1), 1.0)

    # Anti-gaming: require presence in >=2 distinct clusters to reach full score.
    # A candidate with skills in only ONE cluster (e.g. 5 vector DB names and
    # nothing else) is capped at 0.70 — real depth, not full specialist credit.
    distinct_clusters = len(cluster_counts)
    if distinct_clusters < 2:
        return min(base_coherence, 0.70)

    return base_coherence


# ── CORROBORATION TERMS: career text → skill domain evidence ────────────────
# For each skill domain, what terms in career descriptions would corroborate it?
#
# Single source of truth: loaded from config/config.yaml's `semantic.corroboration_terms`
# section so editing config.yaml actually changes behavior (previously this was
# duplicated — a hardcoded copy here that config.yaml's copy had no effect on).
# Falls back to the hardcoded defaults below if config is unavailable, so the
# module still works standalone (e.g. in isolated unit tests).
_DEFAULT_CORROBORATION_MAP: dict[str, list[str]] = {
    "vector_retrieval": [
        "retrieval", "search", "index", "indexed", "indexing", "ranking", "ranked",
        "similarity", "nearest neighbor", "ann", "approximate", "vector",
        "embedding search", "semantic", "dense", "sparse",
    ],
    "embeddings_encoding": [
        "embedding", "embed", "vector representation", "encode", "encoding",
        "sentence", "document", "dense", "semantic",
    ],
    "ranking_systems": [
        "ranking", "ranked", "relevance", "precision", "recall", "ndcg",
        "mrr", "retrieval", "search quality", "rerank", "ltr",
    ],
    "llm_ecosystem": [
        "language model", "llm", "gpt", "claude", "gemini", "llama",
        "generation", "prompt", "rag", "fine-tun", "instruction",
    ],
    "production_ml": [
        "production", "deployed", "serving", "inference", "latency", "throughput",
        "real users", "scale", "monitoring", "a/b test", "endpoint",
    ],
    "nlp_core": [
        "nlp", "natural language", "text", "sentiment", "classification",
        "extraction", "entity", "summarization", "translation",
    ],
}


def _load_corroboration_map() -> dict[str, list[str]]:
    """Load corroboration terms from config.yaml; fall back to defaults."""
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        terms = cfg.get('semantic', {}).get('corroboration_terms')
        if terms and isinstance(terms, dict):
            return terms  # type: ignore[no-any-return]
    except Exception:
        pass
    return _DEFAULT_CORROBORATION_MAP


CORROBORATION_MAP: dict[str, list[str]] = _load_corroboration_map()


def get_corroboration_score(canonical_skill: str, career_text: str) -> float:
    """
    Returns 0.0-1.0 indicating how much career text corroborates a claimed skill.

    Scoring tiers (intended behavior):
      1.0  — canonical skill name or a close alias appears directly in career text
             (e.g. "faiss" in text when skill is "faiss" → strongest signal)
      0.65-0.99 — multiple cluster-specific technical terms present
      0.40-0.64 — a few cluster terms present, some specificity
      0.30      — zero cluster-term matches; career text provides no evidence
      0.65      — unknown cluster (benefit of the doubt, can't assess)

    The prior version applied a blanket floor of 0.40 for ALL zero-match cases,
    meaning generic "production at scale" text could achieve 0.40 on any skill
    simply because it triggered the floor rather than any actual term match.
    This is now fixed: zero matches returns 0.30 (weak-evidence floor), not 0.40.
    """
    career_lower = career_text.lower()
    clusters = get_skill_cluster(canonical_skill)

    if not clusters:
        return 0.65  # Unknown cluster → mild benefit of the doubt

    # Tier 1: canonical skill name or known alias appears directly in career text.
    # This is the strongest possible corroboration signal — they literally named
    # the specific technology, which is nearly impossible to fake credibly.
    # Check both the canonical form and all known aliases.
    skill_aliases = SKILL_ALIASES.get(canonical_skill, frozenset())
    all_forms = {canonical_skill} | set(skill_aliases)
    for form in all_forms:
        if len(form) >= 3 and form in career_lower:
            return 1.0  # Direct name mention → full corroboration

    # Tier 2: cluster-level term matching (indirect corroboration).
    # How many of the skill's cluster's characteristic terms appear?
    best_corroboration = 0.0
    for cluster in clusters:
        terms = CORROBORATION_MAP.get(cluster, [])
        if not terms:
            best_corroboration = max(best_corroboration, 0.65)
            continue

        matches = sum(1 for term in terms if term in career_lower)

        if matches == 0:
            # No cluster-specific terms at all → no career evidence for this skill.
            # Return a LOW floor (0.30), not the old 0.40 floor.
            # The gap between 0.30 (no evidence) and 0.40+ (some evidence) creates
            # a meaningful incentive to actually demonstrate the skill in descriptions.
            cluster_score = 0.30
        elif matches == 1:
            # One matching term is weak evidence (many terms are semi-generic)
            cluster_score = 0.42
        else:
            # 2+ matches: genuine corroboration, scale linearly
            # threshold = 30% of terms for full score
            cluster_score = min(matches / max(len(terms) * 0.25, 2), 1.0)
            # Cap at 0.90 for cluster-only evidence (reserve 0.91-1.0 for direct name mention)
            cluster_score = min(cluster_score, 0.90)

        best_corroboration = max(best_corroboration, cluster_score)

    return best_corroboration
