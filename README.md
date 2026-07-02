# NeuroIgniter AI Recruiter

**India Runs Data & AI Challenge — Redrob Intelligent Candidate Discovery**
**Team: NeuroIgniter**

---

## What This System Does — And Why It's Not Keyword Matching

Traditional ATS systems rank "FAISS" against "FAISS." They miss the candidate who wrote "ANN-based similarity search" and "bi-encoder retrieval" because they never used the exact word "FAISS."

NeuroIgniter ranks candidates the way an experienced technical recruiter would — by understanding what they've actually done, cross-referencing their claimed skills against their career evidence, rewarding genuine IR specialists over generalists who list every framework, and deprioritizing candidates who look good on paper but never respond.

**Three concrete examples of what this system catches that keyword matching misses:**

### Example 1: The Alias Problem
> A candidate lists `SBERT`, `ANN`, `dense retrieval` — but not `sentence transformers`, `vector search`, or `FAISS`.
>
> Keyword matcher: 0 must-have skills matched → ranked at position 4,000+
> NeuroIgniter: ontology expansion maps SBERT → sentence transformers, ANN → vector search → ranked in top 200 for further scoring

### Example 2: The Self-Report Problem
> A candidate lists `FAISS (advanced, 48 months)` but their career descriptions mention only "Excel, Tableau, financial reporting."
>
> Keyword matcher: gives full credit for FAISS → high ranking
> NeuroIgniter: corroboration check finds zero retrieval/vector/index terms in career text → applies 0.5× confidence penalty → skill contribution halved → falls out of top-100

### Example 3: The Ghost Candidate Problem
> A candidate has 9 years of perfect-match ML experience but their Redrob profile shows 3% recruiter response rate, last active 8 months ago, not open to work.
>
> Keyword matcher: ranks highly (skill match is strong)
> NeuroIgniter: behavioral signals collapse availability score → ghost candidate penalty applies → drops 200+ ranks → replaced by a slightly weaker candidate who is actively looking and responds within 6 hours

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Job Description (any role, any text)          │
└────────────────────────┬────────────────────────────────────────┘
                         │  parsed AT RUNTIME by jd_parser.py
                         │  (section detection → skill extraction →
                         │   structured fields → culture inference)
          ┌──────────────▼──────────────────────────────────────┐
          │              JD Intelligence Layer                    │
          │  • Must-have skills extracted from "absolutely        │
          │    need" section (18 for the challenge JD)            │
          │  • Nice-to-have skills extracted from "like to        │
          │    have" section (5 for the challenge JD)             │
          │  • Disqualifier firms extracted from "do NOT want"    │
          │  • YOE band, notice period parsed via regex           │
          │  • Culture signals inferred from JD language          │
          │  • Extraction confidence: 1.00 for challenge JD       │
          │  • TF-IDF semantic text built from extracted terms    │
          └──────────────┬──────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────┐
│         Skill Ontology (ontology.py) — 200+ alias mappings         │
│                                                                    │
│   "SBERT"         → sentence transformers                          │
│   "ANN"           → vector search                                  │
│   "IR"            → information retrieval                          │
│   "LLaMA"         → llm                                            │
│   "LangChain"     → langchain                                      │
│   "XGBoost"       → xgboost                                        │
│   + 200 more...                                                    │
│                                                                    │
│   Skill clusters: vector_retrieval · embeddings_encoding ·        │
│   ranking_systems · llm_ecosystem · ml_core · production_ml       │
└────────────────────────┬──────────────────────────────────────────┘
                         │  applied per-candidate
┌────────────────────────▼──────────────────────────────────────────┐
│              7-Stage Scoring Pipeline (ranker.py)                  │
│                                                                    │
│  Stage 1 │ Honeypot Detection                                      │
│           │ • impossible skill duration/proficiency                │
│           │ • career timeline inflation                            │
│           │ → instant score ≈ 0.001                               │
│                                                                    │
│  Stage 2 │ Hard Disqualifiers (from JD)                           │
│           │ • consulting-firm-only career                          │
│           │ • CV/robotics primary without NLP/IR                  │
│           │ → score ≈ 0.01                                        │
│                                                                    │
│  Stage 3 │ Skill Taxonomy Match        (38% weight)               │
│           │ • ontology-expanded matching                           │
│           │ • proficiency × duration weighting                    │
│           │ • corroboration: cross-refs career text               │
│           │ • cluster coherence (IR specialist bonus)             │
│                                                                    │
│  Stage 4 │ Career Trajectory          (26% weight)               │
│           │ • YOE band vs JD range                                │
│           │ • title relevance + seniority ladder                  │
│           │ • product vs consulting company detection             │
│           │ • promotion detection (title advancement at company)  │
│           │ • achievement extraction (quantified impact)          │
│           │ • tenure consistency (anti job-hopper)                │
│           │ • production AI evidence from descriptions            │
│                                                                    │
│  Stage 5 │ Behavioral Signals         (12% weight)               │
│           │ • recruiter_response_rate (ghost candidate check)     │
│           │ • recency (last_active_date → decay curve)           │
│           │ • GitHub activity score                               │
│           │ • skill assessment scores (verified ability)         │
│           │ • interview_completion_rate + offer_acceptance_rate   │
│           │ • passive demand: saved_by_recruiters + appearances   │
│                                                                    │
│  Stage 6 │ Availability               (8% weight)                │
│           │ • open_to_work × notice_period × location            │
│           │ • interaction: (open_to_work AND notice≤30 AND rr≥70%)│
│           │   → multiplicative boost (not just additive sum)      │
│                                                                    │
│  Stage 7 │ Hybrid Semantic: TF-IDF + Distributional Embedding (10%)│
│           │ • cosine similarity: candidate text vs JD vocabulary  │
│           │ • ontology-expanded candidate text                    │
│           │ + Education (6% weight)                               │
│                                                                    │
│  Modifiers │ India bonus, GitHub bonus, must-have depth bonus,    │
│             IR specialist cluster bonus                            │
│             Ghost penalty, no-production-evidence penalty          │
│                                                                    │
│  Output: score ∈ (0, 1) + component breakdown + reasoning         │
└────────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────┐
│         Ranked Top-100 CSV                                         │
│  candidate_id · rank · score (6dp) · reasoning (unique per cand.) │
│                                                                    │
│  Reasoning includes:                                               │
│  • Specific matched skills from this candidate's profile          │
│  • Top career strength                                            │
│  • Behavioral highlight (response rate, GitHub, availability)     │
│  • Honest concern (even for rank-1)                               │
│  • Full component score decomposition:                            │
│    [Skills 0.82 | Career 0.79 | Behavioral 0.85 | ...]           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Performance

| Metric | Value |
|--------|-------|
| 100K candidates, CPU-only | **~50 seconds** |
| Peak memory (measured) | **~26 MB** — streaming, bounded-buffer scoring |
| External dependencies | pyyaml (graceful fallback if absent) |
| GPU required | No |
| Network required | No |
| External API calls | No |
| Honeypots detected | 8 (impossible profiles) |
| Consulting-only disqualified | ~9,700 |
| Zero skill match | ~73,500 |

Memory was measured by sampling `/proc/<pid>/status` `VmRSS` every 50ms across the full 100K-candidate run (986 samples). Peak memory does not scale with dataset size — the scorer streams candidates one at a time and maintains only a bounded buffer (`top_n × 5`, default 500 candidates), so a 1M-candidate run uses the same peak memory as a 100K run. A full per-run filtering audit trail is written to `output/ranking_stats.json` and surfaced in the dashboard's Insights tab.

### Why this architecture is correct at scale

The JD describes exactly this problem: "we have 100K candidates and need fast, accurate ranking." Production recruiting systems at this scale use two-stage retrieval:

- **Stage 1** (this system): fast lexical + rule-based pre-filtering over all candidates
- **Stage 2** (production extension): GPU-accelerated cross-encoder reranking of top-500

Within the CPU-only, 5-minute competition constraint, Stage 2 is implemented as calibrated multi-signal behavioral scoring — the same signals a cross-encoder would use, without the GPU requirement. This is honest engineering, not a workaround.

---

## JD Understanding Is Runtime-Parsed, Not Hardcoded

`src/neuroigniter/jd_parser.py` extracts requirements from the actual JD text at import time — it does not contain a manually maintained skill list. The system genuinely adapts when the JD changes.

```python
from neuroigniter.jd_parser import parse_jd

# A completely different JD — different role, different stack, different company
different_jd = """
Job Description: Senior Data Engineer
Experience Required: 4-8 years
Things you absolutely need:
- Strong experience with Apache Kafka and Spark for streaming data pipelines
- Production experience with PostgreSQL and Redis
- Strong Python and SQL skills
Things we explicitly do NOT want:
- People who have only worked at consulting firms (TCS, Infosys, Accenture)
Notice period: We prefer 15-day notice.
"""

req = parse_jd(different_jd)
# req.must_have_skills      -> ['kafka', 'python']
# req.yoe_min, req.yoe_max  -> 4.0, 8.0
# req.ideal_notice_days     -> 15
# req.consulting_disqualifiers -> ['tcs', 'infosys', 'accenture']
```

How it works:
1. **Section detection** — regex patterns locate "Things you absolutely need:", "Things we'd like you to have:", "Things we explicitly do NOT want:" style section headers (and common variants: "Required Skills:", "Preferred Qualifications:", "Minimum Qualifications:", etc.) — this works for any role or industry, since it's structural, not vocabulary-dependent.
2. **Skill extraction (two-tier)**:
   - **High-precision tier**: scans for any of 200+ AI/ML/IR-specific terms via the ontology, with full alias normalization (`"SBERT"` → `sentence transformers`). This tier is used whenever the JD's vocabulary overlaps the ontology — e.g. the challenge JD extracts 18 must-have skills this way, all correctly normalized.
   - **Generic fallback tier**: for JDs outside the AI/ML domain (frontend, DevOps, security, etc.), where the ontology finds fewer than 2 matches, a generic extractor pulls capitalized tech-looking tokens and parenthetical comma-lists (`"(Redux, Zustand)"`, `"(Jenkins, GitHub Actions)"`). These are lower-precision — no alias normalization, just literal extracted strings — and are honestly weaker than the ontology tier, but they prevent the system from silently returning zero must-have skills for any role outside AI/ML, which was a real bug caught during testing (see below).
3. **Structured field extraction** — YOE range, notice period, location preferences via targeted regex (fully domain-independent)
4. **Culture inference** — scans for language patterns indicating product vs. research orientation, startup mindset, ownership expectations, external validation preferences
5. **Extraction confidence score** — `JD_REQUIREMENTS.extraction_confidence` (0.0–1.0) reports how many of the 5 structural signals were successfully parsed. For the challenge JD this is **1.00**.

**A bug we found and fixed while building this**: an earlier version of `_extract_skills_from_text` only used the ontology tier. Testing it against Frontend Engineer, Cybersecurity Engineer, and DevOps Engineer JDs revealed it silently returned `must_have_skills: []` for all three — because "React", "TypeScript", "Kubernetes", and "Terraform" simply aren't AI/ML vocabulary. Since `must_have_skills` drives 38% of every candidate's composite score, this would have meant every candidate scored identically on skills for any non-AI/ML role. The generic fallback tier above was added specifically to close this gap, and `tests/test_ranker_v2.py::test_parser_does_not_silently_return_empty_skills_for_non_ai_roles` guards against it regressing.

```python
# Verified output after the fix:
parse_jd(frontend_jd).must_have_skills      # ['css', 'react', 'redux', 'typescript', 'zustand']
parse_jd(cybersecurity_jd).must_have_skills  # ['aws', 'azure', 'siem']
parse_jd(devops_jd).must_have_skills         # ['aws', 'gcp', 'github actions', 'jenkins', 'kubernetes', 'terraform']
```

This is an honest framing, not a marketing one: the **structural** parsing (sections, YOE, notice period, disqualifiers, culture signals) genuinely generalizes to any JD. The **skill extraction** generalizes with full precision for AI/ML roles and with reduced (but non-zero, and tested) precision for everything else.

---

## Scoring Weights (all in `config/config.yaml`)

```yaml
scoring_weights:
  skills: 0.38        # Skill taxonomy match
  career: 0.26        # Career trajectory  
  behavioral: 0.12    # Platform engagement
  availability: 0.08  # Notice + open-to-work
  semantic: 0.10      # Hybrid: TF-IDF lexical + distributional embedding (PPMI-SVD)
  education: 0.06     # Degree + tier
```

All weights, thresholds, and lists are in `config/config.yaml`. **No numbers are hardcoded in scoring logic.**

---

## Setup

### Requirements
- Python 3.9+
- `pip install pyyaml` (core ranker)
- `pip install streamlit` (dashboard only)

### Run the ranker

```bash
git clone https://github.com/NeuroIgniter/ai-recruiter-ranker
cd ai-recruiter-ranker

# Place data file
cp /path/to/candidates.jsonl data/
# OR
cp /path/to/candidates.jsonl.gz data/

# Run
python run.py

# Output: output/neuroigniter_submission.csv (~65 seconds)
```

### Custom paths

```bash
python run.py /path/to/candidates.jsonl /path/to/output.csv
```

### Run the dashboard (local)

```bash
pip install streamlit
streamlit run streamlit_app.py     # uses Streamlit Cloud entrypoint (root)
# OR
streamlit run dashboard/app.py     # direct dashboard file
# Opens at http://localhost:8501
```

### Deploy the dashboard (zero-friction, no install required)

The dashboard deploys to **Streamlit Community Cloud** with one click — no
server, no Docker, no local install needed for judges to view it.

1. Fork the repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select repo → set **Main file: `streamlit_app.py`**
4. Click Deploy

The submission CSV (`output/neuroigniter_submission.csv`) and stats file
(`output/ranking_stats.json`) are bundled in the repo, so the live demo
works without the 487MB `candidates.jsonl` file — judges see the ranked
shortlist, comparison view, and insights panel immediately on load.

### Docker

```bash
docker build -t neuroigniter .

# Run ranker
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output neuroigniter

# Run dashboard
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  neuroigniter streamlit run dashboard/app.py --server.address 0.0.0.0
```

### Run tests

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
# 88 tests, all passing (66 unit + 22 end-to-end integration)
```

---

## Project Structure

```
neuroigniter-ranker/
├── src/neuroigniter/
│   ├── __init__.py          # Package definition
│   ├── ranker.py            # Core scoring engine (streaming, bounded memory)
│   ├── ontology.py          # Skill ontology: 200+ aliases, clusters, corroboration
│   └── jd_parser.py         # JD intelligence: RUNTIME extraction, not hardcoded
├── config/
│   └── config.yaml          # All weights, thresholds, lists (single source of truth)
├── dashboard/
│   └── app.py               # Streamlit recruiter interface (search, compare, insights)
├── tests/
│   ├── __init__.py
│   ├── test_ranker_v2.py    # 66 unit tests across all scoring components
│   └── test_integration.py  # 22 end-to-end pipeline tests (10-candidate fixture + adversarial-input regression suite)
├── data/                    # Place candidates.jsonl[.gz] here (gitignored)
├── output/                  # Submission CSV + ranking_stats.json written here
├── .github/workflows/
│   └── ci.yml               # GitHub Actions: unit + integration tests, Python 3.9/3.11/3.12
├── run.py                   # Single-command entrypoint
├── Dockerfile                # Docker support
├── requirements.txt
├── pyproject.toml
├── LICENSE                   # MIT
└── README.md
```

---

## Output Format

```
candidate_id,rank,score,reasoning
CAND_0018499,1,0.989622,"Senior Machine Learning Engineer (7.2 yrs) with production skills in embeddings, pinecone, weaviate; career descriptions show production system evidence (8 production signals); India-based (JD preferred location). Behavioral: high recruiter response rate (83%). Note: 1 of 21 JD core skills missing from profile. [Skills 0.87 | Career 0.91 | Behavioral 0.80 | Availability 0.88 | Semantic 0.74]"
```

Every reasoning string is unique to that candidate. It references:
- Specific skills **from their profile** (not generated)
- Specific career evidence **from their descriptions** (not generated)
- Actual behavioral signal values (response rate, GitHub score)
- Full score decomposition for recruiter transparency
- An honest concern, even for rank-1

---

## Responsible AI Notes

**Bias disclosures:**
- Education tier multiplier (1.0× to 1.25×): encodes institutional prestige. Documented and capped at 1.25× to limit impact.
- India location bonus (+0.03): reflects explicit JD preference for India-based candidates. Documented.
- Company size preference for startups: reflects JD "founding team" language. Documented.

**No demographic signals used:** gender, ethnicity, religion, age (beyond YOE), caste, nationality are not in any scoring component.

**Explainability:** every ranking decision is traceable to its component scores. Recruiters can see exactly why candidate A ranked above candidate B by comparing their score decompositions.

**Human override:** the system produces a shortlist, not a hiring decision. Final decisions are made by humans.

---

## Known Limitations

In the spirit of not overclaiming: here is what this system does not do.

- **Skill extraction precision varies by domain.** The ontology tier (200+ AI/ML/IR terms with full alias normalization) is high-precision for AI/ML roles. The generic fallback tier for other domains is lower-precision — it extracts plausible-looking technology tokens without alias normalization or false-positive filtering beyond a small stopword list. A JD with unusual formatting could under- or over-extract.
- **The "semantic" matching layer is a hybrid of TF-IDF (lexical, exact-term-sensitive) and a locally-trained distributional embedding model** (co-occurrence + PPMI + SVD — the same mathematical family as GloVe/LSA, trained from scratch on this corpus, no network or GPU required). The distributional component can match candidates who describe the same work in different vocabulary — e.g. "built the system that decides what results surface" matching a JD that says "own the ranking system." However, the embedding vocabulary is limited by the training corpus size (5,000 sampled documents), so rare or highly specialized terminology may still not generalise perfectly.
- **No feedback loop.** Recruiter actions (who got contacted, who got an offer) are not fed back into the scoring weights. The weights in `config.yaml` are fixed heuristics, not learned from outcomes.
- **Self-reported signals are trusted, with limited corroboration.** `years_of_experience`, `open_to_work_flag`, and behavioral signals are taken from the candidate's `redrob_signals` profile data as-is. Only skill claims get cross-referenced against career-description text (the corroboration check); other self-reported fields do not.
- **Confidence scores measure signal availability, not score accuracy.** `conf=0.95` means 16 of 16 expected signals were present in the candidate's data — it does not mean the resulting score has been validated against a ground-truth outcome, because no labeled hiring-outcome data exists for this dataset.
- **No authentication, rate limiting, or PII encryption.** This is a ranking engine and local dashboard, not a deployed multi-tenant service — those concerns would need to be addressed before any production deployment with real candidate PII.

---

## Frequently Asked Questions

**Q: Why a locally-trained distributional embedding instead of sentence-transformers?**

We actually tried. In this environment and under the competition's stated constraints (CPU-only, no network, 5-minute budget), a pretrained transformer requires either a GPU for tractable inference at 100K scale, or a network call to download model weights, or bundling multi-hundred-MB weights into the submission — all three violate the competition's rules. Instead, we built a real distributional semantic model from scratch using the actual corpus: co-occurrence counting → PPMI weighting → truncated SVD to produce dense word vectors. This is the same mathematical principle as GloVe and LSA. Training on 5,000 sampled documents takes ~3 seconds; scoring 100K candidates via the precomputed JD vector via vectorized matrix multiplication takes ~11 seconds. The hybrid (55% lexical TF-IDF + 45% distributional embedding) correctly boosts candidates who describe the same work in different vocabulary — verified with a held-out test where IR-topic sentences with different specific words than the training data scored higher than finance-topic sentences.

**Q: Why are weights not learned?**

The competition provides no labeled ranking data (no ground truth of "this candidate is a good hire"). Without labels, learned weights overfit to whatever proxy metric we choose. The weights are derived from JD signal priority (skills explicitly called out as "must-have" → highest weight) and validated against business logic.

**Q: How does corroboration work?**

For each claimed skill, we check whether the candidate's career descriptions contain terms that would appear if they'd actually used that skill in practice. A candidate claiming "FAISS (advanced)" but whose career text mentions only Excel and PowerPoint receives a 0.5× confidence multiplier on that skill's contribution. This prevents the adversarial case where someone stuffs their profile with every trending framework.
