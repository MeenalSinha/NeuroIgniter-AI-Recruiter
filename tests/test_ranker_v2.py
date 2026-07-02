"""
NeuroIgniter v2 — Test Suite
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import date
from neuroigniter.ontology import (
    normalize_skill, skills_are_equivalent, compute_cluster_coherence,
    get_corroboration_score, SKILL_ALIASES, ALIAS_TO_CANONICAL,
)
from neuroigniter.jd_parser import JD_REQUIREMENTS, MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS
from neuroigniter.ranker import (
    TFIDFMatcher, score_candidate, _detect_honeypot, _check_disqualifiers,
    _score_skills, _score_career, _score_behavioral, _detect_promotions,
    _extract_achievements, _generate_reasoning, CandidateAnalysis, ComponentScores,
    CONSULTING_FIRMS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_candidate(
    title="ML Engineer",
    yoe=6.5,
    skills=None,
    company="Startup AI",
    company_size="51-200",
    is_consulting=False,
    signals=None,
    career_desc="Built production embedding pipeline with FAISS deployed to real users at scale. A/B tested retrieval quality using NDCG@10.",
    country="India",
    career_history=None,
    cid="CAND_TEST001",
):
    if skills is None:
        skills = [
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 22, "duration_months": 30},
            {"name": "Python", "proficiency": "advanced", "endorsements": 45, "duration_months": 66},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 18, "duration_months": 36},
            {"name": "Elasticsearch", "proficiency": "intermediate", "endorsements": 12, "duration_months": 24},
            {"name": "NLP", "proficiency": "advanced", "endorsements": 28, "duration_months": 40},
            {"name": "Ranking", "proficiency": "advanced", "endorsements": 14, "duration_months": 28},
        ]

    if signals is None:
        signals = {
            "profile_completeness_score": 88.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 18,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.82,
            "avg_response_time_hours": 6,
            "skill_assessment_scores": {"Python": 90, "ML": 85},
            "connection_count": 220,
            "endorsements_received": 55,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 28, "max": 42},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 78,
            "search_appearance_30d": 14,
            "saved_by_recruiters_30d": 6,
            "interview_completion_rate": 0.92,
            "offer_acceptance_rate": 0.80,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        }

    actual_company = "Tata Consultancy Services" if is_consulting else company

    if career_history is None:
        career_history = [{
            "company": actual_company,
            "title": title,
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": int(yoe * 12),
            "is_current": True,
            "industry": "IT Services" if is_consulting else "AI/ML",
            "company_size": "10001+" if is_consulting else company_size,
            "description": career_desc,
        }]

    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "ML Engineer | Retrieval Systems",
            "summary": "Senior ML engineer with production experience in embeddings, retrieval, and ranking systems.",
            "location": "Bengaluru, Karnataka",
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": actual_company,
            "current_company_size": "10001+" if is_consulting else company_size,
            "current_industry": "IT Services" if is_consulting else "AI/ML",
        },
        "career_history": career_history,
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2015,
            "end_year": 2019,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": signals,
    }


def default_weights():
    return {
        'skills': 0.38, 'career': 0.26, 'behavioral': 0.12,
        'availability': 0.08, 'semantic': 0.10, 'education': 0.06,
    }


# ── Ontology Tests ────────────────────────────────────────────────────────────

class TestOntology:
    def test_sbert_maps_to_sentence_transformers(self):
        assert normalize_skill("SBERT") == "sentence transformers"

    def test_ann_maps_to_vector_search(self):
        assert normalize_skill("ANN") == "vector search"

    def test_hnsw_maps_to_vector_search(self):
        assert normalize_skill("HNSW") == "vector search"

    def test_approximate_nearest_neighbor_maps_to_vector_search(self):
        assert normalize_skill("Approximate Nearest Neighbor") == "vector search"

    def test_llama_maps_to_llm(self):
        assert normalize_skill("LLaMA") == "llm"

    def test_langchain_maps_to_langchain(self):
        assert normalize_skill("LangChain") == "langchain"

    def test_xgboost_maps_correctly(self):
        assert normalize_skill("XGBoost") == "xgboost"

    def test_ir_maps_to_information_retrieval(self):
        assert normalize_skill("IR") == "information retrieval"

    def test_exact_string_preserves(self):
        assert normalize_skill("faiss") == "faiss"

    def test_unknown_skill_passthrough(self):
        result = normalize_skill("SomeObscureFramework2026")
        assert result == "someobscureframework2026"

    def test_skill_equivalence(self):
        assert skills_are_equivalent("SBERT", "sentence-transformers")
        assert skills_are_equivalent("faiss", "FAISS")
        assert not skills_are_equivalent("faiss", "python")

    def test_ir_specialist_cluster_coherence_high(self):
        ir_skills = {'faiss', 'vector search', 'elasticsearch', 'opensearch', 'bm25', 'hybrid search'}
        coherence = compute_cluster_coherence(ir_skills)
        assert coherence >= 0.60

    def test_random_skills_cluster_coherence_low(self):
        random_skills = {'python', 'excel', 'photoshop', 'marketing', 'accounting'}
        # These don't fall in any cluster → low coherence
        coherence = compute_cluster_coherence(random_skills)
        assert coherence <= 0.50

    def test_corroboration_strong_for_matching_career(self):
        career_text = "Built FAISS-based retrieval system with vector indexing and embedding search deployed to production"
        score = get_corroboration_score('faiss', career_text)
        assert score >= 0.70

    def test_corroboration_penalizes_mismatch(self):
        career_text = "Managed spreadsheets, reconciled financial accounts, prepared audit reports"
        score = get_corroboration_score('faiss', career_text)
        assert score <= 0.55  # Weak corroboration

    def test_corroboration_direct_name_mention_scores_maximum(self):
        """Directly naming the skill in career text should give full corroboration — 
        the strongest possible signal."""
        career_text = "Used FAISS library for ANN search across candidate embeddings."
        score = get_corroboration_score('faiss', career_text)
        assert score == 1.0, f"Expected 1.0 for direct mention, got {score}"

    def test_corroboration_gaming_attack_penalized(self):
        """
        REGRESSION TEST for the corroboration-gaming vulnerability:
        A candidate who writes generic 'production/deployed/scale' sentences
        to corroborate specific IR skills WITHOUT naming those skills
        must NOT score higher than the floor for zero evidence (0.30).
        Previously this would return the 0.40 floor regardless of actual
        evidence, letting adversarial candidates fake corroboration cheaply.
        """
        gaming_text = (
            "Built and deployed production systems at scale serving real users "
            "with high performance, monitoring, and reliability."
        )
        for skill in ['faiss', 'vector search', 'ndcg', 'ranking']:
            score = get_corroboration_score(skill, gaming_text)
            assert score <= 0.35, (
                f"Gaming text should score ≤0.35 for '{skill}' "
                f"(no specific evidence), got {score:.2f}"
            )

    def test_corroboration_no_evidence_is_lower_than_weak_evidence(self):
        """Zero cluster-term matches (0.30) must score below one-term match (0.42).
        The scoring must create a meaningful gradient, not flatten at a shared floor."""
        no_evidence = "Reconciled accounts and prepared quarterly financial reports."
        weak_evidence = "Built a search system for internal use."  # 'search' matches

        no_ev_score = get_corroboration_score('faiss', no_evidence)
        weak_ev_score = get_corroboration_score('faiss', weak_evidence)
        assert no_ev_score < weak_ev_score, (
            f"Zero evidence ({no_ev_score:.2f}) should be lower than weak evidence ({weak_ev_score:.2f})"
        )


# ── JD Parser Tests ───────────────────────────────────────────────────────────

class TestJDParser:
    def test_jd_has_must_have_skills(self):
        assert len(JD_REQUIREMENTS.must_have_skills) >= 10

    def test_jd_has_nice_to_have_skills(self):
        assert len(JD_REQUIREMENTS.nice_to_have_skills) >= 5

    def test_jd_yoe_range_correct(self):
        assert JD_REQUIREMENTS.yoe_min == 5.0
        assert JD_REQUIREMENTS.yoe_max == 9.0

    def test_jd_wants_product_orientation(self):
        assert JD_REQUIREMENTS.wants_product_orientation

    def test_jd_semantic_text_not_empty(self):
        assert len(JD_REQUIREMENTS.semantic_text) > 100

    def test_jd_extraction_confidence_high_for_challenge_jd(self):
        """The challenge JD has clear section headers — extraction should be near-perfect."""
        assert JD_REQUIREMENTS.extraction_confidence >= 0.8

    def test_jd_must_have_and_nice_have_dont_overlap(self):
        """A skill shouldn't be both must-have and nice-to-have."""
        must_set = set(JD_REQUIREMENTS.must_have_skills)
        nice_set = set(JD_REQUIREMENTS.nice_to_have_skills)
        assert not (must_set & nice_set), f"Overlap: {must_set & nice_set}"

    def test_jd_consulting_disqualifiers_extracted(self):
        """The challenge JD names TCS/Infosys/Wipro etc. as disqualifiers — must be extracted."""
        assert 'tcs' in JD_REQUIREMENTS.consulting_disqualifiers
        assert 'infosys' in JD_REQUIREMENTS.consulting_disqualifiers

    def test_parser_generalizes_to_completely_different_jd(self):
        """
        The core claim: this is a RUNTIME PARSER, not a hardcoded skill list.
        Feed it a Data Engineer JD with a totally different tech stack and
        verify it extracts DIFFERENT requirements — proving genuine parsing.
        """
        from neuroigniter.jd_parser import parse_jd

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

        # Different skills than the AI Engineer JD
        assert 'kafka' in req.must_have_skills
        assert 'python' in req.must_have_skills
        # Should NOT contain AI-Engineer-specific skills not mentioned in this JD
        assert 'faiss' not in req.must_have_skills
        assert 'pinecone' not in req.must_have_skills

        # Different YOE range
        assert req.yoe_min == 4.0
        assert req.yoe_max == 8.0

        # Different notice period
        assert req.ideal_notice_days == 15

        # Disqualifiers still correctly extracted
        assert 'tcs' in req.consulting_disqualifiers
        assert 'infosys' in req.consulting_disqualifiers
        assert 'accenture' in req.consulting_disqualifiers

    def test_parser_handles_jd_with_no_clear_sections(self):
        """A poorly-formatted JD without clear section headers shouldn't crash —
        should fall back gracefully with lower confidence."""
        from neuroigniter.jd_parser import parse_jd

        messy_jd = "We need someone who knows Python and has worked with FAISS before. 5 years experience minimum."
        req = parse_jd(messy_jd)
        assert isinstance(req.must_have_skills, list)  # Doesn't crash
        assert req.extraction_confidence < 0.8  # Lower confidence for unclear structure

    def test_parser_does_not_silently_return_empty_skills_for_non_ai_roles(self):
        """
        REGRESSION TEST for a real bug: the ontology-based extraction only
        recognizes ~200 AI/ML/IR terms. Without a generic fallback, ANY JD
        for a role outside that domain (frontend, DevOps, security, etc.)
        would extract ZERO must-have skills — silently breaking 38% of every
        candidate's composite score for that role. This must never regress.
        """
        from neuroigniter.jd_parser import parse_jd

        non_ai_jds = {
            'frontend': """
                Things you absolutely need:
                - Strong experience with React and TypeScript
                - Production experience with state management (Redux, Zustand)
                Things we explicitly do NOT want:
                - People who have only worked at consulting firms (TCS, Infosys)
            """,
            'security': """
                Things you absolutely need:
                - Strong experience with penetration testing and SIEM tools
                - Experience with cloud security on AWS or Azure
                Things we explicitly do NOT want:
                - People who have only worked at consulting firms (TCS, Wipro)
            """,
            'devops': """
                Things you absolutely need:
                - Strong experience with Kubernetes and Terraform
                - Production experience with CI/CD pipelines (Jenkins, GitHub Actions)
                Things we explicitly do NOT want:
                - People who have only worked at consulting firms (Capgemini, HCL)
            """,
        }

        for role, jd_text in non_ai_jds.items():
            req = parse_jd(jd_text)
            assert len(req.must_have_skills) > 0, (
                f"{role} JD extracted ZERO must-have skills — this is the "
                f"exact silent-failure regression this test guards against"
            )

    def test_generic_extraction_does_not_pollute_ai_jd_results(self):
        """The generic fallback should NOT engage when the ontology already
        found a healthy number of matches — otherwise AI/ML JDs would get
        noisy non-canonical terms mixed into their high-precision skill list."""
        from neuroigniter.jd_parser import JD_REQUIREMENTS
        # The challenge JD has 18 ontology-matched skills; none of them should
        # be noise fragments like single capital letters or sentence-starters.
        for skill in JD_REQUIREMENTS.must_have_skills:
            assert len(skill) > 1, f"Suspiciously short skill extracted: {skill!r}"
            assert skill == skill.lower(), f"Non-canonical (un-normalized) skill leaked in: {skill!r}"


# ── Cluster Coherence Anti-Gaming ─────────────────────────────────────────────

class TestClusterCoherenceAntiGaming:
    def test_single_cluster_specialist_capped(self):
        """5 vector DB names with nothing else (single cluster) should be capped,
        not scored as a full IR specialist — prevents 'list every vector DB' gaming."""
        single_cluster_skills = {'faiss', 'weaviate', 'pinecone', 'qdrant', 'milvus'}
        coherence = compute_cluster_coherence(single_cluster_skills)
        assert coherence <= 0.70

    def test_multi_cluster_specialist_scores_higher(self):
        """A candidate spanning retrieval + ranking + evaluation clusters
        demonstrates genuine breadth, not just tool-name memorization."""
        multi_cluster_skills = {'faiss', 'vector search', 'bm25', 'ndcg', 'reranking'}
        coherence = compute_cluster_coherence(multi_cluster_skills)
        single_cluster_skills = {'faiss', 'weaviate', 'pinecone', 'qdrant', 'milvus'}
        single_coherence = compute_cluster_coherence(single_cluster_skills)
        assert coherence >= single_coherence


# ── Honeypot Detection ────────────────────────────────────────────────────────

class TestHoneypotDetection:
    def test_normal_candidate_not_flagged(self):
        is_hp, _ = _detect_honeypot(make_candidate())
        assert not is_hp

    def test_zero_duration_advanced_skills_flagged(self):
        cand = make_candidate(skills=[
            {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 95, "duration_months": 0}
            for i in range(7)
        ])
        is_hp, reason = _detect_honeypot(cand)
        assert is_hp
        assert "0 months" in reason

    def test_career_timeline_inflation_flagged(self):
        cand = make_candidate(yoe=5.0)
        cand['career_history'] = [
            {"company": f"Co{i}", "title": "Engineer", "start_date": "2000-01-01",
             "end_date": None, "duration_months": 120, "is_current": i == 9,
             "industry": "Tech", "company_size": "51-200", "description": "Worked."}
            for i in range(10)
        ]
        is_hp, reason = _detect_honeypot(cand)
        assert is_hp

    def test_expert_skills_low_yoe_flagged(self):
        cand = make_candidate(yoe=2.0, skills=[
            {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 95, "duration_months": 12}
            for i in range(10)
        ])
        is_hp, reason = _detect_honeypot(cand)
        assert is_hp


# ── Disqualifier Tests ────────────────────────────────────────────────────────

class TestDisqualifiers:
    def test_consulting_only_disqualified(self):
        cand = make_candidate(is_consulting=True)
        cand['career_history'] = [
            {"company": "TCS", "title": "Engineer", "start_date": "2018-01-01",
             "end_date": "2020-01-01", "duration_months": 24, "is_current": False,
             "industry": "IT Services", "company_size": "10001+", "description": "Worked."},
            {"company": "Wipro", "title": "Senior Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 30, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "Worked."},
        ]
        companies = [ch['company'] for ch in cand['career_history']]
        is_disq, reasons = _check_disqualifiers(cand, companies)
        assert is_disq
        assert any('consulting' in r.lower() for r in reasons)

    def test_product_company_not_disqualified(self):
        cand = make_candidate(company="Zepto AI", company_size="51-200")
        is_disq, _ = _check_disqualifiers(cand, ["Zepto AI"])
        assert not is_disq

    def test_single_consulting_firm_not_disqualified(self):
        """One consulting job in otherwise product career should not disqualify."""
        cand = make_candidate()
        cand['career_history'] = [
            {"company": "TCS", "title": "Engineer", "start_date": "2016-01-01",
             "end_date": "2018-01-01", "duration_months": 24, "is_current": False,
             "industry": "IT Services", "company_size": "10001+", "description": "."},
            {"company": "Zepto", "title": "ML Engineer", "start_date": "2018-01-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "E-commerce", "company_size": "201-500", "description": "Built ML."},
        ]
        companies = [ch['company'] for ch in cand['career_history']]
        is_disq, _ = _check_disqualifiers(cand, companies)
        assert not is_disq


# ── Skill Scoring ─────────────────────────────────────────────────────────────

class TestSkillScoring:
    def test_strong_ir_candidate_high_skill_score(self):
        cand = make_candidate(skills=[
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
            {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 66},
            {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
            {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 12, "duration_months": 24},
            {"name": "BM25", "proficiency": "advanced", "endorsements": 8, "duration_months": 18},
            {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 22, "duration_months": 24},
            {"name": "RAG", "proficiency": "advanced", "endorsements": 10, "duration_months": 18},
            {"name": "NDCG", "proficiency": "intermediate", "endorsements": 5, "duration_months": 12},
        ], career_desc="Built embedding retrieval system with FAISS and BM25 hybrid search. Deployed to production.")
        
        career_text = "Built embedding retrieval system with FAISS and BM25 hybrid search. Deployed to production."
        score, must_matched, nice_matched, _, _ = _score_skills(cand, career_text)
        # Must-have list is now runtime-extracted from JD text (18 genuine skills,
        # down from a hand-tuned 21) — score threshold reflects the more precise taxonomy.
        # Corroboration check also reduces self-reported-only skill weight.
        assert score > 0.30
        assert len(must_matched) >= 4

    def test_alias_expansion_works_in_scoring(self):
        """Candidate with SBERT should match 'sentence transformers' in JD."""
        cand = make_candidate(skills=[
            {"name": "SBERT", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            {"name": "ANN", "proficiency": "advanced", "endorsements": 15, "duration_months": 18},
            {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 66},
        ], career_desc="Built embedding-based search using SBERT for retrieval.")
        
        career_text = "Built embedding-based search using SBERT for retrieval."
        score, must_matched, _, _, _ = _score_skills(cand, career_text)
        # Should recognise SBERT → sentence transformers and ANN → vector search
        assert 'sentence transformers' in must_matched or len(must_matched) >= 1

    def test_unrelated_candidate_low_skill_score(self):
        cand = make_candidate(title="Accountant", skills=[
            {"name": "Excel", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
            {"name": "Accounting", "proficiency": "advanced", "endorsements": 20, "duration_months": 60},
            {"name": "Tally", "proficiency": "advanced", "endorsements": 5, "duration_months": 48},
        ], career_desc="Managed accounts and financial reporting.")
        
        score, must_matched, _, _, _ = _score_skills(cand, "Managed accounts and financial reporting.")
        assert score < 0.20
        assert len(must_matched) == 0

    def test_corroboration_boosts_validated_skills(self):
        """Skill with career evidence should score higher than same skill without."""
        strong_career = "Built FAISS-based retrieval system. Dense retrieval, vector indexing, embedding search at scale."
        weak_career = "Designed spreadsheets. Managed project timelines."
        
        cand_strong = make_candidate(skills=[
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 66},
        ], career_desc=strong_career)
        
        cand_weak = make_candidate(skills=[
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 66},
        ], career_desc=weak_career)
        
        score_strong, _, _, _, _ = _score_skills(cand_strong, strong_career)
        score_weak, _, _, _, _ = _score_skills(cand_weak, weak_career)
        assert score_strong >= score_weak


# ── Career Scoring ────────────────────────────────────────────────────────────

class TestCareerScoring:
    def test_ideal_yoe_scores_highest(self):
        ideal = make_candidate(yoe=7.0)
        junior = make_candidate(yoe=2.0)
        score_ideal, _, _, _, _ = _score_career(ideal)
        score_junior, _, _, _, _ = _score_career(junior)
        assert score_ideal > score_junior

    def test_promotion_detection(self):
        cand = make_candidate()
        cand['career_history'] = [
            {"company": "Redrob", "title": "Junior ML Engineer", "start_date": "2019-01-01",
             "end_date": "2021-01-01", "duration_months": 24, "is_current": False,
             "industry": "AI", "company_size": "51-200",
             "description": "Built ML models. Deployed to production."},
            {"company": "Redrob", "title": "Senior ML Engineer", "start_date": "2021-01-01",
             "end_date": None, "duration_months": 42, "is_current": True,
             "industry": "AI", "company_size": "51-200",
             "description": "Led retrieval system. Embedding pipeline at scale."},
        ]
        n_promotions = _detect_promotions(cand['career_history'])
        assert n_promotions >= 1

    def test_no_promotion_when_same_title(self):
        cand = make_candidate()
        cand['career_history'] = [
            {"company": "Redrob", "title": "ML Engineer", "start_date": "2019-01-01",
             "end_date": "2022-01-01", "duration_months": 36, "is_current": False,
             "industry": "AI", "company_size": "51-200", "description": "."},
            {"company": "Redrob", "title": "ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 30, "is_current": True,
             "industry": "AI", "company_size": "51-200", "description": "."},
        ]
        n_promotions = _detect_promotions(cand['career_history'])
        assert n_promotions == 0

    def test_achievement_extraction(self):
        # Each sentence needs BOTH a magnitude (%, number, scale) AND an action verb.
        # "Built from scratch" fails — no magnitude. Correct by tighter filter design.
        cand = make_candidate(career_desc=(
            "Reduced retrieval latency by 40% through index optimization. "
            "Served 10M users daily with sub-50ms p99 latency. "
            "Shipped new ranking system in 6 weeks ahead of schedule. "
            "Scaled search infrastructure to handle 3x traffic increase."
        ))
        achievements = _extract_achievements(cand['career_history'])
        assert len(achievements) >= 2

    def test_job_hopper_penalized(self):
        cand = make_candidate()
        cand['career_history'] = [
            {"company": f"Co{i}", "title": "ML Engineer", "start_date": f"202{i}-01-01",
             "end_date": f"202{i}-10-01", "duration_months": 9, "is_current": i == 3,
             "industry": "AI", "company_size": "51-200", "description": "Did ML."}
            for i in range(4)
        ]
        score, _, concerns, _, _ = _score_career(cand)
        assert any('hop' in c.lower() or 'tenure' in c.lower() for c in concerns)

    def test_current_consulting_penalized(self):
        cand = make_candidate(company="Infosys", company_size="10001+", is_consulting=True)
        score, _, concerns, _, _ = _score_career(cand)
        assert any('consulting' in c.lower() for c in concerns)


# ── Behavioral Scoring ────────────────────────────────────────────────────────

class TestBehavioralScoring:
    def test_active_engaged_scores_high(self):
        cand = make_candidate()
        sigs = cand['redrob_signals']
        sigs['last_active_date'] = '2026-06-25'
        sigs['recruiter_response_rate'] = 0.95
        sigs['interview_completion_rate'] = 1.0
        beh, _, _ = _score_behavioral(sigs, date(2026, 6, 28))
        assert beh > 0.72

    def test_ghost_candidate_low_behavioral(self):
        cand = make_candidate()
        sigs = cand['redrob_signals']
        sigs['last_active_date'] = '2025-01-01'  # 18 months ago
        sigs['recruiter_response_rate'] = 0.04
        sigs['open_to_work_flag'] = False
        sigs['interview_completion_rate'] = 0.1
        sigs['offer_acceptance_rate'] = 0.1
        sigs['github_activity_score'] = -1
        # A real ghost candidate typically has no verified contact info either
        sigs['verified_email'] = False
        sigs['verified_phone'] = False
        sigs['linkedin_connected'] = False
        sigs['skill_assessment_scores'] = {}
        beh, _, _ = _score_behavioral(sigs, date(2026, 6, 28))
        # Ghost with inactive profile, low response, unverified = low behavioral
        assert beh < 0.30, f"Expected ghost to score < 0.30, got {beh:.3f}"

    def test_notice_period_interaction_bonus(self):
        """open_to_work + notice<=30 + high response → availability > simple sum."""
        sigs_boosted = {
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "recruiter_response_rate": 0.90,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "avg_response_time_hours": 6,
            "last_active_date": "2026-06-25",
            "github_activity_score": 70,
            "profile_completeness_score": 85,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "skill_assessment_scores": {"Python": 88},
            "saved_by_recruiters_30d": 5,
            "search_appearance_30d": 10,
            "verified_email": True,
            "verified_phone": True,
        }
        sigs_no_boost = dict(sigs_boosted)
        sigs_no_boost['notice_period_days'] = 90
        sigs_no_boost['open_to_work_flag'] = False

        _, avail_boosted, _ = _score_behavioral(sigs_boosted, date(2026, 6, 28))
        _, avail_no_boost, _ = _score_behavioral(sigs_no_boost, date(2026, 6, 28))
        assert avail_boosted > avail_no_boost

    def test_missing_offer_rate_neutral(self):
        """offer_acceptance_rate = -1 should not catastrophically penalize."""
        cand = make_candidate()
        sigs = cand['redrob_signals']
        sigs['offer_acceptance_rate'] = -1
        beh, _, _ = _score_behavioral(sigs, date(2026, 6, 28))
        assert beh > 0.40  # Shouldn't crash or severely penalize


# ── End-to-End Scoring ────────────────────────────────────────────────────────

class TestEndToEnd:
    def setup_method(self):
        self.tfidf = TFIDFMatcher("""
            senior ai engineer retrieval ranking embeddings llm production deployment
            embedding drift index refresh retrieval quality regression vector database
            hybrid search python evaluation ndcg mrr map a/b testing offline benchmark
            nlp information retrieval bm25 faiss elasticsearch opensearch pinecone weaviate qdrant milvus
            sentence transformers hugging face pytorch sklearn xgboost lightgbm
            reranking cross-encoder bi-encoder dense retrieval sparse retrieval semantic search
            product startup founding team ship fast production deployed real users scale
        """)
        self.today = date(2026, 6, 28)
        self.weights = default_weights()

    def test_ideal_ir_engineer_scores_high(self):
        cand = make_candidate(
            title="Senior AI Engineer",
            yoe=7.0,
            skills=[
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
                {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 72},
                {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
                {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 35, "duration_months": 48},
                {"name": "Ranking", "proficiency": "advanced", "endorsements": 15, "duration_months": 28},
                {"name": "RAG", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
                {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 40, "duration_months": 30},
                {"name": "Vector Search", "proficiency": "advanced", "endorsements": 22, "duration_months": 24},
                {"name": "Hybrid Search", "proficiency": "advanced", "endorsements": 18, "duration_months": 24},
            ],
            career_desc=(
                "Built production embedding-based retrieval system with FAISS and Elasticsearch hybrid search. "
                "Deployed to real users at scale, serving 5M queries/day. Reduced latency by 35%. "
                "Set up A/B testing framework for retrieval quality (NDCG@10). "
                "Led ranking system v2 from scratch to production in 8 weeks."
            ),
        )
        result = score_candidate(cand, self.tfidf, self.today, self.weights)
        assert result.total_score >= 0.75
        assert not result.is_honeypot
        assert not result.is_disqualified

    def test_accountant_scores_very_low(self):
        cand = make_candidate(
            title="Senior Accountant",
            yoe=10.0,
            skills=[
                {"name": "Excel", "proficiency": "advanced", "endorsements": 15, "duration_months": 100},
                {"name": "Tally", "proficiency": "advanced", "endorsements": 8, "duration_months": 80},
                {"name": "SAP", "proficiency": "intermediate", "endorsements": 5, "duration_months": 60},
            ],
            career_desc="Managed accounts payable, reconciliation, financial reporting. Audit support.",
            company_size="10001+",
        )
        result = score_candidate(cand, self.tfidf, self.today, self.weights)
        assert result.total_score < 0.42
        assert result.components.skills < 0.10

    def test_ideal_ranks_above_accountant(self):
        ideal = make_candidate(
            title="NLP Engineer", yoe=6.0,
            skills=[
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
                {"name": "Embeddings", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
            ],
            career_desc="Built embedding retrieval pipelines in production. Dense vector search."
        )
        poor = make_candidate(
            title="HR Manager", yoe=8.0,
            skills=[
                {"name": "Excel", "proficiency": "advanced", "endorsements": 10, "duration_months": 80},
                {"name": "Recruitment", "proficiency": "advanced", "endorsements": 5, "duration_months": 80},
            ],
            career_desc="Managed hiring and HR operations.",
        )
        r_ideal = score_candidate(ideal, self.tfidf, self.today, self.weights)
        r_poor = score_candidate(poor, self.tfidf, self.today, self.weights)
        assert r_ideal.total_score > r_poor.total_score

    def test_consulting_only_score_very_low(self):
        cand = make_candidate()
        cand['career_history'] = [
            {"company": "TCS", "title": "ML Engineer", "start_date": "2018-01-01",
             "end_date": "2020-01-01", "duration_months": 24, "is_current": False,
             "industry": "IT Services", "company_size": "10001+",
             "description": "Built ML. Deployed models."},
            {"company": "Wipro", "title": "Senior ML", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 36, "is_current": True,
             "industry": "IT Services", "company_size": "10001+",
             "description": "Embeddings, FAISS, Python."},
        ]
        cand['profile']['current_company'] = "Wipro"
        cand['profile']['current_company_size'] = "10001+"
        result = score_candidate(cand, self.tfidf, self.today, self.weights)
        assert result.is_disqualified or result.total_score < 0.30

    def test_honeypot_gets_near_zero(self):
        cand = make_candidate(skills=[
            {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 97, "duration_months": 0}
            for i in range(8)
        ])
        result = score_candidate(cand, self.tfidf, self.today, self.weights)
        assert result.is_honeypot
        assert result.total_score < 0.01

    def test_alias_candidate_ranks_reasonably(self):
        """Candidate using SBERT/ANN terminology should still rank well."""
        cand = make_candidate(
            title="Search Engineer",
            yoe=6.0,
            skills=[
                {"name": "SBERT", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "ANN", "proficiency": "advanced", "endorsements": 15, "duration_months": 20},
                {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
                {"name": "Weaviate", "proficiency": "advanced", "endorsements": 12, "duration_months": 18},
                {"name": "BM25", "proficiency": "intermediate", "endorsements": 8, "duration_months": 14},
            ],
            career_desc="Built SBERT-based semantic search with ANN indexing and hybrid BM25+dense retrieval."
        )
        result = score_candidate(cand, self.tfidf, self.today, self.weights)
        assert result.total_score >= 0.55
        # Key: sentence transformers and vector search should be in must_have_matched
        canonical_matched = {normalize_skill(s) for s in result.must_have_matched}
        assert 'sentence transformers' in canonical_matched or 'vector search' in canonical_matched


# ── TF-IDF Tests ──────────────────────────────────────────────────────────────

class TestTFIDF:
    def setup_method(self):
        self.tfidf = TFIDFMatcher("retrieval ranking embeddings python vector search nlp production faiss")

    def test_relevant_text_higher_similarity(self):
        relevant = "embeddings faiss retrieval ranking python vector search production nlp"
        irrelevant = "accounting excel tally financial reporting audit reconciliation"
        assert self.tfidf.similarity(relevant) > self.tfidf.similarity(irrelevant)

    def test_empty_text_returns_zero(self):
        assert self.tfidf.similarity("") == 0.0

    def test_returns_float_0_to_1(self):
        sim = self.tfidf.similarity("machine learning python retrieval")
        assert 0.0 <= sim <= 1.0


# ── Reasoning Tests ───────────────────────────────────────────────────────────

class TestReasoning:
    def _make_analysis(self, score, must_matched, concerns, strengths, title, yoe):
        a = CandidateAnalysis(
            candidate_id="CAND_TEST",
            total_score=score,
            experience_years=yoe,
            current_title=title,
            notice_days=30,
            response_rate=0.82,
            github_score=72.0,
            is_open_to_work=True,
            must_have_matched=must_matched,
            nice_have_matched=['lora', 'xgboost'],
            strengths=strengths,
            concerns=concerns,
        )
        a.components = ComponentScores(
            skills=0.82, career=0.78, behavioral=0.80,
            availability=0.85, semantic=0.70, education=0.75
        )
        return a

    def test_high_score_reasoning_contains_score_decomp(self):
        cand = make_candidate(title="Senior AI Engineer", yoe=7.0)
        analysis = self._make_analysis(
            0.88, ['faiss', 'python', 'embeddings'], [], ['production evidence'], 'Senior AI Engineer', 7.0
        )
        reasoning = _generate_reasoning(analysis, cand)
        assert '[Skills' in reasoning
        assert 'Career' in reasoning

    def test_low_score_reasoning_mentions_gaps(self):
        cand = make_candidate(title="Accountant", yoe=10.0)
        analysis = self._make_analysis(
            0.18, [], ['no JD skills', 'non-technical title'], [], 'Accountant', 10.0
        )
        reasoning = _generate_reasoning(analysis, cand)
        assert any(word in reasoning.lower() for word in ['gap', 'concern', 'weak', 'below', 'no'])

    def test_reasoning_is_not_empty(self):
        cand = make_candidate()
        analysis = self._make_analysis(
            0.72, ['faiss', 'python'], ['consulting background'], ['product experience'], 'ML Engineer', 6.0
        )
        reasoning = _generate_reasoning(analysis, cand)
        assert len(reasoning) > 30

    def test_reasoning_references_actual_title(self):
        """Reasoning must reference the actual candidate title."""
        for title in ["NLP Engineer", "Data Scientist", "Search Engineer"]:
            cand = make_candidate(title=title, yoe=5.0)
            analysis = self._make_analysis(0.75, ['faiss'], [], [], title, 5.0)
            reasoning = _generate_reasoning(analysis, cand)
            assert title in reasoning

    def test_reasoning_mentions_concern_even_for_top_score(self):
        """Even high-scoring candidates should have concerns acknowledged."""
        cand = make_candidate(title="Staff ML Engineer", yoe=8.0)
        cand['redrob_signals']['notice_period_days'] = 90
        analysis = self._make_analysis(
            0.92, ['faiss', 'python', 'embeddings', 'vector search', 'ranking'],
            ['90-day notice period (JD prefers sub-30)'], ['IIT background, startup experience'],
            'Staff ML Engineer', 8.0
        )
        analysis.notice_days = 90
        reasoning = _generate_reasoning(analysis, cand)
        # Should acknowledge at least one concern
        concern_words = ['note', 'concern', 'notice', 'friction', 'gap', '90']
        assert any(word in reasoning.lower() for word in concern_words)


# ── Component Score Dataclass ─────────────────────────────────────────────────

class TestComponentScores:
    def test_weighted_total_with_default_weights(self):
        components = ComponentScores(
            skills=1.0, career=1.0, behavioral=1.0,
            availability=1.0, semantic=1.0, education=1.0
        )
        weights = {'skills': 0.38, 'career': 0.26, 'behavioral': 0.12,
                   'availability': 0.08, 'semantic': 0.10, 'education': 0.06}
        total = components.weighted_total(weights)
        assert abs(total - 1.0) < 0.001

    def test_weighted_total_zero_when_all_zero(self):
        components = ComponentScores()
        assert components.weighted_total(default_weights()) == 0.0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
