"""
NeuroIgniter — Integration Test
================================
Runs the complete ranking pipeline on a 10-candidate JSONL fixture.
Verifies: output format, score monotonicity, CSV validity, top-candidate quality.
This test catches runtime errors that unit tests cannot.

Run: python -m pytest tests/test_integration.py -v
"""

import sys
import os
import csv
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import date
from neuroigniter.ranker import rank_candidates


# ── Fixture: 10 synthetic candidates ─────────────────────────────────────────

def _make_cand(cid, title, yoe, skills, career_desc, country="India",
               signals=None, company="Redrob AI", size="51-200", is_consulting=False):
    actual_company = "Tata Consultancy Services" if is_consulting else company
    if signals is None:
        signals = {
            "profile_completeness_score": 85.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 12,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.78,
            "avg_response_time_hours": 8,
            "skill_assessment_scores": {"Python": 85},
            "connection_count": 180,
            "endorsements_received": 40,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 72,
            "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 4,
            "interview_completion_rate": 0.88,
            "offer_acceptance_rate": 0.75,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        }
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": f"Candidate {cid}",
            "headline": title,
            "summary": f"Experienced {title} with {yoe:.0f} years in ML/AI.",
            "location": "Bengaluru, Karnataka",
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": actual_company,
            "current_company_size": "10001+" if is_consulting else size,
            "current_industry": "IT Services" if is_consulting else "AI/ML",
        },
        "career_history": [{
            "company": actual_company,
            "title": title,
            "start_date": "2019-01-01",
            "end_date": None,
            "duration_months": int(yoe * 12),
            "is_current": True,
            "industry": "IT Services" if is_consulting else "AI/ML",
            "company_size": "10001+" if is_consulting else size,
            "description": career_desc,
        }],
        "education": [{
            "institution": "IIT Delhi",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2013, "end_year": 2017,
            "grade": "8.2 CGPA", "tier": "tier_1",
        }],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": signals,
    }


def build_fixture() -> list[dict]:
    return [
        # ── Tier 1: Strong IR engineers ──────────────────────────────────
        _make_cand(
            "CAND_INT_001", "Senior AI Engineer", 7.5,
            skills=[
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
                {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 72},
                {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
                {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
                {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 22, "duration_months": 30},
                {"name": "Ranking", "proficiency": "advanced", "endorsements": 15, "duration_months": 28},
                {"name": "RAG", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
                {"name": "NDCG", "proficiency": "intermediate", "endorsements": 8, "duration_months": 18},
            ],
            career_desc=(
                "Built production embedding-based retrieval with FAISS and Elasticsearch. "
                "Deployed to 5M users, reduced latency by 35%. A/B tested retrieval with NDCG@10. "
                "Shipped ranking v2 from scratch in 8 weeks."
            ),
        ),
        _make_cand(
            "CAND_INT_002", "NLP Engineer", 6.0,
            skills=[
                {"name": "SBERT", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "ANN", "proficiency": "advanced", "endorsements": 15, "duration_months": 20},
                {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
                {"name": "Weaviate", "proficiency": "advanced", "endorsements": 12, "duration_months": 18},
                {"name": "BM25", "proficiency": "intermediate", "endorsements": 8, "duration_months": 14},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 25, "duration_months": 40},
            ],
            career_desc=(
                "Built SBERT-based semantic search with ANN indexing and hybrid BM25 + dense retrieval. "
                "Production deployment serving 10M queries/day."
            ),
        ),
        # ── Tier 2: Adjacent ML with some IR ─────────────────────────────
        _make_cand(
            "CAND_INT_003", "ML Engineer", 5.5,
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
                {"name": "XGBoost", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
                {"name": "NLP", "proficiency": "intermediate", "endorsements": 15, "duration_months": 20},
                {"name": "Embeddings", "proficiency": "intermediate", "endorsements": 10, "duration_months": 12},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 40},
            ],
            career_desc="Trained classification models. Built NLP pipeline. Some embedding work for text similarity.",
        ),
        _make_cand(
            "CAND_INT_004", "Data Scientist", 8.0,
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 45, "duration_months": 80},
                {"name": "LightGBM", "proficiency": "advanced", "endorsements": 25, "duration_months": 50},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 20, "duration_months": 35},
                {"name": "Hugging Face", "proficiency": "intermediate", "endorsements": 12, "duration_months": 18},
                {"name": "Fine-tuning", "proficiency": "intermediate", "endorsements": 8, "duration_months": 14},
            ],
            career_desc="NLP model development. Fine-tuned BERT for classification. Large-scale data analysis.",
        ),
        # ── Tier 3: Low IR / unrelated ────────────────────────────────────
        _make_cand(
            "CAND_INT_005", "Junior ML Engineer", 2.5,
            skills=[
                {"name": "Python", "proficiency": "intermediate", "endorsements": 15, "duration_months": 24},
                {"name": "TensorFlow", "proficiency": "beginner", "endorsements": 5, "duration_months": 12},
                {"name": "Pandas", "proficiency": "intermediate", "endorsements": 8, "duration_months": 20},
            ],
            career_desc="Built data pipelines. Trained basic classification models. Learning NLP.",
        ),
        _make_cand(
            "CAND_INT_006", "Accountant", 10.0,
            skills=[
                {"name": "Excel", "proficiency": "advanced", "endorsements": 15, "duration_months": 100},
                {"name": "Tally", "proficiency": "advanced", "endorsements": 8, "duration_months": 90},
                {"name": "SAP", "proficiency": "intermediate", "endorsements": 5, "duration_months": 60},
            ],
            career_desc="Managed accounts, financial reporting, audit preparation.",
        ),
        _make_cand(
            "CAND_INT_007", "HR Manager", 7.0,
            skills=[
                {"name": "Excel", "proficiency": "advanced", "endorsements": 10, "duration_months": 70},
                {"name": "Recruitment", "proficiency": "advanced", "endorsements": 20, "duration_months": 70},
            ],
            career_desc="Managed hiring processes, onboarding, performance reviews.",
        ),
        # ── Disqualified: consulting-only ────────────────────────────────
        _make_cand(
            "CAND_INT_008", "ML Engineer", 6.0,
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 30, "duration_months": 60},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
                {"name": "NLP", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
            ],
            career_desc="Built ML models for clients. Deployed classification systems.",
            is_consulting=True,
            company="Tata Consultancy Services",
        ),
        # ── Honeypot ─────────────────────────────────────────────────────
        _make_cand(
            "CAND_INT_009", "AI Wizard", 3.0,
            skills=[
                {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 99, "duration_months": 0}
                for i in range(8)
            ],
            career_desc="Expert in everything.",
        ),
        # ── Alias candidate: uses non-canonical names ────────────────────
        _make_cand(
            "CAND_INT_010", "Search Engineer", 6.5,
            skills=[
                {"name": "HNSW", "proficiency": "advanced", "endorsements": 18, "duration_months": 24},
                {"name": "Approximate Nearest Neighbor", "proficiency": "advanced", "endorsements": 15, "duration_months": 20},
                {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 66},
                {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 22, "duration_months": 30},
                {"name": "Learning to Rank", "proficiency": "intermediate", "endorsements": 10, "duration_months": 18},
                {"name": "Weaviate", "proficiency": "advanced", "endorsements": 12, "duration_months": 20},
            ],
            career_desc=(
                "Built approximate nearest neighbor search using HNSW indexing. "
                "Production information retrieval system with learning-to-rank models. "
                "Deployed at scale, serving 2M queries/day."
            ),
        ),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests using 10-candidate fixture."""

    def _run_pipeline(self, top_n=10):
        """Write fixture to temp JSONL, run pipeline, return results + CSV rows."""
        fixture = build_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'candidates.jsonl')
            output_path = os.path.join(tmpdir, 'output.csv')

            with open(input_path, 'w') as f:
                for cand in fixture:
                    f.write(json.dumps(cand) + '\n')

            results = rank_candidates(input_path, output_path, top_n=top_n)

            with open(output_path, 'r') as f:
                rows = list(csv.DictReader(f))

        return results, rows, fixture

    def test_pipeline_produces_output(self):
        _, rows, _ = self._run_pipeline()
        assert len(rows) > 0

    def test_output_csv_has_required_columns(self):
        _, rows, _ = self._run_pipeline()
        required = {'candidate_id', 'rank', 'score', 'reasoning'}
        assert required.issubset(set(rows[0].keys()))

    def test_scores_are_monotonically_nonincreasing(self):
        _, rows, _ = self._run_pipeline()
        scores = [float(r['score']) for r in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i+1} ({scores[i]:.4f}) < rank {i+2} ({scores[i+1]:.4f})"
            )

    def test_all_scores_in_unit_interval(self):
        _, rows, _ = self._run_pipeline()
        for r in rows:
            score = float(r['score'])
            assert 0.0 < score < 1.0, f"{r['candidate_id']} has out-of-range score {score}"

    def test_all_ranks_unique(self):
        _, rows, _ = self._run_pipeline()
        ranks = [int(r['rank']) for r in rows]
        assert len(ranks) == len(set(ranks)), "Duplicate ranks detected"

    def test_ranks_start_at_1_and_are_sequential(self):
        _, rows, _ = self._run_pipeline()
        ranks = sorted(int(r['rank']) for r in rows)
        assert ranks[0] == 1
        for i in range(1, len(ranks)):
            assert ranks[i] == ranks[i - 1] + 1

    def test_reasoning_not_empty_for_any_candidate(self):
        _, rows, _ = self._run_pipeline()
        for r in rows:
            assert len(r['reasoning'].strip()) > 10, (
                f"{r['candidate_id']} has empty reasoning"
            )

    def test_all_candidate_ids_in_output_are_valid(self):
        _, rows, fixture = self._run_pipeline()
        valid_ids = {c['candidate_id'] for c in fixture}
        for r in rows:
            assert r['candidate_id'] in valid_ids, (
                f"Unknown candidate_id in output: {r['candidate_id']}"
            )

    def test_ir_engineer_ranks_above_accountant(self):
        """The core value proposition: skill-matched candidates rank above irrelevant ones."""
        results, rows, _ = self._run_pipeline()
        id_to_rank = {r['candidate_id']: int(r['rank']) for r in rows}

        # CAND_INT_001 (Senior AI Engineer with FAISS/embeddings/semantic search)
        # must rank above CAND_INT_006 (Accountant)
        ir_rank = id_to_rank.get('CAND_INT_001', 999)
        acct_rank = id_to_rank.get('CAND_INT_006', 999)
        assert ir_rank < acct_rank, (
            f"IR engineer (rank {ir_rank}) should rank above accountant (rank {acct_rank})"
        )

    def test_alias_candidate_ranks_competitively(self):
        """Candidate using HNSW/ANN/IR terminology should rank near top IR engineers."""
        _, rows, _ = self._run_pipeline()
        id_to_rank = {r['candidate_id']: int(r['rank']) for r in rows}

        alias_rank = id_to_rank.get('CAND_INT_010', 999)  # uses HNSW, ANN, IR
        canonical_rank = id_to_rank.get('CAND_INT_001', 999)  # uses FAISS, embeddings

        # Alias candidate should be within 5 ranks of canonical candidate
        assert abs(alias_rank - canonical_rank) <= 5, (
            f"Alias candidate (rank {alias_rank}) should be close to canonical (rank {canonical_rank}). "
            f"Gap of {abs(alias_rank - canonical_rank)} suggests alias expansion isn't working."
        )

    def test_honeypot_gets_lowest_score(self):
        """Honeypot (impossible skills) should score near zero."""
        results, rows, _ = self._run_pipeline()
        id_to_rank = {r['candidate_id']: int(r['rank']) for r in rows}
        id_to_score = {r['candidate_id']: float(r['score']) for r in rows}

        honeypot_score = id_to_score.get('CAND_INT_009', 1.0)
        ir_score = id_to_score.get('CAND_INT_001', 0.0)

        assert honeypot_score < 0.01, f"Honeypot should score near 0, got {honeypot_score}"
        assert ir_score > honeypot_score

    def test_consulting_only_candidate_scores_very_low(self):
        """Consulting-only career is a JD explicit disqualifier."""
        _, rows, _ = self._run_pipeline()
        id_to_score = {r['candidate_id']: float(r['score']) for r in rows}

        consulting_score = id_to_score.get('CAND_INT_008', 1.0)
        ir_score = id_to_score.get('CAND_INT_001', 0.0)

        assert consulting_score < 0.15, (
            f"Consulting-only candidate should score < 0.15, got {consulting_score}"
        )
        assert ir_score > consulting_score * 4

    def test_score_decomposition_in_reasoning(self):
        """Every reasoning string should contain score decomposition."""
        _, rows, _ = self._run_pipeline()
        import re
        decomp_pattern = re.compile(r'\[Skills [\d.]+ \|')
        for r in rows:
            if float(r['score']) > 0.05:  # Skip disqualified
                assert decomp_pattern.search(r['reasoning']), (
                    f"{r['candidate_id']} reasoning lacks score decomposition: {r['reasoning'][:80]}"
                )

    def test_top_n_respected(self):
        """Pipeline should return exactly top_n results."""
        _, rows, _ = self._run_pipeline(top_n=5)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

    def test_tie_breaking_by_candidate_id(self):
        """When scores tie, lower candidate_id should rank first."""
        _, rows, _ = self._run_pipeline()
        for i in range(len(rows) - 1):
            r1, r2 = rows[i], rows[i + 1]
            s1, s2 = float(r1['score']), float(r2['score'])
            if abs(s1 - s2) < 1e-9:  # Tie
                assert r1['candidate_id'] <= r2['candidate_id'], (
                    f"Tie-break failed: {r1['candidate_id']} should sort before {r2['candidate_id']}"
                )

    def test_reasoning_mentions_candidate_title(self):
        """Each reasoning string should reference the candidate's actual title."""
        results, rows, fixture = self._run_pipeline()
        id_to_title = {c['candidate_id']: c['profile']['current_title'] for c in fixture}

        for r in rows:
            cid = r['candidate_id']
            title = id_to_title.get(cid, '')
            if float(r['score']) > 0.10 and title:
                # Title or YOE should appear in reasoning
                assert title in r['reasoning'] or str(id_to_title[cid]) in r['reasoning'], (
                    f"{cid}: title '{title}' not in reasoning: {r['reasoning'][:100]}"
                )

    def test_confidence_values_in_reasoning(self):
        """Reasoning should include confidence signal count."""
        _, rows, _ = self._run_pipeline()
        import re
        conf_pattern = re.compile(r'conf=[\d.]+')
        count_with_conf = sum(1 for r in rows if conf_pattern.search(r['reasoning']) and float(r['score']) > 0.05)
        total_non_disq = sum(1 for r in rows if float(r['score']) > 0.05)
        # At least 80% of non-disqualified candidates should have confidence
        assert count_with_conf >= total_non_disq * 0.8, (
            f"Only {count_with_conf}/{total_non_disq} candidates have confidence in reasoning"
        )

    def test_stats_sidecar_written_when_requested(self):
        """rank_candidates should write a stats JSON when stats_output_path is given,
        and the stats should be internally consistent with the actual run."""
        fixture = build_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'candidates.jsonl')
            output_path = os.path.join(tmpdir, 'output.csv')
            stats_path = os.path.join(tmpdir, 'stats.json')

            with open(input_path, 'w') as f:
                for cand in fixture:
                    f.write(json.dumps(cand) + '\n')

            rank_candidates(input_path, output_path, top_n=10, stats_output_path=stats_path)

            assert os.path.exists(stats_path), "Stats sidecar was not written"
            with open(stats_path) as f:
                stats = json.load(f)

            assert stats['total_candidates'] == len(fixture)
            assert stats['honeypot_count'] >= 1  # CAND_INT_009 is a honeypot
            assert stats['disqualified_consulting'] >= 1  # CAND_INT_008 is consulting-only
            assert 'must_have_skills_parsed' in stats
            assert isinstance(stats['must_have_skills_parsed'], list)
            assert len(stats['must_have_skills_parsed']) > 0
            assert 'jd_extraction_confidence' in stats
            assert 0.0 <= stats['jd_extraction_confidence'] <= 1.0

    def test_no_stats_sidecar_when_not_requested(self):
        """Stats file should NOT be written if stats_output_path is None (default)."""
        fixture = build_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'candidates.jsonl')
            output_path = os.path.join(tmpdir, 'output.csv')

            with open(input_path, 'w') as f:
                for cand in fixture:
                    f.write(json.dumps(cand) + '\n')

            rank_candidates(input_path, output_path, top_n=10)

            stats_default_path = os.path.join(tmpdir, 'stats.json')
            assert not os.path.exists(stats_default_path)

    def test_streaming_does_not_load_full_dataset_into_results_list(self):
        """
        The pipeline should bound memory by capping the working buffer, not by
        holding every scored candidate. With top_n=3 and a buffer multiplier of 5,
        the internal buffer should never need to exceed a small bound even though
        we process 10 fixture candidates.
        """
        results, rows, fixture = self._run_pipeline(top_n=3)
        assert len(rows) == 3
        # The returned top_results list itself should never exceed top_n
        assert len(results) == 3

    def test_pipeline_survives_malformed_candidate_records(self):
        """
        REGRESSION TEST for a real production-fragility bug: a single
        malformed candidate (missing 'profile', missing 'candidate_id', or
        a skill entry missing its 'name' key) used to raise an unhandled
        KeyError deep inside score_candidate() and crash the ENTIRE run —
        discarding every other candidate's work. A 100K-line JSONL file is
        exactly the kind of input where one bad record is statistically
        likely. The pipeline must skip and count malformed records, not die.
        """
        good_fixture = build_fixture()[:3]

        malformed_no_profile = {
            'candidate_id': 'CAND_BAD_NO_PROFILE',
            'career_history': [], 'skills': [],
        }
        malformed_no_id = {
            'profile': {'current_title': 'X', 'years_of_experience': 5},
            'career_history': [], 'skills': [],
        }
        malformed_bad_skill = json.loads(json.dumps(good_fixture[0]))  # deep copy
        malformed_bad_skill['candidate_id'] = 'CAND_BAD_SKILL_NAME'
        malformed_bad_skill['skills'] = [{'proficiency': 'advanced'}]  # missing 'name'

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'candidates.jsonl')
            output_path = os.path.join(tmpdir, 'output.csv')
            stats_path = os.path.join(tmpdir, 'stats.json')

            with open(input_path, 'w') as f:
                for cand in good_fixture:
                    f.write(json.dumps(cand) + '\n')
                f.write(json.dumps(malformed_no_profile) + '\n')
                f.write(json.dumps(malformed_no_id) + '\n')
                f.write(json.dumps(malformed_bad_skill) + '\n')

            # Must not raise — this is the core assertion
            results = rank_candidates(input_path, output_path, top_n=10, stats_output_path=stats_path)

            assert len(results) == 3, f"Expected 3 valid candidates ranked, got {len(results)}"

            with open(stats_path) as f:
                stats = json.load(f)
            assert stats['malformed_skipped'] == 3, (
                f"Expected 3 malformed records counted, got {stats['malformed_skipped']}"
            )
            assert stats['total_candidates'] == 6  # 3 good + 3 malformed processed

    def test_pipeline_survives_null_valued_fields(self):
        """
        REGRESSION TEST for a distinct failure mode from missing keys: a
        candidate record where keys are PRESENT but their values are JSON
        null/None. dict.get(key, default) only falls back when the key is
        ABSENT — a present key with value None still returns None, which
        then crashes arithmetic deep in scoring (e.g. `notice + 5` on a
        None notice_period_days). This is realistic: malformed upstream
        data pipelines often emit explicit nulls rather than omitting keys.
        """
        null_heavy = {
            'candidate_id': 'CAND_NULL_VALUES',
            'profile': {
                'current_title': None, 'years_of_experience': None,
                'current_company': None, 'current_company_size': None,
                'country': None, 'location': None, 'summary': None,
                'headline': None, 'current_industry': None,
            },
            'career_history': [{
                'company': None, 'title': None, 'start_date': None, 'end_date': None,
                'duration_months': None, 'is_current': None, 'industry': None,
                'company_size': None, 'description': None,
            }],
            'education': [{'institution': None, 'degree': None, 'field_of_study': None, 'tier': None}],
            'skills': [{'name': 'Python', 'proficiency': None, 'endorsements': None, 'duration_months': None}],
            'certifications': [],
            'redrob_signals': {
                'last_active_date': None, 'open_to_work_flag': None,
                'recruiter_response_rate': None, 'notice_period_days': None,
            },
        }
        good_fixture = build_fixture()[:1]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'candidates.jsonl')
            output_path = os.path.join(tmpdir, 'output.csv')
            stats_path = os.path.join(tmpdir, 'stats.json')

            with open(input_path, 'w') as f:
                for cand in good_fixture:
                    f.write(json.dumps(cand) + '\n')
                f.write(json.dumps(null_heavy) + '\n')

            # Must not raise
            results = rank_candidates(input_path, output_path, top_n=10, stats_output_path=stats_path)
            assert len(results) == 1, f"Expected 1 valid candidate, got {len(results)}"

            with open(stats_path) as f:
                stats = json.load(f)
            assert stats['malformed_skipped'] == 1


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
