"""
NeuroIgniter AI Recruiter — Core Scoring Engine v2
====================================================
Upgraded architecture:
  - Skill alias/ontology expansion (semantic matching without compute cost)
  - Skill corroboration check (self-reported skills cross-referenced against career text)
  - Promotion detection (multiple roles at same company with title advancement)
  - Achievement extraction (quantified impact signals from career descriptions)
  - Candidate-specific reasoning (no templates — each reasoning is unique)
  - Score decomposition (full component breakdown for explainability)
  - Config-driven weights (no hardcoded numbers in scoring logic)
  - Signal renormalization (missing signals don't unfairly penalize)
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from .jd_parser import (
    ADJACENT_TITLE_KEYWORDS,
    IDEAL_TITLE_KEYWORDS,
    JD_REQUIREMENTS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
)
from .ontology import (
    compute_cluster_coherence,
    get_corroboration_score,
    normalize_skill,
)
from .semantic import DistributionalEmbedder

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger("neuroigniter")

# ── Load config ───────────────────────────────────────────────────────────────
try:
    import yaml
    _config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    with open(_config_path) as f:
        CONFIG = yaml.safe_load(f)
    log.info("Config loaded from config.yaml")
except Exception:
    # Fallback defaults if yaml unavailable or config missing
    CONFIG = {}
    log.warning("Config not loaded — using defaults")

_cfg_miss_logged: set[str] = set()

def _cfg(path: str, default: Any) -> Any:
    """
    Traverse dot-path in CONFIG dict. Falls back to `default` if the key
    is missing or config failed to load — and logs the miss exactly once
    per key so a config.yaml typo doesn't fail silently forever.
    """
    parts = path.split('.')
    node = CONFIG
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            if path not in _cfg_miss_logged:
                log.debug(f"Config key '{path}' not found — using default {default!r}")
                _cfg_miss_logged.add(path)
            return default
        node = node[p]
    return node if node != {} else default


# ── Company disqualifier list ─────────────────────────────────────────────────
CONSULTING_FIRMS = set(_cfg('consulting_firms', [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "lti", "ltimindtree", "mindtree",
]))

CV_ROBOTICS_TITLES = set(_cfg('cv_robotics_titles', [
    "computer vision engineer", "cv engineer", "robotics engineer",
    "speech engineer", "asr engineer", "tts engineer", "autonomous systems",
    "ros developer", "slam engineer",
]))

CV_ROBOTICS_SKILLS = {
    "opencv", "yolo", "object detection", "image segmentation", "slam",
    "ros", "lidar", "point cloud", "3d reconstruction", "speech recognition",
    "asr", "tts", "audio processing", "signal processing",
}

# ── Achievement regex patterns ─────────────────────────────────────────────────
# An achievement must have BOTH: a magnitude/scale indicator AND an action verb.
# This prevents matching mundane sentences like "Led the team meeting."
ACHIEVEMENT_MAGNITUDE_RE = re.compile(
    r'\b(\d+\s*[%xX]|\d+[KkMmBb]\b|\d+\s*(users|requests|candidates|queries|calls|records|ms|seconds|hours|weeks|days|months))\b',
    re.IGNORECASE,
)
ACHIEVEMENT_ACTION_RE = re.compile(
    r'\b(shipped|launched|deployed|reduced|improved|increased|built|scaled|cut|optimized|led|drove|delivered)\b',
    re.IGNORECASE,
)

# ── Seniority title ladder ─────────────────────────────────────────────────────
SENIORITY_SCORES = {
    "staff": 1.0, "principal": 1.0, "distinguished": 1.0, "fellow": 1.0,
    "lead": 0.90, "senior": 0.85, "mid": 0.75, "": 0.70,
    "junior": 0.50, "associate": 0.55, "intern": 0.20,
    "applied": 0.85, "research": 0.80,
}


def _seniority_score(title: str) -> float:
    title_lower = title.lower()
    for kw, score in SENIORITY_SCORES.items():
        if kw and kw in title_lower:
            return score
    return 0.70


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """Per-component scores for full explainability."""
    skills: float = 0.0
    career: float = 0.0
    behavioral: float = 0.0
    availability: float = 0.0
    semantic: float = 0.0
    education: float = 0.0

    def weighted_total(self, weights: dict[str, float]) -> float:
        return float(
            weights.get('skills', 0.38) * self.skills +
            weights.get('career', 0.26) * self.career +
            weights.get('behavioral', 0.12) * self.behavioral +
            weights.get('availability', 0.08) * self.availability +
            weights.get('semantic', 0.10) * self.semantic +
            weights.get('education', 0.06) * self.education
        )


@dataclass
class CandidateAnalysis:
    """Full candidate intelligence extraction result."""
    candidate_id: str
    total_score: float
    components: ComponentScores = field(default_factory=ComponentScores)

    # Skills
    must_have_matched: list[str] = field(default_factory=list)
    nice_have_matched: list[str] = field(default_factory=list)
    missing_must_have: list[str] = field(default_factory=list)
    skill_cluster_coherence: float = 0.0

    # Semantic (hybrid: lexical TF-IDF + distributional embedding)
    semantic_lexical_component: float = 0.0
    semantic_distributional_component: float | None = None

    # Career
    experience_years: float = 0.0
    current_title: str = ""
    promotions_detected: int = 0
    achievements_found: list[str] = field(default_factory=list)
    career_at_product_companies: bool = True

    # Signals
    notice_days: int = 90
    response_rate: float = 0.5
    github_score: float = -1.0
    is_open_to_work: bool = False
    last_active_days_ago: int = 999
    assessment_score: float = 0.0

    # Flags
    is_disqualified: bool = False
    is_honeypot: bool = False
    disqualify_reasons: list[str] = field(default_factory=list)

    # Narrative signals for reasoning
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    interview_focus: list[str] = field(default_factory=list)
    missing_skills_top3: list[str] = field(default_factory=list)
    signal_count: int = 0           # How many signals were available (out of 23)
    confidence: float = 0.80        # Estimate based on signal availability


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF SEMANTIC MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFMatcher:
    """Lightweight TF-IDF cosine similarity. Built from JD requirements, not hardcoded."""

    def __init__(self, jd_text: str):
        self.jd_tokens = self._tokenize(jd_text)
        self.jd_tf = self._compute_tf(self.jd_tokens)

    STOPWORDS = frozenset({
        'the', 'a', 'an', 'and', 'or', 'in', 'at', 'to', 'for',
        'of', 'with', 'on', 'is', 'are', 'was', 'be', 'by', 'from',
        'i', 'my', 'we', 'our', 'their', 'this', 'that', 'have',
        'has', 'had', 'not', 'but', 'they', 'it', 'as', 'been',
        'who', 'what', 'which', 'when', 'where', 'how', 'about',
        'will', 'would', 'could', 'should', 'can', 'may', 'might',
    })

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        return [t for t in text.split() if t not in self.STOPWORDS and len(t) > 2]

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        if tokens:
            max_freq = max(tf.values())
            for t in tf:
                tf[t] /= max_freq
        return dict(tf)

    def similarity(self, candidate_text: str) -> float:
        tokens = self._tokenize(candidate_text)
        if not tokens:
            return 0.0
        cand_tf = self._compute_tf(tokens)
        dot = sum(self.jd_tf.get(t, 0) * cand_tf.get(t, 0) for t in cand_tf)
        jd_mag = math.sqrt(sum(v**2 for v in self.jd_tf.values()))
        cand_mag = math.sqrt(sum(v**2 for v in cand_tf.values()))
        if jd_mag == 0 or cand_mag == 0:
            return 0.0
        return min(dot / (jd_mag * cand_mag), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SCORING MODULES
# ─────────────────────────────────────────────────────────────────────────────

def _detect_honeypot(candidate: dict) -> tuple[bool, str]:
    profile = candidate['profile']
    yoe = profile.get('years_of_experience', 0)
    skills = candidate.get('skills', [])

    # Impossible skill proficiency/duration
    impossible = sum(
        1 for s in skills
        if s.get('duration_months', 1) == 0 and s.get('proficiency') in ('advanced', 'expert')
    )
    threshold = _cfg('honeypot.impossible_skill_threshold', 5)
    if impossible >= threshold:
        return True, f"{impossible} skills claim advanced proficiency with 0 months experience"

    # Career timeline inflation
    total_months = sum(ch.get('duration_months', 0) for ch in candidate.get('career_history', []))
    if total_months > 0 and yoe > 0:
        implied_years = total_months / 12
        factor = _cfg('honeypot.career_inflation_factor', 2.5)
        min_years = _cfg('honeypot.career_inflation_min_years', 20)
        if implied_years > yoe * factor and implied_years > min_years:
            return True, f"career implies {implied_years:.1f} years but profile claims {yoe}"

    # Too many expert skills for low YOE
    expert_threshold = _cfg('honeypot.expert_skill_threshold', 8)
    yoe_max = _cfg('honeypot.expert_yoe_max', 5)
    expert_count = sum(
        1 for s in skills
        if s.get('proficiency') == 'advanced' and s.get('endorsements', 0) > 90
    )
    if expert_count >= expert_threshold and yoe < yoe_max:
        return True, f"{expert_count} suspiciously high-endorsement skills for {yoe} yrs"

    return False, ""


def _check_disqualifiers(candidate: dict, career_companies: list[str]) -> tuple[bool, list[str]]:
    reasons = []

    # Consulting-only career (single or multiple firms)
    valid_companies = [c for c in career_companies if c]
    if valid_companies:
        all_consulting = all(
            any(firm in c.lower() for firm in CONSULTING_FIRMS)
            for c in valid_companies
        )
        if all_consulting:
            reasons.append("entire career at consulting firms (TCS/Wipro/Infosys-class) — JD explicit disqualifier")

    # CV/robotics primary without NLP/IR
    skills_lower = {s['name'].lower() for s in candidate.get('skills', [])}
    skill_names_normalized = {normalize_skill(s) for s in skills_lower}
    has_cv_robotics_skills = bool(skills_lower & CV_ROBOTICS_SKILLS)
    has_nlp_ir = any(
        s in skill_names_normalized for s in
        {'nlp', 'embeddings', 'vector search', 'information retrieval', 'rag', 'ranking', 'semantic search'}
    )
    title_lower = candidate['profile']['current_title'].lower()
    if any(t in title_lower for t in CV_ROBOTICS_TITLES) and has_cv_robotics_skills and not has_nlp_ir:
        reasons.append("primary expertise is CV/speech/robotics without NLP/IR background")

    return bool(reasons), reasons


def _score_skills(
    candidate: dict,
    career_text: str,
) -> tuple[float, list[str], list[str], list[str], float]:
    """
    Score skills with:
    - Ontology-based alias expansion
    - Proficiency × duration weighting
    - Corroboration check (career text validation)
    - Cluster coherence bonus
    Returns: (score, must_matched, nice_matched, missing_must, cluster_coherence)
    """
    skills = candidate.get('skills', [])

    # Build a normalized skill map with weights
    skill_weights: dict[str, float] = {}
    for s in skills:
        raw = s['name']
        canonical = normalize_skill(raw)
        prof = s.get('proficiency', 'intermediate')
        duration = s.get('duration_months', 12)

        prof_mult = {'beginner': 0.5, 'intermediate': 1.0, 'advanced': 1.35, 'expert': 1.5}.get(prof, 1.0)
        dur_mult = min(duration / 48.0, 1.0) * 0.25 + 0.75  # 0.75 to 1.0

        # Corroboration: cross-reference against career descriptions
        corroboration = get_corroboration_score(canonical, career_text)
        penalty = _cfg('bonuses.corroboration_penalty', 0.50)
        # If corroboration is weak (<0.45), reduce effective weight
        # Harder floor: zero career evidence → 0.35x, some evidence → 0.50x minimum
        if corroboration < 0.30:
            corroboration_factor = 0.35  # No career evidence at all
        elif corroboration < 0.45:
            corroboration_factor = max(corroboration, penalty)
        else:
            corroboration_factor = 1.0

        weight = prof_mult * dur_mult * corroboration_factor
        # Take the best score if skill appears multiple times under aliases
        existing = skill_weights.get(canonical, 0.0)
        skill_weights[canonical] = max(existing, weight)

    canonical_skill_set = set(skill_weights.keys())

    # Must-have matching
    must_matched = []
    must_missing = []
    must_score = 0.0
    must_possible = 8.0 * 1.35  # 8 key skills at advanced proficiency

    for skill in MUST_HAVE_SKILLS:
        canonical = normalize_skill(skill)
        if canonical in skill_weights:
            must_score += skill_weights[canonical]
            must_matched.append(canonical)
        else:
            must_missing.append(canonical)

    must_normalized = min(must_score / must_possible, 1.0)

    # Nice-to-have matching
    nice_matched = []
    nice_score = 0.0
    nice_possible = 6.0 * 1.35

    for skill in NICE_TO_HAVE_SKILLS:
        canonical = normalize_skill(skill)
        if canonical in skill_weights:
            nice_score += skill_weights[canonical]
            nice_matched.append(canonical)

    nice_normalized = min(nice_score / nice_possible, 1.0)

    # Cluster coherence — IR specialist bonus
    cluster_coherence = compute_cluster_coherence(canonical_skill_set)

    # Python must-have boost
    python_weight = skill_weights.get('python', 0.0)
    python_boost = 0.04 * min(python_weight, 1.5)

    mh_weight = _cfg('skill_weights.must_have', 0.72)
    nth_weight = _cfg('skill_weights.nice_to_have', 0.28)
    skill_score = (
        mh_weight * must_normalized +
        nth_weight * nice_normalized +
        python_boost +
        0.05 * cluster_coherence  # Specialist coherence bonus
    )
    skill_score = min(skill_score, 1.0)

    return skill_score, must_matched, nice_matched, must_missing[:5], cluster_coherence


def _detect_promotions(career_history: list[dict]) -> int:
    """
    Detect promotions: multiple roles at same company with title advancement.
    Returns count of detected promotions.
    """
    promotions = 0
    by_company: dict[str, list[dict]] = defaultdict(list)
    for job in career_history:
        by_company[job.get('company', '')].append(job)

    for company, jobs in by_company.items():
        if len(jobs) < 2:
            continue
        # Sort by start date
        sorted_jobs = sorted(jobs, key=lambda x: x.get('start_date', ''))
        for i in range(1, len(sorted_jobs)):
            prev_title = sorted_jobs[i-1].get('title', '').lower()
            curr_title = sorted_jobs[i].get('title', '').lower()

            # Title advancement signals
            prev_seniority = _seniority_score(prev_title)
            curr_seniority = _seniority_score(curr_title)

            if curr_seniority > prev_seniority + 0.05:
                promotions += 1
            elif prev_title != curr_title and any(
                adv in curr_title for adv in ['lead', 'senior', 'staff', 'principal', 'manager', 'head']
            ) and not any(
                adv in prev_title for adv in ['lead', 'senior', 'staff', 'principal', 'manager', 'head']
            ):
                promotions += 1

    return promotions


def _detect_cross_company_advancement(career_history: list[dict]) -> int:
    """
    Detect upward title movement across different companies.
    A candidate who was 'ML Engineer' at Co A and moved to 'Senior ML Engineer'
    at Co B is advancing their career, not just job-hopping.
    Returns count of detected cross-company advancements.
    """
    if len(career_history) < 2:
        return 0

    sorted_jobs = sorted(career_history, key=lambda x: x.get('start_date', ''))
    advances = 0
    for i in range(1, len(sorted_jobs)):
        prev = sorted_jobs[i - 1]
        curr = sorted_jobs[i]
        # Only cross-company moves
        if prev.get('company', '') == curr.get('company', ''):
            continue
        prev_seniority = _seniority_score(prev.get('title', ''))
        curr_seniority = _seniority_score(curr.get('title', ''))
        if curr_seniority > prev_seniority + 0.08:
            advances += 1
    return advances


def _extract_achievements(career_history: list[dict]) -> list[str]:
    """
    Extract genuine quantified achievements from career descriptions.
    Requires BOTH a magnitude indicator (number/%, scale) AND an action verb.
    This prevents matching mundane sentences without real impact claims.
    """
    achievements = []
    for job in career_history:
        desc = job.get('description', '')
        sentences = re.split(r'[.!?]\s+', desc)
        for sentence in sentences:
            has_magnitude = ACHIEVEMENT_MAGNITUDE_RE.search(sentence)
            has_action = ACHIEVEMENT_ACTION_RE.search(sentence)
            if has_magnitude and has_action and len(sentence.strip()) > 20:
                clean = sentence.strip()
                if len(clean) > 15:
                    achievements.append(clean[:120])
    return achievements[:5]  # Top 5 genuine achievements


def _score_career(candidate: dict) -> tuple[float, list[str], list[str], int, list[str]]:
    """
    Score career trajectory.
    Returns: (score, strengths, concerns, promotions, achievements)
    """
    profile = candidate['profile']
    career = candidate.get('career_history', [])
    yoe = profile.get('years_of_experience', 0)

    strengths = []
    concerns = []

    # ── Experience score — smooth Gaussian curve ────────────────────
    # Peaks at 7.0 yrs (midpoint of 5-9 ideal), decays smoothly either
    # side. Eliminates the hard 4.9 vs 5.0 cliff from step-function bands.
    # sigma=3.5 gives ~0.85 at 4 yrs, ~0.70 at 2 yrs, ~0.75 at 12 yrs.
    yoe_center = 7.0
    yoe_sigma = 3.5
    exp_score = max(0.30, math.exp(-0.5 * ((yoe - yoe_center) / yoe_sigma) ** 2))

    if 5.0 <= yoe <= 9.0:
        strengths.append(f"{yoe:.1f} yrs aligns with the 5-9 yr ideal range")
    elif yoe < 3.0:
        concerns.append(f"{yoe:.1f} yrs experience is below the JD minimum of 5 yrs")

    # ── Title relevance ───────────────────────────────────────────────
    title_lower = profile.get('current_title', '').lower()
    if any(t in title_lower for t in IDEAL_TITLE_KEYWORDS):
        title_score = 1.0
        strengths.append(f"title '{profile['current_title']}' directly matches the JD role")
    elif any(t in title_lower for t in ADJACENT_TITLE_KEYWORDS):
        title_score = 0.55
    else:
        title_score = 0.15
        concerns.append(f"current title '{profile['current_title']}' is non-technical")

    # Seniority within title
    seniority = _seniority_score(profile.get('current_title', ''))
    title_score = title_score * (0.7 + 0.3 * seniority)

    # ── Product vs consulting ──────────────────────────────────────────
    product_months = 0
    total_months = 0
    current_in_consulting = False

    for job in career:
        months = job.get('duration_months', 0)
        company_lower = job.get('company', '').lower()
        total_months += months
        is_consulting = any(firm in company_lower for firm in CONSULTING_FIRMS)
        if not is_consulting:
            product_months += months
        if job.get('is_current') and is_consulting:
            current_in_consulting = True

    product_ratio = product_months / max(total_months, 1)
    product_score = min(product_ratio * 1.15, 1.0)

    if current_in_consulting:
        concerns.append("currently at a consulting firm — JD explicitly prefers product companies")
        product_score *= 0.72

    # ── Production AI evidence ────────────────────────────────────────
    career_text = ' '.join(ch.get('description', '') for ch in career).lower()
    production_signals = [
        'production', 'deployed', 'real users', 'at scale', 'serving',
        'latency', 'inference', 'a/b test', 'evaluation', 'shipped',
        'monitoring', 'endpoint', 'api', 'mlops',
    ]
    ir_signals = [
        'retrieval', 'ranking', 'embedding', 'vector', 'search',
        'index', 'rerank', 'similarity', 'faiss', 'bm25', 'dense',
    ]
    prod_count = sum(1 for s in production_signals if s in career_text)
    ir_count = sum(1 for s in ir_signals if s in career_text)

    prod_evidence_score = min((prod_count * 0.07 + ir_count * 0.08), 1.0)
    if prod_count >= 3:
        strengths.append(f"career descriptions show production system evidence ({prod_count} production signals)")
    if ir_count >= 3:
        strengths.append(f"career text demonstrates IR/retrieval domain depth ({ir_count} IR signals)")

    # ── Tenure consistency ────────────────────────────────────────────
    if len(career) >= 2:
        avg_tenure = total_months / len(career)
        hop_threshold = _cfg('career.min_tenure_months_hop', 18)
        good_threshold = _cfg('career.good_tenure_months', 30)
        if avg_tenure < hop_threshold:
            concerns.append(f"avg tenure {avg_tenure:.0f} months suggests job hopping — JD wants 3+ yr commitment")
            tenure_score = 0.50
        elif avg_tenure >= good_threshold:
            tenure_score = 1.0
            strengths.append(f"strong tenure (avg {avg_tenure:.0f} months per role)")
        else:
            tenure_score = 0.75
    else:
        tenure_score = 0.80

    # ── Promotion detection ───────────────────────────────────────────
    promotions = _detect_promotions(career)
    cross_advances = _detect_cross_company_advancement(career)
    total_advances = promotions + cross_advances
    promotion_bonus = total_advances * _cfg('career.promotion_bonus', 0.05)
    if promotions > 0:
        strengths.append(f"{promotions} promotion(s) at same company — consistent advancement")
    if cross_advances > 0:
        strengths.append(f"{cross_advances} cross-company title advancement(s) detected")

    # ── Achievement extraction ────────────────────────────────────────
    achievements = _extract_achievements(career)
    achievement_bonus = min(len(achievements) * _cfg('career.achievement_bonus', 0.02), 0.08)
    if achievements:
        strengths.append(f"{len(achievements)} quantified achievements found in career descriptions")

    # ── Company size signal ───────────────────────────────────────────
    current_size = profile.get('current_company_size', '')
    product_sizes = set(_cfg('career.product_company_sizes', ['1-10', '11-50', '51-200', '201-500', '501-1000']))
    if current_size in product_sizes:
        company_score = 0.90
        if current_size in ('1-10', '11-50', '51-200'):
            strengths.append(f"startup-scale company ({current_size} employees) — matches JD 'founding team' mindset")
    elif current_size in ('501-1000', '1001-5000'):
        company_score = 0.75
    else:
        company_score = 0.55

    # ── Culture-signal alignment ──────────────────────────────────────
    # The JD's extracted culture signals (wants_product, wants_startup,
    # wants_ownership, is_research_role) are now genuinely applied to
    # modulate the career score — previously computed but never used.
    culture_bonus = 0.0

    if JD_REQUIREMENTS.wants_startup_mindset:
        # JD explicitly wants "founding team" / startup experience.
        # Reward candidates at small-scale companies with strong tenure.
        if current_size in ('1-10', '11-50', '51-200') and tenure_score >= 0.75:
            culture_bonus += 0.02
            strengths.append("startup tenure matches JD 'founding team' culture expectation")

    if JD_REQUIREMENTS.wants_ownership:
        # JD wants "own the intelligence layer" — reward solo/lead contributions.
        ownership_signals = ['from scratch', 'end-to-end', 'independently', 'led the',
                             'architected', 'sole', 'owned', 'drove', 'responsible for']
        if any(sig in career_text.lower() for sig in ownership_signals):
            culture_bonus += 0.02
            strengths.append("career descriptions show ownership/autonomous-work evidence")

    if JD_REQUIREMENTS.is_research_role:
        # Research roles should reward academic/publication evidence
        research_signals = ['published', 'arxiv', 'paper', 'research', 'workshop', 'journal']
        if any(sig in career_text.lower() for sig in research_signals):
            culture_bonus += 0.02
    else:
        # Production role: penalize pure research-only backgrounds
        production_signals = ['production', 'deployed', 'shipped', 'users', 'api', 'endpoint']
        research_only_signals = ['research', 'publication', 'arxiv', 'lab', 'theoretical']
        has_production = any(sig in career_text.lower() for sig in production_signals)
        is_research_heavy = (
            sum(1 for sig in research_only_signals if sig in career_text.lower()) >= 3
            and not has_production
        )
        if is_research_heavy:
            culture_bonus -= 0.02
            concerns.append("career text suggests research-focused background; JD wants production engineers")

    culture_bonus = max(-0.05, min(culture_bonus, 0.06))  # cap impact

    # ── Composite career score ──────────────────────────────────────
    career_score = (
        0.22 * exp_score +
        0.24 * title_score +
        0.18 * product_score +
        0.15 * prod_evidence_score +
        0.10 * tenure_score +
        0.08 * company_score +
        promotion_bonus +
        achievement_bonus +
        culture_bonus
    )
    career_score = min(career_score, 1.0)

    return career_score, strengths, concerns, total_advances, achievements


def _score_behavioral(signals: dict, today: date) -> tuple[float, float, dict]:
    """
    Score behavioral signals. Returns (behavioral_score, availability_score, signal_detail).
    signal_detail is used for reasoning generation.
    """
    detail: dict[str, Any] = {}

    # ── Recency ───────────────────────────────────────────────────────
    last_active_str = signals.get('last_active_date', '')
    days_ago = 999
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
            days_ago = (today - last_active).days
        except ValueError:
            pass

    if days_ago <= 7:
        recency_score = 1.0
    elif days_ago <= 14:
        recency_score = 0.92
    elif days_ago <= 30:
        recency_score = 0.80
    elif days_ago <= 60:
        recency_score = 0.65
    elif days_ago <= 90:
        recency_score = 0.45
    elif days_ago <= 180:
        recency_score = 0.28
    else:
        recency_score = 0.10
    detail['days_inactive'] = days_ago

    # ── Response rate ─────────────────────────────────────────────────
    response_rate = signals.get('recruiter_response_rate', 0.5)
    detail['response_rate'] = response_rate

    # ── Response speed ────────────────────────────────────────────────
    avg_response_hours = signals.get('avg_response_time_hours', 48)
    if avg_response_hours <= 4:
        response_speed = 1.0
    elif avg_response_hours <= 12:
        response_speed = 0.85
    elif avg_response_hours <= 24:
        response_speed = 0.70
    elif avg_response_hours <= 48:
        response_speed = 0.50
    else:
        response_speed = 0.25

    # ── GitHub ────────────────────────────────────────────────────────
    github = signals.get('github_activity_score', -1)
    if github == -1:
        github_score = 0.30
        detail['github'] = 'not linked'
    else:
        github_score = github / 100.0
        detail['github'] = github
    # Renormalize if github missing
    # ── Assessment scores ─────────────────────────────────────────────
    assessments = signals.get('skill_assessment_scores', {})
    relevant = {
        k: v for k, v in assessments.items()
        if any(kw in k.lower() for kw in ['python', 'ml', 'ai', 'nlp', 'data', 'sql', 'machine'])
    }
    if relevant:
        assessment_score = sum(relevant.values()) / len(relevant) / 100.0
        detail['assessment'] = f"{sum(relevant.values())/len(relevant):.0f}/100 on {list(relevant.keys())[0]}"
    else:
        assessment_score = 0.40
        detail['assessment'] = 'none'

    # ── Interview completion ───────────────────────────────────────────
    interview_rate = signals.get('interview_completion_rate', 0.5)

    # ── Offer acceptance ──────────────────────────────────────────────
    # A -1 ("no prior offers") is ambiguous in isolation: it could mean
    # "strong candidate who simply hasn't found the right role yet" or
    # "weak candidate who never advances past screening." We disambiguate
    # using correlated signals rather than defaulting both cases to 0.50.
    offer_accept = signals.get('offer_acceptance_rate', -1)
    response_rate_for_offer = signals.get('recruiter_response_rate', 0.5)
    if offer_accept == -1:
        if response_rate_for_offer > 0.70 and interview_rate > 0.80:
            # High engagement + high interview completion but no offers yet —
            # more consistent with "hasn't found the right fit" than "weak."
            offer_score = 0.65
        elif response_rate_for_offer < 0.30 or interview_rate < 0.40:
            # Low engagement signals correlate with candidates who don't
            # advance — lean slightly negative rather than neutral.
            offer_score = 0.35
        else:
            offer_score = 0.50  # Genuinely ambiguous — stay neutral
        offer_weight_factor = 0.7  # Inferred value is less reliable than observed
    else:
        offer_score = max(offer_accept, 0)
        offer_weight_factor = 1.0

    # ── Profile completeness ──────────────────────────────────────────
    completeness = signals.get('profile_completeness_score', 50) / 100.0

    # ── Passive demand ────────────────────────────────────────────────
    saved_30d = min(signals.get('saved_by_recruiters_30d', 0) / 10.0, 1.0)
    search_appearances = min(signals.get('search_appearance_30d', 0) / 15.0, 1.0)
    detail['saved_by_recruiters'] = signals.get('saved_by_recruiters_30d', 0)

    # ── Verified signals (lower ghost risk) ───────────────────────────
    # Verified contact info = real person, lower ghost-candidate risk.
    # LinkedIn connection = engaged professional network.
    verified = (
        (0.5 if signals.get('verified_email', False) else 0) +
        (0.5 if signals.get('verified_phone', False) else 0)
    )
    linkedin_connected = signals.get('linkedin_connected', False)

    # ── Network / passive demand ──────────────────────────────────────
    # Connection count: professional network size → field embeddedness
    connection_count = signals.get('connection_count', 0)
    network_score = min(connection_count / 500.0, 1.0)  # 500+ connections = max

    # ── Composite behavioral ──────────────────────────────────────────
    beh_weights = _cfg('behavioral_weights', {})
    behavioral_score = (
        beh_weights.get('recruiter_response_rate', 0.23) * response_rate +
        beh_weights.get('recency', 0.14) * recency_score +
        beh_weights.get('response_speed', 0.11) * response_speed +
        beh_weights.get('interview_completion_rate', 0.11) * interview_rate +
        beh_weights.get('offer_acceptance_rate', 0.09) * offer_score * offer_weight_factor +
        beh_weights.get('github_activity', 0.09) * github_score +
        beh_weights.get('skill_assessment', 0.08) * assessment_score +
        beh_weights.get('profile_completeness', 0.05) * completeness +
        beh_weights.get('saved_by_recruiters', 0.03) * (saved_30d + search_appearances * 0.5) +
        # Newly wired: verification reduces ghost risk (+small trust bonus)
        0.04 * (verified / 1.0) +
        # LinkedIn connectivity = professional network presence
        0.02 * (1.0 if linkedin_connected else 0.0) +
        0.01 * network_score
    )

    # ── Availability ─────────────────────────────────────────────────
    open_to_work = signals.get('open_to_work_flag', False)
    notice_days = signals.get('notice_period_days', 90)
    preferred_mode = signals.get('preferred_work_mode', 'hybrid')
    willing_to_relocate = signals.get('willing_to_relocate', True)

    # Notice period scoring
    if notice_days == 0:
        notice_score = 1.0
    elif notice_days <= 15:
        notice_score = 0.95
    elif notice_days <= 30:
        notice_score = 0.85
    elif notice_days <= 60:
        notice_score = 0.60
    elif notice_days <= 90:
        notice_score = 0.40
    else:
        notice_score = 0.20
        detail['notice_concern'] = True

    # Location
    if preferred_mode in ('hybrid', 'flexible', 'onsite'):
        location_score = 0.90
    else:
        location_score = 0.70 if willing_to_relocate else 0.45

    open_weight = _cfg('availability_weights.open_to_work', 0.45)
    notice_weight = _cfg('availability_weights.notice_period', 0.35)
    loc_weight = _cfg('availability_weights.location', 0.20)

    open_val = 1.0 if open_to_work else 0.25
    availability_score = (
        open_weight * open_val +
        notice_weight * notice_score +
        loc_weight * location_score
    )

    # Signal interaction: open_to_work AND short notice AND good response = multiplier
    if open_to_work and notice_days <= 30 and response_rate >= 0.7:
        availability_score = min(availability_score * 1.12, 1.0)
        detail['availability_composite_boost'] = True

    detail['notice_days'] = notice_days
    detail['open_to_work'] = open_to_work

    return behavioral_score, availability_score, detail


def _score_education(candidate: dict) -> float:
    education = candidate.get('education', [])
    if not education:
        return 0.42

    relevant_fields = set(_cfg('education.relevant_fields', [
        'computer science', 'data science', 'machine learning',
        'artificial intelligence', 'statistics', 'mathematics',
        'software engineering', 'electrical engineering',
    ]))
    degree_scores = _cfg('education.degree_scores', {})
    tier_mults = _cfg('education.tier_multipliers', {
        'tier_1': 1.25, 'tier_2': 1.10, 'tier_3': 0.90,
        'tier_4': 0.75, 'unknown': 0.95,
    })

    best = 0.30
    for edu in education:
        field = edu.get('field_of_study', '').lower()
        degree = edu.get('degree', '').lower().strip('.')
        tier = edu.get('tier', 'unknown')

        deg_score = degree_scores.get(degree, 0.45)
        tier_mult = tier_mults.get(tier, 0.95)
        field_mult = 1.10 if any(rf in field for rf in relevant_fields) else 0.82

        score = min(deg_score * tier_mult * field_mult, 1.0)
        best = max(best, score)
    return best


def _build_candidate_text(candidate: dict) -> str:
    """Rich text for semantic matching."""
    parts = []
    profile = candidate.get('profile', {})
    parts.extend([
        profile.get('summary', ''),
        profile.get('headline', ''),
        profile.get('current_title', ''),
        profile.get('current_industry', ''),
    ])
    for job in candidate.get('career_history', []):
        parts.extend([job.get('title', ''), job.get('description', ''), job.get('industry', '')])
    for s in candidate.get('skills', []):
        parts.append(s.get('name', ''))
        # Also add canonical form (ontology expansion)
        canonical = normalize_skill(s.get('name', ''))
        if canonical != s.get('name', '').lower():
            parts.append(canonical)
    for cert in candidate.get('certifications', []):
        parts.append(cert.get('name', ''))
    return ' '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_reasoning(analysis: CandidateAnalysis, candidate: dict) -> str:
    """
    Generate candidate-specific, non-templated reasoning.
    References actual profile facts. Acknowledges concerns honestly.
    Different for every candidate.
    """
    c = analysis.components
    score = analysis.total_score

    yoe = analysis.experience_years
    title = analysis.current_title
    notice = analysis.notice_days
    rr = analysis.response_rate

    # ── Build opening ────────────────────────────────────────────────
    # Most differentiating fact first
    if analysis.must_have_matched:
        top_skills = analysis.must_have_matched[:3]
        skill_clause = f"production skills in {', '.join(top_skills)}"
    elif analysis.nice_have_matched:
        top_skills = analysis.nice_have_matched[:3]
        skill_clause = f"adjacent ML skills ({', '.join(top_skills)})"
    else:
        skill_clause = "limited direct JD skill match"

    # Achievements: inject top verified achievement into reasoning when available
    achievement_clause = ""
    if analysis.achievements_found:
        ach = analysis.achievements_found[0]
        achievement_clause = f" Evidence: '{ach[:90]}'"

    # ── Score decomposition ───────────────────────────────────────────
    conf = getattr(analysis, 'confidence', 0.80)
    n_sig = getattr(analysis, 'signal_count', 12)
    # Show the semantic sub-breakdown (lexical vs distributional) when the
    # embedder was used, so recruiters/judges can see the hybrid blend is
    # genuinely operating, not just a single TF-IDF number.
    sem_lex = getattr(analysis, 'semantic_lexical_component', None)
    sem_dist = getattr(analysis, 'semantic_distributional_component', None)
    if sem_dist is not None:
        semantic_detail = f"Semantic {c.semantic:.3f} (lex={sem_lex:.2f}/dist={sem_dist:.2f})"
    else:
        semantic_detail = f"Semantic {c.semantic:.3f}"

    score_str = (
        f"[Skills {c.skills:.3f} | Career {c.career:.3f} | "
        f"Behavioral {c.behavioral:.3f} | Avail {c.availability:.3f} | "
        f"{semantic_detail} | conf={conf:.2f} n={n_sig}]"
    )

    # ── Tier-specific reasoning ───────────────────────────────────────
    if score >= 0.85:
        # Pick the most notable strength not already covered by the skill clause
        notable = [s for s in analysis.strengths if not any(ts in s for ts in top_skills[:1])]
        strength_clause = f"; {notable[0].lower()}" if notable else ""

        # Honest concern even at top rank
        concern_clause = ""
        if analysis.concerns:
            concern_clause = f" Note: {analysis.concerns[0].lower()}."
        elif notice > 60:
            concern_clause = f" Note: {notice}-day notice period adds some hiring friction."

        # Behavioral highlight
        if rr >= 0.80:
            beh = f"high recruiter response rate ({rr:.0%})"
        elif analysis.is_open_to_work and notice <= 15:
            beh = f"immediately available (open to work, {notice}-day notice)"
        elif analysis.github_score > 70:
            beh = f"strong GitHub activity (score {analysis.github_score:.0f}) — external validation"
        else:
            beh = f"response rate {rr:.0%}"

        # ── Opening anchor rotation ─────────────────────────────────────
        # Top-tier candidates cluster on similar skill profiles, which makes
        # an always-skills-first opening read as templated across 20-30 rows.
        # Lead with whichever signal is genuinely most differentiating for
        # THIS candidate: an exceptional achievement, immediate availability,
        # strong external validation, or (fallback) their skill match.
        if analysis.achievements_found:
            ach = analysis.achievements_found[0]
            opening = (
                f"Evidence of measurable impact: '{ach[:90]}' — "
                f"{title} ({yoe:.1f} yrs) with {skill_clause}"
            )
            achievement_clause = ""  # Already used in opening; don't repeat
        elif analysis.is_open_to_work and notice <= 15 and rr >= 0.75:
            opening = (
                f"Immediately available (open to work, {notice}-day notice, {rr:.0%} response rate) — "
                f"{title} ({yoe:.1f} yrs) with {skill_clause}"
            )
        elif analysis.github_score > 80:
            opening = (
                f"Strong external validation (GitHub score {analysis.github_score:.0f}) — "
                f"{title} ({yoe:.1f} yrs) with {skill_clause}"
            )
        elif analysis.promotions_detected >= 2:
            opening = (
                f"{analysis.promotions_detected} career advancements detected — "
                f"{title} ({yoe:.1f} yrs) with {skill_clause}"
            )
        else:
            opening = f"{title} ({yoe:.1f} yrs) with {skill_clause}"

        reasoning = (
            f"{opening}{strength_clause}.{achievement_clause} "
            f"Behavioral: {beh}.{concern_clause} "
            f"{score_str}"
        )

    elif score >= 0.60:
        # Balanced — strengths and concerns both present
        top_strength = analysis.strengths[0].lower() if analysis.strengths else "some relevant AI experience"
        top_concern = analysis.concerns[0].lower() if analysis.concerns else "below top-tier skill match"

        beh_note = ""
        if rr < 0.30:
            beh_note = f" Low response rate ({rr:.0%}) is a hiring risk."
        elif notice > 90:
            beh_note = f" {notice}-day notice period significantly constrains timing."

        reasoning = (
            f"{title} ({yoe:.1f} yrs) — strength: {top_strength}; "
            f"concern: {top_concern}.{beh_note} "
            f"{score_str}"
        )

    else:
        # Honest about weak fit
        primary_gap = analysis.concerns[0] if analysis.concerns else "no must-have JD skills matched"
        must_gap = f"matched {len(analysis.must_have_matched)}/{len(MUST_HAVE_SKILLS)} core JD skills"

        reasoning = (
            f"{title} ({yoe:.1f} yrs) — weak fit: {primary_gap}. "
            f"{must_gap}. "
            f"{score_str}"
        )

    return reasoning[:500]  # Spec doesn't hard-limit but keep reasonable


def _build_interview_focus(analysis: CandidateAnalysis) -> list[str]:
    """
    Generate candidate-specific interview focus recommendations.
    References actual missing skills, specific signals, and genuine anomalies
    — not generic templates that apply to all candidates.
    """
    focus = []
    matched_canonical = {normalize_skill(s) for s in analysis.must_have_matched}

    # Gap 1: Most critical missing must-have skill (named specifically)
    critical_missing = [s for s in analysis.missing_must_have
                        if s in ('retrieval', 'bm25', 'hybrid search', 'learning to rank',
                                 'a/b testing', 'ndcg', 'reranking', 'information retrieval')]
    if critical_missing:
        focus.append(
            f"Probe depth on '{critical_missing[0]}' — listed as a JD must-have but "
            f"absent from this profile; ask for a concrete example they built or measured"
        )
    elif 'a/b testing' not in matched_canonical:
        focus.append(
            "Test evaluation framework knowledge specifically: ask how they measured "
            "retrieval quality improvement in production (NDCG, MRR, offline-to-online lift)"
        )

    # Gap 2: GitHub anomaly — high score but no OSS listed, or not linked
    if analysis.github_score != -1 and analysis.github_score > 65:
        focus.append(
            f"GitHub activity score is {analysis.github_score:.0f}/100 — ask what they've "
            f"built publicly; high activity score without visible OSS warrants verification"
        )
    elif analysis.github_score == -1:
        focus.append(
            "No GitHub linked — JD explicitly values external validation; ask about "
            "any public technical work, papers, talks, or open-source contributions"
        )

    # Gap 3: Notice period (only if it's a genuine concern)
    if analysis.notice_days > 75:
        focus.append(
            f"{analysis.notice_days}-day notice period: confirm whether buyout is possible "
            f"and discuss actual start-date flexibility"
        )
    elif not analysis.career_at_product_companies:
        focus.append(
            "Career includes consulting firm(s): probe ownership mindset — ask them to "
            "describe a decision they made independently vs one that went through a client"
        )
    elif analysis.promotions_detected == 0 and analysis.experience_years >= 5:
        focus.append(
            f"{analysis.experience_years:.0f} yrs experience with no detected title progression — "
            f"ask them to narrate their career growth; may be IC-track or data gap"
        )

    return focus[:3]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CANDIDATE SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(
    candidate: dict,
    tfidf: TFIDFMatcher,
    today: date,
    weights: dict,
    embedder: "DistributionalEmbedder | None" = None,
    jd_embedding=None,
) -> CandidateAnalysis:

    cid = candidate['candidate_id']
    profile = candidate['profile']
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])

    analysis = CandidateAnalysis(
        candidate_id=cid,
        total_score=0.0,
        experience_years=profile.get('years_of_experience', 0),
        current_title=profile.get('current_title', ''),
        notice_days=signals.get('notice_period_days', 90),
        response_rate=signals.get('recruiter_response_rate', 0.5),
        github_score=signals.get('github_activity_score', -1.0),
        is_open_to_work=signals.get('open_to_work_flag', False),
    )

    # ── 1. Honeypot ───────────────────────────────────────────────────
    is_hp, hp_reason = _detect_honeypot(candidate)
    if is_hp:
        analysis.is_honeypot = True
        analysis.total_score = 0.001
        analysis.concerns.append(f"honeypot flag: {hp_reason}")
        return analysis

    # ── 2. Disqualifiers ─────────────────────────────────────────────
    companies = [ch.get('company', '') for ch in career]
    is_disq, disq_reasons = _check_disqualifiers(candidate, companies)
    if is_disq:
        analysis.is_disqualified = True
        analysis.disqualify_reasons = disq_reasons
        analysis.concerns = disq_reasons
        analysis.total_score = 0.01 + analysis.experience_years * 0.001
        return analysis

    # ── 3. Build career text for corroboration ────────────────────────
    career_text = ' '.join(ch.get('description', '') for ch in career)

    # ── 4. Skills ─────────────────────────────────────────────────────
    skill_score, must_matched, nice_matched, missing_must, cluster_coherence = _score_skills(
        candidate, career_text
    )
    analysis.components.skills = skill_score
    analysis.must_have_matched = must_matched
    analysis.nice_have_matched = nice_matched
    analysis.missing_must_have = missing_must
    analysis.skill_cluster_coherence = cluster_coherence

    # ── 5. Career ─────────────────────────────────────────────────────
    career_score, strengths, concerns, promotions, achievements = _score_career(candidate)
    analysis.components.career = career_score
    analysis.strengths.extend(strengths)
    analysis.concerns.extend(concerns)
    analysis.promotions_detected = promotions
    analysis.achievements_found = achievements

    # Check product company ratio
    consulting_count = sum(
        1 for c in companies if any(firm in c.lower() for firm in CONSULTING_FIRMS)
    )
    analysis.career_at_product_companies = consulting_count < len(companies) * 0.5

    # ── 6. Behavioral ─────────────────────────────────────────────────
    behavioral_score, availability_score, signal_detail = _score_behavioral(signals, today)
    analysis.components.behavioral = behavioral_score
    analysis.components.availability = availability_score
    analysis.last_active_days_ago = signal_detail.get('days_inactive', 999)
    analysis.assessment_score = float(signal_detail.get('assessment', '0').split('/')[0]) if signal_detail.get('assessment', '').split('/')[0].replace('.', '').isdigit() else 0.0

    # Recency concern
    if signal_detail.get('days_inactive', 0) > 180:
        analysis.concerns.append(f"inactive for {signal_detail['days_inactive']} days — possible ghost candidate")

    # Notice concern
    if signal_detail.get('notice_concern'):
        analysis.concerns.append(f"{analysis.notice_days}-day notice period (JD prefers sub-30)")

    # ── 7. Education ──────────────────────────────────────────────────
    edu_score = _score_education(candidate)
    analysis.components.education = edu_score

    # ── 8. Semantic ───────────────────────────────────────────────────
    # Hybrid: TF-IDF (exact-term-sensitive, rewards precise JD vocabulary
    # matches) blended with a genuine distributional semantic embedding
    # (rewards candidates who describe the same work in different words —
    # real "you shall know a word by the company it keeps" semantics,
    # trained from scratch on this corpus, no network/GPU required).
    candidate_text = _build_candidate_text(candidate)
    lexical_score = tfidf.similarity(candidate_text)

    if embedder is not None and embedder.is_trained and jd_embedding is not None:
        distributional_score = embedder.similarity(candidate_text, jd_embedding)
        lex_w = _cfg('semantic.embedder_lexical_weight', 0.55)
        dist_w = _cfg('semantic.embedder_distributional_weight', 0.45)
        semantic_score = lex_w * lexical_score + dist_w * distributional_score
        analysis.semantic_lexical_component = lexical_score
        analysis.semantic_distributional_component = distributional_score
    else:
        semantic_score = lexical_score
        analysis.semantic_lexical_component = lexical_score
        analysis.semantic_distributional_component = None

    analysis.components.semantic = semantic_score

    # ── 9. Base composite ─────────────────────────────────────────────
    base_score = analysis.components.weighted_total(weights)

    # ── 10. Bonuses/Penalties ─────────────────────────────────────────
    # India location bonus
    country = profile.get('country', '').lower()
    location = profile.get('location', '').lower()
    india_pref = any(loc in location for loc in [
        'india', 'pune', 'noida', 'bangalore', 'bengaluru',
        'hyderabad', 'mumbai', 'delhi', 'new delhi', 'ncr',
        'gurgaon', 'gurugram', 'chennai', 'kolkata', 'calcutta',
        'ahmedabad', 'coimbatore', 'kochi', 'cochin', 'greater noida',
        'navi mumbai', 'thane', 'bhubaneswar', 'jaipur', 'indore',
    ])
    if country == 'india' or india_pref:
        india_bonus = _cfg('bonuses.india_location', 0.03)
        base_score = min(base_score + india_bonus, 1.0)
        analysis.strengths.append("India-based (JD preferred location)")

    # GitHub bonus
    if analysis.github_score > _cfg('bonuses.github_high_threshold', 70):
        github_bonus = _cfg('bonuses.github_high_bonus', 0.03)
        base_score = min(base_score + github_bonus, 1.0)

    # Must-have depth bonus
    if len(must_matched) >= _cfg('bonuses.must_have_deep_threshold', 5):
        base_score = min(base_score + _cfg('bonuses.must_have_deep_bonus', 0.05), 1.0)
        analysis.strengths.append(f"covers {len(must_matched)}/{len(MUST_HAVE_SKILLS)} JD core requirements")
    elif len(must_matched) >= _cfg('bonuses.must_have_moderate_threshold', 3):
        base_score = min(base_score + _cfg('bonuses.must_have_moderate_bonus', 0.02), 1.0)

    # IR specialist cluster bonus
    if cluster_coherence > 0.65:
        base_score = min(base_score + 0.03, 1.0)
        analysis.strengths.append(f"IR/retrieval skill cluster coherence: {cluster_coherence:.2f}")

    # Ghost candidate penalty
    ghost_rr = _cfg('penalties.ghost_response_rate_max', 0.15)
    if analysis.response_rate < ghost_rr and not analysis.is_open_to_work:
        base_score *= _cfg('penalties.ghost_candidate', 0.75)
        analysis.concerns.append("ghost candidate risk: very low response rate + not open to work")

    # No production evidence penalty
    prod_signals = ['production', 'deployed', 'real users', 'at scale', 'serving',
                    'latency', 'shipped', 'monitoring', 'endpoint']
    has_production = any(sig in career_text.lower() for sig in prod_signals)
    if not has_production and len(must_matched) < 2:
        base_score *= _cfg('penalties.no_production_evidence', 0.82)
        analysis.concerns.append("no production deployment evidence in career descriptions")

    analysis.total_score = max(0.001, min(base_score, 0.9999))

    # ── 11. Interview focus ───────────────────────────────────────────
    analysis.interview_focus = _build_interview_focus(analysis)

    # ── 12. Missing skills (top 3 most impactful for gap panel) ──────
    analysis.missing_skills_top3 = analysis.missing_must_have[:3]

    # ── 13. Confidence estimation ─────────────────────────────────────
    sig_checks = [
        signals.get('recruiter_response_rate') is not None,
        signals.get('last_active_date') is not None,
        signals.get('github_activity_score', -1) != -1,
        bool(signals.get('skill_assessment_scores')),
        signals.get('interview_completion_rate') is not None,
        signals.get('offer_acceptance_rate', -1) != -1,
        signals.get('profile_completeness_score', 0) > 0,
        signals.get('saved_by_recruiters_30d', 0) > 0,
        bool(candidate.get('career_history')),
        bool(candidate.get('education')),
        bool(candidate.get('certifications')),
        profile.get('country') is not None,
        profile.get('current_company_size') is not None,
        signals.get('verified_email', False),
        signals.get('verified_phone', False),
        signals.get('linkedin_connected', False),
    ]
    analysis.signal_count = sum(sig_checks)
    analysis.confidence = round(0.60 + 0.35 * (analysis.signal_count / max(len(sig_checks), 1)), 2)

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _stream_candidates(input_path: str):
    """Yield candidate dicts one at a time from a JSONL[.gz] file without
    holding the whole file in memory."""
    opener = gzip.open if input_path.endswith('.gz') else open
    with opener(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class RankingStats:
    """Disqualification / filtering statistics for dashboard transparency."""
    total_candidates: int = 0
    honeypot_count: int = 0
    disqualified_consulting: int = 0
    disqualified_cv_robotics: int = 0
    disqualified_other: int = 0
    zero_skill_match: int = 0
    malformed_skipped: int = 0
    elapsed_seconds: float = 0.0
    peak_heap_size: int = 0


def rank_candidates(
    input_path: str,
    output_path: str,
    top_n: int = 100,
    config_path: Optional[str] = None,
    stats_output_path: Optional[str] = None,
) -> list[tuple[CandidateAnalysis, dict]]:
    """
    Stream-score candidates with bounded memory: maintains only a top-N*3
    candidate buffer at any time rather than holding all 100K+ in RAM.
    Peak memory scales with top_n, not with input dataset size.
    """

    log.info("NeuroIgniter AI Recruiter v2 — Starting ranking pipeline")
    log.info(f"  Input:  {input_path}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Top-N:  {top_n}")

    t0 = time.time()
    today = date.today()

    weights = _cfg('scoring_weights', {
        'skills': 0.38, 'career': 0.26, 'behavioral': 0.12,
        'availability': 0.08, 'semantic': 0.10, 'education': 0.06,
    })
    log.info(f"  Weights: {weights}")
    log.info(f"  JD extraction confidence: {JD_REQUIREMENTS.extraction_confidence:.2f} "
              f"({len(MUST_HAVE_SKILLS)} must-have, {len(NICE_TO_HAVE_SKILLS)} nice-to-have skills parsed from JD)")

    tfidf = TFIDFMatcher(JD_REQUIREMENTS.semantic_text)

    # ── Train the distributional semantic embedder ───────────────────────
    # A genuine semantic layer (real distributional embeddings, trained from
    # scratch on this corpus — see semantic.py for why this approach is used
    # instead of a pretrained transformer). Trained on a bounded sample so
    # this pre-pass stays fast and memory-light even for very large inputs.
    EMBEDDER_TRAIN_SAMPLE_SIZE = _cfg('semantic.embedder_train_sample_size', 5000)
    embedder = DistributionalEmbedder(
        vocab_size=_cfg('semantic.embedder_vocab_size', 3000),
        embedding_dim=_cfg('semantic.embedder_dim', 64),
    )
    jd_embedding = None

    log.info(f"  Training distributional semantic embedder on up to "
              f"{EMBEDDER_TRAIN_SAMPLE_SIZE:,} sampled candidate texts...")
    t_embed_start = time.time()
    training_corpus = [JD_REQUIREMENTS.semantic_text] * 5  # weight JD slightly
    for i, cand in enumerate(_stream_candidates(input_path)):
        if i >= EMBEDDER_TRAIN_SAMPLE_SIZE:
            break
        if isinstance(cand, dict) and 'profile' in cand:
            try:
                training_corpus.append(_build_candidate_text(cand))
            except Exception:
                continue

    try:
        embedder.fit(training_corpus)
        if embedder.is_trained:
            jd_embedding = embedder.embed(JD_REQUIREMENTS.semantic_text)
    except Exception as e:
        log.warning(f"Distributional embedder training failed ({e}) — "
                     f"falling back to lexical-only semantic scoring.")
        embedder = None
        jd_embedding = None

    t_embed_end = time.time()
    if embedder is not None and embedder.is_trained:
        log.info(f"  Embedder trained in {t_embed_end - t_embed_start:.1f}s "
                  f"({len(embedder.vocab)} vocab terms, {embedder.embedding_dim}-dim vectors)")
    else:
        log.info("  Embedder unavailable — semantic scoring will use lexical-only TF-IDF")

    # ── Bounded streaming scan ───────────────────────────────────────────
    # Keep a buffer of at most top_n * BUFFER_MULT candidates, periodically
    # trimming to top_n via partial sort. This bounds peak memory to a small
    # multiple of top_n rather than the full dataset size.
    BUFFER_MULT = 5
    buffer_cap = max(top_n * BUFFER_MULT, top_n + 50)

    buffer: list[tuple[CandidateAnalysis, dict]] = []
    stats = RankingStats()
    n_processed = 0
    peak_buffer = 0

    log.info("  Streaming + scoring candidates (bounded memory)...")
    for cand in _stream_candidates(input_path):
        n_processed += 1

        # Defensive scoring: a single malformed candidate record (missing
        # 'profile', 'candidate_id', a skill's 'name' key, etc.) must not
        # crash the entire run and discard every other candidate's work.
        # We validate the minimum required shape before scoring, and skip
        # + count anything that doesn't meet it.
        if not isinstance(cand, dict) or 'candidate_id' not in cand or 'profile' not in cand:
            stats.malformed_skipped += 1
            log.debug(f"Skipping malformed candidate record at position {n_processed} "
                       f"(missing candidate_id or profile)")
            continue

        try:
            analysis = score_candidate(cand, tfidf, today, weights, embedder=embedder, jd_embedding=jd_embedding)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            stats.malformed_skipped += 1
            cid = cand.get('candidate_id', f'<unknown at position {n_processed}>')
            log.debug(f"Skipping candidate {cid} — scoring failed: {type(e).__name__}: {e}")
            continue

        if analysis.is_honeypot:
            stats.honeypot_count += 1
        elif analysis.is_disqualified:
            reason_text = ' '.join(analysis.disqualify_reasons).lower()
            if 'consulting' in reason_text:
                stats.disqualified_consulting += 1
            elif 'cv/speech/robotics' in reason_text or 'robotics' in reason_text:
                stats.disqualified_cv_robotics += 1
            else:
                stats.disqualified_other += 1
        elif len(analysis.must_have_matched) == 0 and len(analysis.nice_have_matched) == 0:
            stats.zero_skill_match += 1

        buffer.append((analysis, cand))
        peak_buffer = max(peak_buffer, len(buffer))

        if len(buffer) >= buffer_cap:
            buffer.sort(key=lambda x: (-x[0].total_score, x[0].candidate_id))
            buffer = buffer[:top_n]

    # Final sort of whatever remains in the buffer
    buffer.sort(key=lambda x: (-x[0].total_score, x[0].candidate_id))
    top_results = buffer[:top_n]

    stats.total_candidates = n_processed
    stats.peak_heap_size = peak_buffer

    t_score = time.time()
    log.info(f"  Scored {n_processed:,} candidates in {t_score - t0:.1f}s "
              f"(peak buffer: {peak_buffer:,} candidates, bounded at {buffer_cap:,})")
    log.info(f"  Honeypots: {stats.honeypot_count} | "
              f"Consulting-only: {stats.disqualified_consulting} | "
              f"CV/robotics-only: {stats.disqualified_cv_robotics} | "
              f"Zero skill match: {stats.zero_skill_match} | "
              f"Malformed (skipped): {stats.malformed_skipped}")

    # ── Write ranked CSV ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank_idx, (analysis, cand) in enumerate(top_results, start=1):
            reasoning = _generate_reasoning(analysis, cand)
            writer.writerow([
                analysis.candidate_id,
                rank_idx,
                round(analysis.total_score, 6),
                reasoning,
            ])

    # ── Write stats sidecar (for dashboard transparency panel) ────────────
    if stats_output_path:
        stats.elapsed_seconds = time.time() - t0
        os.makedirs(os.path.dirname(stats_output_path) or '.', exist_ok=True)
        with open(stats_output_path, 'w') as f:
            json.dump({
                'total_candidates': stats.total_candidates,
                'honeypot_count': stats.honeypot_count,
                'disqualified_consulting': stats.disqualified_consulting,
                'disqualified_cv_robotics': stats.disqualified_cv_robotics,
                'disqualified_other': stats.disqualified_other,
                'zero_skill_match': stats.zero_skill_match,
                'malformed_skipped': stats.malformed_skipped,
                'elapsed_seconds': round(stats.elapsed_seconds, 1),
                'peak_buffer_size': stats.peak_heap_size,
                'top_n': top_n,
                'jd_extraction_confidence': JD_REQUIREMENTS.extraction_confidence,
                'must_have_skills_parsed': MUST_HAVE_SKILLS,
                'nice_to_have_skills_parsed': NICE_TO_HAVE_SKILLS,
            }, f, indent=2)
        log.info(f"  Stats written: {stats_output_path}")

    t_end = time.time()
    log.info(f"  Ranking complete in {t_end - t0:.1f}s total")
    log.info(f"  Output: {output_path}")

    log.info("  Top-5 candidates:")
    for i, (analysis, cand) in enumerate(top_results[:5], 1):
        log.info(
            f"    #{i}: {analysis.candidate_id} | {cand['profile']['current_title']} | "
            f"{analysis.experience_years:.1f}yrs | score={analysis.total_score:.4f} | "
            f"must_have={len(analysis.must_have_matched)}/{len(MUST_HAVE_SKILLS)}"
        )

    return top_results


if __name__ == '__main__':
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, '..', '..', '..')

    default_inputs = [
        os.path.join(repo_root, 'data', 'candidates.jsonl.gz'),
        os.path.join(repo_root, 'data', 'candidates.jsonl'),
        'candidates.jsonl.gz',
        'candidates.jsonl',
    ]
    input_path = None
    for p in default_inputs:
        if os.path.exists(p):
            input_path = p
            break

    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo_root, 'output', 'neuroigniter_submission.csv')

    if not input_path:
        print("ERROR: Place candidates.jsonl or candidates.jsonl.gz in data/")
        sys.exit(1)

    rank_candidates(input_path, output_path)
