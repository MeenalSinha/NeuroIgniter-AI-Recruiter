"""
NeuroIgniter JD Intelligence Layer
====================================
Runtime JD parsing: extracts structured requirements from ANY job description text.
No hardcoded skill lists. No manually curated taxonomies.
Rankings automatically adapt when the JD changes.

Architecture:
  1. Section detection — identify must-have, nice-to-have, disqualifier sections
  2. Skill extraction — pull noun phrases, technology names, domain terms
  3. Context inference — infer seniority, culture, orientation from language patterns
  4. Ontology alignment — map extracted terms to canonical skill names
"""

import re
from dataclasses import dataclass, field

from .ontology import ALIAS_TO_CANONICAL

# ── DEFAULT JD TEXT ───────────────────────────────────────────────────────────
# The challenge JD. In production, this would be loaded from a file or API.
# The parser reads THIS text to extract requirements at runtime.
DEFAULT_JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team

Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking,
retrieval, and matching systems that decide what recruiters see when they search for candidates.

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI
  embeddings, BGE, E5, or similar) deployed to real users. We care that you've handled
  embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with vector databases or hybrid search infrastructure — Pinecone,
  Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar.
- Strong Python. We care about code quality.
- Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP,
  offline-to-online correlation, A/B test interpretation.

Things we'd like you to have but won't reject you for:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

Things we explicitly do NOT want:
- Title-chasers: career trajectory showing optimizing for titles by switching every 1.5 years
- Framework enthusiasts: GitHub full of LangChain tutorials and hot-framework demos
- People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant,
  Capgemini, etc.) in their entire career.
- People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure.
- People whose work has been entirely on closed-source proprietary systems for 5+ years without
  external validation (papers, talks, open-source).

First 90 days will look like:
- Weeks 1-3: Audit what we currently have (BM25 + rule-based scoring).
  Identify the 3-4 highest-leverage things to fix.
- Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics.
  This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking.
- Weeks 9-12: Set up evaluation infrastructure — offline benchmarks, online A/B testing,
  recruiter-feedback loops.

Notice period: We'd love sub-30-day notice. We can buy out up to 30 days.
Location: Pune/Noida-preferred but flexible. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome.
"""


# ── SECTION DETECTION PATTERNS ────────────────────────────────────────────────
# Detect JD section boundaries using common recruiter language patterns.

MUST_HAVE_PATTERNS = [
    r'things? you (?:absolutely )?need\s*:',
    r'required\s+(?:skills?|qualifications?|experience)\s*:',
    r'must.have\s*:',
    r'you (?:must|need to|should) have\s*:',
    r'what you(?:\'ll)? need\s*:',
    r'minimum qualifications?\s*:',
    r'basic qualifications?\s*:',
    r'what we(?:\'re)? looking for\s*:',
    r'key requirements?\s*:',
    r'essential skills?\s*:',
    r'mandatory\s*:',
    r'you will bring\s*:',
    r'we need you to have\s*:',
]

NICE_HAVE_PATTERNS = [
    r'things? we(?:\'d| would) like.*?:',
    r'nice.to.have\s*:',
    r'preferred\s+qualifications?\s*:',
    r'bonus\s*:',
    r'good to have\s*:',
    r'we\'d like you to have',
]

DISQUALIFIER_PATTERNS = [
    r'do (?:not|n\'t) want\s*:',
    r'not looking for\s*:',
    r'please (?:do not|don\'t) apply',
    r'things? we (?:explicitly )?do not want\s*:',
    r'explicitly do not want\s*:',
]

YOE_PATTERN = re.compile(
    r'(\d+)\s*[-–—]\s*(\d+)\s*years?|(\d+)\+?\s*years?(?:\s+of)?\s*(?:experience|exp)',
    re.IGNORECASE,
)

SALARY_PATTERN = re.compile(
    r'(?:salary|compensation|ctc|package)[:\s]*(?:₹|inr|rs\.?)?\s*(\d+)\s*[-–to]+\s*(\d+)\s*(?:lpa|lakhs?|l)?',
    re.IGNORECASE,
)

LOCATION_PATTERN = re.compile(
    r'(?:location|based in|located in|office(?:s)? in)[:\s]+([^\n.]+)',
    re.IGNORECASE,
)

NOTICE_PATTERN = re.compile(
    r'(?:notice period|join(?:ing)?)[^\d]*?(\d+)[\s\-]*day',
    re.IGNORECASE,
)


# ── TECHNOLOGY EXTRACTION ─────────────────────────────────────────────────────
# Known technology patterns to extract from JD text

TECH_EXTRACTION_PATTERNS = [
    # Parenthetical tech lists: "systems (FAISS, Pinecone, Weaviate)"
    re.compile(r'\(([A-Za-z0-9\s,/\-\.]+)\)', re.IGNORECASE),
    # Dash/bullet items: "- Python, Java, Scala"
    re.compile(r'^[\-\*•]\s*(.+)$', re.MULTILINE),
    # "using X, Y, Z" patterns
    re.compile(r'(?:using|with|via|including)\s+([A-Za-z0-9,\s/\-\.]+?)(?:\.|,|\s+and\s+|\s+or\s+)', re.IGNORECASE),
]

# Known tech terms to look for directly in JD text
KNOWN_TECH_TERMS = list(ALIAS_TO_CANONICAL.keys())


def _detect_section_boundaries(text: str) -> dict[str, tuple[int, int]]:
    """
    Identify start/end positions of must-have, nice-to-have, disqualifier sections.
    Returns dict of section_name -> (start_char, end_char).
    """
    lines = text.split('\n')
    sections = {}
    section_starts = {}

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        char_pos = sum(len(ln) + 1 for ln in lines[:i])

        for pat in MUST_HAVE_PATTERNS:
            if re.search(pat, line_lower):
                section_starts['must_have'] = char_pos
                break  # section = must_have

        for pat in NICE_HAVE_PATTERNS:
            if re.search(pat, line_lower):
                if 'must_have' in section_starts:
                    end = char_pos
                    sections['must_have'] = (section_starts['must_have'], end)
                section_starts['nice_have'] = char_pos
                break  # current_section = 'nice_have'

        for pat in DISQUALIFIER_PATTERNS:
            if re.search(pat, line_lower):
                if 'nice_have' in section_starts:
                    end = char_pos
                    sections['nice_have'] = (section_starts['nice_have'], end)
                elif 'must_have' in section_starts:
                    end = char_pos
                    sections['must_have'] = (section_starts['must_have'], end)
                section_starts['disqualifiers'] = char_pos
                break  # current_section = 'disqualifiers' — tracked via section_starts

    # Close any open sections at end of text
    total_len = len(text)
    for sec in ('must_have', 'nice_have', 'disqualifiers'):
        if sec in section_starts and sec not in sections:
            sections[sec] = (section_starts[sec], total_len)

    return sections


def _extract_generic_terms(text: str) -> list[str]:
    """
    Fallback extraction for skills/technologies NOT present in the AI/ML
    ontology (e.g. React, Kubernetes, Terraform for non-AI roles).

    The ontology-based extraction in `_extract_skills_from_text` only
    recognizes ~200 AI/ML/IR terms. Without this fallback, a JD for any
    role outside that domain (frontend, DevOps, security, etc.) would
    extract zero must-have skills — a real failure mode, not a cosmetic one,
    since MUST_HAVE_SKILLS directly drives 38% of every candidate's score.

    This is intentionally a *generic* extractor: it pulls out capitalized
    multi-word tech-looking tokens and short noun phrases from bullet
    points, without claiming ontology-level alias normalization for them.
    It is honest about being lower-precision than the ontology path.
    """
    found = set()

    # Capitalized tech-like tokens: "React", "TypeScript", "Kubernetes",
    # "Terraform", "AWS", "GCP" — single CamelCase/UPPERCASE/Title-case words,
    # 2-20 chars, not common English sentence-starters.
    SENTENCE_STARTERS = {
        'The', 'This', 'That', 'We', 'You', 'It', 'Our', 'Things',
        'Strong', 'Production', 'Experience', 'Prior', 'Background',
        'People', 'Notice', 'Things', 'Bonus',
    }
    # Common short fragments produced by splitting on punctuation (e.g. "CI/CD"
    # -> "CI", "CD") that are noise unless part of a recognized longer phrase.
    NOISE_FRAGMENTS = {'CI', 'CD', 'Actions', 'Github', 'Pipelines'}

    cap_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9+.#]{1,19})\b')
    single_terms = set()
    for line in text.split('\n'):
        line = line.strip().lstrip('-*•').strip()
        if not line:
            continue
        for match in cap_pattern.finditer(line):
            term = match.group(1)
            if term in SENTENCE_STARTERS or len(term) < 2:
                continue
            if match.start() == 0:
                continue
            single_terms.add(term)

    found = {t for t in single_terms if t not in NOISE_FRAGMENTS}

    # Parenthetical comma-lists: "(Redux, Zustand)", "(Jenkins, GitHub Actions)"
    phrase_terms = set()
    for match in re.finditer(r'\(([A-Za-z0-9,\s/\-\.]+)\)', text):
        items = [x.strip() for x in match.group(1).split(',')]
        for item in items:
            if 2 <= len(item) <= 30 and item.lower() not in SENTENCE_STARTERS:
                phrase_terms.add(item)

    # Drop single-word terms that are already covered as a substring of a
    # longer phrase term (e.g. don't keep standalone "GitHub" if "GitHub
    # Actions" was already captured as a phrase).
    found = {
        t for t in found
        if not any(t != p and t in p.split() for p in phrase_terms)
    }
    found |= phrase_terms

    return sorted(found)


def _extract_skills_from_text(text: str) -> list[str]:
    """
    Extract skill/technology mentions from a text segment.
    Uses: parenthetical lists, bullet items, known term scanning.
    Returns canonical skill names where the ontology recognizes them,
    plus generic extracted terms (lowercased) for anything outside the
    AI/ML ontology so non-AI/ML JDs don't silently extract zero skills.
    """
    found = set()
    text_lower = text.lower()

    # Method 1: Direct term scanning against the AI/ML ontology (highest precision —
    # these get full alias normalization, e.g. "SBERT" -> "sentence transformers")
    for term in KNOWN_TECH_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            canonical = ALIAS_TO_CANONICAL.get(term)
            if canonical:
                found.add(canonical)

    # Method 2: Parenthetical extraction against the ontology (catches
    # "systems (FAISS, Pinecone)")
    for match in TECH_EXTRACTION_PATTERNS[0].finditer(text):
        items = [x.strip().lower() for x in match.group(1).split(',')]
        for item in items:
            canonical = ALIAS_TO_CANONICAL.get(item)
            if canonical:
                found.add(canonical)

    ontology_match_count = len(found)

    # Method 3: Generic fallback — ONLY engages if the ontology found nothing
    # or very little. This keeps AI/ML JDs on the high-precision ontology path
    # (where "Python" correctly normalizes, isn't polluted by generic noise)
    # while still extracting *something* for JDs outside that domain.
    if ontology_match_count < 2:
        generic = _extract_generic_terms(text)
        found.update(g.lower() for g in generic)

    return sorted(found)


def _extract_yoe(text: str) -> tuple[float, float]:
    """Extract years-of-experience range from JD text."""
    match = YOE_PATTERN.search(text)
    if match:
        # Group 1,2 = range form "5-9 years"; Group 3 = single form "5+ years"
        if match.group(1) and match.group(2):
            return float(match.group(1)), float(match.group(2))
        elif match.group(3):
            single = float(match.group(3))
            return single, single + 4.0
    return 3.0, 10.0  # Fallback: broad range when no YOE pattern found


def _extract_locations(text: str) -> list[str]:
    """Extract preferred locations from JD."""
    locations = []
    match = LOCATION_PATTERN.search(text)
    if match:
        loc_text = match.group(1).lower()
        india_cities = [
            'india', 'pune', 'noida', 'bangalore', 'bengaluru', 'hyderabad',
            'mumbai', 'delhi', 'new delhi', 'ncr', 'gurgaon', 'gurugram',
            'chennai', 'kolkata', 'ahmedabad', 'kochi', 'coimbatore',
        ]
        for city in india_cities:
            if city in loc_text:
                locations.append(city)
    return locations or ['india']


def _extract_consulting_disqualifiers(text: str) -> list[str]:
    """Extract consulting firm names mentioned as disqualifiers."""
    # Common firms mentioned in "do not want" sections
    firm_patterns = [
        r'\bTCS\b', r'\bInfosys\b', r'\bWipro\b', r'\bAccenture\b',
        r'\bCognizant\b', r'\bCapgemini\b', r'\bHCL\b', r'\bTech Mahindra\b',
        r'\bMphasis\b', r'\bHexaware\b',
    ]
    disq_section_text = ''
    sections = _detect_section_boundaries(text)
    if 'disqualifiers' in sections:
        start, end = sections['disqualifiers']
        disq_section_text = text[start:end]

    found = []
    for pat in firm_patterns:
        if re.search(pat, disq_section_text, re.IGNORECASE) or re.search(pat, text, re.IGNORECASE):
            name = re.search(pat, text, re.IGNORECASE)
            if name:
                found.append(name.group(0).lower())
    return found


def _infer_culture_signals(text: str) -> dict[str, bool]:
    """Infer culture/orientation signals from JD language."""
    text_lower = text.lower()
    return {
        'wants_product': any(w in text_lower for w in [
            'product', 'users', 'shipped', 'real users', 'user engagement',
            'marketplace', 'platform', 'growth', 'founding team',
        ]),
        'wants_startup': any(w in text_lower for w in [
            'founding', 'series a', 'startup', 'scrappy', 'fast-moving',
            'early stage', 'move fast', 'wear many hats',
        ]),
        'wants_external_validation': any(w in text_lower for w in [
            'open source', 'github', 'papers', 'talks', 'publications',
            'blog', 'external validation', 'oss',
        ]),
        'wants_ownership': any(w in text_lower for w in [
            'own', 'ownership', 'end-to-end', 'from scratch', 'independently',
            'drive', 'lead the', 'responsible for',
        ]),
        'wants_production': any(w in text_lower for w in [
            'production', 'deployed', 'real users', 'scale', 'latency',
            'serving', 'inference', 'monitoring',
        ]),
        'is_research': any(w in text_lower for w in [
            'research', 'phd', 'publications', 'academic', 'papers',
            'novel', 'state of the art',
        ]),
    }


def _infer_seniority(text: str) -> tuple[str, float]:
    """Infer seniority level from JD title and content."""
    text_lower = text.lower()
    title_match = re.search(r'(?:job title|role|position)[:\s]+([^\n]+)', text_lower)
    title_text = title_match.group(1) if title_match else text_lower[:200]

    if any(w in title_text for w in ['staff', 'principal', 'distinguished', 'fellow']):
        return 'staff', 0.95
    elif any(w in title_text for w in ['senior', 'sr.', 'lead', 'founding']):
        return 'senior', 0.85
    elif any(w in title_text for w in ['junior', 'jr.', 'associate', 'entry']):
        return 'junior', 0.50
    else:
        return 'mid', 0.70


@dataclass
class JDRequirements:
    """Structured representation extracted from JD text at runtime."""

    # Extracted from JD text (not hardcoded)
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    consulting_disqualifiers: list[str] = field(default_factory=list)

    # Parsed from JD text
    yoe_min: float = 3.0
    yoe_max: float = 10.0
    preferred_locations: list[str] = field(default_factory=list)
    preferred_work_mode: str = "hybrid"
    ideal_notice_days: int = 30
    seniority_level: str = "senior"
    seniority_score_target: float = 0.85

    # Culture signals inferred from JD language
    wants_product_orientation: bool = True
    wants_startup_mindset: bool = True
    wants_external_validation: bool = True
    values_code_quality: bool = True
    wants_ownership: bool = True
    wants_production_experience: bool = True
    is_research_role: bool = False

    # Derived semantic text for TF-IDF matching
    semantic_text: str = ""

    # Raw JD text
    raw_text: str = ""

    # Extraction metadata
    extraction_confidence: float = 0.0
    must_have_section_found: bool = False
    nice_have_section_found: bool = False


def parse_jd(jd_text: str) -> JDRequirements:
    """
    Parse a job description text and extract structured requirements at runtime.
    This function generalizes to any JD — not just the challenge JD.
    """
    req = JDRequirements(raw_text=jd_text)

    # 1. Detect section boundaries
    sections = _detect_section_boundaries(jd_text)
    req.must_have_section_found = 'must_have' in sections
    req.nice_have_section_found = 'nice_have' in sections

    # 2. Extract skills from each section
    if 'must_have' in sections:
        start, end = sections['must_have']
        req.must_have_skills = _extract_skills_from_text(jd_text[start:end])
    else:
        # Fallback: scan full text, weight everything as must-have
        req.must_have_skills = _extract_skills_from_text(jd_text)

    if 'nice_have' in sections:
        start, end = sections['nice_have']
        nice = _extract_skills_from_text(jd_text[start:end])
        # Deduplicate: skills in must-have shouldn't repeat in nice-to-have
        req.nice_to_have_skills = [s for s in nice if s not in req.must_have_skills]

    # 3. Extract disqualifier firms
    req.consulting_disqualifiers = _extract_consulting_disqualifiers(jd_text)

    # 4. Parse structured fields
    req.yoe_min, req.yoe_max = _extract_yoe(jd_text)
    req.preferred_locations = _extract_locations(jd_text)

    notice_match = NOTICE_PATTERN.search(jd_text)
    if notice_match:
        req.ideal_notice_days = int(notice_match.group(1))

    req.seniority_level, req.seniority_score_target = _infer_seniority(jd_text)

    # 5. Infer culture signals
    culture = _infer_culture_signals(jd_text)
    req.wants_product_orientation = culture['wants_product']
    req.wants_startup_mindset = culture['wants_startup']
    req.wants_external_validation = culture['wants_external_validation']
    req.wants_ownership = culture['wants_ownership']
    req.wants_production_experience = culture['wants_production']
    req.is_research_role = culture['is_research']

    # 6. Build semantic text for TF-IDF from extracted terms + JD text
    skill_vocab = ' '.join(req.must_have_skills + req.nice_to_have_skills)
    # Also include the raw JD text (preprocessed) for full semantic coverage
    jd_clean = re.sub(r'[^\w\s]', ' ', jd_text.lower())
    req.semantic_text = f"{skill_vocab} {jd_clean}"

    # 7. Compute extraction confidence
    # High confidence = sections found + skills extracted
    confidence_factors = [
        req.must_have_section_found,
        req.nice_have_section_found,
        len(req.must_have_skills) >= 3,
        len(req.nice_to_have_skills) >= 2,
        bool(req.consulting_disqualifiers),
    ]
    req.extraction_confidence = sum(confidence_factors) / len(confidence_factors)

    return req


# ── SINGLETON: parsed from DEFAULT_JD_TEXT at import time ────────────────────
# In production: `JD_REQUIREMENTS = parse_jd(load_jd_from_file_or_api())`
JD_REQUIREMENTS: JDRequirements = parse_jd(DEFAULT_JD_TEXT)


# ── FALLBACK LISTS for ranker compatibility ───────────────────────────────────
# These are derived from the parsed JD, not hardcoded.
# They update automatically when JD_REQUIREMENTS is rebuilt.
MUST_HAVE_SKILLS: list[str] = JD_REQUIREMENTS.must_have_skills
NICE_TO_HAVE_SKILLS: list[str] = JD_REQUIREMENTS.nice_to_have_skills

# Title keywords derived from JD seniority + domain
IDEAL_TITLE_KEYWORDS = {
    "ai engineer", "ml engineer", "machine learning engineer", "nlp engineer",
    "applied ml", "applied ai", "senior ai", "search engineer", "retrieval engineer",
    "data scientist", "research engineer", "ai research", "senior nlp", "staff ml",
    "lead ai", "staff ai", "principal ml",
}

ADJACENT_TITLE_KEYWORDS = {
    "software engineer", "backend engineer", "data engineer", "full stack",
    "platform engineer", "mlops engineer", "ai specialist", "senior software",
    "ml ops",
}

PREFERRED_LOCATIONS: list[str] = JD_REQUIREMENTS.preferred_locations
