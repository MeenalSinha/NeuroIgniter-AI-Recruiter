"""
NeuroIgniter AI Recruiter — Recruiter Dashboard v2
===================================================
Run: streamlit run dashboard/app.py
     (from the repo root directory)
"""

import sys
import os

# Ensure src/ is on path regardless of how the dashboard is launched
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import csv
import json
import re

from neuroigniter.jd_parser import MUST_HAVE_SKILLS, JD_REQUIREMENTS
from neuroigniter.ontology import normalize_skill
from neuroigniter.ranker import CONSULTING_FIRMS

st.set_page_config(
    page_title="NeuroIgniter AI Recruiter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Pastel gradient background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(249, 229, 216, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Headers with gradient */
    h1, h2, h3, h4, h5, h6 {
        color: #6A5D7B !important;
        font-weight: 700;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hero section with glassmorphism */
    .hero-section {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.6), rgba(163, 201, 168, 0.6));
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        animation: heroFadeIn 1s ease-out;
    }
    
    @keyframes heroFadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .hero-logo {
        font-size: 4rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        color: white !important;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .hero-subtitle {
        color: white;
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Pastel buttons */
    .stButton>button {
        background: linear-gradient(135deg, #A3C9A8 0%, #B8D4BE 100%);
        color: white;
        border-radius: 15px;
        height: 3.5em;
        width: 100%;
        font-size: 1.1em;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(163, 201, 168, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #9EB5A5 0%, #B0C8B7 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(163, 201, 168, 0.5);
    }
    
    /* Metric cards with glassmorphism */
    .metric-glass-card {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.7), rgba(212, 196, 232, 0.7));
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-glass-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 40px rgba(200, 184, 219, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert boxes with glassmorphism */
    .glass-alert-success {
        background: rgba(212, 241, 221, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #A3C9A8;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(163, 201, 168, 0.2);
    }
    
    .glass-alert-warning {
        background: rgba(255, 243, 205, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #F9C74F;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(249, 199, 79, 0.2);
    }
    
    .glass-alert-danger {
        background: rgba(255, 229, 229, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #F4978E;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 151, 142, 0.2);
    }
    
    .glass-alert-info {
        background: rgba(227, 242, 253, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #90CAF9;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(144, 202, 249, 0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(245, 223, 208, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 15px 15px 0 0;
        color: #6A5D7B;
        padding: 12px 24px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(245, 223, 208, 0.8);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.8), rgba(212, 196, 232, 0.8));
        color: white;
        box-shadow: 0 4px 15px rgba(200, 184, 219, 0.3);
    }
    
    /* Badge styling */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #A3C9A8, #C8B8DB);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Original Dashboard tags adapted for new theme */
    .skill-tag {
        display: inline-block; background: rgba(227, 242, 253, 0.7); color: #1565c0;
        border-radius: 12px; padding: 2px 10px; margin: 2px;
        font-size: 0.80rem; font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .missing-tag {
        display: inline-block; background: rgba(255, 229, 229, 0.7); color: #c62828;
        border-radius: 12px; padding: 2px 10px; margin: 2px;
        font-size: 0.80rem; border: 1px dashed #F4978E;
    }
    .concern-tag {
        display: inline-block; background: rgba(255, 243, 205, 0.7); color: #f57c00;
        border-radius: 12px; padding: 2px 10px; margin: 2px;
        font-size: 0.80rem;
    }
    .conf-badge {
        display: inline-block; background: rgba(212, 241, 221, 0.7); color: #2e7d32;
        border-radius: 8px; padding: 1px 8px; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.85: return "#2e7d32"
    elif score >= 0.70: return "#f57c00"
    elif score >= 0.55: return "#e65100"
    else: return "#c62828"

def render_score_bar(label: str, score: float, color: str = "#6c63ff", width: str = "90px"):
    """Render a horizontal score bar with ARIA attributes for screen reader accessibility."""
    pct = int(score * 100)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:2px 0"
         role="meter" aria-label="{label} score" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100">
        <span style="width:{width};font-size:0.78rem;color:#555;flex-shrink:0">{label}</span>
        <div style="flex:1;background:#eee;border-radius:4px;height:6px" role="presentation">
            <div style="width:{pct}%;background:{color};border-radius:4px;height:6px;transition:width 0.3s"
                 aria-hidden="true"></div>
        </div>
        <span style="width:32px;text-align:right;font-size:0.78rem;font-weight:600;color:{color}"
              aria-hidden="true">{score:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

def parse_decomp(reasoning: str) -> dict:
    m = re.search(
        r'\[Skills ([\d.]+) \| Career ([\d.]+) \| Behavioral ([\d.]+) \| Avail ([\d.]+) \| Semantic ([\d.]+)',
        reasoning
    )
    if m:
        return {
            'Skills': float(m.group(1)),
            'Career': float(m.group(2)),
            'Behavioral': float(m.group(3)),
            'Availability': float(m.group(4)),
            'Semantic': float(m.group(5)),
        }
    # Fallback for older format
    m2 = re.search(
        r'\[Skills ([\d.]+) \| Career ([\d.]+) \| Behavioral ([\d.]+) \| Availability ([\d.]+) \| Semantic ([\d.]+)',
        reasoning
    )
    if m2:
        return {
            'Skills': float(m2.group(1)),
            'Career': float(m2.group(2)),
            'Behavioral': float(m2.group(3)),
            'Availability': float(m2.group(4)),
            'Semantic': float(m2.group(5)),
        }
    return {}

def parse_semantic_subcomponents(reasoning: str) -> tuple[float, float] | None:
    """Extract the lexical/distributional sub-breakdown from the hybrid
    semantic score, e.g. 'Semantic 0.617 (lex=0.36/dist=0.93)'."""
    m = re.search(r'lex=([\d.]+)/dist=([\d.]+)', reasoning)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_confidence(reasoning: str) -> tuple[float, int]:
    m = re.search(r'conf=([\d.]+)\s+n=(\d+)', reasoning)
    if m:
        return float(m.group(1)), int(m.group(2))
    return 0.80, 12

def clean_reasoning(reasoning: str) -> str:
    return re.sub(r'\[Skills.*?\]', '', reasoning).strip()

def get_ai_skills(skills: list) -> list[str]:
    """Extract AI/ML-relevant skill names for display tags. Defensive against
    malformed skill entries (missing 'name' key) — the dashboard may load a
    JSONL from a different source than the one that produced the CSV, so it
    shouldn't assume the ranker's upstream validation already ran."""
    ai_kw = ['embed', 'vector', 'faiss', 'python', 'nlp', 'lora', 'rag',
             'search', 'bert', 'transformer', 'ranking', 'retrieval',
             'elasticsearch', 'qdrant', 'weaviate', 'pinecone', 'milvus',
             'learning to rank', 'xgboost', 'llm', 'hugging', 'bm25',
             'semantic', 'dense', 'sparse', 'encoder', 'embedding']
    names = [s.get('name', '') for s in skills if isinstance(s, dict) and s.get('name')]
    matched = [n for n in names if any(k in n.lower() for k in ai_kw)]
    other = [n for n in names if n not in matched]
    return (matched + other)[:8]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_submission(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

@st.cache_data
def load_candidates_sample(jsonl_path: str, ids: frozenset) -> dict:
    result = {}
    if not os.path.exists(jsonl_path):
        return result
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                if c['candidate_id'] in ids:
                    result[c['candidate_id']] = c
                    if len(result) == len(ids):
                        break
            except json.JSONDecodeError:
                pass
    return result

@st.cache_data
def load_ranking_stats(stats_path: str) -> dict:
    """Load the disqualification/filtering transparency stats sidecar file."""
    if not os.path.exists(stats_path):
        return {}
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ── Paths ─────────────────────────────────────────────────────────────────────
_curr = os.path.abspath(__file__)
if os.path.basename(_curr) == 'streamlit_app.py':
    repo_root = os.path.dirname(_curr)
else:
    repo_root = os.path.dirname(os.path.dirname(_curr))

default_csv = os.path.join(repo_root, 'output', 'neuroigniter_submission.csv')
default_jsonl = os.path.join(repo_root, 'India_runs_data_and_ai_challenge', 'candidates.jsonl')
default_stats = os.path.join(repo_root, 'output', 'ranking_stats.json')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚙️ Advanced: Override Data Sources", expanded=False):
        csv_path = st.text_input("Submission CSV", value=default_csv)
        jsonl_path = st.text_input("Candidates JSONL", value=default_jsonl)
        stats_path = st.text_input("Ranking Stats JSON", value=default_stats)

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    search_query = st.text_input("Search (title, company, location)", value="", placeholder="e.g. Bengaluru, Senior...")
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    max_rank = st.slider("Show top N", 10, 100, 50, 10)
    max_notice = st.slider("Max notice period (days)", 0, 120, 120, 5)
    filter_open = st.checkbox("Open to work only", value=False)

    st.markdown("---")
    st.markdown("### 📋 Role Context")
    st.caption("Auto-extracted from JD text at runtime — adapts automatically if the JD changes")
    skills_preview = ', '.join(MUST_HAVE_SKILLS[:6]) + ('...' if len(MUST_HAVE_SKILLS) > 6 else '')
    st.markdown(
        f"""
        <div class="glass-alert-info">
            <strong>📋 Role Context:</strong><br>
            <strong>Must-have ({len(MUST_HAVE_SKILLS)}):</strong> {skills_preview}<br>
            <strong>YOE:</strong> {JD_REQUIREMENTS.yoe_min:.0f}–{JD_REQUIREMENTS.yoe_max:.0f} yrs | 
            <strong>Location:</strong> {', '.join(JD_REQUIREMENTS.preferred_locations[:3])}<br>
            <strong>Notice:</strong> Sub-{JD_REQUIREMENTS.ideal_notice_days}-day preferred<br>
            <strong>Extraction confidence:</strong> {JD_REQUIREMENTS.extraction_confidence:.0%}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📤 Export")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-section">
    <div class="hero-logo">🧠</div>
    <h1 class="hero-title">NeuroIgniter AI Recruiter</h1>
    <p class="hero-subtitle">
        Intelligent Candidate Ranking · {JD_REQUIREMENTS.seniority_level.title()}-level role · JD parsed at runtime
    </p>
    <div style="margin-top: 1.5rem;">
        <span class="tech-badge">📊 {len(MUST_HAVE_SKILLS)} Key Skills</span>
        <span class="tech-badge">⚡ Lexical & Semantic</span>
        <span class="tech-badge">🎯 Top-Tier Fit</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ── Load data ─────────────────────────────────────────────────────────────────
rows = load_submission(csv_path)
if not rows:
    st.error(f"No submission CSV at `{csv_path}`. Run `python run.py` first.")
    st.stop()

top_ids = frozenset(r['candidate_id'] for r in rows)
with st.spinner("Loading candidate profiles..."):
    candidates = load_candidates_sample(jsonl_path, top_ids) if os.path.exists(jsonl_path) else {}

ranking_stats = load_ranking_stats(stats_path)

# Apply filters
filtered = [r for r in rows if float(r['score']) >= min_score]

if filter_open:
    filtered = [r for r in filtered
                if candidates.get(r['candidate_id'], {}).get('redrob_signals', {}).get('open_to_work_flag', False)]

if max_notice < 120:
    filtered = [r for r in filtered
                if candidates.get(r['candidate_id'], {}).get('redrob_signals', {}).get('notice_period_days', 999) <= max_notice]

if search_query.strip():
    q = search_query.strip().lower()
    def _matches_search(r):
        c = candidates.get(r['candidate_id'], {})
        p = c.get('profile', {})
        haystack = ' '.join([
            p.get('current_title', ''), p.get('current_company', ''),
            p.get('location', ''), p.get('country', ''),
            r['candidate_id'],
        ]).lower()
        return q in haystack
    filtered = [r for r in filtered if _matches_search(r)]

filtered = filtered[:max_rank]

# ── Stats ─────────────────────────────────────────────────────────────────────
scores = [float(r['score']) for r in rows]
c1, c2, c3, c4, c5, c6 = st.columns(6)
stats = [
    (len(rows), "Ranked"),
    (f"{scores[0]:.3f}", "Top Score"),
    (sum(1 for s in scores if s >= 0.85), "Tier 1 ≥0.85"),
    (sum(1 for s in scores if 0.70 <= s < 0.85), "Tier 2 0.70-0.85"),
    (sum(1 for s in scores if s < 0.70), "Tier 3 <0.70"),
    (len(candidates), "Profiles Loaded"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5, c6], stats):
    with col:
        st.markdown(f"""
        <div class="metric-glass-card">
            <div class="metric-value" style="font-size: 2rem;">{val}</div>
            <div class="metric-label" style="font-size: 0.75rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ── Tabs: Ranked List | Compare | Insights ────────────────────────────────────
tab_list, tab_compare, tab_insights = st.tabs(["📋 Ranked Shortlist", "⚖️ Compare Candidates", "📊 Insights"])

# ── TAB 1: Ranked List ────────────────────────────────────────────────────────
with tab_list:
    st.markdown(f"### Showing {len(filtered)} of {len(rows)} ranked candidates")

    for row in filtered:
        rank = int(row['rank'])
        cid = row['candidate_id']
        score = float(row['score'])
        reasoning = row.get('reasoning', '')

        cand = candidates.get(cid, {})
        profile = cand.get('profile', {})
        signals = cand.get('redrob_signals', {})
        skills = cand.get('skills', [])

        title = profile.get('current_title', '—')
        yoe = profile.get('years_of_experience', '?')
        company = profile.get('current_company', '—')
        country = profile.get('country', '')
        location = profile.get('location', '—')

        decomp = parse_decomp(reasoning)
        conf, n_sig = parse_confidence(reasoning)
        reasoning_text = clean_reasoning(reasoning)

        tier_color = score_color(score)
        badge_bg = "#e63946" if rank <= 3 else ("#6c63ff" if rank <= 10 else "#555")

        with st.container():
            col_rank, col_main, col_score = st.columns([1, 7, 2])

            with col_rank:
                st.markdown(
                    f'<div style="background:{badge_bg};color:white;font-weight:700;'
                    f'font-size:1rem;width:40px;height:40px;border-radius:50%;'
                    f'line-height:40px;text-align:center;margin-top:4px" '
                    f'role="img" aria-label="Rank {rank}">#{rank}</div>',
                    unsafe_allow_html=True
                )

            with col_main:
                loc_flag = "🇮🇳 " if country.lower() == 'india' else "🌍 "
                st.markdown(f"**{title}** · {yoe} yrs · {loc_flag}{location}")

                salary = signals.get('expected_salary_range_inr_lpa', {})
                salary_str = ""
                if salary and salary.get('min') and salary.get('max'):
                    salary_str = f" · 💰 ₹{salary['min']:.0f}–{salary['max']:.0f} LPA"
                st.caption(f"{cid} · {company}{salary_str}")

                if skills:
                    ai_skills = get_ai_skills(skills)
                    tags = ' '.join(f'<span class="skill-tag">{s}</span>' for s in ai_skills)
                    st.markdown(tags, unsafe_allow_html=True)

            with col_score:
                st.markdown(
                    f'<div style="text-align:center;padding:4px">'
                    f'<div style="font-size:1.9rem;font-weight:700;color:{tier_color}" '
                    f'role="meter" aria-label="Match score {score:.3f}" '
                    f'aria-valuenow="{int(score*100)}" aria-valuemin="0" aria-valuemax="100">{score:.3f}</div>'
                    f'<div class="conf-badge">conf {conf:.0%} · {n_sig} signals</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                open_work = signals.get('open_to_work_flag', False)
                notice = signals.get('notice_period_days', 90)
                rr = signals.get('recruiter_response_rate', 0)
                avail_color = "#2e7d32" if open_work and notice <= 30 else ("#f57c00" if open_work else "#888")
                st.markdown(
                    f'<div style="font-size:0.75rem;color:{avail_color};text-align:center">'
                    f'{"✅ Open" if open_work else "⭕ Passive"} · {notice}d · {rr:.0%} resp'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Expandable detail (auto-open top 3)
            with st.expander(f"Full analysis — #{rank} {cid}", expanded=(rank <= 3)):
                t1, t2, t3, t4 = st.tabs(["🤖 AI Analysis", "📊 Score Breakdown", "👤 Profile", "🎯 Interview"])

                with t1:
                    if reasoning_text:
                        st.markdown(f"> {reasoning_text}")

                    # Skill gap panel
                    matched_set = set()
                    if cand:
                        matched_set = {normalize_skill(s['name']) for s in skills}

                    missing_critical = [s for s in MUST_HAVE_SKILLS[:10]
                                        if normalize_skill(s) not in matched_set][:3]
                    if missing_critical:
                        st.markdown("**Key JD gaps:**")
                        tags = ' '.join(f'<span class="missing-tag">missing: {s}</span>'
                                        for s in missing_critical)
                        st.markdown(tags, unsafe_allow_html=True)

                with t2:
                    if decomp:
                        col_a, col_b = st.columns(2)
                        items = list(decomp.items())
                        with col_a:
                            for label, val in items[:3]:
                                render_score_bar(label, val)
                        with col_b:
                            for label, val in items[3:]:
                                render_score_bar(label, val)

                        # Show the hybrid semantic sub-breakdown if present —
                        # makes the lexical-vs-distributional blend visible,
                        # not just a single opaque "Semantic" number.
                        subcomp = parse_semantic_subcomponents(reasoning)
                        if subcomp:
                            lex_val, dist_val = subcomp
                            st.caption("Semantic score breakdown (hybrid):")
                            render_score_bar("  Lexical (TF-IDF)", lex_val, color="#9575cd", width="140px")
                            render_score_bar("  Distributional", dist_val, color="#26a69a", width="140px")
                            st.caption(
                                "Lexical rewards exact JD-vocabulary matches; Distributional is a "
                                "locally-trained semantic embedding that can match candidates who "
                                "describe the same work in different words."
                            )

                        st.markdown(f"**Composite: {score:.4f}** · Confidence: {conf:.0%} ({n_sig}/16 signals available)")
                        st.caption("Weights: Skills 38% | Career 26% | Semantic 10% | Behavioral 12% | Availability 8% | Education 6%")
                    else:
                        st.info("Score breakdown not available")

                with t3:
                    if cand:
                        career = cand.get('career_history', [])
                        edu = cand.get('education', [])
                        sigs_p = cand.get('redrob_signals', {})
                        certs = cand.get('certifications', [])

                        summ = profile.get('summary', '')
                        if summ:
                            st.markdown(f"**Summary:** {summ[:350]}{'...' if len(summ)>350 else ''}")

                        if career:
                            st.markdown("**Career:**")
                            for job in career[:4]:
                                dur = job.get('duration_months', 0)
                                st.markdown(
                                    f"- **{job.get('title')}** @ {job.get('company')} "
                                    f"({dur//12}y {dur%12}m) · {job.get('company_size','?')} employees"
                                )

                        cols_edu = st.columns(2)
                        with cols_edu[0]:
                            if edu:
                                st.markdown("**Education:**")
                                for e in edu[:2]:
                                    st.markdown(f"- {e.get('degree')} · {e.get('institution')} ({e.get('tier','')})")
                        with cols_edu[1]:
                            if certs:
                                st.markdown("**Certifications:**")
                                for cert in certs[:3]:
                                    st.markdown(f"- {cert.get('name')} ({cert.get('year','')})")

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        gh = sigs_p.get('github_activity_score', -1)
                        col_m1.metric("GitHub", f"{gh:.0f}" if gh != -1 else "—")
                        col_m2.metric("Response Rate", f"{sigs_p.get('recruiter_response_rate',0):.0%}")
                        col_m3.metric("Profile Views 30d", sigs_p.get('profile_views_received_30d', 0))
                        col_m4.metric("Saved by Recruiters", sigs_p.get('saved_by_recruiters_30d', 0))
                    else:
                        st.info("Profile data unavailable — place candidates.jsonl in data/")

                with t4:
                    st.markdown("**Recommended Interview Focus:**")
                    # Build a candidate-specific question list; only fall back to a
                    # generic prompt if fewer than 2 specific ones were generated.
                    specific_questions = []

                    if decomp:
                        lowest_comp = min(decomp, key=decomp.get)
                        specific_questions.append(
                            f"🔍 Lowest-scoring component: **{lowest_comp}** ({decomp[lowest_comp]:.3f}) — "
                            f"probe this area with specific examples from their career"
                        )

                    if cand:
                        career_companies = [ch.get('company','') for ch in cand.get('career_history',[])]
                        has_consulting = any(any(f in c.lower() for f in CONSULTING_FIRMS) for c in career_companies)
                        gh_s = cand.get('redrob_signals',{}).get('github_activity_score',-1)
                        notice_s = cand.get('redrob_signals',{}).get('notice_period_days', 30)

                        if missing_critical:
                            specific_questions.append(
                                f"📋 Ask for a hands-on example involving **{missing_critical[0]}** "
                                f"— this is a JD must-have that doesn't appear in their profile"
                            )
                        if gh_s == -1:
                            specific_questions.append("💻 Ask about **public technical work** (OSS, papers, talks) — JD values external validation")
                        elif gh_s > 65:
                            specific_questions.append(f"💻 GitHub score is **{gh_s:.0f}** — ask them to share a specific repo or contribution")
                        if has_consulting:
                            specific_questions.append("🏢 Ask: *'Describe a technical decision you made autonomously'* — probe product vs consulting mindset")
                        if notice_s > 75:
                            specific_questions.append(f"📅 Discuss **{notice_s}-day notice** — explore buyout possibility and actual start flexibility")

                    for q in specific_questions:
                        st.markdown(f"- {q}")

                    # Generic fallback ONLY if we couldn't generate enough specific
                    # questions for this candidate — not shown for every candidate.
                    if len(specific_questions) < 2:
                        # Pick a fallback tied to their weakest decomp component, not a
                        # one-size-fits-all line repeated for everyone.
                        if decomp:
                            weakest = min(decomp, key=decomp.get)
                            fallback_map = {
                                'Skills': "🧠 Walk through a production retrieval/ranking system they built end-to-end",
                                'Career': "📈 Ask them to narrate their career trajectory and what drove each transition",
                                'Behavioral': "📬 Ask why their platform engagement signals are lower than typical — gauge current job-search intent",
                                'Availability': "📅 Clarify actual availability and notice-period flexibility directly",
                                'Semantic': "🎯 Ask them to describe, in their own words, how their background fits this specific role",
                            }
                            st.markdown(f"- {fallback_map.get(weakest, 'Ask for specific, quantified examples of their impact')}")
                        else:
                            st.markdown("- Ask for specific, quantified examples of production impact")

        st.markdown("---")

    # Export
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_data = f.read()
    with st.sidebar:
        st.download_button("⬇️ Download CSV", data=csv_data,
                           file_name="neuroigniter_submission.csv", mime="text/csv")

# ── TAB 2: Compare Candidates ─────────────────────────────────────────────────
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Candidate Comparison")
    st.caption("Select two candidates to see a direct score comparison — answering 'why is A ranked above B?'")

    all_ids = [r['candidate_id'] for r in rows[:50]]
    all_labels = {r['candidate_id']: f"#{r['rank']} · {r['candidate_id']} · score {float(r['score']):.3f}"
                  for r in rows[:50]}

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_a = st.selectbox("Candidate A", all_ids, index=0,
                             format_func=lambda x: all_labels.get(x, x))
    with col_sel2:
        sel_b = st.selectbox("Candidate B", all_ids, index=1,
                             format_func=lambda x: all_labels.get(x, x))

    if sel_a == sel_b:
        st.warning("Select two different candidates to compare.")
    else:
        row_a = next((r for r in rows if r['candidate_id'] == sel_a), None)
        row_b = next((r for r in rows if r['candidate_id'] == sel_b), None)
        cand_a = candidates.get(sel_a, {})
        cand_b = candidates.get(sel_b, {})

        if row_a and row_b:
            score_a, score_b = float(row_a['score']), float(row_b['score'])
            decomp_a = parse_decomp(row_a['reasoning'])
            decomp_b = parse_decomp(row_b['reasoning'])
            prof_a = cand_a.get('profile', {})
            prof_b = cand_b.get('profile', {})
            sig_a = cand_a.get('redrob_signals', {})
            sig_b = cand_b.get('redrob_signals', {})

            # Header row
            hc1, hc2, hc3 = st.columns([2, 3, 3])
            with hc1:
                st.markdown("**Component**")
            with hc2:
                winner_a = "🏆 " if score_a > score_b else ""
                st.markdown(f"**{winner_a}#{row_a['rank']} — {sel_a}**")
                st.caption(f"{prof_a.get('current_title','—')} · {prof_a.get('years_of_experience','?')} yrs")
            with hc3:
                winner_b = "🏆 " if score_b > score_a else ""
                st.markdown(f"**{winner_b}#{row_b['rank']} — {sel_b}**")
                st.caption(f"{prof_b.get('current_title','—')} · {prof_b.get('years_of_experience','?')} yrs")

            st.markdown("---")

            # Composite
            cc1, cc2, cc3 = st.columns([2, 3, 3])
            with cc1:
                st.markdown("**Overall Score**")
            with cc2:
                col_a_color = score_color(score_a)
                st.markdown(f"<span style='font-size:1.5rem;font-weight:700;color:{col_a_color}'>{score_a:.4f}</span>", unsafe_allow_html=True)
            with cc3:
                col_b_color = score_color(score_b)
                st.markdown(f"<span style='font-size:1.5rem;font-weight:700;color:{col_b_color}'>{score_b:.4f}</span>", unsafe_allow_html=True)

            st.markdown("---")

            # Component breakdown
            if decomp_a and decomp_b:
                components = list(decomp_a.keys())
                weights = {'Skills': 0.38, 'Career': 0.26, 'Behavioral': 0.12,
                           'Availability': 0.08, 'Semantic': 0.10}

                for comp in components:
                    va = decomp_a.get(comp, 0)
                    vb = decomp_b.get(comp, 0)
                    diff = va - vb
                    wt = weights.get(comp, 0.05)

                    rc1, rc2, rc3 = st.columns([2, 3, 3])
                    with rc1:
                        st.markdown(f"**{comp}** _(wt {wt:.0%})_")
                    with rc2:
                        diff_indicator = "▲" if diff > 0.01 else ("▽" if diff < -0.01 else "≈")
                        render_score_bar("", va, color="#6c63ff" if diff >= 0 else "#aaa")
                    with rc3:
                        render_score_bar("", vb, color="#6c63ff" if diff <= 0 else "#aaa")

                    if abs(diff) > 0.05:
                        winner_name = sel_a if diff > 0 else sel_b
                        st.caption(f"  → {winner_name} leads on {comp} by {abs(diff):.3f} ({abs(diff)*wt:.3f} composite points)")
                    st.markdown("")

            st.markdown("---")

            # Why A ranks above B
            winner_id = sel_a if score_a > score_b else sel_b
            loser_id = sel_b if score_a > score_b else sel_a
            score_diff = abs(score_a - score_b)
            if decomp_a and decomp_b:
                diffs = {k: decomp_a.get(k, 0) - decomp_b.get(k, 0) for k in components}
                top_advantage = max(diffs, key=lambda k: abs(diffs[k]))
                adv_direction = "A" if diffs[top_advantage] > 0 else "B"
                winner_label = sel_a if adv_direction == "A" else sel_b

                st.markdown(f"**Why #{row_a['rank'] if score_a > score_b else row_b['rank']} ranks above #{row_b['rank'] if score_a > score_b else row_a['rank']}:**")
                st.info(
                    f"{winner_id} leads by **{score_diff:.4f}** composite points. "
                    f"Biggest differentiator: **{top_advantage}** (gap of {abs(diffs[top_advantage]):.3f}). "
                    f"To close the gap, {loser_id} would need stronger {top_advantage.lower()} signals."
                )

# ── TAB 3: Insights ───────────────────────────────────────────────────────────
with tab_insights:
    st.markdown("### 📊 Shortlist Insights")

    if not candidates:
        st.info("Load candidates.jsonl to see insights.")
    else:
        top_rows = rows[:20]
        top_cands = [candidates.get(r['candidate_id'], {}) for r in top_rows]

        # ── "So what" actionable summary ─────────────────────────────────
        st.markdown("#### 🎯 Recommended Actions")

        # Fastest available: open + sub-15-day notice + decent score
        fastest = [
            (r, c) for r, c in zip(top_rows, top_cands)
            if c.get('redrob_signals', {}).get('open_to_work_flag')
            and c.get('redrob_signals', {}).get('notice_period_days', 999) <= 15
        ]
        # Highest confidence: parse conf= from reasoning
        conf_scored = []
        for r in top_rows:
            conf, n_sig = parse_confidence(r['reasoning'])
            conf_scored.append((conf, r))
        conf_scored.sort(key=lambda x: -x[0])

        # Most interview-ready: high interview_completion_rate + high response rate
        ready = [
            (r, c) for r, c in zip(top_rows, top_cands)
            if c.get('redrob_signals', {}).get('interview_completion_rate', 0) >= 0.85
            and c.get('redrob_signals', {}).get('recruiter_response_rate', 0) >= 0.70
        ]

        ai1, ai2, ai3 = st.columns(3)
        with ai1:
            if fastest:
                top_fast = fastest[0][0]
                st.markdown(f'<div class="glass-alert-success"><strong>⚡ Fastest hire:</strong> #{top_fast["rank"]} {top_fast["candidate_id"]} — open to work, ≤15-day notice</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-alert-warning"><strong>⚡ Fastest hire:</strong> No top-20 candidate has both open-to-work + sub-15-day notice</div>', unsafe_allow_html=True)
        with ai2:
            if conf_scored:
                top_conf = conf_scored[0][1]
                st.markdown(f'<div class="glass-alert-success"><strong>🎯 Highest confidence:</strong> #{top_conf["rank"]} {top_conf["candidate_id"]} — {conf_scored[0][0]:.0%} signal coverage</div>', unsafe_allow_html=True)
        with ai3:
            if ready:
                top_ready = ready[0][0]
                st.markdown(f'<div class="glass-alert-success"><strong>📞 Most interview-ready:</strong> #{top_ready["rank"]} {top_ready["candidate_id"]} — high completion + response rate</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-alert-warning"><strong>📞 Most interview-ready:</strong> No standout candidate — verify availability before reaching out</div>', unsafe_allow_html=True)

        # Most common missing skill across top 20
        from collections import Counter
        missing_counter = Counter()
        for c in top_cands:
            if not c:
                continue
            cand_skills = {normalize_skill(s['name']) for s in c.get('skills', [])}
            for must in MUST_HAVE_SKILLS:
                if normalize_skill(must) not in cand_skills:
                    missing_counter[must] += 1
        if missing_counter:
            top_gap, gap_count = missing_counter.most_common(1)[0]
            st.markdown(
                f'<div class="glass-alert-info"><strong>📋 Most common gap:</strong> <strong>{top_gap}</strong> is missing from {gap_count}/{len(top_cands)} top-20 profiles — '
                f'consider whether this is a hard requirement or can be trained post-hire</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Charts ───────────────────────────────────────────────────────
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Score distribution — Top 20**")
            # Limited to top 20 (not 50) to avoid label overlap in bar chart
            score_data = {f"#{r['rank']}": float(r['score']) for r in top_rows}
            st.bar_chart(score_data, height=220)

        with col_d2:
            st.markdown("**Top-20 candidates by YOE**")
            yoe_data = {f"#{r['rank']}": c.get('profile', {}).get('years_of_experience', 0)
                       for r, c in zip(top_rows, top_cands)}
            st.bar_chart(yoe_data, height=220)

        # Notice period distribution
        st.markdown("**Availability summary — Top 20**")
        notice_ranges = {"0-15d": 0, "16-30d": 0, "31-60d": 0, "61-90d": 0, "90d+": 0}
        open_count = 0
        salary_values = []
        for c in top_cands:
            sigs = c.get('redrob_signals', {})
            if sigs.get('open_to_work_flag'):
                open_count += 1
            notice = sigs.get('notice_period_days', 90)
            if notice <= 15: notice_ranges["0-15d"] += 1
            elif notice <= 30: notice_ranges["16-30d"] += 1
            elif notice <= 60: notice_ranges["31-60d"] += 1
            elif notice <= 90: notice_ranges["61-90d"] += 1
            else: notice_ranges["90d+"] += 1
            sal = sigs.get('expected_salary_range_inr_lpa', {})
            if sal and sal.get('max'):
                salary_values.append(sal['max'])

        ni1, ni2, ni3, ni4 = st.columns(4)
        ni1.metric("Open to work", f"{open_count}/20")
        ni2.metric("Sub-30d notice", f"{notice_ranges['0-15d'] + notice_ranges['16-30d']}/20")
        ni3.metric("90d+ notice", f"{notice_ranges['90d+']}/20")
        if salary_values:
            ni4.metric("Avg salary ask (max)", f"₹{sum(salary_values)/len(salary_values):.0f} LPA")

        st.bar_chart(notice_ranges, height=180)

        # ── Disqualification transparency panel ─────────────────────────
        st.markdown("---")
        st.markdown("#### 🔍 Filtering Transparency")
        st.caption("What happened to candidates who didn't make the shortlist — full pipeline audit trail")

        if ranking_stats:
            total = ranking_stats.get('total_candidates', 0)
            hp = ranking_stats.get('honeypot_count', 0)
            cons = ranking_stats.get('disqualified_consulting', 0)
            cvr = ranking_stats.get('disqualified_cv_robotics', 0)
            zero = ranking_stats.get('zero_skill_match', 0)
            elapsed = ranking_stats.get('elapsed_seconds', 0)
            peak_buf = ranking_stats.get('peak_buffer_size', 0)

            st.markdown(
                f"Of **{total:,}** candidates processed: **{hp}** flagged as honeypot profiles (impossible "
                f"skill/timeline data), **{cons:,}** disqualified for consulting-only careers (explicit JD "
                f"exclusion), **{cvr}** disqualified for CV/robotics-only background without NLP/IR exposure, "
                f"and **{zero:,}** had zero overlap with any JD-extracted must-have or nice-to-have skill."
            )

            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Total processed", f"{total:,}")
            fc2.metric("Consulting-only", f"{cons:,}", delta=f"-{cons/max(total,1):.0%}", delta_color="inverse")
            fc3.metric("Zero skill match", f"{zero:,}", delta=f"-{zero/max(total,1):.0%}", delta_color="inverse")
            fc4.metric("Pipeline runtime", f"{elapsed:.0f}s")

            st.caption(
                f"Peak memory buffer: {peak_buf:,} candidates held in memory at once "
                f"(bounded streaming — does not scale with dataset size)."
            )
        else:
            st.info("No ranking_stats.json found — run `python run.py` to generate the filtering audit trail.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.78rem'>"
    "NeuroIgniter AI Recruiter v2 · India Runs Data & AI Challenge · "
    "Skills 38% | Career 26% | Semantic 10% | Behavioral 12% | Availability 8% | Education 6%"
    "</div>",
    unsafe_allow_html=True,
)
