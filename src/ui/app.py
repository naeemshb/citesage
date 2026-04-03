"""CiteSage — AI Research Copilot."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import httpx
import streamlit as st

from src.agent.graph import stream_agent
from src.ingestion.loader import load_pdf, extract_metadata_from_first_page
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import ingest_documents, get_paper_count, get_chunk_count, get_ingested_papers_info
from src.retrieval.retriever import retrieve_documents

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CiteSage — AI Research Copilot",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS with animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Reset */
    .stApp { font-family: 'Inter', -apple-system, sans-serif; background: #f9fafb; color: #1a1a2e; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp li, .stApp td, .stApp th { color: inherit; }
    .stMarkdown, .stMarkdown p { color: #1a1a2e !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0rem; max-width: 960px; }
    .stDeployButton { display: none; }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-12px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes dotPulse {
        0%, 80%, 100% { opacity: 0; transform: scale(0.6); }
        40% { opacity: 1; transform: scale(1); }
    }
    @keyframes progressBar {
        0% { width: 0%; }
        100% { width: 100%; }
    }

    /* Search loading animation */
    .search-loading {
        padding: 16px 20px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin: 8px 0;
        animation: fadeInUp 0.3s ease-out;
    }
    .search-loading .step-text {
        font-size: 0.88rem;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .loading-dots {
        display: inline-flex;
        gap: 4px;
        margin-left: 4px;
    }
    .loading-dots span {
        width: 6px; height: 6px;
        background: #5b5fc7;
        border-radius: 50%;
        animation: dotPulse 1.4s infinite ease-in-out;
    }
    .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.4s; }

    .step-done {
        color: #059669;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        animation: slideInLeft 0.3s ease-out;
    }
    .step-done::before {
        content: '';
        width: 18px; height: 18px;
        background: #dcfce7;
        border: 2px solid #059669;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%23059669' stroke-width='3'%3E%3Cpath d='M20 6L9 17l-5-5'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: center;
    }

    .progress-bar-container {
        width: 100%;
        height: 3px;
        background: #e2e8f0;
        border-radius: 2px;
        margin: 12px 0 4px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #5b5fc7, #e85d26);
        border-radius: 2px;
        animation: progressBar 12s ease-out forwards;
    }

    /* Gradient banner */
    .gradient-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #e85d26 50%, #5b5fc7 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 14px 24px;
        border-radius: 0 0 16px 16px;
        margin: -1rem -1rem 20px -1rem;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .gradient-banner a {
        text-decoration: none;
        color: white;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .gradient-banner a:hover { opacity: 0.85; }
    .gradient-banner-tag {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.7);
        font-weight: 400;
    }

    /* Landing */
    .landing {
        text-align: center;
        padding: 60px 0 10px 0;
        animation: fadeInUp 0.6s ease-out;
    }
    .landing-logo {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #1a1a2e;
        margin-bottom: 8px;
    }
    .landing-logo .cite { color: #e85d26; }
    .landing-logo .sage { color: #1a1a2e; }
    .landing-tagline {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 8px;
    }
    .landing-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 32px;
    }

    /* Feature cards */
    .feature-cards {
        display: flex;
        gap: 16px;
        margin: 30px 0 10px 0;
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    .feature-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(91,95,199,0.1);
        border-color: #c7d2fe;
    }
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* Force all inputs to dark text — overrides Streamlit dark theme */
    .stApp input,
    .stApp input[type="text"],
    .stApp textarea,
    .stTextInput input,
    .stTextInput > div > div > input,
    [data-testid="stTextInput"] input,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input {
        font-size: 1.05rem !important;
        padding: 16px 24px !important;
        border-radius: 14px !important;
        border: 2px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
        caret-color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        transition: all 0.25s ease !important;
    }
    .stApp input:focus,
    .stApp input[type="text"]:focus,
    .stTextInput input:focus,
    [data-testid="stTextInput"] input:focus,
    [data-baseweb="input"] input:focus,
    [data-baseweb="base-input"] input:focus {
        border-color: #5b5fc7 !important;
        box-shadow: 0 4px 20px rgba(91,95,199,0.12) !important;
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    /* Placeholder text */
    .stApp input::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    /* Powered by */
    .powered-by {
        text-align: center;
        margin: 28px 0;
        font-size: 0.78rem;
        color: #94a3b8;
        animation: fadeIn 1s ease-out 0.4s both;
    }
    .tech-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 0 2px;
        font-weight: 500;
        color: #64748b;
        font-size: 0.75rem;
    }

    /* Result card */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 28px;
        margin: 16px 0;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.06); }

    /* Badges */
    .badges {
        display: flex; gap: 8px; align-items: center;
        margin: 12px 0;
        animation: fadeIn 0.4s ease-out;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-route { background: #ede9fe; color: #5b5fc7; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-yellow { background: #fef3c7; color: #92400e; }
    .badge-red { background: #fee2e2; color: #991b1b; }

    /* Trace */
    .trace-line {
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #475569;
        border-left: 3px solid #5b5fc7;
        margin-bottom: 4px;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
        animation: slideInLeft 0.3s ease-out;
    }

    /* Citation */
    .cite-item {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 10px 0; border-bottom: 1px solid #f1f5f9;
        animation: fadeIn 0.3s ease-out;
    }
    .cite-item:last-child { border-bottom: none; }
    .cite-num {
        background: #5b5fc7; color: white;
        width: 22px; height: 22px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
    }
    .cite-title { font-weight: 500; color: #1e293b; }
    .cite-page { color: #94a3b8; font-size: 0.78rem; }

    /* Paper card */
    .paper-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.2s;
        animation: fadeInUp 0.4s ease-out;
    }
    .paper-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); border-color: #c7d2fe; }
    .paper-title { font-weight: 600; font-size: 0.95rem; color: #1e293b; margin-bottom: 4px; }
    .paper-meta { font-size: 0.78rem; color: #64748b; }
    .paper-chips { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
    .paper-chip {
        background: #f1f5f9; border-radius: 4px; padding: 2px 8px;
        font-size: 0.72rem; color: #475569; font-weight: 500;
    }

    /* Similar paper card */
    .similar-card {
        background: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        animation: fadeInUp 0.4s ease-out;
    }
    .similar-title { font-weight: 600; font-size: 0.88rem; color: #1e293b; }
    .similar-meta { font-size: 0.76rem; color: #64748b; margin-top: 2px; }
    .similar-citations { color: #5b5fc7; font-weight: 600; font-size: 0.76rem; }

    /* Doc card */
    .doc-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #334155;
    }
    .doc-meta {
        font-size: 0.75rem;
        color: #5b5fc7;
        font-weight: 600;
        margin-bottom: 6px;
    }

    /* Step cards */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 28px 16px;
        text-align: center;
        height: 100%;
        transition: all 0.2s;
    }
    .step-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
    .step-num {
        background: linear-gradient(135deg, #5b5fc7, #818cf8);
        color: white; width: 40px; height: 40px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; margin-bottom: 14px; font-size: 1rem;
    }
    .step-title { font-weight: 600; font-size: 0.95rem; color: #1e293b; margin-bottom: 8px; }
    .step-desc { font-size: 0.82rem; color: #64748b; line-height: 1.5; }

    /* Metric card */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 28px 16px;
        text-align: center;
        transition: all 0.2s;
    }
    .metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
    .metric-val { font-size: 2.4rem; font-weight: 800; color: #5b5fc7; }
    .metric-label { font-size: 0.78rem; color: #64748b; margin-top: 4px; font-weight: 500; }

    /* Section title */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        margin: 24px 0 16px 0; letter-spacing: -0.02em;
    }

    /* Loading shimmer */
    .shimmer {
        background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
        height: 16px;
        margin-bottom: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #f1f5f9; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 500; color: #64748b; }
    .stTabs [aria-selected="true"] { color: #5b5fc7 !important; }

    /* Papers tab section backgrounds */
    .upload-section {
        background: #f0f4ff;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #dbe4ff;
    }
    .collection-section {
        background: #f8faf5;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e2e8d8;
    }
    .search-section {
        background: #fdf8f0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #f0e6d4;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "similar_papers" not in st.session_state:
    st.session_state.similar_papers = {}

# ---------------------------------------------------------------------------
# Auto-ingest demo papers
# ---------------------------------------------------------------------------
DEMO_PAPERS = {
    "federated_recsys_survey.pdf": "https://arxiv.org/pdf/2301.00767v1",
    "federated_learning_mcmahan.pdf": "https://arxiv.org/pdf/1602.05629v4",
    "matrix_factorization_recsys.pdf": "https://arxiv.org/pdf/2205.01708v1",
}


@st.cache_resource(show_spinner="Loading knowledge base...")
def ensure_demo_papers():
    if get_paper_count() > 0:
        return
    papers_dir = Path(__file__).parent.parent.parent / "data" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DEMO_PAPERS.items():
        pdf_path = papers_dir / filename
        if not pdf_path.exists():
            try:
                urllib.request.urlretrieve(url, str(pdf_path))
            except Exception:
                continue
        try:
            pages = load_pdf(str(pdf_path))
            chunks = chunk_documents(pages)
            ingest_documents(chunks)
        except Exception:
            continue


ensure_demo_papers()
n_papers = get_paper_count()
n_chunks = get_chunk_count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_similar_papers(title: str, abstract: str = "") -> list[dict]:
    """Search Semantic Scholar for papers similar to the given one."""
    query = title[:100]
    if abstract:
        query += " " + abstract[:100]
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        fields = "title,authors,year,citationCount,abstract,url"
        resp = httpx.get(url, params={"query": query, "limit": 8, "fields": fields}, timeout=10)
        if resp.status_code != 200:
            return []
        papers = resp.json().get("data", [])
        return [
            {
                "title": p.get("title", ""),
                "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:3]),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (p.get("abstract") or "")[:200],
                "url": p.get("url", ""),
                "paperId": p.get("paperId", ""),
            }
            for p in papers if p.get("title", "").lower() != title.lower()
        ][:6]
    except Exception:
        return []


def deep_paper_search(title: str) -> dict:
    """Find a paper by title and return its full network: citations, references, and similar."""
    BASE = "https://api.semanticscholar.org/graph/v1"
    result = {"paper": None, "citations": [], "references": [], "similar": []}

    try:
        # Step 1: Find the paper
        fields = "title,authors,year,citationCount,abstract,url,paperId"
        resp = httpx.get(f"{BASE}/paper/search", params={"query": title, "limit": 1, "fields": fields}, timeout=10)
        if resp.status_code != 200:
            return result
        data = resp.json().get("data", [])
        if not data:
            return result

        paper = data[0]
        paper_id = paper["paperId"]
        result["paper"] = {
            "title": paper.get("title", ""),
            "authors": ", ".join(a["name"] for a in (paper.get("authors") or [])[:5]),
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
            "abstract": (paper.get("abstract") or "")[:300],
            "url": paper.get("url", ""),
            "paperId": paper_id,
        }

        # Step 2: Get papers that cite this one
        cite_fields = "title,authors,year,citationCount,url"
        resp = httpx.get(f"{BASE}/paper/{paper_id}/citations",
                         params={"fields": cite_fields, "limit": 8}, timeout=10)
        if resp.status_code == 200:
            for c in resp.json().get("data", []):
                p = c.get("citingPaper", {})
                if p.get("title"):
                    result["citations"].append({
                        "title": p["title"],
                        "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:3]),
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                        "url": p.get("url", ""),
                    })

        # Step 3: Get papers this one references
        resp = httpx.get(f"{BASE}/paper/{paper_id}/references",
                         params={"fields": cite_fields, "limit": 8}, timeout=10)
        if resp.status_code == 200:
            for r in resp.json().get("data", []):
                p = r.get("citedPaper", {})
                if p.get("title"):
                    result["references"].append({
                        "title": p["title"],
                        "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:3]),
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                        "url": p.get("url", ""),
                    })

        # Step 4: Get recommended similar papers
        resp = httpx.post(f"https://api.semanticscholar.org/recommendations/v1/papers/",
                          json={"positivePaperIds": [paper_id]},
                          params={"fields": cite_fields, "limit": 6}, timeout=10)
        if resp.status_code == 200:
            for p in resp.json().get("recommendedPapers", []):
                if p.get("title"):
                    result["similar"].append({
                        "title": p["title"],
                        "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:3]),
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                        "url": p.get("url", ""),
                    })

    except Exception:
        pass

    return result


ROUTE_LABELS = {
    "vectorstore": "Local Papers", "web_search": "Web Search",
    "direct": "Knowledge", "compare": "Comparison",
    "literature_review": "Literature Review",
}


def g_badge(score):
    if score >= 0.8:
        return '<span class="badge badge-green">Grounded</span>'
    elif score >= 0.5:
        return '<span class="badge badge-yellow">Partial</span>'
    return '<span class="badge badge-red">Unverified</span>'


def render_result(answer, route, score, citations, trace, documents):
    """Render a search result with answer, badges, citations, and trace."""

    # Answer first, rendered as native Markdown inside a styled container
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

    # Horizontal badge row
    badges_html = '<div class="badges">'
    if route:
        badges_html += f'<span class="badge badge-route">{ROUTE_LABELS.get(route, route)}</span>'
    badges_html += g_badge(score)
    badges_html += f'<span style="font-size:0.73rem;color:#94a3b8;">{score:.0%} grounded</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

    # Citations and Agent Trace as two columns below
    col_cite, col_trace = st.columns(2)
    with col_cite:
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for i, c in enumerate(citations, 1):
                    pg = f" &middot; p.{c['page']}" if c.get("page") else ""
                    st.markdown(
                        f'<div class="cite-item"><div class="cite-num">{i}</div>'
                        f'<div><span class="cite-title">{c["title"]}</span>'
                        f'<span class="cite-page">{pg}</span></div></div>',
                        unsafe_allow_html=True)
        if documents:
            with st.expander(f"Sources ({len(documents)})", expanded=False):
                for doc in documents[:5]:
                    title = doc.get("title") or doc.get("source", "")
                    st.markdown(
                        f'<div class="doc-card"><div class="doc-meta">{title} &middot; '
                        f'p.{doc.get("page", "?")}</div>{doc["content"][:200]}...</div>',
                        unsafe_allow_html=True)
    with col_trace:
        if trace:
            with st.expander("Agent Trace", expanded=False):
                for t in trace:
                    st.markdown(f'<div class="trace-line">{t}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation — handle home clicks
# ---------------------------------------------------------------------------
if st.session_state.get("_go_home"):
    st.session_state._go_home = False
    st.session_state.messages = []
    st.session_state.last_result = None
    st.rerun()

has_results = len(st.session_state.messages) > 0

# ---------------------------------------------------------------------------
# Gradient banner (always visible)
# ---------------------------------------------------------------------------
if has_results:
    st.markdown(
        '<div class="gradient-banner">'
        '<a href="?_go_home=1" onclick="window.location.search=\'\';">'
        '<span style="color:#e85d26;">Cite</span><span style="color:#1a1a2e;">Sage</span></a>'
        '<span class="gradient-banner-tag">AI Research Copilot</span>'
        '</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="gradient-banner">'
        '<a href="#"><span style="color:#e85d26;">Cite</span><span style="color:#1a1a2e;">Sage</span></a>'
        '<span class="gradient-banner-tag">AI Research Copilot</span>'
        '</div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
if not has_results:
    st.markdown("""
    <div class="landing">
        <div class="landing-logo"><span class="cite">Cite</span><span class="sage">Sage</span></div>
        <div class="landing-tagline">Search research papers with AI-powered, cited answers</div>
        <div class="landing-sub">Upload papers, ask questions, get verified answers with real citations</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Search bar
# ---------------------------------------------------------------------------
if has_results:
    # Clickable logo in the banner handles going home.
    # A small home button for reliability via session state.
    col_home, col_search = st.columns([1, 10])
    with col_home:
        if st.button("Home", width='stretch', type="secondary"):
            st.session_state._go_home = True
            st.rerun()
    with col_search:
        query = st.text_input("s", placeholder="Ask a research question...",
                              label_visibility="collapsed", key="main_search")
else:
    query = st.text_input("s", placeholder="Ask a research question...",
                          label_visibility="collapsed", key="main_search")

    # Feature cards on landing
    st.markdown("""
    <div class="feature-cards">
        <div class="feature-card">
            <div class="feature-icon">&#128196;</div>
            <div class="feature-title">Upload Papers</div>
            <div class="feature-desc">Add your research PDFs. We read and index them automatically.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#128269;</div>
            <div class="feature-title">Ask Questions</div>
            <div class="feature-desc">Ask anything. Our AI agent finds the best answer across all your papers.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#9989;</div>
            <div class="feature-title">Get Cited Answers</div>
            <div class="feature-desc">Every answer cites exact papers and pages. Verified for accuracy.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="powered-by">
        Powered by
        <span class="tech-chip">LangGraph</span>
        <span class="tech-chip">MCP</span>
        <span class="tech-chip">ChromaDB</span>
        <span class="tech-chip">RAGAS</span>
        &nbsp;&middot;&nbsp; {n_papers} papers indexed
    </div>
    """, unsafe_allow_html=True)

pending = st.session_state.pop("_run_query", None)
active_query = query or pending

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_results, tab_explore, tab_eval, tab_how = st.tabs([
    "Search", "Papers", "Quality", "About"
])

# ===== TAB 1: SEARCH =====
with tab_results:
    # Previous messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"##### {msg['content']}")
        else:
            meta = msg.get("metadata", {})
            render_result(
                msg["content"], meta.get("route", ""),
                meta.get("groundedness_score", 0),
                meta.get("citations", []),
                meta.get("trace", []),
                meta.get("documents", []))
            st.markdown("---")

    # New query
    if active_query:
        st.session_state.messages.append({"role": "user", "content": active_query})
        st.markdown(f"##### {active_query}")

        with st.status("", expanded=True) as status:
            # Animated progress bar
            status.markdown(
                '<div class="search-loading">'
                '<div class="step-text">Searching papers and generating answer'
                '<span class="loading-dots"><span></span><span></span><span></span></span>'
                '</div>'
                '<div class="progress-bar-container"><div class="progress-bar"></div></div>'
                '</div>', unsafe_allow_html=True)

            collected_trace = []
            final = {}
            try:
                for node_name, update in stream_agent(active_query):
                    for t in update.get("trace", []):
                        collected_trace.append(t)
                        status.markdown(f'<div class="step-done">{t}</div>',
                                        unsafe_allow_html=True)
                    for key in ("generation", "citations", "groundedness_score", "documents", "route"):
                        if key in update:
                            final[key] = update[key]
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                final = {"generation": f"Error: {e}", "groundedness_score": 0}

        render_result(
            final.get("generation", ""), final.get("route", ""),
            final.get("groundedness_score", 0), final.get("citations", []),
            collected_trace, final.get("documents", []))

        st.session_state.messages.append({
            "role": "assistant", "content": final.get("generation", ""),
            "metadata": {
                "route": final.get("route", ""),
                "groundedness_score": final.get("groundedness_score", 0),
                "citations": final.get("citations", []),
                "trace": collected_trace,
                "documents": final.get("documents", [])},
        })
        st.session_state.last_result = final

    if not has_results and not active_query:
        st.markdown('<p style="text-align:center;color:#94a3b8;margin-top:40px;">'
                    'Type a question above or upload papers in the Papers tab</p>',
                    unsafe_allow_html=True)


# ===== TAB 2: PAPERS =====
with tab_explore:

    # Helper to render a paper card
    def _paper_link(p):
        url = p.get("url", "")
        return f' <a href="{url}" target="_blank" style="color:#e85d26;text-decoration:none;">[open]</a>' if url else ""

    def _render_paper_list(papers, label=""):
        if not papers:
            st.caption(f"No {label} found.")
            return
        for p in papers:
            authors = (p.get("authors") or "")[:60]
            year = f" ({p['year']})" if p.get("year") else ""
            cites = p.get("citations", 0)
            st.markdown(
                f'<div class="similar-card">'
                f'<div class="similar-title">{p["title"]}{_paper_link(p)}</div>'
                f'<div class="similar-meta">{authors}{year} &middot; {cites} citations</div>'
                f'</div>', unsafe_allow_html=True)

    # ---- Discover Papers ----
    st.markdown('<div class="section-title">Discover Related Papers</div>', unsafe_allow_html=True)
    st.caption("Enter a paper title to find its citations, references, and recommended similar papers.")

    discover_q = st.text_input("Paper title", placeholder='e.g., "Federated Learning for Mobile Keyboard Prediction"',
                               key="discover_q", label_visibility="collapsed")

    if discover_q and st.button("Discover", type="primary"):
        with st.status("", expanded=True) as status:
            status.markdown(
                '<div class="search-loading">'
                '<div class="step-text">Searching Semantic Scholar'
                '<span class="loading-dots"><span></span><span></span><span></span></span>'
                '</div>'
                '<div class="progress-bar-container"><div class="progress-bar"></div></div>'
                '</div>', unsafe_allow_html=True)
            network = deep_paper_search(discover_q)
            status.update(label="Done", state="complete")

        st.session_state.discover_result = network

    # Show discover results
    net = st.session_state.get("discover_result")
    if net and net.get("paper"):
        p = net["paper"]
        st.markdown(
            f'<div class="result-card">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#1a1a2e;margin-bottom:4px;">{p["title"]}</div>'
            f'<div style="font-size:0.82rem;color:#64748b;">{p["authors"]}</div>'
            f'<div style="font-size:0.82rem;color:#64748b;margin-top:2px;">'
            f'{p.get("year", "")} &middot; {p.get("citations", 0)} citations'
            f'{_paper_link(p)}</div>'
            f'<div style="font-size:0.85rem;color:#475569;margin-top:10px;line-height:1.6;">{p.get("abstract", "")}</div>'
            f'</div>', unsafe_allow_html=True)

        # Citation graph
        if net["citations"] or net["references"] or net["similar"]:
            from streamlit_agraph import agraph, Node, Edge, Config

            nodes = [Node(id="center", label=p["title"][:30] + "...", size=30, color="#e85d26",
                          font={"size": 11, "color": "#fff"}, shape="dot")]
            edges = []

            for i, c in enumerate(net["citations"][:6]):
                nid = f"cite_{i}"
                nodes.append(Node(id=nid, label=c["title"][:25] + "...", size=18, color="#5b5fc7",
                                  font={"size": 9}, shape="dot"))
                edges.append(Edge(source=nid, target="center", color="#c7d2fe", width=1.5))

            for i, r in enumerate(net["references"][:6]):
                nid = f"ref_{i}"
                nodes.append(Node(id=nid, label=r["title"][:25] + "...", size=18, color="#22c55e",
                                  font={"size": 9}, shape="dot"))
                edges.append(Edge(source="center", target=nid, color="#bbf7d0", width=1.5))

            for i, s in enumerate(net["similar"][:4]):
                nid = f"sim_{i}"
                nodes.append(Node(id=nid, label=s["title"][:25] + "...", size=16, color="#f59e0b",
                                  font={"size": 9}, shape="dot"))
                edges.append(Edge(source="center", target=nid, color="#fde68a", width=1, dashes=True))

            config = Config(width=800, height=350, directed=True, physics=True,
                            nodeHighlightBehavior=True, highlightColor="#e85d26")
            agraph(nodes=nodes, edges=edges, config=config)

            # Legend
            st.markdown(
                '<div style="font-size:0.75rem;color:#94a3b8;text-align:center;">'
                '<span style="color:#e85d26;">&#9679;</span> This paper &nbsp; '
                '<span style="color:#5b5fc7;">&#9679;</span> Cited by &nbsp; '
                '<span style="color:#22c55e;">&#9679;</span> References &nbsp; '
                '<span style="color:#f59e0b;">&#9679;</span> Similar'
                '</div>', unsafe_allow_html=True)

        # Three columns: cited by / references / similar
        tab_cited, tab_refs, tab_sim = st.tabs([
            f"Cited by ({len(net['citations'])})",
            f"References ({len(net['references'])})",
            f"Similar ({len(net['similar'])})",
        ])

        with tab_cited:
            _render_paper_list(net["citations"], "citing papers")
        with tab_refs:
            _render_paper_list(net["references"], "references")
        with tab_sim:
            _render_paper_list(net["similar"], "similar papers")

    st.markdown("---")

    # ---- Upload ----
    st.markdown('<div class="section-title">Upload a Paper</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False,
                                label_visibility="collapsed", key="paper_upload")

    if uploaded:
        if st.button("Add to Collection", type="primary"):
            tmp = Path("/tmp") / uploaded.name
            tmp.write_bytes(uploaded.getvalue())

            with st.status("", expanded=True) as status:
                status.markdown(
                    '<div class="search-loading">'
                    '<div class="step-text">Processing paper'
                    '<span class="loading-dots"><span></span><span></span><span></span></span>'
                    '</div></div>', unsafe_allow_html=True)
                pages = load_pdf(str(tmp))
                status.markdown(f'<div class="step-done">Read {len(pages)} pages</div>',
                                unsafe_allow_html=True)
                chunks = chunk_documents(pages)
                n_new = ingest_documents(chunks)
                status.markdown(f'<div class="step-done">Indexed {n_new} sections</div>',
                                unsafe_allow_html=True)
                status.update(label="Done", state="complete")

            import fitz
            doc = fitz.open(str(tmp))
            meta = extract_metadata_from_first_page(doc, uploaded.name)
            doc.close()
            st.success(f"**{meta.title or uploaded.name}** added to collection")

    # ---- Your Papers ----
    st.markdown("---")
    papers_info = get_ingested_papers_info()
    if papers_info:
        st.markdown(f'<div class="section-title">Your Collection ({len(papers_info)})</div>',
                    unsafe_allow_html=True)
        for p in papers_info:
            title = p["title"] if p["title"] and p["title"] != p["source"] else p["source"]
            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-title">{title}</div>'
                f'<div class="paper-meta">{p["total_pages"]} pages</div>'
                f'</div>', unsafe_allow_html=True)


# ===== TAB 3: QUALITY =====
with tab_eval:
    st.markdown('<div class="section-title">Quality Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:0.85rem;color:#64748b;">'
                '30-question evaluation across 5 categories</span>', unsafe_allow_html=True)
    st.markdown("")

    eval_dir = Path(__file__).parent.parent.parent / "evals" / "experiments"
    eval_files = sorted(eval_dir.glob("eval_*.json"), reverse=True)

    if eval_files:
        selected = st.selectbox("Select run", eval_files, format_func=lambda p: p.stem)
        report = json.loads(selected.read_text())
        agg = report.get("aggregate_metrics", {})

        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("Groundedness", "groundedness_score"),
            ("Citation Acc.", "citation_accuracy"),
            ("Routing Acc.", "routing_accuracy"),
            ("Completeness", "answer_completeness"),
            ("Citation Rate", "citation_rate"),
        ]):
            val = agg.get(key, 0)
            col.markdown(
                f'<div class="metric-card"><div class="metric-val">{val:.0%}</div>'
                f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"**{report.get('successful', 0)}/{report.get('total_questions', 0)}** passed "
                    f"&middot; Avg latency: **{report.get('avg_latency_seconds', 0):.1f}s**")

        breakdown = report.get("category_breakdown", {})
        if breakdown:
            import pandas as pd
            st.dataframe(pd.DataFrame(breakdown).T, width='stretch')
    else:
        st.info("No evaluation results yet.")


# ===== TAB 4: ABOUT =====
with tab_how:
    st.markdown('<div class="section-title">How CiteSage Works</div>', unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(5)
    steps = [
        ("1", "Route", "Classifies your query into one of 5 types"),
        ("2", "Retrieve", "Searches papers + ArXiv + Semantic Scholar"),
        ("3", "Grade", "Filters out irrelevant results"),
        ("4", "Generate", "Creates answer with paper citations"),
        ("5", "Verify", "Detects hallucinations, retries if needed"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Tech Stack**

        | Component | Technology |
        |---|---|
        | Agent Orchestration | LangGraph |
        | RAG | LangChain + ChromaDB |
        | External Search | Custom MCP Servers |
        | Evaluation | Custom + RAGAS |
        | LLM | OpenAI / Anthropic / Ollama |
        """)
    with col2:
        st.markdown("""
        **Agent Flow**
        ```
        Query
          |-> Router -> [Papers | Web | Direct]
                            |
                        Grade docs
                            |
                     Generate (cited)
                            |
                     Hallucination check
                          |       |
                       [Pass]  [Retry]
                          |
                       Answer
        ```
        """)

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/naeemshb/citesage) &middot; "
                "Built with LangGraph + MCP + ChromaDB + RAGAS")
