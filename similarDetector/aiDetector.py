"""공공기관 재정사업 유사·중복 탐지 웹앱 (Streamlit).

신규 사업을 벡터로 변환 후 Supabase pgvector로 유사 사업 검색,
GPT-4o로 유사·중복 분석합니다.

실행:
    streamlit run aiDetector.py

환경변수 (.env):
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_MODEL = "text-embedding-3-small"
SEARCH_TOP_K = 10        # 벡터 검색 후보 수
SEARCH_THRESHOLD = 0.5   # 코사인 유사도 최소값
GPT_TOP_K = 5            # GPT에 전달할 최대 사업 수

FIELD_OPTIONS = [
    "공공질서및안전",
    "과학기술",
    "교육",
    "교통및물류",
    "국방",
    "국토및지역개발",
    "농림수산",
    "문화및관광",
    "보건",
    "사회복지",
    "산업·중소기업및에너지",
    "예비비",
    "일반·지방행정",
    "통신",
    "통일·외교",
    "환경",
    "기타",
]

NAVY = "#003366"
GNB_NAVY = "#1a3a6b"
PAGE_BG = "#F0F2F6"
INPUT_BG = "#F0F2F6"
INPUT_BORDER = "#CBD5E0"
BREADCRUMB_BG = "#E8EDF5"
OPENFISCAL_URL = "https://www.openfiscaldata.go.kr/"
PAGE_INPUT = "input"
PAGE_RESULTS = "results"

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
.block-container {padding-top: 0 !important; padding-bottom: 0 !important; max-width: 100% !important;}
.main .block-container {padding-left: 0 !important; padding-right: 0 !important;}
</style>
"""

SYSTEM_PROMPT = """당신은 재정사업 유사·중복을 검토하는 예산 분석 전문가입니다.
신규 사업과 기존 사업 목록을 비교하여 아래 JSON 형식으로만 응답하세요.
{
  "similar_projects": [
    {
      "id": "기존사업 id",
      "similarity_score": 0~100,
      "risk_level": "높음|중간|낮음",
      "similar_points": "유사한 점 설명 (사업 목적, 대상, 방법 등 구체적으로)",
      "different_points": "다른 점 설명 (차별성, 보완 관계 등 구체적으로)",
      "review_comment": "담당자 검토 의견 (중복 여부 판단 근거, 예산 낭비 가능성, 부처 간 협의 필요 여부, 사업 조정 방향 등을 2~3문장으로 구체적으로 작성)"
    }
  ],
  "overall_opinion": "종합 의견 (전체적인 중복 위험도 평가, 주요 중복 사업과의 관계, 예산 효율화를 위한 정책 제언을 3~5문장으로 작성)"
}

규칙:
- similarity_score 70 이상인 사업만 similar_projects에 포함 (최대 5건, 점수 내림차순)
- 해당 없으면 similar_projects는 빈 배열 []
- risk_level: 80 이상 높음, 60~79 중간, 70 미만은 목록에서 제외
- 검토 의견은 단순 요약이 아닌 실제 예산 담당자 관점에서 구체적으로 작성
- 부처가 다른 경우 부처 간 협의 또는 통합 필요성 언급
- 예산 규모 차이가 있으면 그 의미도 분석
"""

RISK_COLORS = {
    "높음": {"bg": "#fef2f2", "border": "#fca5a5", "bar": "#ef4444", "badge": "#991b1b"},
    "중간": {"bg": "#fffbeb", "border": "#fcd34d", "bar": "#f59e0b", "badge": "#92400e"},
    "낮음": {"bg": "#ecfdf5", "border": "#6ee7b7", "bar": "#10b981", "badge": "#065f46"},
}


# ── 환경변수 ──────────────────────────────────────────────────────────────────

def get_env_keys() -> dict[str, str]:
    return {
        "openai": os.getenv("OPENAI_API_KEY", "").strip(),
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
    }


def missing_env_message(keys: dict[str, str]) -> str | None:
    missing: list[str] = []
    if not keys["openai"]:
        missing.append("OPENAI_API_KEY")
    if not keys["supabase_url"]:
        missing.append("SUPABASE_URL")
    if not keys["supabase_key"]:
        missing.append("SUPABASE_ANON_KEY")
    if not missing:
        return None
    return "다음 환경변수가 `.env`에 설정되어 있지 않습니다: " + ", ".join(missing)


def get_supabase_client(keys: dict[str, str]) -> Client | None:
    if not keys["supabase_url"] or not keys["supabase_key"]:
        return None
    return create_client(keys["supabase_url"], keys["supabase_key"])


# ── 프로젝트 수 조회 ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_project_count(supabase_url: str, supabase_key: str) -> int:
    try:
        client = create_client(supabase_url, supabase_key)
        response = client.table("projects").select("id", count="exact").execute()
        return response.count or 0
    except Exception:  # noqa: BLE001
        return 0


# ── 임베딩 ────────────────────────────────────────────────────────────────────

def embed_text(openai_key: str, text: str) -> list[float]:
    """텍스트 → 벡터 변환."""
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ── 벡터 검색 ─────────────────────────────────────────────────────────────────

def search_similar_projects(
    supabase: Client,
    embedding: list[float],
    top_k: int = SEARCH_TOP_K,
    threshold: float = SEARCH_THRESHOLD,
) -> list[dict[str, Any]]:
    """pgvector 코사인 유사도 검색."""
    try:
        response = supabase.rpc(
            "match_projects",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": top_k,
            },
        ).execute()
        return response.data or []
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"벡터 검색 실패: {exc}") from exc


# ── GPT 분석 ──────────────────────────────────────────────────────────────────

def analyze_similar_projects(
    openai_key: str,
    supabase: Client,
    project_name: str,
    ministry: str,
    field: str,
    overview: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    1. 신규 사업 임베딩
    2. Supabase 벡터 검색
    3. GPT-4o 분석
    반환: (gpt_result, vector_search_results)
    """
    # 1. 임베딩 텍스트 조합
    query_text = f"{field} {project_name} {overview}"
    embedding = embed_text(openai_key, query_text)

    # 2. 벡터 검색
    vector_results = search_similar_projects(supabase, embedding)

    if not vector_results:
        return {
            "similar_projects": [],
            "overall_opinion": "벡터 검색 결과 유사한 기존 사업이 없습니다.",
        }, []

    # 3. GPT에 전달할 사업 목록 (상위 GPT_TOP_K개)
    gpt_candidates = [
        {
            "id": r["id"],
            "사업명": r.get("project_name", ""),
            "부처명": r.get("ministry", ""),
            "분야": r.get("category", ""),
            "사업개요": r.get("overview", ""),
            "벡터유사도": round(float(r.get("similarity", 0)) * 100, 1),
        }
        for r in vector_results[:GPT_TOP_K]
    ]

    user_payload = {
        "new_project": {
            "사업명": project_name,
            "부처명": ministry,
            "분야": field,
            "사업개요": overview,
        },
        "existing_projects": gpt_candidates,
    }

    completion = OpenAI(api_key=openai_key).chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ],
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("GPT-4o 응답이 비어 있습니다.")

    result: dict[str, Any] = json.loads(content)
    if "overall_opinion" not in result and "summary" in result:
        result["overall_opinion"] = result["summary"]

    return result, vector_results


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def normalize_risk(risk: str | None) -> str:
    r = (risk or "중간").strip()
    if "높" in r or r.lower() == "high":
        return "높음"
    if "낮" in r or r.lower() == "low":
        return "낮음"
    return "중간"


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_review_copy_text(
    result: dict[str, Any],
    form: dict[str, str],
    ref_map: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "[재정사업 유사·중복 검토 의견]",
        "",
        "■ 신규 사업",
        f"- 사업명: {form.get('name', '-')}",
        f"- 부처: {form.get('ministry', '-')}",
        f"- 분야: {form.get('field', '-')}",
        f"- 개요: {form.get('overview', '-')}",
        "",
        "■ 종합 의견",
        result.get("overall_opinion") or "(없음)",
        "",
    ]
    similar = result.get("similar_projects") or []
    if similar:
        lines.append("■ 유사·중복 검토 사업")
        for i, item in enumerate(similar, 1):
            ref = ref_map.get(str(item.get("id", "")), {})
            title = ref.get("project_name") or item.get("id", "-")
            lines.extend([
                "",
                f"{i}. {title} (유사도 {item.get('similarity_score', '-')}%, "
                f"위험 {item.get('risk_level', '중간')})",
                f"   - 유사: {item.get('similar_points', '-')}",
                f"   - 차이: {item.get('different_points', '-')}",
                f"   - 검토: {item.get('review_comment', '-')}",
            ])
    return "\n".join(lines)


# ── 스타일 ────────────────────────────────────────────────────────────────────

def _inject_global_styles() -> None:
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
  }}
  .stApp {{
    background-color: {PAGE_BG} !important;
    color-scheme: light;
    --primary-color: {NAVY} !important;
    --background-color: {PAGE_BG};
    --secondary-background-color: #ffffff;
  }}
  .govt-gnb {{
    height: 48px; background: {GNB_NAVY}; display: flex; align-items: center;
    justify-content: space-between; padding: 0 1.5rem; color: #fff;
    font-size: 0.8125rem; font-weight: 500;
  }}
  .govt-gnb a {{ color: #fff; text-decoration: none; }}
  .govt-gnb a:hover {{ text-decoration: underline; }}
  .govt-main-header {{
    min-height: 100px; background: {NAVY}; display: flex; align-items: center;
    padding: 0.75rem 1.75rem 1.15rem; gap: 1.25rem;
  }}
  .govt-logo-wrap {{ display: flex; align-items: center; flex-shrink: 0; }}
  .govt-logo-badge {{
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.97); border: 2px solid rgba(255,255,255,0.9);
    border-radius: 28px; padding: 6px 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.18);
  }}
  .govt-logo-badge img {{ height: 44px; width: auto; max-width: 220px; display: block; }}
  .govt-header-center {{ flex: 1; text-align: center; }}
  .govt-header-center h1 {{
    margin: 0; color: #fff; font-size: 26px; font-weight: 700; letter-spacing: -0.03em;
  }}
  .govt-header-center .govt-subtitle {{
    margin: 0.35rem 0 0.65rem; color: rgba(255,255,255,0.7); font-size: 14px; line-height: 1.5;
  }}
  .govt-header-spacer {{ width: 140px; flex-shrink: 0; }}
  .govt-breadcrumb {{
    height: 36px; background: {BREADCRUMB_BG}; border-bottom: 1px solid #CBD5E0;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.5rem; font-size: 0.8125rem; color: #4A5568;
  }}
  .govt-breadcrumb-path strong {{ color: #2D3748; font-weight: 600; }}
  .govt-db-status {{ font-size: 0.75rem; color: #4A5568; }}
  .main .block-container {{ padding: 1.5rem 1.25rem 3rem !important; }}
  .st-key-govt-content-panel {{ max-width: 900px; margin: 0.25rem auto 0 !important; width: 100% !important; }}
  .st-key-govt-content-panel,
  .st-key-govt-content-panel [data-testid="stVerticalBlockBorderWrapper"],
  .st-key-govt-content-panel > div {{
    background: #ffffff !important;
    border: 1px solid #B8C4D4 !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 24px rgba(0, 51, 102, 0.12) !important;
    padding: 1.75rem 2rem 2rem !important;
  }}
  .st-key-govt-content-panel [data-testid="stVerticalBlock"],
  .st-key-govt-content-panel [data-testid="stVerticalBlock"] > div {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
  }}
  .govt-section-title {{
    display: flex; align-items: center; gap: 0.65rem; margin: 0 0 1.25rem;
    font-size: 20px; font-weight: 700; color: {NAVY};
  }}
  .govt-section-title .accent-bar {{ width: 4px; height: 1.35rem; background: {NAVY}; border-radius: 1px; }}
  .govt-field-label {{ display: block; color: {NAVY}; font-weight: 700; font-size: 14px; margin: 0 0 0.4rem; }}
  .govt-field-label .req {{ color: #E53E3E; }}
  .st-key-govt-content-panel div[data-testid="stForm"] {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important; margin-top: 0 !important;
  }}
  .st-key-govt-content-panel div[data-testid="stForm"] input,
  .st-key-govt-content-panel div[data-testid="stForm"] textarea,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="input"],
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="textarea"],
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="select"],
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="input"] input,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="textarea"] textarea,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="input"] > div,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="textarea"] > div,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important; background: {INPUT_BG} !important;
    color: #1a1a2e !important; -webkit-text-fill-color: #1a1a2e !important;
    border: 1px solid {INPUT_BORDER} !important; border-radius: 4px !important;
    color-scheme: light !important;
  }}
  .st-key-govt-content-panel div[data-testid="stForm"] input::placeholder,
  .st-key-govt-content-panel div[data-testid="stForm"] textarea::placeholder {{
    color: #64748b !important; -webkit-text-fill-color: #64748b !important; opacity: 1 !important;
  }}
  .st-key-govt-content-panel div[data-testid="stForm"] input:focus,
  .st-key-govt-content-panel div[data-testid="stForm"] textarea:focus,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="input"]:focus-within,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="textarea"]:focus-within,
  .st-key-govt-content-panel div[data-testid="stForm"] [data-baseweb="select"]:focus-within {{
    background-color: {INPUT_BG} !important; border: 2px solid {NAVY} !important;
    box-shadow: none !important; outline: none !important;
  }}
  .stApp .main button,
  .stApp .main button[kind="primary"],
  .stApp .main button[kind="secondary"],
  .stApp .main [data-testid="stFormSubmitButton"],
  .stApp .main [data-testid="stBaseButton-primary"],
  .stApp .main [data-testid="stBaseButton-secondary"],
  .stApp .main div.stButton > button,
  .stApp .main .stFormSubmitButton button,
  .stApp .main [data-testid="stDownloadButton"] button,
  .stApp .st-key-btn_go_back button,
  .stApp .st-key-btn_copy_text button,
  .stApp .st-key-btn_download_text button {{
    background-color: {NAVY} !important; background: {NAVY} !important;
    color: #ffffff !important; border: 1px solid {NAVY} !important;
    opacity: 1 !important; font-weight: 600 !important;
    border-radius: 4px !important; box-shadow: none !important; filter: none !important;
  }}
  .stApp .main button:hover,
  .stApp .main [data-testid="stButton"] button:hover,
  .stApp .main [data-testid="stDownloadButton"] button:hover {{
    background-color: #004499 !important; background: #004499 !important;
    color: #ffffff !important; border-color: #004499 !important;
    opacity: 1 !important; box-shadow: none !important; filter: none !important;
  }}
  .stApp .main button:disabled {{
    background-color: {NAVY} !important; border-color: {NAVY} !important;
    opacity: 0.55 !important; color: #ffffff !important;
  }}
  .stApp .main [data-testid="stFormSubmitButton"] button {{
    width: 100% !important; min-height: 52px !important;
    font-size: 16px !important; font-weight: 700 !important;
  }}
  .stApp [data-testid="stFormSubmitButton"] > button,
  .stApp [data-testid="stForm"] .stFormSubmitButton > button {{
    background-color: {NAVY} !important; background: {NAVY} !important;
    color: #ffffff !important; border-color: {NAVY} !important;
    opacity: 1 !important; box-shadow: none !important;
  }}
  .summary-box {{
    background: #fff; border: 1px solid #d1d5db; border-left: 4px solid {NAVY};
    border-radius: 4px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
  }}
  .summary-box h3 {{ margin: 0 0 0.75rem; color: {NAVY}; font-size: 1rem; font-weight: 600; }}
  .summary-row {{ font-size: 0.9rem; color: #4b5563; margin: 0.25rem 0; }}
  .summary-row strong {{ color: #111827; }}
  .govt-section-heading {{
    color: {NAVY}; font-size: 1.125rem; font-weight: 700;
    margin: 1.75rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #E2E8F0;
  }}
  .opinion-box {{
    background: #EEF2FF; border: 1px solid #C5CDE8; border-radius: 8px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; color: #1A202C;
    line-height: 1.7; font-size: 0.9375rem; display: flex; gap: 0.75rem;
  }}
  .result-card {{
    background: #fff; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 1.35rem 1.5rem; margin-bottom: 1.25rem;
    box-shadow: 0 2px 10px rgba(0,51,102,0.06);
  }}
  .result-card-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem; }}
  .result-card-title {{ color: {NAVY}; font-size: 1.0625rem; font-weight: 700; }}
  .risk-badge {{ padding: 0.25rem 0.7rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }}
  .card-meta {{ font-size: 0.8rem; color: #718096; margin: 0.35rem 0 0.5rem; }}
  .sim-bar-wrap {{ margin: 0.75rem 0 1rem; }}
  .sim-bar-label {{ display: flex; justify-content: space-between; font-size: 0.8125rem; color: #4A5568; margin-bottom: 0.35rem; }}
  .sim-bar-track {{ height: 10px; background: #E2E8F0; border-radius: 4px; overflow: hidden; }}
  .sim-bar-fill {{ height: 100%; background: {NAVY}; border-radius: 4px; }}
  .compare-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 640px) {{ .compare-cols {{ grid-template-columns: 1fr; }} }}
  .compare-box {{ background: #F7FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1rem; }}
  .compare-box h4 {{ margin: 0 0 0.5rem; font-size: 0.8125rem; font-weight: 700; color: {NAVY}; }}
  .compare-box p {{ margin: 0; font-size: 0.875rem; color: #2D3748; line-height: 1.6; }}
  .review-box-yellow {{
    background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 6px;
    padding: 0.85rem 1rem; margin-top: 1rem; font-size: 0.875rem; color: #744210; line-height: 1.6;
  }}
  .review-box-yellow strong {{ color: #92400E; display: block; margin-bottom: 0.35rem; }}
  .vector-result-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid #E2E8F0; font-size: 0.875rem;
  }}
  .card-path {{
    font-size: 0.8rem; color: #4A5568; margin: 0.25rem 0 0.5rem;
  }}
  .vector-result-row:last-child {{ border-bottom: none; }}
  .vector-score {{ color: {NAVY}; font-weight: 700; font-size: 0.8125rem; }}
  .govt-footer {{
    min-height: 60px; background: {GNB_NAVY}; color: rgba(255,255,255,0.92);
    display: flex; align-items: center; justify-content: center; text-align: center;
    font-size: 0.8125rem; padding: 0.75rem 1rem; margin-top: 2rem;
  }}
  .govt-status-strip {{
    background: #F7F8FA; border: 1px solid #E2E8F0; border-radius: 6px;
    padding: 0.65rem 1rem; margin-bottom: 1.25rem; font-size: 0.8125rem; color: #4A5568;
  }}
  .govt-analyzing-msg {{
    margin: 0.85rem 0 0; padding: 0; color: #4A5568 !important;
    font-size: 0.9375rem; font-weight: 500; text-align: center;
    background: none !important; border: none !important; box-shadow: none !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── 헤더/푸터 ─────────────────────────────────────────────────────────────────

def _logo_data_uri() -> str:
    if not LOGO_PATH.is_file():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_header_logo_html() -> str:
    logo_src = _logo_data_uri()
    if logo_src:
        return (
            f'<div class="govt-logo-wrap">'
            f'<div class="govt-logo-badge">'
            f'<img src="{logo_src}" alt="기획예산처" />'
            f"</div></div>"
        )
    return (
        '<div class="govt-logo-wrap">'
        '<div class="govt-logo-badge" style="font-size:18px;font-weight:700;color:#003366;">'
        "기획예산처</div></div>"
    )


def _render_site_chrome(*, breadcrumb_current: str, project_count: int) -> None:
    st.markdown(
        f"""
<div class="govt-gnb">
  <span>기획예산처</span>
  <a href="{OPENFISCAL_URL}" target="_blank" rel="noopener noreferrer">열린재정 바로가기</a>
</div>
<div class="govt-main-header">
  {_render_header_logo_html()}
  <div class="govt-header-center">
    <h1>재정사업 유사·중복 탐지 시스템</h1>
    <p class="govt-subtitle">AI 기반 재정사업 중복 투자 사전 검토 도구</p>
  </div>
  <div class="govt-header-spacer" aria-hidden="true"></div>
</div>
<div class="govt-breadcrumb">
  <span class="govt-breadcrumb-path">홈 &gt; 재정사업 관리 &gt; <strong>{_html_escape(breadcrumb_current)}</strong></span>
  <span class="govt-db-status">기존 사업 DB: {project_count}건</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        """
<div class="govt-footer">
  ⓒ 2026 기획예산처 | 열린재정 정보공개시스템
</div>
""",
        unsafe_allow_html=True,
    )


def _render_section_title(title: str) -> None:
    st.markdown(
        f"""
<div class="govt-section-title">
  <span class="accent-bar"></span>
  <span>{_html_escape(title)}</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ── 결과 카드 ─────────────────────────────────────────────────────────────────

def _render_similarity_bar_html(score: int | float) -> str:
    pct = max(0, min(100, int(score or 0)))
    return f"""
<div class="sim-bar-wrap">
  <div class="sim-bar-label"><span>유사도</span><strong>{pct}%</strong></div>
  <div class="sim-bar-track"><div class="sim-bar-fill" style="width:{pct}%;"></div></div>
</div>
"""


def _render_result_card_html(item: dict[str, Any], ref: dict[str, Any]) -> str:
    risk = normalize_risk(str(item.get("risk_level", "")))
    colors = RISK_COLORS.get(risk, RISK_COLORS["중간"])
    title = _html_escape(ref.get("project_name") or item.get("id", "-"))
    ministry = _html_escape(ref.get("ministry") or "-")
    category = _html_escape(ref.get("category") or "-")
    vector_sim = ref.get("similarity")
    vector_sim_str = f" · 벡터유사도 {round(float(vector_sim)*100, 1)}%" if vector_sim else ""
    pid = _html_escape(item.get("id", ""))
    similar = _html_escape(item.get("similar_points", "-"))
    different = _html_escape(item.get("different_points", "-"))
    review = _html_escape(item.get("review_comment", "-"))
    bar = _render_similarity_bar_html(item.get("similarity_score", 0))

    fiscal_year = ref.get("fiscal_year") or "-"
    budget = ref.get("budget_100m_krw")
    budget_str = f"{budget:,}억원" if budget else "-"
    path_parts = [ref.get("fld_nm"), ref.get("sect_nm"), ref.get("pgm_nm")]
    path_str = " > ".join(p for p in path_parts if p)

    return f"""
<div class="result-card" style="border-top:3px solid {colors['border']};">
  <div class="result-card-header">
    <div class="result-card-title">{title}</div>
    <span class="risk-badge" style="background:{colors['bg']};color:{colors['badge']};border:1px solid {colors['border']};">
      위험도 {risk}
    </span>
  </div>
  <div class="card-meta">{ministry} · {category}{vector_sim_str}</div>
  {"<div class='card-path'>📂 " + _html_escape(path_str) + "</div>" if path_str else ""}
  <div class="card-meta">📅 {fiscal_year}년 · 💰 {budget_str}</div>
  <div class="card-meta">🔗 출처: <a href="https://www.openfiscaldata.go.kr" target="_blank" style="color:{NAVY};">열린재정 OpenAPI</a></div>
  {bar}
  <div class="compare-cols">
    <div class="compare-box"><h4>유사한 점</h4><p>{similar}</p></div>
    <div class="compare-box"><h4>다른 점</h4><p>{different}</p></div>
  </div>
  <div class="review-box-yellow"><strong>담당자 검토 의견</strong>{review}</div>
</div>
"""


# ── 세션 ──────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "page": PAGE_INPUT,
        "analysis_result": None,
        "vector_results": [],
        "last_form": {},
        "show_copy_area": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 입력 페이지 ───────────────────────────────────────────────────────────────

def _render_input_page(
    env_keys: dict[str, str],
    env_error: str | None,
    supabase: Client | None,
) -> None:
    _render_section_title("신규 재정사업 정보 입력")

    with st.form("new_project_form", clear_on_submit=False):
        st.markdown(
            '<label class="govt-field-label">사업명<span class="req"> *</span></label>',
            unsafe_allow_html=True,
        )
        project_name = st.text_input(
            "사업명", placeholder="신규 사업명을 입력하세요", label_visibility="collapsed",
        )
        st.markdown('<label class="govt-field-label">부처명</label>', unsafe_allow_html=True)
        ministry = st.text_input(
            "부처명", placeholder="예: 기획예산처", label_visibility="collapsed",
        )
        st.markdown('<label class="govt-field-label">분야</label>', unsafe_allow_html=True)
        field = st.selectbox("분야", FIELD_OPTIONS, index=0, label_visibility="collapsed")
        st.markdown(
            '<label class="govt-field-label">사업개요<span class="req"> *</span></label>',
            unsafe_allow_html=True,
        )
        overview = st.text_area(
            "사업개요", height=200,
            placeholder="사업 목적, 주요 추진 내용, 기대 효과 등을 기술하세요.",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "🔍 유사·중복 분석 시작", type="primary",
            use_container_width=True, disabled=bool(env_error),
        )

    if submitted:
        if not project_name.strip() or not overview.strip():
            st.error("사업명과 사업개요는 필수 입력 항목입니다.")
            return
        if supabase is None:
            st.error("Supabase 연결 실패. 환경변수를 확인하세요.")
            return

        st.session_state.last_form = {
            "name": project_name.strip(),
            "ministry": ministry.strip(),
            "field": field.strip(),
            "overview": overview.strip(),
        }
        st.markdown(
            '<p class="govt-analyzing-msg">AI가 유사 사업을 분석 중입니다...</p>',
            unsafe_allow_html=True,
        )
        try:
            result, vector_results = analyze_similar_projects(
                env_keys["openai"],
                supabase,
                st.session_state.last_form["name"],
                st.session_state.last_form["ministry"],
                st.session_state.last_form["field"],
                st.session_state.last_form["overview"],
            )
            st.session_state.analysis_result = result
            st.session_state.vector_results = vector_results
            st.session_state.page = PAGE_RESULTS
            st.session_state.show_copy_area = False
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state.analysis_result = None
            st.error(f"분석 중 오류가 발생했습니다: {exc}")


# ── 결과 페이지 ───────────────────────────────────────────────────────────────

def _render_results_page(ref_map: dict[str, dict[str, Any]]) -> None:
    result = st.session_state.analysis_result
    vector_results = st.session_state.vector_results
    form = st.session_state.last_form

    if not result:
        st.session_state.page = PAGE_INPUT
        st.rerun()
        return

    if st.button("← 이전으로 돌아가기", type="secondary", key="btn_go_back"):
        st.session_state.page = PAGE_INPUT
        st.session_state.show_copy_area = False
        st.rerun()

    # 분석 대상 사업 요약
    st.markdown(
        f"""
<div style="background:#fff;border:1px solid #d1d5db;border-left:4px solid {NAVY};border-radius:4px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
  <p style="margin:0 0 0.75rem;color:{NAVY};font-size:1rem;font-weight:600;">분석 대상 사업</p>
  <p style="font-size:0.9rem;color:#4b5563;margin:0.25rem 0;"><strong style="color:#111827;">사업명</strong> {_html_escape(form.get('name', '-'))}</p>
  <p style="font-size:0.9rem;color:#4b5563;margin:0.25rem 0;"><strong style="color:#111827;">부처명</strong> {_html_escape(form.get('ministry') or '-')}</p>
  <p style="font-size:0.9rem;color:#4b5563;margin:0.25rem 0;"><strong style="color:#111827;">분야</strong> {_html_escape(form.get('field') or '-')}</p>
  <p style="font-size:0.9rem;color:#4b5563;margin:0.25rem 0;"><strong style="color:#111827;">사업개요</strong> {_html_escape(form.get('overview', '-'))}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # 벡터 검색 결과 + 종합 의견을 하나의 카드로
    opinion = result.get("overall_opinion") or result.get("summary")
    vector_rows_html = ""
    if vector_results:
        vector_rows_html = "".join(
            f"""
<div class="vector-result-row">
  <span>{_html_escape(r.get('project_name', '-'))} · {_html_escape(r.get('ministry', '-'))}</span>
  <span class="vector-score">{round(float(r.get('similarity', 0)) * 100, 1)}%</span>
</div>
"""
            for r in vector_results
        )

    st.markdown(
        f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;box-shadow:0 2px 10px rgba(0,51,102,0.06);">
  
  <!-- 벡터 검색 결과 -->
  <p style="color:{NAVY};font-size:1.125rem;font-weight:700;margin:0 0 1rem;padding-bottom:0.5rem;border-bottom:2px solid #E2E8F0;">
    벡터 검색 결과 (상위 {len(vector_results)}건)
  </p>
  {vector_rows_html}

  <!-- 종합 의견 -->
  <p style="color:{NAVY};font-size:1.125rem;font-weight:700;margin:1.25rem 0 0.75rem;padding-bottom:0.5rem;border-bottom:2px solid #E2E8F0;">
    종합 의견
  </p>
  <div style="background:#EEF2FF;border:1px solid #C5CDE8;border-radius:8px;padding:1rem 1.25rem;display:flex;gap:0.75rem;color:#1A202C;line-height:1.7;font-size:0.9375rem;">
    <span>📋</span>
    <div>{_html_escape(opinion or "종합 의견이 없습니다.")}</div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # 유사 사업 검토 결과
    similar: list[dict[str, Any]] = result.get("similar_projects") or []
    cards_html = "".join(
        _render_result_card_html(item, ref_map.get(str(item.get("id", "")), {}))
        for item in similar
    )

    st.markdown(
        f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;box-shadow:0 2px 10px rgba(0,51,102,0.06);">
  <p style="color:{NAVY};font-size:1.125rem;font-weight:700;margin:0 0 1rem;padding-bottom:0.5rem;border-bottom:2px solid #E2E8F0;">
    유사 사업 검토 결과 ({len(similar)}건)
  </p>
  {cards_html if cards_html else '<p style="color:#4A5568;font-size:0.9rem;">유사·중복 가능성이 높은 기존 사업이 발견되지 않았습니다.</p>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # 다운로드
    copy_text = build_review_copy_text(result, form, ref_map)
    if st.button("전체 결과 텍스트 복사", type="secondary", key="btn_copy_text", use_container_width=True):
        st.session_state.show_copy_area = True
    st.download_button(
        "텍스트 파일 저장", data=copy_text,
        file_name="유사중복_검토결과.txt", mime="text/plain",
        type="secondary", key="btn_download_text", use_container_width=True,
    )
    if st.session_state.show_copy_area:
        st.caption("아래 전체를 선택(Ctrl+A) 후 복사(Ctrl+C)하세요.")
        st.text_area("검토 결과 전문", value=copy_text, height=280, label_visibility="collapsed")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="재정사업 탐지 시스템",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session()
    _inject_global_styles()

    env_keys = get_env_keys()
    env_error = missing_env_message(env_keys)
    supabase = get_supabase_client(env_keys) if not env_error else None

    # 프로젝트 수 조회
    project_count = 0
    if supabase and env_keys["supabase_url"] and env_keys["supabase_key"]:
        project_count = fetch_project_count(env_keys["supabase_url"], env_keys["supabase_key"])

    # 벡터 검색 결과 ref_map (결과 페이지용)
    vector_results: list[dict[str, Any]] = st.session_state.get("vector_results", [])
    ref_map: dict[str, dict[str, Any]] = {str(r["id"]): r for r in vector_results}

    is_results = st.session_state.page == PAGE_RESULTS
    _render_site_chrome(
        breadcrumb_current="검토 결과" if is_results else "유사·중복 탐지",
        project_count=project_count,
    )

    with st.container(border=True, key="govt-content-panel"):
        if env_error:
            st.error(env_error)

        if is_results:
            _render_results_page(ref_map)
        else:
            _render_input_page(env_keys, env_error, supabase)

    _render_footer()


if __name__ == "__main__":
    main()