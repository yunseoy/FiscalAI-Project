"""공공기관 재정사업 유사·중복 탐지 시스템

신규 사업을 벡터로 변환 후 Supabase pgvector로 유사 사업 검색,
GPT-4o로 유사·중복 분석합니다.

실행:
    streamlit run aiDetector.py

환경변수 (.env):
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY, Openfiscal_api_key
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
SEARCH_TOP_K = 10
SEARCH_THRESHOLD = 0.5
GPT_TOP_K = 5

FIELD_OPTIONS = [
    "공공질서및안전", "과학기술", "교육", "교통및물류", "국방",
    "국토및지역개발", "농림수산", "문화및관광", "보건", "사회복지",
    "산업·중소기업및에너지", "예비비", "일반·지방행정", "통신",
    "통일·외교", "환경", "기타",
]

PRIMARY       = "#0062B1"
PRIMARY_DARK  = "#004A8C"
PRIMARY_LIGHT = "#E8F0FA"
WHITE         = "#FFFFFF"
GRAY_50       = "#F7F8FA"
GRAY_100      = "#EEF0F4"
GRAY_200      = "#DDE1E9"
GRAY_400      = "#9AA4B2"
GRAY_600      = "#5A6475"
GRAY_900      = "#1A2130"

OPENFISCAL_URL = "https://www.openfiscaldata.go.kr/"
PAGE_INPUT   = "input"
PAGE_RESULTS = "results"

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
    margin-top: 0 !important;
}
.main .block-container { padding-top: 0 !important; margin-top: 0 !important; }
.stMainBlockContainer { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > section:first-child { padding-top: 0 !important; }
.stApp > div:first-child { padding: 0 !important; margin: 0 !important; }
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
- risk_level: 80 이상 높음, 70~79 중간, 70 미만은 목록에서 제외
- 검토 의견은 단순 요약이 아닌 실제 예산 담당자 관점에서 구체적으로 작성
- 부처가 다른 경우 부처 간 협의 또는 통합 필요성 언급
- 예산 규모 차이가 있으면 그 의미도 분석
"""

RISK_COLORS = {
    "높음": {"bg": "#FFF5F5", "border": "#FC8181", "badge_bg": "#C53030", "badge_text": "#FFFFFF"},
    "중간": {"bg": "#FFFAF0", "border": "#F6AD55", "badge_bg": "#C05621", "badge_text": "#FFFFFF"},
    "낮음": {"bg": "#F0FFF4", "border": "#68D391", "badge_bg": "#276749", "badge_text": "#FFFFFF"},
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
    if not keys["openai"]: missing.append("OPENAI_API_KEY")
    if not keys["supabase_url"]: missing.append("SUPABASE_URL")
    if not keys["supabase_key"]: missing.append("SUPABASE_ANON_KEY")
    if not missing: return None
    return "다음 환경변수가 `.env`에 설정되어 있지 않습니다: " + ", ".join(missing)


def get_supabase_client(keys: dict[str, str]) -> Client | None:
    if not keys["supabase_url"] or not keys["supabase_key"]: return None
    return create_client(keys["supabase_url"], keys["supabase_key"])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_project_count(supabase_url: str, supabase_key: str) -> int:
    try:
        client = create_client(supabase_url, supabase_key)
        response = client.table("projects").select("id", count="exact").execute()
        return response.count or 0
    except Exception:
        return 0


def embed_text(openai_key: str, text: str) -> list[float]:
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def search_similar_projects(
    supabase: Client, embedding: list[float],
    top_k: int = SEARCH_TOP_K, threshold: float = SEARCH_THRESHOLD,
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "match_projects",
            {"query_embedding": embedding, "match_threshold": threshold, "match_count": top_k},
        ).execute()
        return response.data or []
    except Exception as exc:
        raise RuntimeError(f"벡터 검색 실패: {exc}") from exc


def analyze_similar_projects(
    openai_key: str, supabase: Client,
    project_name: str, ministry: str, field: str, overview: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_text = f"{field} {project_name} {overview}"
    embedding = embed_text(openai_key, query_text)
    vector_results = search_similar_projects(supabase, embedding)

    if not vector_results:
        return {"similar_projects": [], "overall_opinion": "벡터 검색 결과 유사한 기존 사업이 없습니다."}, []

    gpt_candidates = [
        {
            "id": r["id"], "사업명": r.get("project_name", ""),
            "부처명": r.get("ministry", ""), "분야": r.get("category", ""),
            "사업개요": r.get("overview", ""),
            "벡터유사도": round(float(r.get("similarity", 0)) * 100, 1),
        }
        for r in vector_results[:GPT_TOP_K]
    ]

    completion = OpenAI(api_key=openai_key).chat.completions.create(
        model="gpt-4o", temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"new_project": {"사업명": project_name, "부처명": ministry, "분야": field, "사업개요": overview},
                 "existing_projects": gpt_candidates},
                ensure_ascii=False, indent=2)},
        ],
    )
    content = completion.choices[0].message.content
    if not content: raise ValueError("GPT-4o 응답이 비어 있습니다.")
    result: dict[str, Any] = json.loads(content)
    if "overall_opinion" not in result and "summary" in result:
        result["overall_opinion"] = result["summary"]
    return result, vector_results


def normalize_risk(risk: str | None) -> str:
    r = (risk or "중간").strip()
    if "높" in r or r.lower() == "high": return "높음"
    if "낮" in r or r.lower() == "low": return "낮음"
    return "중간"


def _html_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def build_review_copy_text(
    result: dict[str, Any], form: dict[str, str], ref_map: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "[재정사업 유사·중복 검토 의견]", "",
        "■ 신규 사업",
        f"- 사업명: {form.get('name', '-')}", f"- 부처: {form.get('ministry', '-')}",
        f"- 분야: {form.get('field', '-')}", f"- 개요: {form.get('overview', '-')}", "",
        "■ 종합 의견", result.get("overall_opinion") or "(없음)", "",
    ]
    similar = result.get("similar_projects") or []
    if similar:
        lines.append("■ 유사·중복 검토 사업")
        for i, item in enumerate(similar, 1):
            ref = ref_map.get(str(item.get("id", "")), {})
            title = ref.get("project_name") or item.get("id", "-")
            lines.extend([
                "", f"{i}. {title} (유사도 {item.get('similarity_score', '-')}%, 위험 {item.get('risk_level', '중간')})",
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
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    color: {GRAY_900};
  }}
  .stApp {{ background-color: {GRAY_50} !important; color-scheme: light; }}

  /* ── GNB ── */
  .krds-gnb {{
    background: {PRIMARY}; height: 40px; display: flex; align-items: center;
    justify-content: flex-end; padding: 0 2rem; margin: 0;
  }}
  .krds-gnb a {{ color: rgba(255,255,255,0.9); font-size: 0.8rem; text-decoration: none; }}
  .krds-gnb a:hover {{ color: #fff; text-decoration: underline; }}

  /* ── 헤더 ── */
  .krds-header {{
    background: {WHITE}; border-bottom: 3px solid {PRIMARY};
    padding: 1.25rem 2rem; display: flex; align-items: center;
    justify-content: space-between; margin: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .krds-header-left {{ display: flex; align-items: center; gap: 1rem; }}
  .krds-logo-wrap img {{ height: 56px; width: auto; }}
  .krds-title-wrap h1 {{
    margin: 0; font-size: 1.375rem; font-weight: 700;
    color: {PRIMARY}; letter-spacing: -0.02em; line-height: 1.3;
  }}
  .krds-title-wrap p {{ margin: 0.2rem 0 0; font-size: 0.8125rem; color: {GRAY_600}; }}
  .krds-db-badge {{
    background: {PRIMARY_LIGHT}; border: 1px solid {PRIMARY};
    border-radius: 20px; padding: 0.35rem 0.85rem;
    font-size: 0.8125rem; color: {PRIMARY}; font-weight: 600; white-space: nowrap;
  }}

  /* ── 브레드크럼 ── */
  .krds-breadcrumb {{
    background: {GRAY_100}; border-bottom: 1px solid {GRAY_200};
    padding: 0.6rem 2rem; font-size: 0.8125rem; color: {GRAY_600};
    display: flex; align-items: center; justify-content: space-between; margin: 0;
  }}
  .krds-breadcrumb-left {{ display: flex; align-items: center; gap: 0.4rem; }}
  .krds-breadcrumb strong {{ color: {PRIMARY}; font-weight: 600; }}
  .krds-back-btn {{
    font-size: 0.8125rem; color: {PRIMARY}; font-weight: 600;
    text-decoration: none; display: flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.65rem; border: 1px solid {PRIMARY};
    border-radius: 4px; background: {WHITE};
    transition: background 0.15s;
  }}
  .krds-back-btn:hover {{ background: {PRIMARY_LIGHT}; }}

  /* ── 콘텐츠 영역 ── */
  .main .block-container {{
    padding: 2rem 4rem 3rem !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    }}
    .stTextArea small,
    [data-testid="InputInstructions"] {{
        display: none !important;
    }}

  /* ── 입력 페이지 레이아웃 ── */
  .krds-input-hero {{
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {GRAY_200};
  }}
  .krds-input-hero h2 {{
    font-size: 1.5rem; font-weight: 700; color: {GRAY_900};
    margin: 0 0 0.5rem; letter-spacing: -0.02em;
  }}
  .krds-input-hero p {{
    font-size: 0.9375rem; color: {GRAY_600}; margin: 0;
  }}

  /* ── 입력 그리드 ── */
  .krds-form-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
  }}
  .krds-form-full {{ grid-column: 1 / -1; }}

  /* ── 라벨 ── */
  .krds-label {{
    display: block; font-size: 0.875rem; font-weight: 600;
    color: {GRAY_900}; margin: 0 0 0.4rem;
  }}
  .krds-label .req {{ color: #C53030; margin-left: 2px; }}

  /* ── 폼 입력창 ── */
  div[data-testid="stForm"] {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important; margin-top: 0 !important;
  }}
  div[data-testid="stForm"] input,
  div[data-testid="stForm"] textarea,
  div[data-testid="stForm"] [data-baseweb="input"] input,
  div[data-testid="stForm"] [data-baseweb="textarea"] textarea,
  div[data-testid="stForm"] [data-baseweb="input"] > div,
  div[data-testid="stForm"] [data-baseweb="textarea"] > div,
  div[data-testid="stForm"] [data-baseweb="select"] > div {{
    background-color: {WHITE} !important; color: {GRAY_900} !important;
    -webkit-text-fill-color: {GRAY_900} !important;
    border: 1px solid {GRAY_200} !important; border-radius: 6px !important;
    color-scheme: light !important;
  }}
  div[data-testid="stForm"] input::placeholder,
  div[data-testid="stForm"] textarea::placeholder {{
    color: {GRAY_400} !important; -webkit-text-fill-color: {GRAY_400} !important; opacity: 1 !important;
  }}
  div[data-testid="stForm"] input:focus,
  div[data-testid="stForm"] textarea:focus,
  div[data-testid="stForm"] [data-baseweb="input"]:focus-within > div,
  div[data-testid="stForm"] [data-baseweb="textarea"]:focus-within > div,
  div[data-testid="stForm"] [data-baseweb="select"]:focus-within > div {{
    border: 2px solid {PRIMARY} !important;
    box-shadow: 0 0 0 3px {PRIMARY_LIGHT} !important; outline: none !important;
  }}

  /* ── 버튼 ── */
  .stApp .main button,
  .stApp .main [data-testid="stFormSubmitButton"] button,
  .stApp .main [data-testid="stBaseButton-primary"],
  .stApp .main [data-testid="stBaseButton-secondary"],
  .stApp .main div.stButton > button,
  .stApp .main .stFormSubmitButton button,
  .stApp .main [data-testid="stDownloadButton"] button,
  .stApp .st-key-btn_copy_text button,
  .stApp .st-key-btn_download_text button {{
    background-color: {PRIMARY} !important; background: {PRIMARY} !important;
    color: {WHITE} !important; border: 1px solid {PRIMARY} !important;
    border-radius: 6px !important; font-weight: 600 !important;
    box-shadow: none !important; filter: none !important;
  }}
  .stApp .main button:hover,
  .stApp .main [data-testid="stButton"] button:hover,
  .stApp .main [data-testid="stDownloadButton"] button:hover {{
    background-color: {PRIMARY_DARK} !important; background: {PRIMARY_DARK} !important;
    border-color: {PRIMARY_DARK} !important; color: {WHITE} !important;
  }}
  .stApp .main button:disabled {{ opacity: 0.5 !important; }}
  .stApp .main [data-testid="stFormSubmitButton"] button {{
    width: 100% !important; min-height: 52px !important;
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.02em !important;
  }}
  .stApp [data-testid="stFormSubmitButton"] > button,
  .stApp [data-testid="stForm"] .stFormSubmitButton > button {{
    background-color: {PRIMARY} !important; color: {WHITE} !important; border-color: {PRIMARY} !important;
  }}

  /* ── 결과 섹션 헤딩 ── */
  .krds-section-heading {{
    font-size: 1rem; font-weight: 700; color: {GRAY_900};
    margin: 0 0 0.85rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid {PRIMARY};
    display: flex; align-items: center; gap: 0.5rem;
  }}

  /* ── 분석대상 요약 카드 ── */
  .krds-summary {{
    background: {WHITE}; border: 1px solid {GRAY_200};
    border-left: 4px solid {PRIMARY}; border-radius: 8px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}

  /* ── 벡터+의견 카드 ── */
  .krds-result-section {{
    background: {WHITE}; border: 1px solid {GRAY_200};
    border-radius: 8px; padding: 1.5rem;
    margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}

  /* ── 벡터 행 ── */
  .krds-vector-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid {GRAY_100};
    font-size: 0.875rem; color: {GRAY_600};
  }}
  .krds-vector-row:last-child {{ border-bottom: none; }}
  .krds-vector-score {{
    color: {PRIMARY}; font-weight: 700; font-size: 0.875rem;
    background: {PRIMARY_LIGHT}; padding: 0.15rem 0.55rem;
    border-radius: 12px; white-space: nowrap;
  }}

  /* ── 종합 의견 ── */
  .krds-opinion-box {{
    background: {PRIMARY_LIGHT}; border: 1px solid #C3D9F5;
    border-left: 4px solid {PRIMARY}; border-radius: 6px;
    padding: 1rem 1.25rem; font-size: 0.9375rem;
    color: {GRAY_900}; line-height: 1.75;
    display: flex; gap: 0.75rem; align-items: flex-start;
    margin-top: 1rem;
  }}

  /* ── 결과 카드 ── */
  .krds-card {{
    background: {WHITE}; border: 1px solid {GRAY_200};
    border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}
  .krds-card-header {{
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 0.75rem; margin-bottom: 0.5rem;
  }}
  .krds-card-title {{ font-size: 1rem; font-weight: 700; color: {GRAY_900}; line-height: 1.4; }}
  .krds-risk-badge {{
    padding: 0.25rem 0.65rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 700; white-space: nowrap; flex-shrink: 0;
  }}
  .krds-meta {{ font-size: 0.8125rem; color: {GRAY_600}; margin: 0.25rem 0; line-height: 1.5; }}
  .krds-meta a {{ color: {PRIMARY}; text-decoration: none; }}
  .krds-path {{ font-size: 0.8rem; color: {GRAY_400}; margin: 0.2rem 0 0.5rem; }}

  /* ── 유사도 바 ── */
  .krds-bar-wrap {{ margin: 0.85rem 0 1rem; }}
  .krds-bar-label {{
    display: flex; justify-content: space-between;
    font-size: 0.8125rem; color: {GRAY_600}; margin-bottom: 0.4rem; font-weight: 500;
  }}
  .krds-bar-label strong {{ color: {PRIMARY}; font-weight: 700; }}
  .krds-bar-track {{ height: 8px; background: {GRAY_100}; border-radius: 4px; overflow: hidden; }}
  .krds-bar-fill {{
    height: 100%; background: linear-gradient(90deg, {PRIMARY} 0%, #3B82F6 100%);
    border-radius: 4px;
  }}

  /* ── 비교 컬럼 ── */
  .krds-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem; }}
  @media (max-width: 640px) {{ .krds-compare {{ grid-template-columns: 1fr; }} }}
  .krds-compare-box {{
    background: {GRAY_50}; border: 1px solid {GRAY_200};
    border-radius: 6px; padding: 0.85rem 1rem;
  }}
  .krds-compare-box h4 {{ margin: 0 0 0.4rem; font-size: 0.8125rem; font-weight: 700; color: {PRIMARY}; }}
  .krds-compare-box p {{ margin: 0; font-size: 0.875rem; color: {GRAY_600}; line-height: 1.6; }}

  /* ── 검토의견 ── */
  .krds-review-box {{
    background: #FFFBF0; border: 1px solid #F6AD55;
    border-left: 4px solid #DD6B20; border-radius: 6px;
    padding: 0.85rem 1rem; margin-top: 0.85rem;
    font-size: 0.875rem; color: #744210; line-height: 1.7;
  }}
  .krds-review-box strong {{ color: #C05621; display: block; margin-bottom: 0.3rem; font-size: 0.8125rem; }}

  /* ── 분석중 ── */
  .krds-analyzing {{
    text-align: center; color: {GRAY_600}; font-size: 0.9375rem;
    margin: 0.75rem 0 0; padding: 0; background: none !important; border: none !important;
  }}

  /* ── 푸터 ── */
  .krds-footer {{
    background: {GRAY_900}; color: rgba(255,255,255,0.7); text-align: center;
    padding: 1.25rem 1rem; font-size: 0.8125rem; margin-top: 3rem; line-height: 1.6;
  }}
  .krds-footer a {{ color: rgba(255,255,255,0.85); text-decoration: none; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ── 헤더/푸터 ─────────────────────────────────────────────────────────────────

def _logo_data_uri() -> str:
    if not LOGO_PATH.is_file(): return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_header_logo_html() -> str:
    logo_src = _logo_data_uri()
    if logo_src:
        return f'<div class="krds-logo-wrap"><img src="{logo_src}" alt="기획예산처" /></div>'
    return f'<div class="krds-logo-wrap"><span style="font-size:1rem;font-weight:700;color:{PRIMARY};">기획예산처</span></div>'


def _render_site_chrome(*, breadcrumb_current: str, project_count: int, is_results: bool) -> None:
    back_btn = (
        f'<a class="krds-back-btn" href="javascript:void(0);" '
        f'onclick="window.parent.document.querySelector(\'[data-testid=stBaseButton-secondary]\').click();">'
        f'← 이전으로 돌아가기</a>'
        if is_results else ""
    )
    st.markdown(
        f"""
<div class="krds-gnb">
  <a href="{OPENFISCAL_URL}" target="_blank" rel="noopener noreferrer">열린재정 바로가기 →</a>
</div>
<div class="krds-header">
  <div class="krds-header-left">
    {_render_header_logo_html()}
    <div class="krds-title-wrap">
      <h1>재정사업 유사·중복 탐지 시스템</h1>
      <p>AI 기반 재정사업 중복 투자 사전 검토 도구</p>
    </div>
  </div>
  <div class="krds-db-badge">기존 사업: {project_count:,}건</div>
</div>
<div class="krds-breadcrumb">
  <div class="krds-breadcrumb-left">
    <span>홈</span><span>›</span>
    <span>재정사업 관리</span><span>›</span>
    <strong>{_html_escape(breadcrumb_current)}</strong>
  </div>
  {back_btn}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        """
<div class="krds-footer">
  ⓒ 2026 기획예산처 | 재정사업 유사·중복 탐지 시스템 &nbsp;·&nbsp;
</div>
""",
        unsafe_allow_html=True,
    )


# ── 결과 카드 ─────────────────────────────────────────────────────────────────

def _render_similarity_bar_html(score: int | float) -> str:
    pct = max(0, min(100, int(score or 0)))
    return f"""
<div class="krds-bar-wrap">
  <div class="krds-bar-label"><span>GPT 유사도 점수</span><strong>{pct}점</strong></div>
  <div class="krds-bar-track"><div class="krds-bar-fill" style="width:{pct}%;"></div></div>
</div>
"""


def _render_result_card_html(item: dict[str, Any], ref: dict[str, Any]) -> str:
    risk = normalize_risk(str(item.get("risk_level", "")))
    colors = RISK_COLORS.get(risk, RISK_COLORS["중간"])
    title = _html_escape(ref.get("project_name") or item.get("id", "-"))
    ministry = _html_escape(ref.get("ministry") or "-")
    category = _html_escape(ref.get("category") or "-")
    vector_sim = ref.get("similarity")
    vector_sim_str = f" &nbsp;·&nbsp; 벡터유사도 {round(float(vector_sim)*100, 1)}%" if vector_sim else ""
    similar = _html_escape(item.get("similar_points", "-"))
    different = _html_escape(item.get("different_points", "-"))
    review = _html_escape(item.get("review_comment", "-"))
    bar = _render_similarity_bar_html(item.get("similarity_score", 0))
    fiscal_year = ref.get("fiscal_year") or "-"
    budget = ref.get("budget_100m_krw")
    budget_str = f"{budget:,}억원" if budget else "-"
    path_parts = [ref.get("fld_nm"), ref.get("sect_nm"), ref.get("pgm_nm")]
    path_str = " › ".join(p for p in path_parts if p)

    return f"""
<div class="krds-card" style="border-left:4px solid {colors['border']};">
  <div class="krds-card-header">
    <div class="krds-card-title">{title}</div>
    <span class="krds-risk-badge" style="background:{colors['badge_bg']};color:{colors['badge_text']};">{risk}</span>
  </div>
  <div class="krds-meta">{ministry} &nbsp;·&nbsp; {category}{vector_sim_str}</div>
  {"<div class='krds-path'>📂 " + _html_escape(path_str) + "</div>" if path_str else ""}
  <div class="krds-meta">📅 {fiscal_year}년 &nbsp;·&nbsp; 💰 {budget_str} &nbsp;·&nbsp; 🔗 <a href="https://www.openfiscaldata.go.kr" target="_blank">열린재정 OpenAPI</a></div>
  {bar}
  <div class="krds-compare">
    <div class="krds-compare-box"><h4>유사한 점</h4><p>{similar}</p></div>
    <div class="krds-compare-box"><h4>다른 점</h4><p>{different}</p></div>
  </div>
  <div class="krds-review-box"><strong>담당자 검토 의견</strong>{review}</div>
</div>
"""


# ── 세션 ──────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "page": PAGE_INPUT, "analysis_result": None,
        "vector_results": [], "last_form": {}, "show_copy_area": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 입력 페이지 ───────────────────────────────────────────────────────────────

def _render_input_page(env_keys, env_error, supabase, project_count: int = 0) -> None:

    st.markdown(
        f"""
<div class="krds-input-hero">
  <h2>신규 재정사업 유사·중복 검토</h2>
  <p>사업 정보를 입력하면 AI가 기존 사업 {project_count:,}건과 비교하여 유사·중복 여부를 분석합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("new_project_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<label class="krds-label">사업명 <span class="req">*</span></label>', unsafe_allow_html=True)
            project_name = st.text_input("사업명", placeholder="신규 사업명을 입력하세요", label_visibility="collapsed")

        with col2:
            st.markdown('<label class="krds-label">부처명</label>', unsafe_allow_html=True)
            ministry = st.text_input("부처명", placeholder="예: 기획예산처", label_visibility="collapsed")

        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown('<label class="krds-label">분야</label>', unsafe_allow_html=True)
            field = st.selectbox("분야", FIELD_OPTIONS, index=0, label_visibility="collapsed")

        st.markdown('<label class="krds-label">사업개요 <span class="req">*</span></label>', unsafe_allow_html=True)
        overview = st.text_area(
            "사업개요", height=160,
            placeholder="사업 목적, 주요 추진 내용, 기대 효과 등을 기술하세요.",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "유사·중복 분석 시작", type="primary",
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
            "name": project_name.strip(), "ministry": ministry.strip(),
            "field": field.strip(), "overview": overview.strip(),
        }
        with st.spinner("유사 사업을 분석 중입니다..."):
            try:
                result, vector_results = analyze_similar_projects(
                    env_keys["openai"], supabase,
                    st.session_state.last_form["name"], st.session_state.last_form["ministry"],
                    st.session_state.last_form["field"], st.session_state.last_form["overview"],
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

    # 분석 대상 요약
    st.markdown(
        f"""
<div class="krds-summary">
  <p style="margin:0 0 0.75rem;font-size:0.9375rem;font-weight:700;color:{PRIMARY};">분석 대상 사업</p>
  <table style="width:100%;border-collapse:collapse;font-size:0.875rem;">
    <tr>
      <td style="width:80px;font-weight:600;color:{GRAY_600};padding:0.2rem 0.5rem 0.2rem 0;vertical-align:top;">사업명</td>
      <td style="color:{GRAY_900};padding:0.2rem 0;">{_html_escape(form.get('name', '-'))}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.2rem 0.5rem 0.2rem 0;vertical-align:top;">부처명</td>
      <td style="color:{GRAY_900};padding:0.2rem 0;">{_html_escape(form.get('ministry') or '-')}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.2rem 0.5rem 0.2rem 0;vertical-align:top;">분야</td>
      <td style="color:{GRAY_900};padding:0.2rem 0;">{_html_escape(form.get('field') or '-')}</td>
    </tr>
    <tr>
      <td style="font-weight:600;color:{GRAY_600};padding:0.2rem 0.5rem 0.2rem 0;vertical-align:top;">사업개요</td>
      <td style="color:{GRAY_900};padding:0.2rem 0;line-height:1.6;">{_html_escape(form.get('overview', '-'))}</td>
    </tr>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    # 벡터 검색 결과 + 종합 의견
    opinion = result.get("overall_opinion") or result.get("summary")
    vector_rows_html = "".join(
        f"""
<div class="krds-vector-row">
  <span>{_html_escape(r.get('project_name', '-'))} &nbsp;·&nbsp; {_html_escape(r.get('ministry', '-'))}</span>
  <span class="krds-vector-score">{round(float(r.get('similarity', 0)) * 100, 1)}%</span>
</div>
"""
        for r in vector_results
    ) if vector_results else ""

    st.markdown(
        f"""
<div class="krds-result-section">
  <p class="krds-section-heading">벡터 검색 결과 (상위 {len(vector_results)}건)</p>
  {vector_rows_html if vector_rows_html else f'<p style="color:{GRAY_400};font-size:0.875rem;">검색 결과가 없습니다.</p>'}
  <p class="krds-section-heading" style="margin-top:1.5rem;">종합 의견</p>
  <div class="krds-opinion-box">
    <span style="font-size:1.25rem;flex-shrink:0;"></span>
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
<div class="krds-result-section">
  <p class="krds-section-heading">유사·중복 사업 검토 결과 ({len(similar)}건)</p>
  {cards_html if cards_html else f'<p style="color:{GRAY_400};font-size:0.9rem;text-align:center;padding:1rem 0;">유사·중복 가능성이 높은 기존 사업이 발견되지 않았습니다.</p>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # 다운로드
    copy_text = build_review_copy_text(result, form, ref_map)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("전체 결과 텍스트 복사", type="secondary", key="btn_copy_text", use_container_width=True):
            st.session_state.show_copy_area = True
    with col2:
        st.download_button(
            "텍스트 파일 저장", data=copy_text,
            file_name="유사중복_검토결과.txt", mime="text/plain",
            type="secondary", key="btn_download_text", use_container_width=True,
        )
    if st.session_state.show_copy_area:
        st.caption("아래 전체를 선택(Ctrl+A) 후 복사(Ctrl+C)하세요.")
        st.text_area("검토 결과 전문", value=copy_text, height=280, label_visibility="collapsed")

    # 숨겨진 이전으로 버튼 (브레드크럼 버튼이 트리거)
    if st.button("← 이전으로 돌아가기", type="secondary", key="btn_go_back"):
        st.session_state.page = PAGE_INPUT
        st.session_state.show_copy_area = False
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="재정사업 유사·중복 탐지 시스템",
        page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed",
    )
    _init_session()
    _inject_global_styles()

    env_keys = get_env_keys()
    env_error = missing_env_message(env_keys)
    supabase = get_supabase_client(env_keys) if not env_error else None

    project_count = 0
    if supabase and env_keys["supabase_url"] and env_keys["supabase_key"]:
        project_count = fetch_project_count(env_keys["supabase_url"], env_keys["supabase_key"])

    vector_results: list[dict[str, Any]] = st.session_state.get("vector_results", [])
    ref_map: dict[str, dict[str, Any]] = {str(r["id"]): r for r in vector_results}

    is_results = st.session_state.page == PAGE_RESULTS

    _render_site_chrome(
        breadcrumb_current="검토 결과" if is_results else "유사·중복 탐지",
        project_count=project_count,
        is_results=is_results,
    )

    if env_error:
        st.error(env_error)

    if is_results:
        _render_results_page(ref_map)
    else:
        _render_input_page(env_keys, env_error, supabase)

    _render_footer()


if __name__ == "__main__":
    main()