"""열린재정 OpenAPI 재정사업 데이터 수집 → Supabase projects 저장.

API 응답 필드만 저장합니다. GPT·템플릿 등 가상 데이터는 생성하지 않습니다.

실행:
    python collect_data.py

환경변수 (.env):
    OPENFISCAL_API_KEY   — 열린재정 인증키 (Openfiscal_api_key 별칭 지원)
    SUPABASE_URL, SUPABASE_ANON_KEY
    OPENFISCAL_API_BASE  — 기본 https://www.openfiscaldata.go.kr/openapi/
    OPENFISCAL_API_SERVICE — 기본 ExpenditureBudgetInit5 (세부사업 예산편성)
    OPENFISCAL_FISCAL_YEAR — 수집 회계연도, 기본 2026
    COLLECT_LIMIT        — 저장 건수 상한, 기본 5000 (0이면 제한 없음)
    COLLECT_MAX_PAGES    — (선택) API 페이지 상한
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_API_BASE = "https://www.openfiscaldata.go.kr/openapi/"
FALLBACK_API_HOST = "https://openapi.openfiscaldata.go.kr/"
DEFAULT_SERVICE = "ExpenditureBudgetInit5"
DEFAULT_COLLECT_LIMIT = 5000
PAGE_SIZE = 100
REQUEST_DELAY_SEC = 0.15

# API에 사업개요 단일 필드는 없음 — 계층·분류 필드를 그대로 연결
OVERVIEW_FIELD_KEYS = (
    "FLD_NM",
    "SECT_NM",
    "PGM_NM",
    "ACTV_NM",
    "SACTV_NM",
    "BZ_CLS_NM",
    "FIN_DE_EP_NM",
    "FSCL_NM",
)


def load_config() -> dict[str, Any]:
    fiscal_year = os.getenv("OPENFISCAL_FISCAL_YEAR", "2026").strip()
    max_pages_raw = os.getenv("COLLECT_MAX_PAGES", "").strip()
    max_pages = int(max_pages_raw) if max_pages_raw.isdigit() else None

    limit_raw = os.getenv("COLLECT_LIMIT", str(DEFAULT_COLLECT_LIMIT)).strip()
    collect_limit = int(limit_raw) if limit_raw.isdigit() else DEFAULT_COLLECT_LIMIT
    if collect_limit <= 0:
        collect_limit = None

    openfiscal_key = (
        os.getenv("OPENFISCAL_API_KEY", "").strip()
        or os.getenv("Openfiscal_api_key", "").strip()
    )

    return {
        "openfiscal_key": openfiscal_key,
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "api_base": os.getenv("OPENFISCAL_API_BASE", DEFAULT_API_BASE).strip(),
        "api_service": os.getenv("OPENFISCAL_API_SERVICE", DEFAULT_SERVICE).strip(),
        "fiscal_year": fiscal_year,
        "max_pages": max_pages,
        "collect_limit": collect_limit,
    }


def validate_config(cfg: dict[str, Any]) -> None:
    missing: list[str] = []
    if not cfg["openfiscal_key"]:
        missing.append("OPENFISCAL_API_KEY")
    if not cfg["supabase_url"]:
        missing.append("SUPABASE_URL")
    if not cfg["supabase_key"]:
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        raise SystemExit(
            "다음 환경변수가 `.env`에 설정되어 있지 않습니다: " + ", ".join(missing)
        )


def _join_url(base: str, service: str) -> str:
    base = base if base.endswith("/") else base + "/"
    service = service.lstrip("/")
    return base + service


def _http_get_json(url: str, params: dict[str, str]) -> Any:
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _api_params(cfg: dict[str, Any], p_index: int) -> dict[str, str]:
    """열린재정 OpenAPI 공통 파라미터 (Key/Type 대소문자 규격)."""
    return {
        "Key": cfg["openfiscal_key"],
        "Type": "json",
        "pIndex": str(p_index),
        "pSize": str(PAGE_SIZE),
        "FSCL_YY": cfg["fiscal_year"],
    }


def fetch_openfiscal_page(
    cfg: dict[str, Any],
    *,
    p_index: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """한 페이지 조회. (rows, total_count) 반환."""
    params = _api_params(cfg, p_index)
    primary = _join_url(cfg["api_base"], cfg["api_service"])
    fallback = _join_url(FALLBACK_API_HOST, cfg["api_service"])

    last_err: Exception | None = None
    for url in (primary, fallback):
        try:
            payload = _http_get_json(url, params)
            return _parse_openfiscal_payload(payload, cfg["api_service"])
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 404 and url == primary:
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if url == primary:
                continue
            raise

    raise RuntimeError(f"열린재정 API 호출 실패: {last_err}") from last_err


def _parse_openfiscal_payload(
    payload: Any,
    service: str,
) -> tuple[list[dict[str, Any]], int | None]:
    if isinstance(payload, dict) and "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(
            f"열린재정 API 오류: {result.get('CODE')} — {result.get('MESSAGE')}"
        )

    blocks = payload.get(service) if isinstance(payload, dict) else None
    if not blocks:
        raise RuntimeError(f"응답에 `{service}` 블록이 없습니다.")

    total: int | None = None
    rows: list[dict[str, Any]] = []
    for block in blocks:
        if "head" in block:
            for head in block["head"]:
                if "list_total_count" in head:
                    total = int(head["list_total_count"])
        if "row" in block:
            rows.extend(block["row"])

    return rows, total


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_project_name(row: dict[str, Any]) -> str | None:
    """API 계층명을 조합한 사업명 (가상 문구 없음)."""
    parts: list[str] = []
    for key in ("PGM_NM", "ACTV_NM", "SACTV_NM"):
        val = _clean(row.get(key))
        if val and (not parts or parts[-1] != val):
            parts.append(val)
    if parts:
        return " / ".join(parts)
    for key in ("FIN_DE_EP_NM", "BZ_CLS_NM"):
        val = _clean(row.get(key))
        if val:
            return val
    return None


def _pick_overview(row: dict[str, Any]) -> str:
    """API 분류·계층 필드만 이어 붙인 사업개요."""
    parts: list[str] = []
    seen: set[str] = set()
    for key in OVERVIEW_FIELD_KEYS:
        val = _clean(row.get(key))
        if val and val not in seen:
            seen.add(val)
            parts.append(val)
    return " | ".join(parts)


def _pick_ministry(row: dict[str, Any]) -> str | None:
    return _clean(row.get("OFFC_NM"))


def _pick_category(row: dict[str, Any]) -> str | None:
    return _clean(row.get("FLD_NM")) or _clean(row.get("SECT_NM"))


def _budget_to_100m_krw(amount: Any) -> int | None:
    """천원 단위 예산액 → 억원(정수, 반올림)."""
    if amount is None:
        return None
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    return max(0, int(round(value / 100_000)))


def make_project_id(project_name: str, ministry: str, fiscal_year: int) -> str:
    digest = hashlib.sha256(
        f"{fiscal_year}|{ministry}|{project_name}".encode("utf-8")
    ).hexdigest()[:20]
    return f"P{fiscal_year}-{digest}"


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    project_name = _pick_project_name(row)
    ministry = _pick_ministry(row)
    category = _pick_category(row)
    budget = _budget_to_100m_krw(
        row.get("Y_YY_DFN_MEDI_KCUR_AMT") or row.get("Y_YY_MEDI_KCUR_AMT")
    )

    if not project_name or not ministry or not category or budget is None:
        return None

    try:
        fiscal_year = int(str(row.get("FSCL_YY", "")).strip())
    except ValueError:
        return None

    return {
        "id": make_project_id(project_name, ministry, fiscal_year),
        "project_name": project_name,
        "ministry": ministry,
        "category": category,
        "fiscal_year": fiscal_year,
        "overview": _pick_overview(row),
        "budget_100m_krw": budget,
        "fscl_nm": _clean(row.get("FSCL_NM")),
        "acct_nm": _clean(row.get("ACCT_NM")),
        "fld_nm": _clean(row.get("FLD_NM")),
        "sect_nm": _clean(row.get("SECT_NM")),
        "pgm_nm": _clean(row.get("PGM_NM")),
        "actv_nm": _clean(row.get("ACTV_NM")),
        "sactv_nm": _clean(row.get("SACTV_NM")),
        "bz_cls_nm": _clean(row.get("BZ_CLS_NM")),
    }


def collect_raw_projects(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    p_index = 1
    total_count: int | None = None
    fetched_rows = 0

    while True:
        if cfg["max_pages"] is not None and p_index > cfg["max_pages"]:
            break

        rows, total = fetch_openfiscal_page(cfg, p_index=p_index)
        if total is not None:
            total_count = total

        if not rows:
            break

        fetched_rows += len(rows)
        for row in rows:
            item = normalize_row(row)
            if item is None:
                continue
            key = (item["project_name"], item["ministry"], item["fiscal_year"])
            existing = merged.get(key)
            if existing is None or item["budget_100m_krw"] > existing["budget_100m_krw"]:
                merged[key] = item

            limit = cfg.get("collect_limit")
            if limit is not None and len(merged) >= limit:
                break

        limit = cfg.get("collect_limit")
        if limit is not None and len(merged) >= limit:
            break

        if total_count is not None and fetched_rows >= total_count:
            break

        p_index += 1
        time.sleep(REQUEST_DELAY_SEC)

    projects = list(merged.values())
    limit = cfg.get("collect_limit")
    if limit is not None:
        projects = projects[:limit]
    return projects


def _upsert_chunk_native(client: Client, chunk: list[dict[str, Any]]) -> None:
    client.table("projects").upsert(
        chunk,
        on_conflict="project_name,fiscal_year",
    ).execute()


def _upsert_chunk_fallback(client: Client, chunk: list[dict[str, Any]]) -> None:
    """unique 인덱스 없을 때 사업명+연도 기준 수동 upsert."""
    for item in chunk:
        existing = (
            client.table("projects")
            .select("id")
            .eq("project_name", item["project_name"])
            .eq("fiscal_year", item["fiscal_year"])
            .limit(1)
            .execute()
        )
        rows = existing.data or []
        if rows:
            client.table("projects").update(item).eq("id", rows[0]["id"]).execute()
        else:
            client.table("projects").insert(item).execute()


def upsert_projects(client: Client, projects: list[dict[str, Any]]) -> int:
    if not projects:
        return 0

    use_native = True
    saved = 0
    chunk_size = 200
    for i in range(0, len(projects), chunk_size):
        chunk = projects[i : i + chunk_size]
        if use_native:
            try:
                _upsert_chunk_native(client, chunk)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "on conflict" in msg or "42p10" in msg:
                    use_native = False
                    _upsert_chunk_fallback(client, chunk)
                else:
                    raise
        else:
            _upsert_chunk_fallback(client, chunk)
        saved += len(chunk)
    return saved


def main() -> None:
    cfg = load_config()
    validate_config(cfg)

    limit_msg = (
        f", 최대 {cfg['collect_limit']}건"
        if cfg.get("collect_limit") is not None
        else ", 건수 제한 없음"
    )
    print(
        f"열린재정 데이터 수집 시작 "
        f"(서비스={cfg['api_service']}, 연도={cfg['fiscal_year']}{limit_msg})..."
    )
    projects = collect_raw_projects(cfg)
    print(f"API 수집·중복 병합 완료: {len(projects)}건")

    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    saved = upsert_projects(client, projects)
    print(f"총 {saved}건 저장 완료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
