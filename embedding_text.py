"""저장·검색 공통 임베딩 텍스트 조합."""

from __future__ import annotations

from typing import Any


def make_embedding_text(
    *,
    fld_nm: str = "",
    sect_nm: str = "",
    pgm_nm: str = "",
    actv_nm: str = "",
    sactv_nm: str = "",
    bz_cls_nm: str = "",
    project_name: str = "",
    category: str = "",
    overview: str = "",
) -> str:
    """계층 분류·사업명·개요를 중복 없이 이어 붙인 임베딩용 텍스트."""
    parts = [
        fld_nm or category,
        sect_nm,
        pgm_nm,
        actv_nm,
        sactv_nm,
        bz_cls_nm,
        project_name,
        overview,
    ]
    seen: set[str] = set()
    tokens: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in seen:
            seen.add(text)
            tokens.append(text)
    return " ".join(tokens)


def make_embedding_text_from_row(row: dict[str, Any]) -> str:
    return make_embedding_text(
        fld_nm=str(row.get("fld_nm") or ""),
        sect_nm=str(row.get("sect_nm") or ""),
        pgm_nm=str(row.get("pgm_nm") or ""),
        actv_nm=str(row.get("actv_nm") or ""),
        sactv_nm=str(row.get("sactv_nm") or ""),
        bz_cls_nm=str(row.get("bz_cls_nm") or ""),
        project_name=str(row.get("project_name") or ""),
        category=str(row.get("category") or ""),
        overview=str(row.get("overview") or ""),
    )


def make_embedding_text_from_query(field: str, project_name: str, overview: str) -> str:
    return make_embedding_text(
        fld_nm=field,
        project_name=project_name,
        overview=overview,
    )
