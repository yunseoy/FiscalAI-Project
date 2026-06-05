"""Supabase projects 테이블 → OpenAI 임베딩 → embedding 컬럼 저장.

실행:
    python embed_projects.py

환경변수 (.env):
    OPENAI_API_KEY
    SUPABASE_URL, SUPABASE_ANON_KEY
    EMBED_BATCH_SIZE  — 한 번에 임베딩할 건수, 기본 100
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
REQUEST_DELAY_SEC = 0.1


def load_config() -> dict[str, Any]:
    batch_raw = os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)).strip()
    try:
        batch_size = int(batch_raw)
    except ValueError:
        batch_size = DEFAULT_BATCH_SIZE

    return {
        "openai_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "batch_size": batch_size,
    }


def validate_config(cfg: dict[str, Any]) -> None:
    missing: list[str] = []
    if not cfg["openai_key"]:
        missing.append("OPENAI_API_KEY")
    if not cfg["supabase_url"]:
        missing.append("SUPABASE_URL")
    if not cfg["supabase_key"]:
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        raise SystemExit(
            "다음 환경변수가 `.env`에 설정되어 있지 않습니다: " + ", ".join(missing)
        )


def make_embedding_text(row: dict[str, Any]) -> str:
    """임베딩용 텍스트 조합 (분류명 + 사업명, 소관명 제외)."""
    parts = [
        row.get("fld_nm") or "",    # 분야명
        row.get("sect_nm") or "",   # 부문명
        row.get("pgm_nm") or "",    # 프로그램명
        row.get("actv_nm") or "",   # 단위사업명
        row.get("sactv_nm") or "",  # 세부사업명
        row.get("bz_cls_nm") or "", # 사업분류명
        row.get("project_name") or "", # 사업명
    ]
    return " ".join(p for p in parts if p.strip())


def fetch_unembedded_projects(client: Client) -> list[dict[str, Any]]:
    """embedding이 NULL인 프로젝트 전체 조회 (페이지네이션)."""
    all_rows = []
    offset = 0
    batch = 1000

    while True:
        response = (
            client.table("projects")
            .select(
                "id, project_name, ministry, "
                "fld_nm, sect_nm, pgm_nm, actv_nm, sactv_nm, bz_cls_nm"
            )
            .is_("embedding", "null")
            .range(offset, offset + batch - 1)
            .execute()
        )
        rows = response.data or []
        all_rows.extend(rows)
        print(f"  조회 중: {len(all_rows)}건...")

        if len(rows) < batch:
            break
        offset += batch

    return all_rows


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """텍스트 목록 → 벡터 목록."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def save_embeddings(
    supabase: Client,
    ids: list[str],
    embeddings: list[list[float]],
) -> None:
    """embedding 컬럼 업데이트."""
    for pid, embedding in zip(ids, embeddings):
        supabase.table("projects").update(
            {"embedding": embedding}
        ).eq("id", pid).execute()


def main() -> None:
    cfg = load_config()
    validate_config(cfg)

    supabase = create_client(cfg["supabase_url"], cfg["supabase_key"])
    openai_client = OpenAI(api_key=cfg["openai_key"])

    print("임베딩 대상 프로젝트 조회 중...")
    projects = fetch_unembedded_projects(supabase)

    if not projects:
        print("임베딩할 프로젝트가 없습니다. (이미 모두 완료됐거나 데이터 없음)")
        return

    print(f"총 {len(projects)}건 임베딩 시작...")

    batch_size = cfg["batch_size"]
    total_saved = 0

    for i in range(0, len(projects), batch_size):
        batch = projects[i: i + batch_size]

        ids = [row["id"] for row in batch]
        texts = [make_embedding_text(row) for row in batch]

        # 빈 텍스트 체크
        for j, (pid, text) in enumerate(zip(ids, texts)):
            if not text.strip():
                print(f"  ⚠️  텍스트 없음 (id={pid}, project_name={batch[j].get('project_name')})")
                texts[j] = batch[j].get("project_name") or pid  # fallback

        try:
            embeddings = embed_texts(openai_client, texts)
            save_embeddings(supabase, ids, embeddings)
            total_saved += len(batch)
            print(f"  ✓ {total_saved}/{len(projects)}건 완료")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 오류 발생 (배치 {i}~{i + batch_size}): {exc}")
            raise

        time.sleep(REQUEST_DELAY_SEC)

    print(f"\n임베딩 완료: 총 {total_saved}건 저장됨")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)