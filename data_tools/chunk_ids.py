"""청크 결정적 식별자 — 트리플↔청크 정밀 역추적 + 내용 기반 증분 재추출용 공용 헬퍼.

두 필드를 부여한다(목표가 서로 달라 분리):
- chunk_id      = "{source_id}#{seq:04d}"  안정 주소. 본문이 바뀌어도 유지 → (a) 역추적 참조 불변.
- content_hash  = sha1(text)[:12]          변경 감지. 본문이 바뀌면 값이 바뀜 → (b) 증분 재추출.

source_id 는 build_v3 의 문서 IRI(`doc:d…`) localname 과 동일한 규칙(uuid5(정규화 제목))이라
트리플(`dct:identifier`)·벡터 메타·청크가 같은 키로 자연스럽게 조인된다. NS 는 build_v3/embed_corpus 와 동일.
"""
from __future__ import annotations

import hashlib
import re
import uuid

NS = uuid.UUID("c1d2e3f4-5a6b-7c8d-9e0f-1a2b3c4d5e6f")  # build_v3 / embed_corpus 와 동일 네임스페이스


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def source_id(doc_title: str) -> str:
    """문서 단위 결정적 id. build_v3 의 doc IRI(`doc:d` + uuid5(정규화 제목)[:16]) localname 과 일치."""
    return "d" + uuid.uuid5(NS, _norm(doc_title)).hex[:16]


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def assign_chunk_ids(chunks: list[dict]) -> list[dict]:
    """한 문서의 청크 리스트(문서 내 순서대로)에 chunk_id / content_hash 를 in-place 부여."""
    for seq, c in enumerate(chunks):
        sid = source_id(c.get("doc_title") or "")
        c["chunk_id"] = f"{sid}#{seq:04d}"
        c["content_hash"] = content_hash(c.get("text") or "")
    return chunks
