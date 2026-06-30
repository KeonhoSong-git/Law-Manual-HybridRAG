"""data_tools.ingest_hwpx — hwpx/txt 문서를 읽어 build_v3 입력(by_document/*.json)으로 변환.

hwpx = OWPML(zip). **외부 의존 없이** section*.xml의 `<hp:t>`(텍스트 런)을 `<hp:p>`(단락)
단위로 추출한 뒤, `제N조(제목)` 경계로 청킹한다(조문 구조가 없으면 전문 1청크).

> 표·도형·이미지·페이지 순서까지 정밀 복원이 필요하면 별도 `hwpx_chunker`(rhwp 엔진)를
> 쓴다. 이 모듈은 **본문 텍스트 위주의 경량 추출**이다.

지원 입력: **.hwpx, .txt**  (구버전 .hwp / .pdf 는 미지원 — 외부 변환 후 .hwpx/.txt 로)
사용: python -m data_tools.ingest_hwpx --in <문서_디렉터리> --out datasets/by_document
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

from data_tools.chunk_ids import assign_chunk_ids
import zipfile

_HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"  # HWPML paragraph 네임스페이스
_SEC = re.compile(r"section\d+\.xml$", re.I)
_ART_SPLIT = re.compile(r"(?=제\s*\d+\s*조\s*\()")          # 제N조(제목) 경계
_ART_HEAD = re.compile(r"제\s*(\d+)\s*조\s*\(([^)]{1,40})\)")
_DATE = re.compile(r"(\d{4})\s*[.\-]\s*(\d{1,2})\s*[.\-]\s*(\d{1,2})")


def read_hwpx(path: str) -> list[str]:
    """hwpx → 단락 텍스트 리스트 (<hp:p> 단위로 <hp:t> 합침)."""
    paras = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if _SEC.search(n)):
            try:
                root = ET.fromstring(z.read(name))
            except ET.ParseError:
                continue
            for p in root.iter(_HP + "p"):
                txt = "".join(t.text or "" for t in p.iter(_HP + "t"))
                txt = re.sub(r"\s+", " ", txt.replace(" ", " ")).strip()
                if txt:
                    paras.append(txt)
    return paras


def read_txt(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.strip() for ln in f if ln.strip()]


def to_chunks(title: str, paras: list[str], source: str) -> list[dict]:
    """단락 → build_v3 청크 배열. 제N조(제목) 단위 분할, 없으면 전문 1청크."""
    full = re.sub(r"\s+", " ", " ".join(paras)).strip()
    if not full:
        return []
    family = "법규체" if _ART_HEAD.search(full) else "공문체"
    m = _DATE.search(full)
    enacted = f"{m.group(1)}. {int(m.group(2))}. {int(m.group(3))}." if m else ""
    meta = {"doc_title": title, "source_file": source, "doc_family": family,
            "doc_type": "문서", "enacted": enacted, "last_amended": ""}
    chunks = []
    for seg in _ART_SPLIT.split(full):
        seg = seg.strip()
        mm = _ART_HEAD.match(seg)
        if mm and len(seg) > 15 and "삭제" not in seg[:22]:
            chunks.append({**meta, "unit": "조", "article_label": f"제{mm.group(1)}조",
                           "article_title": mm.group(2).strip(), "text": seg})
    if not chunks:  # 조문 구조 없음(공문/매뉴얼) → 전문 1청크
        chunks = [{**meta, "unit": "문서", "article_label": "", "article_title": "", "text": full[:8000]}]
    return assign_chunk_ids(chunks)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="hwpx/txt → by_document 청크 변환")
    ap.add_argument("--in", dest="src", required=True, help="문서 디렉터리(.hwpx/.txt)")
    ap.add_argument("--out", required=True, help="by_document 출력 디렉터리")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = (glob.glob(os.path.join(a.src, "**", "*.hwpx"), recursive=True)
             + glob.glob(os.path.join(a.src, "**", "*.txt"), recursive=True))
    n = 0
    for fp in sorted(files):
        title = re.sub(r"\.(hwpx|txt)$", "", os.path.basename(fp), flags=re.I)
        try:
            paras = read_hwpx(fp) if fp.lower().endswith(".hwpx") else read_txt(fp)
            chunks = to_chunks(title, paras, os.path.basename(fp))
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {title}: {type(e).__name__}", flush=True)
            continue
        if not chunks:
            print(f"  EMPTY {title}", flush=True)
            continue
        safe = re.sub(r"[^0-9A-Za-z가-힣]+", "_", title)[:60]
        with open(os.path.join(a.out, f"hwpx_{safe}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1)
        n += 1
        print(f"  [{n}] {title} — {len(chunks)} 청크", flush=True)
    print(f"\n완료: {n}개 문서 → {a.out}")


if __name__ == "__main__":
    main()
