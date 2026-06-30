"""data_tools.fetch_manual — 국가법령정보센터(law.go.kr) OPEN API에서 **매뉴얼·지침서(공문체)** 수집기.

법령·행정규칙(조문체)과 달리, 실제 업무 **매뉴얼/안내서/가이드**는 law.go.kr 본문(조문내용)이
비어 있고 **PDF 첨부파일**로만 제공된다(`fetch_law.py`로는 본문이 안 잡힘). 이 모듈은
행정규칙을 검색해 **첨부 PDF를 받아 텍스트를 추출**하고, 조문 구조가 없는 공문체 문서를
**본문 윈도우 단위로 청킹**해 `by_document/<문서>.json`(build_v3 입력 형식)으로 저장한다.

[조문체 vs 공문체]
- `fetch_law.py`  → 법령·행정규칙(제N조 조문 구조) = 법규체
- `fetch_manual.py` → 매뉴얼·지침서(아웃라인/서술형) = **공문체** (KG는 문서 노드까지만, 세부는 벡터)

[품질 게이트] 스캔(이미지) PDF·글자깨짐(ToUnicode 없는 CID 폰트)은 한글 추출이 안 되므로,
추출 한글 글자수가 임계 미만이면 건너뛴다.

요구: pymupdf(fitz) — PDF 텍스트 추출용(수집 단계 한정). `pip install pymupdf`
env: LAW_OC (필수, open.law.go.kr 발급 OC + 호출 기기 IP 등록)
사용:
  python -m data_tools.fetch_manual --out datasets/by_document --max 40
  python -m data_tools.fetch_manual --queries 매뉴얼,안내서,가이드,업무편람,지침서,운영요령 --max 60 --out datasets/by_document
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request

from data_tools.chunk_ids import assign_chunk_ids

BASE = "http://www.law.go.kr/DRF"
_UA = {"User-Agent": "Mozilla/5.0"}
_DEFAULT_QUERIES = ["매뉴얼", "안내서", "가이드", "업무편람", "지침서", "운영요령"]
# 제목이 이걸 포함하면 '실무 매뉴얼류'로 본다(고시/훈령이어도 첨부 PDF가 실질 매뉴얼인 경우 포함).
_MANUAL_HINT = re.compile(r"매뉴얼|안내서|가이드|편람|지침서|운영요령|핸드북|업무처리요령")
# 본문이 아니라 별표·서식·양식인 첨부는 후순위.
_AUX_NAME = re.compile(r"별표|별지|서식|양식|샘플|템플릿|표지")


def _get_json(url: str) -> dict:
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60).read().decode("utf-8")
    if "사용자 정보 검증" in raw:
        raise SystemExit("OPEN API 인증 실패 — OC와 호출 기기 IP/도메인 등록을 확인하세요(open.law.go.kr).")
    return json.loads(raw)


def _key(d: dict, sub: str) -> str | None:
    for k in d:
        if sub in k:
            return k
    return None


def _find(obj, sub):
    """중첩 dict/list에서 key 부분일치하는 첫 값."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if sub in k:
                return v
        for v in obj.values():
            r = _find(v, sub)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, sub)
            if r is not None:
                return r
    return None


def _fmt_date(s) -> str:
    s = re.sub(r"\D", "", str(s or ""))
    return f"{s[0:4]}. {int(s[4:6])}. {int(s[6:8])}." if len(s) == 8 else ""


def search(oc: str, query: str, display: int, pages: int = 1) -> list[dict]:
    """admrul 검색. display는 페이지당(≤100), pages로 그 이상까지 페이지네이션."""
    out, seen = [], set()
    for page in range(1, pages + 1):
        url = (f"{BASE}/lawSearch.do?OC={oc}&target=admrul&type=JSON"
               f"&display={min(display, 100)}&page={page}&query={urllib.parse.quote(query)}")
        items = _find(_get_json(url), "admrul") or []
        items = [items] if isinstance(items, dict) else items
        if not items:
            break
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = (it.get(_key(it, "규칙명") or "") or "").strip()
            sid = it.get(_key(it, "일련번호") or "")
            if nm and sid and sid not in seen:
                seen.add(sid)
                out.append({"name": nm, "sid": str(sid),
                            "enacted": _fmt_date(it.get(_key(it, "발령일자") or "")),
                            "amended": _fmt_date(it.get(_key(it, "시행일자") or ""))})
        time.sleep(0.2)
    return out


def _pdf_attachments(svc: dict) -> list[tuple[str, str]]:
    """서비스 응답 → [(파일명, 다운로드URL)] 중 PDF만. 본문 우선(별표/서식 후순위)."""
    links = _find(svc, "첨부파일링크")
    names = _find(svc, "첨부파일명")
    if links is None:
        return []
    links = [links] if isinstance(links, str) else list(links)
    names = [names] if isinstance(names, str) else list(names or [])
    pairs = []
    for i, url in enumerate(links):
        nm = names[i] if i < len(names) else ""
        if str(nm).lower().endswith(".pdf"):
            pairs.append((nm, url))
    pairs.sort(key=lambda p: 1 if _AUX_NAME.search(p[0]) else 0)  # 본문 먼저
    return pairs


def extract_pdf(url: str, max_pages: int) -> str:
    import fitz  # pymupdf — 지연 임포트(수집 단계에서만 필요)
    data = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120).read()
    if data[:4] != b"%PDF":
        return ""
    fp = os.path.join(tempfile.gettempdir(), "_fm.pdf")
    with open(fp, "wb") as f:
        f.write(data)
    doc = fitz.open(fp)
    try:
        txt = "".join(page.get_text() for page in doc[:max_pages])
    finally:
        doc.close()
    return re.sub(r"[ \t]+", " ", txt)


def clean_text(t: str) -> str:
    """PDF 추출 잔여물 정리 — 목차 점선(@@@/…)·장식 글리프·PUA 문자 제거."""
    t = re.sub(r"@{2,}", " ", t)                 # 목차 dot leader가 @로 추출됨
    t = re.sub(r"[·∙‥…]{2,}", " ", t)            # 점선 leader
    t = re.sub(r"[-]", " ", t)       # 사설영역(PUA) — 깨진 글리프
    t = re.sub(r"[•■-◿☀-➿]", " ", t)  # 불릿/화살표/기호
    return re.sub(r"[ 	]{2,}", " ", t)


def chunk_outline(full: str, win: int = 900) -> list[str]:
    """공문체 본문 → 문장 경계 기준 ~win자 윈도우 청크. 아웃라인 헤더(1.,제N장 등)에서 우선 분절."""
    full = clean_text(re.sub(r"\n{2,}", "\n", full)).strip()
    # 1차: 최상위 아웃라인 경계로 큰 절 분리(있으면).
    parts = re.split(r"(?=\n\s*(?:제\s*\d+\s*[장절]|[ⅠⅡⅢⅣⅤⅥ]\.|\d{1,2}\.\s))", full)
    segs: list[str] = []
    for p in (parts if len(parts) > 1 else [full]):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) <= win:
            if p:
                segs.append(p)
            continue
        # 2차: 긴 절은 문장 단위로 윈도우.
        buf = ""
        for sent in re.split(r"(?<=[.。!?·])\s+|(?<=다\.)\s+", p):
            if len(buf) + len(sent) > win and buf:
                segs.append(buf.strip())
                buf = ""
            buf += sent + " "
        if buf.strip():
            segs.append(buf.strip())
    return [s for s in segs if len(s) >= 30]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="law.go.kr 매뉴얼/지침서(PDF 첨부) 수집기")
    ap.add_argument("--out", required=True, help="by_document 출력 디렉터리")
    ap.add_argument("--queries", default=",".join(_DEFAULT_QUERIES), help="콤마구분 검색어")
    ap.add_argument("--max", type=int, default=100, help="검색어당 페이지당 후보 수(≤100)")
    ap.add_argument("--pages", type=int, default=1, help="검색어당 페이지 수(페이지네이션)")
    ap.add_argument("--min-korean", type=int, default=400, help="추출 한글 글자수 최소(스캔/깨짐 제외)")
    ap.add_argument("--max-pages", type=int, default=60, help="PDF 추출 최대 페이지")
    a = ap.parse_args()

    oc = os.environ.get("LAW_OC")
    if not oc:
        raise SystemExit("LAW_OC 환경변수가 필요합니다(open.law.go.kr 발급 OC).")
    try:
        import fitz  # noqa: F401
    except ImportError:
        raise SystemExit("pymupdf가 필요합니다: pip install pymupdf")
    os.makedirs(a.out, exist_ok=True)

    seen: set[str] = set()
    n = 0
    for q in [s.strip() for s in a.queries.split(",") if s.strip()]:
        hits = search(oc, q, a.max, a.pages)
        print(f"[{q}] 검색 {len(hits)}건", flush=True)
        for h in hits:
            if h["sid"] in seen or not _MANUAL_HINT.search(h["name"]):
                continue
            seen.add(h["sid"])
            safe = re.sub(r"[^0-9A-Za-z가-힣]+", "_", h["name"])[:60]
            out_fp = os.path.join(a.out, f"manual_{safe}.json")
            if os.path.exists(out_fp):
                continue
            try:
                svc = _get_json(f"{BASE}/lawService.do?OC={oc}&target=admrul&type=JSON&ID={h['sid']}")
                pdfs = _pdf_attachments(_find(svc, "AdmRulService") or svc)
                full = ""
                for _nm, url in pdfs:
                    full = extract_pdf(url, a.max_pages)
                    if sum(1 for c in full if "가" <= c <= "힣") >= a.min_korean:
                        break
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {h['name']}: {type(e).__name__}", flush=True)
                continue
            ko = sum(1 for c in full if "가" <= c <= "힣")
            if ko < a.min_korean:
                print(f"  SCAN/EMPTY {h['name']} (한글 {ko})", flush=True)
                continue
            meta = {"doc_title": h["name"], "source_file": f"매뉴얼_{h['name']}.pdf",
                    "doc_family": "공문체", "doc_type": "매뉴얼",
                    "enacted": h["enacted"], "last_amended": h["amended"]}
            texts = chunk_outline(full)
            chunks = [{**meta, "unit": "본문", "article_label": f"§{i + 1}",
                       "article_title": "", "text": t} for i, t in enumerate(texts)]
            if not chunks:
                continue
            assign_chunk_ids(chunks)
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=1)
            n += 1
            print(f"  [{n}] {h['name']} — {len(chunks)} 청크 (한글 {ko})", flush=True)
            time.sleep(0.3)
    print(f"\n완료: {n}개 매뉴얼 → {a.out}")


if __name__ == "__main__":
    main()
