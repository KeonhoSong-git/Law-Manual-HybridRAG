"""data_tools.fetch_law — 국가법령정보센터(law.go.kr) OPEN API 수집기.

법령(target=law)·행정규칙(target=admrul)을 받아 `kg.build_v3` 입력 형식
(`by_document/<문서>.json`: 청크 dict 배열)으로 변환한다.

[수집 방법 메모]
law.go.kr 본문은 JS로 렌더되는 SPA라 HTML 직접 스크래핑으로는 조문 본문이 안 나온다
(원문 페이지가 제목 수준만 반환). 그래서 **OPEN API(DRF lawSearch.do / lawService.do)** 로
조문 구조를 직접 받는다.

[사전조건]
open.law.go.kr 에서 OPEN API 신청 → OC(보통 이메일 ID) 발급 + **호출 기기의 IP/도메인 등록**
(미등록 IP에서 호출하면 "사용자 정보 검증 실패"). 즉 이 스크립트는 등록된 기기에서 실행한다.

env: LAW_OC (필수)
사용:
  python -m data_tools.fetch_law --query 신용보증 --target law    --out datasets/by_document --max 20
  python -m data_tools.fetch_law --query 보증     --target admrul --out datasets/by_document --max 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from data_tools.chunk_ids import assign_chunk_ids

BASE = "http://www.law.go.kr/DRF"
# 본문 doc_family: 법령/행정규칙 모두 조문(제N조) 구조 → build_v3의 법규체 경로를 태운다.
_FAMILY = {"law": "법규체", "admrul": "법규체"}
_DOCTYPE = {"law": "법령", "admrul": "행정규칙"}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    if "사용자 정보 검증" in raw:
        raise SystemExit("OPEN API 인증 실패 — OC와 호출 기기 IP/도메인 등록을 확인하세요(open.law.go.kr).")
    return json.loads(raw)


def _walk(obj, key_substr):
    """중첩 dict/list에서 key에 부분일치하는 첫 값을 찾는다(응답 스키마 변동 방어)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if key_substr in k and isinstance(v, str) and v.strip():
                return v
        for v in obj.values():
            r = _walk(v, key_substr)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk(v, key_substr)
            if r:
                return r
    return None


def _find_list(obj, key_substr):
    """중첩에서 key에 부분일치하는 list(조문단위 등)를 찾는다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if key_substr in k and isinstance(v, list):
                return v
        for v in obj.values():
            r = _find_list(v, key_substr)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_list(v, key_substr)
            if r:
                return r
    return None


def search(oc: str, target: str, query: str, display: int) -> list[dict]:
    url = (f"{BASE}/lawSearch.do?OC={oc}&target={target}&type=JSON"
           f"&display={display}&query={urllib.parse.quote(query)}")
    d = _get(url)
    items = _find_list(d, "law" if target == "law" else "admrul") or []
    out = []
    for it in items:
        mst = it.get("법령일련번호") or it.get("행정규칙일련번호") or it.get("MST") or _walk(it, "일련번호")
        name = it.get("법령명한글") or it.get("행정규칙명") or _walk(it, "명")
        if mst and name:
            out.append({"mst": str(mst), "name": name.strip()})
    return out


def fetch_content(oc: str, target: str, key: str) -> dict:
    # 법령은 MST(일련번호), 행정규칙은 LM(행정규칙명)으로 조회한다.
    sel = f"LM={urllib.parse.quote(key)}" if target == "admrul" else f"MST={urllib.parse.quote(str(key))}"
    return _get(f"{BASE}/lawService.do?OC={oc}&target={target}&type=JSON&{sel}")


# 실제 조문 시작 = 제N조(제목). 인라인 인용(제7조제2항 등)에서 쪼개지지 않도록 괄호 제목을 요구.
_ART_SPLIT = re.compile(r"(?=제\s*\d+\s*조\s*\()")
_ART_HEAD = re.compile(r"제\s*(\d+)\s*조\s*\(([^)]{1,40})\)")


def _split_articles(full: str) -> list[tuple[str, str, str]]:
    """행정규칙 본문 문자열을 제N조(제목) 단위로 분할 → (label, title, text)."""
    out = []
    for p in _ART_SPLIT.split(full):
        p = p.strip()
        m = _ART_HEAD.match(p)
        if m and len(p) > 15 and "삭제" not in p[:22]:
            out.append((f"제{m.group(1)}조", m.group(2).strip(), p))
    return out


def _fmt_date(s) -> str:
    """YYYYMMDD → 'YYYY. M. D.' (build_v3 날짜 정규식이 먹는 형식)."""
    s = re.sub(r"\D", "", str(s or ""))
    return f"{s[0:4]}. {int(s[4:6])}. {int(s[6:8])}." if len(s) == 8 else ""


def _collect_text(o) -> list[str]:
    """조문단위 내 모든 '*내용'(조문내용·항내용·호내용·목내용)을 순서대로 수집(항·호 포함)."""
    acc: list[str] = []

    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.endswith("내용") and isinstance(v, str):
                    acc.append(v)
                else:
                    rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(o)
    return acc


def to_chunks(doc: dict, name: str, target: str) -> list[dict]:
    """조문단위 리스트 → build_v3 청크(dict) 배열. 첫 청크가 문서 메타를 운반."""
    enacted = _fmt_date(_walk(doc, "공포일자"))
    amended = _fmt_date(_walk(doc, "시행일자"))
    arts = _find_list(doc, "조문단위") or _find_list(doc, "조문") or []
    meta = {"doc_title": name, "source_file": f"{_DOCTYPE[target]}_{name}.json",
            "doc_family": _FAMILY[target], "doc_type": _DOCTYPE[target],
            "enacted": enacted, "last_amended": amended}
    chunks = []
    for a in arts:
        if not isinstance(a, dict) or a.get("조문여부") == "전문":  # 장/절 헤딩 제외
            continue
        no = str(a.get("조문번호") or "").strip()
        label = f"제{no}조" if no else ""
        title = (a.get("조문제목") or "").strip()
        text = re.sub(r"<[^>]+>", " ", " ".join(_collect_text(a)))
        text = re.sub(r"\s+", " ", text).strip()
        if not text or re.match(r"제\s*\d+\s*조\s*삭제", text):  # 빈/삭제 조문 제외
            continue
        chunks.append({**meta, "unit": "조", "article_label": label,
                       "article_title": title, "text": text})
    if not chunks:  # 조문단위 없음(행정규칙 등) → 본문 텍스트를 제N조 단위로 분할
        body_list = _find_list(doc, "조문내용") or []
        texts = []
        for it in body_list:
            texts.append(it) if isinstance(it, str) else texts.extend(_collect_text(it))
        full = re.sub(r"<[^>]+>", " ", " ".join(texts) or " ".join(_collect_text(doc)))
        full = re.sub(r"\s+", " ", full).strip()
        for label, title, text in _split_articles(full):
            chunks.append({**meta, "unit": "조", "article_label": label,
                           "article_title": title, "text": text})
        if not chunks and full:  # 조 분할도 실패 → 전문 1청크
            chunks = [{**meta, "unit": "문서", "article_label": "", "article_title": "", "text": full[:8000]}]
    return assign_chunk_ids(chunks)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="law.go.kr OPEN API 수집기")
    ap.add_argument("--query", required=True, help="검색어(예: 신용보증)")
    ap.add_argument("--target", default="law", choices=["law", "admrul"], help="law=법령, admrul=행정규칙")
    ap.add_argument("--out", required=True, help="by_document 출력 디렉터리")
    ap.add_argument("--max", type=int, default=20, help="최대 문서 수")
    a = ap.parse_args()

    oc = os.environ.get("LAW_OC")
    if not oc:
        raise SystemExit("LAW_OC 환경변수가 필요합니다(open.law.go.kr 발급 OC).")
    os.makedirs(a.out, exist_ok=True)

    hits = search(oc, a.target, a.query, a.max)
    print(f"검색 {len(hits)}건: {[h['name'] for h in hits][:8]}", flush=True)
    n = 0
    for h in hits:
        try:
            key = h["name"] if a.target == "admrul" else h["mst"]
            doc = fetch_content(oc, a.target, key)
            chunks = to_chunks(doc, h["name"], a.target)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {h['name']}: {type(e).__name__}", flush=True)
            continue
        if not chunks:
            print(f"  EMPTY {h['name']}", flush=True)
            continue
        safe = re.sub(r"[^0-9A-Za-z가-힣]+", "_", h["name"])[:60]
        with open(os.path.join(a.out, f"{a.target}_{safe}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1)
        n += 1
        print(f"  [{n}] {h['name']} — {len(chunks)} 조문", flush=True)
        time.sleep(0.3)  # 호출 간격(서버 부하 완화)
    print(f"\n완료: {n}개 문서 → {a.out}")


if __name__ == "__main__":
    main()
