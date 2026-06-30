"""kg.classify_edges_v3 — 법규체 문서 간 참조의 '종류'를 LLM으로 분류 (제안→검증).

build_v3가 만든 결정적 인용엣지(dct:references) 중 법규체→법규체 참조를, 위임(based_on)/
준용(applies)/개정(amends)/단순참조(cites)로 분류한다. 법령KG 주류 방식: 참조 '존재'는
결정적, 참조 '종류'만 LLM. 게이트: evidence∈원문 + 방향(하위→상위). 소스문서별 캐시로 재개.

출력: out_dir/kg_relations.ttl (eli:based_on/applies/amends/cites 오버레이).
사용: python -m kg.classify_edges_v3 <by_document_dir> <out_dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

from . import llm
from .build_v3 import PREFIXES, _QUOTE, _doc_iri, _norm, load_by_document, resolve_citation

_REL = {"based_on": "eli:based_on", "applies": "eli:applies",
        "amends": "eli:amends", "cites": "eli:cites"}
_SYS = (
    "너는 규정 문서 관계 분류기다. '현재 문서'가 각 '대상 규정'과 맺는 관계를 ELI 기준으로 분류한다.\n"
    "- based_on: 현재 문서가 대상 규정을 **법적 근거**로 제정됨. 단서어 '…에 의한/에 따라/에서 위임받아/근거하여/제N조에 의한'.\n"
    "- applies: 현재 문서가 대상 규정을 **준용**. 단서어 '…를 준용한다'.\n"
    "- amends: 대상 규정을 **개정/변경**.\n"
    "- cites: 위 셋 어디에도 안 맞는 **단순 참조·언급**.\n"
    "예) '이 기준은 「감사규정」 제45조에 의한 세부사항을 정한다' → 감사규정: based_on.\n"
    "예) '본 요령은 「보증규정」을 준용한다' → 보증규정: applies.\n"
    "반드시 본문에 근거가 있는 것만. 근거 없으면 그 대상은 제외.\n"
    '출력: JSON 배열 [{"target":"<대상 규정 제목>","relation":"based_on|applies|amends|cites","evidence":"<본문 인용구>"}].'
)


def _rank(title: str) -> int:
    """문서 위계(낮을수록 상위). 법률 > 시행령 > 시행규칙 > 행정규칙(규정/고시/훈령/예규) > 요령 > 기준/지침/세칙.
    based_on/applies 는 하위(높은 rank) → 상위(낮은 rank) 만 정상. 상위 법령(법/령/규칙)을 누락하면
    하위규정→상위법 의 가장 흔한 근거관계가 '역방향'으로 오판돼 드롭되므로 최상위 티어를 반드시 포함한다.
    (시행령/시행규칙은 '령/규칙' 접미보다 먼저 검사.)"""
    t = title
    if t.endswith("법") or "법률" in t:
        return -3
    if "시행령" in t or t.endswith("령"):
        return -2
    if "시행규칙" in t or t.endswith("규칙"):
        return -1
    if "규정" in t or "정관" in t or "고시" in t or "훈령" in t or "예규" in t:
        return 1
    if "요령" in t:
        return 2
    if "기준" in t or "지침" in t or "세칙" in t:
        return 3
    return 4


def _targets(text: str, ntext: str, src_title: str, legal_titles: dict) -> set:
    """현재 문서가 참조하는 법규체 대상 제목 집합 (「」 + 평문 정확제목)."""
    out = set()
    for m in _QUOTE.finditer(text):
        hit = resolve_citation(m.group(1), legal_titles, min_len=6)
        if hit and hit != src_title:
            out.add(hit)
    for lt_n, lt in legal_titles.items():
        if len(lt_n) >= 7 and lt != src_title and lt_n in ntext:
            out.add(lt)
    return out


# 준용 단서: 「대상규정」 … 준용  (표면형이 규칙적 → 결정적 추출, LLM 불필요)
_JY = re.compile(r"[「『]([^」』]{2,40})[」』]([^。\n]{0,60}?)준용")


def _deterministic_applies(recs: list, edges: set, st: Counter) -> None:
    """본문에서 '「X」…준용' 을 직접 applies 엣지로. 부정문(준용하지 아니/되지 아니)은 제외.
    domain/range: 법규체↔법규체 만(주 분류 루프와 동일 제약 — recs 전체를 쓰면 비법규체 applies 가 샌다)."""
    legal = [r for r in recs if r.get("family") == "법규체"]
    titles = {_norm(r["title"]): r["title"] for r in legal}

    for r in legal:
        src = r["title"]
        for m in _JY.finditer(r["text"]):
            tail = r["text"][m.end():m.end() + 12]
            if "아니" in tail or "않" in tail:          # 준용하지 아니한다 등 부정 제외
                continue
            tgt = resolve_citation(m.group(1), titles, min_len=6)
            if not tgt or tgt == src:
                continue
            if _rank(src) < _rank(tgt):                 # 방향 게이트: 하위→상위만(LLM 경로와 동일)
                st["rej_direction_det"] += 1
                continue
            e = (_doc_iri(src), _REL["applies"], _doc_iri(tgt))
            if e not in edges:
                edges.add(e); st["applies_deterministic"] += 1


def classify(src_dir: str, out_dir: str, max_chars: int = 4000) -> dict:
    recs = load_by_document(src_dir)
    legal = [r for r in recs if r["family"] == "법규체"]
    legal_titles = {_norm(r["title"]): r["title"] for r in legal}
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "edge_llm_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except (ValueError, OSError):
            cache = {}

    edges: set[tuple[str, str, str]] = set()
    st = Counter()
    for i, r in enumerate(legal):
        src_title = r["title"]
        ntext = _norm(r["text"])
        targets = _targets(r["text"], ntext, src_title, legal_titles)
        if not targets:
            continue
        st["src_with_targets"] += 1
        # 캐시 키에 본문 해시를 포함 → 원문이 바뀌면 stale LLM 결과를 재사용하지 않고 재분류.
        ckey = src_title + "\x01" + hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:12]
        if ckey in cache:
            picked = cache[ckey]
        elif llm.available():
            tlist = "\n".join("- " + t for t in sorted(targets))
            try:
                picked = llm.parse_json_array(llm.chat(
                    _SYS, f"현재 문서: {src_title}\n대상 규정:\n{tlist}\n\n본문:\n{r['text'][:max_chars]}"))
            except Exception:  # noqa: BLE001
                continue
            st["llm_calls"] += 1
            cache[ckey] = picked
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            if st["llm_calls"] % 20 == 0:
                print(f"  {i + 1}/{len(legal)} classified (llm_calls={st['llm_calls']})", flush=True)
        else:
            continue
        s = _doc_iri(src_title)
        for p in picked or []:
            tt, rel = p.get("target"), p.get("relation")
            ev = p.get("evidence") or ""
            # 보수적 타이핑(확실한 것만): evidence에 명시 신호어 있을 때만 종류 인정, 아니면 cites.
            if rel == "amends" and not any(k in ev for k in ("개정", "변경")):
                rel = "cites"
            elif rel == "applies" and ("준용" not in ev or any(n in ev for n in ("아니", "않"))):
                rel = "cites"   # 준용 신호 없음 OR 부정문('준용하지 아니')이면 단순참조로 강등
            elif rel == "based_on" and "위임" not in ev:   # 위임 명시일 때만(에 따라/근거는 너무 흔해 제외)
                rel = "cites"
            pred = _REL.get(rel)
            if tt not in targets or not pred:
                st["rej_unknown"] += 1
                continue
            if not _norm(ev) or _norm(ev) not in ntext:     # 충실성: 근거가 비었거나 원문에 없으면 드롭(환각 차단)
                st["rej_evidence"] += 1
                continue
            # 방향: 역방향(상위→하위) OR 양쪽 미분류(rank 4 — 위계 확인 불가)면 based_on/applies 드롭.
            if rel in ("based_on", "applies") and (
                _rank(src_title) < _rank(tt) or _rank(src_title) == _rank(tt) == 4
            ):
                st["rej_direction"] += 1
                continue
            edges.add((s, pred, _doc_iri(tt)))
            st["kept"] += 1

    _deterministic_applies(recs, edges, st)   # 결정적 준용 보강(LLM 게이트가 놓친 것)

    lines = [f"{s} {p} {o} ." for s, p, o in sorted(edges)]
    ttl = "\n".join(PREFIXES) + "\n\n" + "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "kg_relations.ttl"), "w", encoding="utf-8") as f:
        f.write(ttl)
    json.dump(dict(st), open(os.path.join(out_dir, "edge_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return dict(st)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    src, out = sys.argv[1], sys.argv[2]
    print(json.dumps(classify(src, out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
