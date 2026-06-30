"""kg.traverse — 규정 지식그래프 N홉 탐색 (GraphRAG 유기 검색의 1단계).

질문에서 시작 규정(seed)을 찾고, 관계 엣지(위임/준용/참조/개정/주제관련)를 따라
최대 N홉까지 BFS 하여 **관련 규정 집합**을 반환한다. 다운스트림(reg_chatbot)은
이 관련 문서들에 한해 추가 벡터검색을 수행한다(관련 객체 스코프 검색).

관계는 방향 무관하게 '관련성'으로 본다(위임/준용은 방향이 있으나, 관련 문서 발견엔
양방향 탐색이 유용). 결정적·stdlib-only.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

from . import llm

_REL_PREDS = {"eli:based_on", "eli:applies", "eli:cites", "eli:amends",
              "eli:repeals", "dct:relation", "eli:has_part"}
_PRED_KO = {"eli:based_on": "위임", "eli:applies": "준용", "eli:cites": "참조",
            "eli:amends": "개정", "eli:repeals": "폐지", "dct:relation": "관련",
            "eli:has_part": "포함"}
_STOP = {"업무처리", "업무", "관련", "및", "기준", "규정", "요령",
         "지침", "적용", "처리", "관리", "에", "관한", "은", "는", "이", "가"}


def parse_ttl(text: str) -> list[tuple[str, str, str]]:
    """우리 kg.ttl 형식 전용 경량 파서 (들여쓰기=같은 주어 연속, '.'=문장 끝)."""
    triples, subj = [], None
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("@prefix"):
            continue
        cont = raw[0] in " \t"
        body = raw.strip().rstrip(". ;").strip()
        if cont:
            parts = body.split(None, 1)
            if len(parts) == 2 and subj:
                triples.append((subj, parts[0], parts[1]))
        else:
            parts = body.split(None, 2)
            if len(parts) == 3:
                subj = parts[0]
                triples.append((subj, parts[1], parts[2]))
    return triples


def _labels(triples) -> dict[str, str]:
    out = {}
    for s, p, o in triples:
        if p == "rdfs:label" and '"' in o:
            out[s] = o.split('"')[1]
    return out


def _tok(s: str) -> set[str]:
    return {t for t in re.split(r"\s+", s) if t and t not in _STOP and len(t) > 1}


def find_seeds(question: str, reg_label: dict[str, str]) -> list[str]:
    """문자열 폴백: 질문과 규정 라벨의 부분포함/토큰 겹침으로 시작 규정 IRI를 찾는다."""
    qn = re.sub(r"\s+", "", question)
    qtok = _tok(question)
    scored = []
    for iri, lab in reg_label.items():
        ln = re.sub(r"\s+", "", lab)
        score = 0
        if ln in qn or qn in ln:
            score += 5
        score += len(_tok(lab) & qtok)
        if score:
            scored.append((score, iri))
    scored.sort(key=lambda x: -x[0])
    return [iri for _, iri in scored[:3]]


def find_seeds_llm(question: str, reg_label: dict[str, str]) -> list[str]:
    """LLM 의미 판단: 질문이 어느 규정에 관한 건지 LLM이 고른다(규정명을 안 써도 매칭).

    규정 목록으로 제약(목록 밖 금지)해 환각을 막는다. 실패 시 빈 리스트(폴백 유도).
    """
    if not llm.available() or not reg_label:
        return []
    title2iri = {t: i for i, t in reg_label.items()}
    system = (
        "너는 규정 검색 라우터다. 주어진 '규정 목록' 중 사용자 질문과 직접 관련된 "
        "규정의 정확한 제목만 JSON 배열로 출력한다. 목록에 없는 제목은 절대 만들지 마라. "
        "관련 규정이 없으면 []."
    )
    user = "규정 목록:\n" + "\n".join("- " + t for t in title2iri) + f"\n\n질문: {question}"
    try:
        picked = llm.parse_json_array(llm.chat(system, user, max_tokens=300))
    except Exception:  # noqa: BLE001 - LLM 실패는 폴백으로
        return []
    return [title2iri[t] for t in picked if t in title2iri]


def related_docs(ttl_text: str, question: str, hops: int = 3, use_llm: bool = True) -> dict:
    """질문 → 시작 규정 → N홉 관련 규정. 반환: seeds/related/doc_labels/paths.

    시드(질문↔규정 관련 판단)는 ``use_llm`` 시 LLM 의미 매칭(규정명을 안 써도 됨),
    실패/미설정 시 문자열 매칭으로 폴백한다.
    """
    triples = parse_ttl(ttl_text)
    label = _labels(triples)
    # 규정(Regulation) 노드만 seed 후보
    reg_iris = {s for s, p, o in triples if p == "rdf:type" and o == "eli:LegalResource"}
    reg_label = {i: label[i] for i in reg_iris if i in label}
    # 규정↔규정 인접(관계 엣지, 방향 무관). has_part(조)는 규정 경계 식별엔 제외하고
    # 규정-규정 관계만 사용.
    adj: dict[str, set[str]] = defaultdict(set)
    rel_via: dict[tuple[str, str], str] = {}
    for s, p, o in triples:
        if p in _REL_PREDS and p != "eli:has_part" and s in reg_iris and o in reg_iris:
            adj[s].add(o)
            adj[o].add(s)
            rel_via[(s, o)] = _PRED_KO.get(p, p)
            rel_via[(o, s)] = _PRED_KO.get(p, p)

    llm_seeds = find_seeds_llm(question, reg_label) if use_llm else []
    seeds = llm_seeds or find_seeds(question, reg_label)
    seed_method = "llm" if llm_seeds else "string"
    visited: dict[str, int] = {s: 0 for s in seeds}
    parent: dict[str, str] = {}
    q = deque(seeds)
    while q:
        cur = q.popleft()
        if visited[cur] >= hops:
            continue
        for nxt in adj.get(cur, ()):
            if nxt not in visited:
                visited[nxt] = visited[cur] + 1
                parent[nxt] = cur
                q.append(nxt)

    def path_of(iri: str) -> list[str]:
        chain, cur = [], iri
        while cur in parent:
            prev = parent[cur]
            chain.append(f"{reg_label.get(prev, prev)} -{rel_via.get((prev, cur), '')}-> {reg_label.get(cur, cur)}")
            cur = prev
        return list(reversed(chain))

    related = [
        {"label": reg_label.get(i, i), "iri": i, "distance": d, "path": path_of(i)}
        for i, d in sorted(visited.items(), key=lambda x: x[1]) if i not in seeds
    ]
    doc_labels = [reg_label.get(s, s) for s in seeds] + [r["label"] for r in related]
    return {
        "question": question,
        "hops": hops,
        "seed_method": seed_method,  # "llm"(의미 매칭) 또는 "string"(폴백)
        "seeds": [reg_label.get(s, s) for s in seeds],
        "related": related,
        "doc_labels": doc_labels,   # 관련 문서(시드+N홉) — 스코프 벡터검색 대상
    }
