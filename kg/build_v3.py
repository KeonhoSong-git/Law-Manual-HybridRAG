"""kg.build_v3 — 규정 문서 지식그래프 v3 (이질 코퍼스, 실측 기반).

코퍼스 실측: 1,282문서 중 법규체 24%(306)·공문체 76%(976). 「」 인용 14,385개 →
코퍼스 내 해소 48%(문서→문서 확정엣지 3,479) · 외부법령 28% · 시스템/서식 24%(드롭).

설계:
- 노드는 **확실한 것만**: 문서(전체, 유형별 클래스) · 조문(법규체) · 정의용어(정의조항) · (기관은 별도 게이트 단계).
- 엣지는 **검증된 것만**: 「」 → 레지스트리 해소된 문서참조(dct:references). 미해소·시스템은 드롭, 외부법령은 경량 외부노드.
- 어휘: 법규체=ELI(LegalResource·has_part·date_publication), 전체공통=Dublin Core·SKOS. 비법령에 ELI 강제 안 함.
- 구조=결정적(이 파일), 의미(인용 종류분류·기관)=LLM+게이트(별도 단계).

입력: hwpx_chunker 의 by_document/*.json (청크 dict 배열).
출력: out_dir/kg.ttl, registry.json, stats.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import uuid
from collections import Counter

NS = uuid.UUID("c1d2e3f4-5a6b-7c8d-9e0f-1a2b3c4d5e6f")

PREFIXES = [
    "@prefix eli:  <http://data.europa.eu/eli/ontology#> .",
    "@prefix dct:  <http://purl.org/dc/terms/> .",
    "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
    "@prefix org:  <http://www.w3.org/ns/org#> .",
    "@prefix reg: <http://example.org/reg/ontology#> .",
    "@prefix doc:  <http://example.org/reg/doc/> .",
    "@prefix term: <http://example.org/reg/term/> .",
    "@prefix orgn: <http://example.org/reg/org/> .",
    "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
]

_ART = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*\d+)?\s*\(([^)]{1,40})\)")
_DATE = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})")
_DATE2 = re.compile(r"(?<!\d)[‘'`]?(\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})")  # 공문체 '18.01.02 형식 ((?<!\d): 4자리 연도의 일부를 2자리로 오인하지 않게)
_QUOTE = re.compile(r"[「『]([^」』]{2,40})[」』]")
_LAW = re.compile(r"(법|법률|시행령|시행규칙|령|규칙)$")
# 문서형 접미어: 평문 참조를 '제목다운' 것으로 제한해 우연한 substring 거짓참조를 줄인다.
_DOCSUFFIX = ("법", "법률", "시행령", "시행규칙", "령", "규칙", "규정", "지침",
              "기준", "요령", "세칙", "정관", "고시", "훈령", "예규")
# 정의 조항: 「N. "용어"란 …」 (곧은/굽은 따옴표 모두)
_Q = '"“”‘’\''
_DEF = re.compile(
    r'\d+\.\s*[' + _Q + r']([^' + _Q + r']{1,30})[' + _Q + r']\s*(?:이?란|이라\s*함은)\s*(.+?)(?=\n\s*\d+\.|\Z)',
    re.S)
_DEF_ARTICLE = re.compile(r"제\s*\d+\s*조\s*\(\s*정의\s*\)")
_DEF_END = re.compile(r"^.{4,}?다\s*[.\n]")   # 정의문 첫 '…다.' 종결(여러 조항 blob 방지)


def _def_sentence(d: str) -> str:
    """정의문을 첫 '…다.' 종결까지로 자른다. 종결 없으면 180자 캡(공문체 등 붕괴 대비)."""
    d = re.sub(r"\s+", " ", (d or "").strip())
    m = _DEF_END.match(d)
    return (m.group(0).strip() if m else d)[:180]
# 기관(named body): 위원회/심의회/협의회/이사회로 끝나는 명사구(공백 없는 연속 명사구).
_ORG = re.compile(r"[가-힣A-Za-z0-9·]{2,14}(?:위원회|심의회|협의회|이사회)")  # 접두 ≤14: 실제 기관명 길이, 20자+ 절조각 차단
_ORG_LEADJUNK = re.compile(r"^[oO·∙‧\-*ㅇ\s]+")  # 앞 list마커/불릿 제거 (ESG·AI 등 영문명은 보존)
# 절·어미·동사·조항참조가 들어간 매치는 기관명이 아니라 문장 조각 → 거부(과포획 쓰레기 제거).
# 진짜 기관명(○○평가위원회·전문위원회·기술기준위원회 등)엔 안 나타나는 신호만 모음.
# 가장 강한 신호 = 숫자(제N조·N일·N차 등 조항/수치 조각). 형태소는 글루된 조사·동사·어미.
_ORG_REJECT = re.compile(
    r"\d"  # 숫자 = 제N조·제N항·제N장·N일·N차 등 조항/수치 조각
    r"|위원장|위원이|위원을|위원은|위원의"
    r"|따른|따라|위한|위하|위해|참여|갖춘|필요|경우|이내|부터|까지|또는|관한|대한|대하|통하|받아|거쳐|걸쳐|관련|해당|만들|청취|초안|저항|한후|및"
    r"|하여|하며|하기|하고|하는|되는|되어|시키|정한|정하|제출|구성하|구성된|운영하|지정하|평가하|포함|분석|심사를|의결을|검토·조정"
    r"|장관은|장은|부서는|기관은|공무원은|지원은|규정에|에따|에관|에는|으로|에서|등은|등의|또한|기타|그밖|이외|및제|및의|및심|결과를|항목을|지식을"
    r"|^(?:의|를|을|및|와|과|란|로)"   # 선두가 조사·접속사면 문장 조각 (에너지·한국·여성 등 정상 접두는 보존)
)
_EST = re.compile(r"(둔다|설치|구성|운영)")
# 문서 약칭(엔티티 해소): 「전체제목」(이하 「약칭」라 한다) → 약칭을 전체제목 문서로 매핑.
_ALIAS = re.compile(
    r"[「『]([^」』]{3,40})[」』]\s*\(?\s*(?:이하|약칭)\s*[「『“\"']?([^」』”\"',)\n]{3,30}?)[」』”\"']?\s*"
    r"(?:이?라고?|이라\s*함)?\s*(?:한다|함|칭한다)")
_GENERIC_ALIAS = {"규정", "기준", "요령", "지침", "세칙", "법", "이법", "본법", "당규정",
                  "이규정", "본규정", "이기준", "본기준", "동법", "약칭"}

# 비법령 문서 클래스 (파일명 접두 기반)
_PREFIX_CLASS = {
    "지시문서": "reg:Directive",
    "업무매뉴얼": "reg:Manual",
    "매뉴얼": "reg:Manual",
    "잠정조치": "reg:ProvisionalMeasure",
    "규정": "eli:LegalResource",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def resolve_citation(raw: str, registry: dict, min_len: int = 4):
    """인용 제목(raw)을 레지스트리(키=_norm(제목))에 해소: 정확일치 → 부분일치(정규화 길이 min_len 이상).
    세 곳(build _resolve_cite, classify _targets/resolve)이 같은 규칙·임계값을 쓰도록 단일화한다
    (제각각이던 임계값으로 build/classify 엣지가 어긋나던 드리프트 방지). 레지스트리 값(IRI 또는
    정식 제목)을 반환하고, 없으면 None."""
    nt = _norm(raw)
    if nt in registry:
        return registry[nt]
    if len(nt) >= min_len:
        for cand_n, val in registry.items():
            if nt in cand_n or cand_n in nt:
                return val
    return None


_BAD_TITLE = re.compile(r"[|]|---|\.json$|_원문파일_")
_DROP_TOK = {"원문파일", "규정", "지시문서", "업무매뉴얼", "잠정조치", "기준", "요령", "지침"}


def _is_bad_title(t: str) -> bool:
    """깨진 제목 판정: 표마크다운·괄호/날짜/문장부호로 시작·날짜만·너무 짧음."""
    if not t or not t.strip():
        return True
    s = t.strip()
    if _BAD_TITLE.search(s):
        return True
    # 문장부호로 시작 → 첫단락 잡문. (숫자 시작은 제외 — '2026년…','1회용…' 등 정상 제목이 많아
    # 숫자만으로 깨진 제목 판정하면 서로 다른 문서가 파일명 폴백으로 같은 제목에 충돌해 노드가 병합됨.)
    # 괄호 '(' 시작도 제외 — '(서해어업관리단)…','(고용노동부)…' 등 부처/지역 접두가 흔한 정상 제목.
    if re.match(r"""^[.,·\-–—'"]""", s):
        return True
    if len(re.sub(r"\s+", "", s)) < 3:                    # 정규화 3자 미만
        return True
    if re.fullmatch(r"[\d.\s년월일()~\-]+", s):            # 날짜/숫자만(연-월-일 등) → 여전히 깨진 제목
        return True
    return False


def _clean_title(title: str, source: str) -> str:
    """깨진 제목이면 파일명에서 복원. 파일명은 제목을 2~3회 반복하므로 **최빈 세그먼트**를
    택해 끝 날짜 truncate(…_(20)·해시에 강건하게 한다."""
    t = (title or "").strip()
    if not _is_bad_title(t):
        return t
    stem = re.sub(r"\.(hwpx|HWPX|json|pdf|PDF|txt)$", "", source or t)
    stem = re.sub(r"_?\(?\d{4}[.\-]\s*\d{1,2}[.\-]\s*\d{1,2}\)?", "", stem)  # 완전 날짜 제거
    parts = [p.strip() for p in stem.split("_") if p.strip()]
    # 후보: 해시·부분날짜(괄호/숫자 시작)·일반 토큰 제외
    cand = [p for p in parts if not re.fullmatch(r"[0-9a-f]{6,}", p)
            and not re.match(r"^[(\[]?\d", p) and p not in _DROP_TOK and len(p) >= 2]
    if not cand:
        return t
    return max(cand, key=lambda x: (cand.count(x), len(x)))  # 최빈 → 동률이면 최장


def _esc(t: str) -> str:
    return (t or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _doc_iri(title: str) -> str:
    return "doc:d" + uuid.uuid5(NS, _norm(title)).hex[:16]


def _term_iri(term: str) -> str:
    return "term:t" + uuid.uuid5(NS, _norm(term)).hex[:16]


def _org_iri(name: str) -> str:
    return "orgn:o" + uuid.uuid5(NS, _norm(name)).hex[:16]


_ORG_GATE_SYS = (
    "너는 한국 법령·행정규칙 본문에서 정규식으로 추출한 '기관/위원회' 후보 이름들을 검수한다.\n"
    "각 후보가 (A) 진짜 고유한 named body — 위원회·심의회·협의회·이사회 등 실제 조직의 고유명사인지,\n"
    "아니면 (B) 문장에서 잘못 잘린 조각 — 동사·어미·조사·조항참조('제N조에 따른', '장관은', '있는지 여부를' 등)가\n"
    "섞여 명사구가 아닌 것인지 판정한다.\n"
    "예: 진짜=[국민권익위원회, 증권선물위원회, 한국전기기술기준위원회, 운영위원회, 금융정책협의회].\n"
    "조각=[장관은평가위원회, 제16조에따른심의위원회, 있는지여부를현장심사위원회, 행한평가위원회, 의평가위원회]."
)


def _llm_org_gate(labels: list[str]) -> set[str]:
    """LLM 으로 후보 기관명을 '진짜 기관 vs 문장 조각'으로 판정해 통과 라벨 집합 반환.
    LLM 미설정/실패 시 보수적으로 전체 통과(결정적 폴백 — 데이터 손실 방지)."""
    from . import llm
    if not labels or not llm.available():
        return set(labels)
    keep: set[str] = set()
    for i in range(0, len(labels), 40):
        batch = labels[i:i + 40]
        numbered = "\n".join(f"{j}. {b}" for j, b in enumerate(batch))
        user = ("다음 후보들 중 **진짜 기관/위원회 고유명사**인 것의 번호만 JSON 배열로 출력해라"
                "(예: [0,2,5]). 조각·비명사구는 제외. 설명 없이 배열만.\n\n" + numbered)
        try:
            idxs = llm.parse_json_array(llm.chat(_ORG_GATE_SYS, user, max_tokens=400))
            for k in idxs:
                if isinstance(k, int) and 0 <= k < len(batch):
                    keep.add(batch[k])
        except Exception:  # noqa: BLE001
            keep.update(batch)   # 호출 실패 배치는 전체 통과
    return keep


def _iso(y: str, m: str, d: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def load_by_document(src_dir: str) -> list[dict]:
    """hwpx_chunker by_document/*.json → 문서 레코드."""
    out = []
    for p in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            arr = json.load(fh)
        if not arr:
            continue
        f = arr[0]
        text = "\n".join(c.get("text") or "" for c in arr)
        out.append({
            "title": _clean_title(f.get("doc_title"), f.get("source_file") or os.path.basename(p)),
            "family": f.get("doc_family") or "",
            "doc_type": f.get("doc_type") or "",
            "source": f.get("source_file") or "",
            "enacted": f.get("enacted") or "",
            "amended": f.get("last_amended") or "",
            "articles": [(c.get("article_label"), c.get("article_title"), c.get("chunk_id"))
                         for c in arr if c.get("unit") == "조" and c.get("article_label")],
            # 조가 아닌 본문 청크(문서/본문 단위) → 문서 노드에 dct:identifier 로 링크.
            "doc_chunks": [c.get("chunk_id") for c in arr
                           if c.get("unit") != "조" and c.get("chunk_id")],
            # 청크 단위 귀속(정의용어·기관)을 위해 모든 청크의 (id, 본문) 보존.
            "all_chunks": [(c.get("chunk_id"), c.get("text") or "") for c in arr if c.get("chunk_id")],
            "text": text,
        })
    return out


def _doc_class(rec: dict) -> str:
    if rec["family"] == "법규체":
        return "eli:LegalResource"
    pref = (rec["source"].split("_", 1)[0]) if rec["source"] else ""
    return _PREFIX_CLASS.get(pref, "reg:Document")


def build(records: list[dict]) -> tuple[str, dict]:
    # 인용 해소의 정본 IRI(제목정규화→IRI, 첫 문서 우선). 같은 정규화 제목의 서로 다른 문서가
    # 같은 IRI 로 병합되던 문제는 아래 레코드 루프에서 source 로 salt 해 분리한다.
    title_iri: dict[str, str] = {}
    for r in records:
        title_iri.setdefault(_norm(r["title"]), _doc_iri(r["title"]))
    # 엔티티 해소(heritage resolve_entities 방식): 「전체제목」(이하 「약칭」) 의 약칭을
    # 전체제목 '문서'로 매핑. 약칭이 일반어(규정·법 등)면 제외 → 오해소 방지.
    def _resolve_full(full: str):
        nf = _norm(full)
        if nf in title_iri:
            return title_iri[nf]
        if len(nf) >= 4:
            for cn, ci in title_iri.items():
                if nf in cn or cn in nf:
                    return ci
        return None

    alias_iri: dict[str, str] = {}
    alias_emit: dict[str, tuple[str, str]] = {}   # nshort -> (doc_iri, surface)
    for r in records:
        for m in _ALIAS.finditer(r["text"]):
            full, short = m.group(1).strip(), m.group(2).strip()
            ns = _norm(short)
            if len(ns) < 3 or ns in _GENERIC_ALIAS or ns in title_iri or ns in alias_iri:
                continue
            tgt = _resolve_full(full)
            if tgt:
                alias_iri[ns] = tgt
                alias_emit[ns] = (tgt, short)
    lines: list[str] = []
    ext_nodes: dict[str, str] = {}     # norm(lawname) -> iri
    chunk_nodes: dict[str, list[str]] = {}  # chunk_id -> [그 청크에서 추출/참조된 노드 IRI] (청크 메타 역기록)
    created_terms: set[str] = set()          # 노드화된 정의용어 IRI (노드 1회만 생성)
    term_defs: set[tuple[str, str]] = set()  # (term_iri, doc_iri) — 정의한 모든 문서를 isDefinedBy 로 연결
    term_chunks: dict[str, set[str]] = {}    # term_iri  -> {정의가 등장한 chunk_id}
    org_chunks: dict[str, set[str]] = {}     # org canon -> {언급된 chunk_id}
    stats = Counter()
    cite_edges: set[tuple[str, str]] = set()
    org_mentions: dict[str, dict[str, bool]] = {}  # canon -> {doc_iri: established?}
    org_label: dict[str, str] = {}

    # 인용 「」 대상 해소(정식 제목 → 약칭 → 부분일치). 추출 시점에 바로 쓰인다.
    # 정확일치는 정식제목·약칭 둘 다, 부분일치는 정식제목만(약칭 부분일치는 오탐이 많아 제외) — 원동작 유지.
    def _resolve_cite(tgt: str):
        nt = _norm(tgt)
        return title_iri.get(nt) or alias_iri.get(nt) or resolve_citation(tgt, title_iri, min_len=4)

    for ns, (tgt, surf) in alias_emit.items():        # 약칭 → skos:altLabel
        lines.append(f'{tgt} skos:altLabel "{_esc(surf)}" .')
    stats["aliases"] = len(alias_emit)

    seen_titles: Counter = Counter()
    for r in records:
        nt = _norm(r["title"])
        if seen_titles[nt] == 0:
            s = title_iri[nt]                       # 정본(첫 문서) = 인용 대상
        else:                                       # 제목 충돌: source 로 deterministically salt → 별도 노드
            s = "doc:d" + uuid.uuid5(NS, nt + "\x01" + (r["source"] or str(seen_titles[nt]))).hex[:16]
            stats["doc_title_collisions"] += 1
        seen_titles[nt] += 1
        cls = _doc_class(r)
        lines.append(f"{s} rdf:type {cls} .")
        lines.append(f'{s} rdfs:label "{_esc(r["title"])}" .')
        lines.append(f'{s} dct:type "{_esc(r["doc_type"] or r["family"])}" .')
        # 벡터 청크 역추적 링크(Lean KG): 문서 본문 청크 → dct:identifier.
        for ch in r.get("doc_chunks", []):
            lines.append(f'{s} dct:identifier "{_esc(ch)}" .')
            chunk_nodes.setdefault(ch, []).append(s)
            stats["chunk_links"] += 1
        # 날짜: 제정일=발행일, 개정일=수정일. 둘 다 방출(상호배타 아님 — 기존 break 는 둘 중 하나만 내보내
        # 비논리적이었음). 개정일을 eli:date_no_longer_in_force(효력상실)로 쓰면 '개정=폐지'로 날조되므로
        # 표준어휘 dct:modified(수정일) 사용. (4자리 우선, 없으면 2자리연도 '18 형식 → 20xx)
        for field, pred in (
            ("enacted", "eli:date_publication" if cls == "eli:LegalResource" else "dct:issued"),
            ("amended", "dct:modified"),
        ):
            m = _DATE.search(r[field])
            iso = None
            if m:
                iso = _iso(*m.groups())
            else:
                m2 = _DATE2.search(r[field])
                if m2:
                    yy, mm, dd = m2.groups()
                    iso = _iso(str(2000 + int(yy)), mm, dd)
            if iso:
                lines.append(f'{s} {pred} "{iso}"^^xsd:date .')
                stats["dates"] += 1
        # 조문 (법규체)
        if r["family"] == "법규체":
            seen = set()
            for label, atitle, ch_id in r["articles"]:
                if not label or label in seen:
                    continue
                seen.add(label)
                a = f"{s}_{_norm(label)}"
                lines.append(f"{a} rdf:type eli:LegalResourceSubdivision .")
                lines.append(f'{a} rdfs:label "{_esc(label + (" " + atitle if atitle else ""))}" .')
                lines.append(f"{s} eli:has_part {a} .")
                # 조문 → 해당 청크 역추적 링크.
                if ch_id:
                    lines.append(f'{a} dct:identifier "{_esc(ch_id)}" .')
                    chunk_nodes.setdefault(ch_id, []).append(a)
                    stats["chunk_links"] += 1
                stats["articles"] += 1
        # ── 청크 단위 추출(단일 패스) ──────────────────────────────────────────────
        # 정의용어·인용·기관을 **청크별 본문에서 추출하면서 그 자리에서 chunk↔node 링크를 기록**한다.
        # (별도 재스캔 패스 없음 — 추출과 귀속이 한 번에 일어난다.)
        # 정의조항 보유 문서(문서 단위 게이트). 공문체(매뉴얼)는 PDF 띄어쓰기 붕괴로 정의 경계가
        # 무너져 garbled blob·일반어가 노드화되므로 제외(Lean KG: 공문체=문서·관계 수준만).
        has_def_article = r["family"] == "법규체" and bool(_DEF_ARTICLE.search(r["text"]))
        for ch_id, ctext in r.get("all_chunks", []):
            if not ctext:
                continue
            # 정의용어: 정의조항 보유 문서의 각 청크에서 추출(게이트는 문서 단위, 추출은 청크 단위).
            if has_def_article:
                for term, definition in _DEF.findall(ctext):
                    term = term.strip()
                    if not term or len(term) > 30:
                        continue
                    ti = _term_iri(term)
                    if ti not in created_terms:
                        lines.append(f"{ti} rdf:type skos:Concept .")
                        lines.append(f'{ti} skos:prefLabel "{_esc(term)}" .')
                        lines.append(f'{ti} skos:definition "{_esc(_def_sentence(definition))}" .')
                        created_terms.add(ti)
                        stats["def_terms"] += 1
                    if (ti, s) not in term_defs:    # 같은 용어를 정의한 모든 문서를 연결(2번째 문서 누락 방지)
                        lines.append(f"{ti} rdfs:isDefinedBy {s} .")
                        term_defs.add((ti, s))
                    if ch_id:
                        term_chunks.setdefault(ti, set()).add(ch_id)
            # 인용 「」: 해소된 대상은 dct:references + 같은 청크 노드(chunk_nodes)로 기록.
            for m in _QUOTE.finditer(ctext):
                tgt = m.group(1).strip()
                hit = _resolve_cite(tgt) or alias_iri.get(_norm(tgt))
                if not hit and _LAW.search(tgt):
                    nt = _norm(tgt)
                    hit = ext_nodes.get(nt)
                    if not hit:
                        hit = "doc:ext" + uuid.uuid5(NS, nt).hex[:12]
                        ext_nodes[nt] = hit
                        lines.append(f"{hit} rdf:type eli:LegalResource .")
                        lines.append(f'{hit} rdfs:label "{_esc(tgt)}" .')
                        lines.append(f"{hit} reg:external true .")
                        stats["ext_refs_unique"] = len(ext_nodes)
                if hit and hit != s:
                    cite_edges.add((s, hit))
                    if ch_id:
                        chunk_nodes.setdefault(ch_id, []).append(hit)
                        stats["chunk_cite_links"] += 1
            # 기관(named body): list마커 제거·정규화. 설치/구성 동사 근접 시 설치관계. 청크 링크 기록.
            for m in _ORG.finditer(ctext):
                label = _ORG_LEADJUNK.sub("", m.group(0)).strip()
                label = re.sub(r"\s+", " ", label)        # 내부 공백은 보존하되 다중 공백만 정규화
                if _ORG_REJECT.search(label):             # 절·동사 조각이면 기관명 아님 → 드롭
                    continue
                canon = re.sub(r"\s+", "", label)         # 중복키는 공백제거(같은 기관 띄어쓰기 변이 통합)
                if len(canon) < 5:
                    continue
                win = ctext[m.end():m.end() + 25]   # 설치/구성 동사 근접 시 설치관계 — 단 폐지·부정문은 제외
                est = bool(_EST.search(win)) and not re.search(r"폐지|아니|않", win)
                d = org_mentions.setdefault(canon, {})
                d[s] = d.get(s, False) or est
                org_label.setdefault(canon, label)
                if ch_id:
                    org_chunks.setdefault(canon, set()).add(ch_id)
        # 평문 참조: 「」 밖이라 근거가 약하므로 '제목다운' 것만 — 정규화 ≥8자 + 문서형 접미어로 끝나고
        # 본문에 등장(우연한 substring 공기로 거짓참조가 생기던 문제 완화).
        ntext = _norm(r["text"])[:200000]
        for nt, ti in title_iri.items():
            if ti != s and len(nt) >= 8 and nt.endswith(_DOCSUFFIX) and nt in ntext and (s, ti) not in cite_edges:
                cite_edges.add((s, ti))
                stats["plaintext_refs"] += 1

    # 조각 기관 제거: 다른 기관명의 진부분문자열(예: 관리위원회 ⊂ 리스크관리위원회)이면 드롭.
    all_canon = set(org_mentions)
    frags = {c for c in all_canon if any(c != o and c in o for o in all_canon)}
    for c in frags:
        del org_mentions[c]
    stats["org_fragments_dropped"] = len(frags)

    # 기관 LLM 게이트(설계 B): 정규식 후보를 LLM 이 '진짜 기관 vs 문장 조각'으로 최종 판정.
    # LLM 미설정/실패 시 전체 통과(결정적 폴백). KG_ORG_LLM_GATE=0 으로 끌 수 있음.
    if os.environ.get("KG_ORG_LLM_GATE", "1") != "0":
        keep = _llm_org_gate([org_label[c] for c in org_mentions])
        gated = [c for c in list(org_mentions) if org_label[c] not in keep]
        for c in gated:
            del org_mentions[c]
        stats["org_llm_dropped"] = len(gated)

    # 기관 노드 + 관계. 추출 단계에서 모은 org_chunks 로 청크 링크를 그 자리에서 emit(재스캔 없음).
    for canon, docs in sorted(org_mentions.items()):
        oi = _org_iri(canon)
        lines.append(f"{oi} rdf:type org:Organization .")
        lines.append(f'{oi} rdfs:label "{_esc(org_label[canon])}" .')
        stats["orgs"] += 1
        for di, est in docs.items():
            lines.append(f"{oi} reg:mentionedIn {di} .")
            if est:
                lines.append(f"{oi} reg:establishedBy {di} .")
                stats["org_established"] += 1
        for ch in sorted(org_chunks.get(canon, ())):   # 기관 → 등장 청크(node→chunk + chunk_nodes)
            lines.append(f'{oi} dct:identifier "{_esc(ch)}" .')
            chunk_nodes.setdefault(ch, []).append(oi)
            stats["chunk_attr_links"] += 1

    # 정의용어 → 정의 등장 청크(node→chunk + chunk_nodes). 추출 단계에서 모은 term_chunks 사용.
    for ti, chs in sorted(term_chunks.items()):
        for ch in sorted(chs):
            lines.append(f'{ti} dct:identifier "{_esc(ch)}" .')
            chunk_nodes.setdefault(ch, []).append(ti)
            stats["chunk_attr_links"] += 1

    for s, o in sorted(cite_edges):
        lines.append(f"{s} dct:references {o} .")
    stats["documents"] = len(records)
    stats["cite_edges"] = len(cite_edges)
    ttl = "\n".join(PREFIXES) + "\n\n" + "\n".join(lines) + "\n"
    return ttl, {"title_iri": title_iri, "stats": dict(stats),
                 "chunk_nodes": {k: sorted(set(v)) for k, v in chunk_nodes.items()}}


def annotate_chunks(src_dir: str, chunk_nodes: dict[str, list[str]]) -> int:
    """트리플 생성 후, 각 청크가 만든 KG 노드 IRI 를 청크 메타(`kg_nodes`)에 역기록(in-place)."""
    n = 0
    for p in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            arr = json.load(f)
        changed = False
        for c in arr:
            nodes = chunk_nodes.get(c.get("chunk_id"), [])
            if c.get("kg_nodes") != nodes:
                c["kg_nodes"] = nodes
                changed = True
            if nodes:
                n += 1
        if changed:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(arr, f, ensure_ascii=False, indent=1)
    return n


def build_to_dir(src_dir: str, out_dir: str) -> dict:
    records = load_by_document(src_dir)
    ttl, meta = build(records)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kg.ttl"), "w", encoding="utf-8") as f:
        f.write(ttl)
    json.dump(meta["stats"], open(os.path.join(out_dir, "kg_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 청크 → 노드 역기록 (트리플 생성 후 청크 메타데이터에 반영)
    json.dump(meta["chunk_nodes"], open(os.path.join(out_dir, "chunk_nodes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta["stats"]["chunks_annotated"] = annotate_chunks(src_dir, meta["chunk_nodes"])
    return meta["stats"]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    src, out = sys.argv[1], sys.argv[2]
    print(json.dumps(build_to_dir(src, out), ensure_ascii=False, indent=1))
