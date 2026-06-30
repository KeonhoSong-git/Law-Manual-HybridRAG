#!/usr/bin/env python3
"""규정 하이브리드(벡터 RAG + 지식그래프) 질의 웹 — 투명 데모.

한 질문에 대해 **지식그래프가 가져온 것 / 벡터 RAG가 가져온 것 / 최종 답변**을 모두
화면에 보여준다. 자기완결(self-contained): KG 3홉 탐색 + 인메모리 벡터검색 + LLM 답변.

env: KG_TTL, CHUNKS_DIR, EMBEDDING_API_BASE, LLM_API_BASE/API_KEY/LLM_MODEL (.env 자동 로드)
실행: uvicorn kg_api:app --host 0.0.0.0 --port 8800
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import urllib.request
from collections import defaultdict, deque

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ---------- 설정/env ----------
def _load_env() -> None:
    if os.environ.get("LLM_API_BASE"):
        return
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                if "=" in ln and not ln.strip().startswith("#"):
                    k, v = ln.strip().split("=", 1)
                    os.environ.setdefault(k, v)
            return
        d = os.path.dirname(d)


_load_env()
KG_TTL = os.environ.get("KG_TTL", os.path.expanduser("~/reg_chunks/kg.ttl"))
CHUNKS_DIR = os.environ.get("CHUNKS_DIR", os.path.expanduser("~/reg_chunks"))
HOPS = int(os.environ.get("KG_GRAPH_HOPS", "3"))
TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_RELATED = int(os.environ.get("KG_MAX_RELATED", "50"))   # related 패널·노이즈 캡

_REL_PREDS = {"eli:based_on", "eli:applies", "eli:cites", "eli:amends", "eli:repeals",
              "dct:relation", "dct:references"}
_PRED_KO = {"eli:based_on": "위임받음", "eli:applies": "준용", "eli:cites": "참조",
            "eli:amends": "개정", "eli:repeals": "폐지", "dct:relation": "관련",
            "dct:references": "참조", "eli:date_publication": "제정일",
            "eli:first_date_entry_in_force": "시행일"}


# ---------- TTL 파서/그래프 ----------
def parse_ttl(text):
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


# ---------- LLM/임베딩 ----------
def chat(system, user, max_tokens=500):
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({"model": os.environ.get("LLM_MODEL", "instruct"), "temperature": 0,
                       "max_tokens": max_tokens, "messages": [{"role": "system", "content": system},
                                                              {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + os.environ.get("API_KEY", ""),
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())["choices"][0]["message"]["content"]


def embed(texts, batch=32):
    """배치로 임베딩(큰 요청 회피). data[i].embedding (1024)."""
    base = os.environ["EMBEDDING_API_BASE"].rstrip("/")
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"input": texts[i:i + batch], "return_dense": True,
                           "return_sparse": False, "return_colbert": False}).encode()
        req = urllib.request.Request(base + "/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        out += [r["embedding"] for r in json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]]
    return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def strip_markup(t):
    return re.sub(r"<[^>]+>", " ", t).replace("\n", " ").strip()


# ---------- 기동 시 로드 ----------
TTL = open(KG_TTL, encoding="utf-8").read()
TRIPLES = parse_ttl(TTL)
LABEL = {s: o.split('"')[1] for s, p, o in TRIPLES if p == "rdfs:label" and '"' in o}
_DOC_TYPES = {"eli:LegalResource", "reg:Directive", "reg:Manual",
              "reg:ProvisionalMeasure", "reg:Document"}
REG_IRIS = {s for s, p, o in TRIPLES if p == "rdf:type" and o in _DOC_TYPES}
REG_LABEL = {i: LABEL[i] for i in REG_IRIS if i in LABEL}
ADJ = defaultdict(set)
VIA = {}
# 관계(인용)만 — 날짜/속성 술어 제외. 답변용 사실 주입에 사용.
_FACT_PREDS = {"eli:based_on", "eli:applies", "eli:cites", "eli:amends",
               "eli:repeals", "dct:references"}
REL_OUT = defaultdict(list)   # iri -> [(관계ko, 대상라벨)]  (현재문서 → 대상)
REL_IN = defaultdict(list)    # iri -> [(관계ko, 출처라벨)]  (출처 → 현재문서)
for s, p, o in TRIPLES:
    if p in _REL_PREDS and s in REG_IRIS and o in REG_IRIS:
        ADJ[s].add(o); ADJ[o].add(s)
        VIA[(s, o)] = VIA[(o, s)] = _PRED_KO.get(p, p)
    if p in _FACT_PREDS:                         # 사실: 대상이 외부법령(ext)이어도 포함
        ko = _PRED_KO.get(p, p)
        if s in REG_IRIS and o in LABEL:
            REL_OUT[s].append((ko, LABEL[o]))
        if o in REG_IRIS and s in LABEL:
            REL_IN[o].append((ko, LABEL[s]))

# ---------- 노드 인덱스(탐색기용) ----------
_SCHEMA_T = {"Class", "Property", "Ontology", "DatatypeProperty", "ObjectProperty", "NamedIndividual"}


def _lit(o):
    return o.split('"')[1] if o.startswith('"') and '"' in o[1:] else None


def _build_nodes():
    N = {}

    def nd(i):
        return N.setdefault(i, {"iri": i, "label": None, "types": [], "attrs": {}, "out": [], "in": []})

    for s, p, o in TRIPLES:
        n = nd(s)
        if p == "rdf:type":
            t = o.split(":")[-1]
            if t not in n["types"]:
                n["types"].append(t)
        elif p in ("rdfs:label", "skos:prefLabel"):
            lit = _lit(o)
            if lit:
                n["label"] = lit
        elif o.startswith('"'):
            n["attrs"].setdefault(p, []).append(_lit(o))
        elif o in ("true", "false"):
            n["attrs"].setdefault(p, []).append(o)
        else:
            n["out"].append([p, o])
            nd(o)["in"].append([p, s])
    for i, n in N.items():
        if not n["label"]:
            n["label"] = i.split(":")[-1]
        n["types"] = n["types"] or ["(untyped)"]
    return {i: n for i, n in N.items() if not (set(n["types"]) & _SCHEMA_T)}


NODES = _build_nodes()

CHUNKS = []
CHUNK_VECS: list = []
_VECS_READY = False
CHUNK_MAT = None  # numpy (N x 1024) 정규화 행렬 — precomputed 모드

# precomputed 벡터: vectors.npy(+ .meta.jsonl) 우선 — 경량 바이너리(float16).
# 없으면 vectors.jsonl({id,doc,text,embedding}).
_VEC_PATH = os.environ.get("VECTORS_PATH", "/data/vectors.jsonl")
_NPY = (_VEC_PATH[:-6] + ".npy") if _VEC_PATH.endswith(".jsonl") else _VEC_PATH
_META = _NPY[:-4] + ".meta.jsonl"
if os.path.exists(_NPY) and os.path.exists(_META):
    import numpy as _np
    CHUNK_MAT = _np.load(_NPY).astype(_np.float32)
    CHUNK_MAT /= (_np.linalg.norm(CHUNK_MAT, axis=1, keepdims=True) + 1e-9)
    for _ln in open(_META, encoding="utf-8"):
        try:
            _o = json.loads(_ln)
        except ValueError:
            continue
        CHUNKS.append({"doc": _o.get("doc"), "text": _o.get("text") or ""})
    _VECS_READY = True
elif os.path.exists(_VEC_PATH):
    import numpy as _np
    _vlist = []
    for _ln in open(_VEC_PATH, encoding="utf-8"):
        try:
            _o = json.loads(_ln)
        except ValueError:
            continue
        CHUNKS.append({"doc": _o.get("doc"), "text": _o.get("text") or ""})
        _vlist.append(_o["embedding"])
    CHUNK_MAT = _np.asarray(_vlist, dtype=_np.float32)
    CHUNK_MAT /= (_np.linalg.norm(CHUNK_MAT, axis=1, keepdims=True) + 1e-9)
    _VECS_READY = True
else:
    for pth in sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.json"))):
        if os.path.basename(pth).startswith(("_", "chunks_", "kg")):
            continue
        d = json.load(open(pth, encoding="utf-8"))
        title = (d.get("document") or {}).get("doc_title")
        for c in d.get("chunks", []):
            txt = c.get("text") or c.get("content") or ""
            if txt.strip():
                CHUNKS.append({"doc": title, "text": txt})


def ensure_vecs():
    """precomputed 모드면 no-op. 아니면 첫 질의 때 지연 임베딩."""
    global CHUNK_VECS, _VECS_READY
    if not _VECS_READY and CHUNKS and os.environ.get("EMBEDDING_API_BASE"):
        CHUNK_VECS = embed([c["text"][:1000] for c in CHUNKS])
        _VECS_READY = True


# ---------- 하이브리드 ----------
def llm_seeds(question):
    if not os.environ.get("LLM_API_BASE") or not REG_LABEL:
        return []
    t2i = {t: i for i, t in REG_LABEL.items()}
    sysmsg = ("주어진 '규정 목록' 중 질문과 직접 관련된 규정의 정확한 제목만 JSON 배열로. "
              "목록 밖 금지. 없으면 [].")
    try:
        s = chat(sysmsg, "규정 목록:\n" + "\n".join("- " + t for t in t2i) + f"\n\n질문: {question}", 300)
        a, b = s.find("["), s.rfind("]")
        picked = json.loads(s[a:b + 1]) if a >= 0 else []
    except Exception:  # noqa: BLE001
        return []
    return [t2i[t] for t in picked if t in t2i]


# 변별력 없는 접두/일반어 — 시드 토큰에서 제외 (단독으로 쓰이면 의미 없는 접미 명사 포함)
_GENERIC_TOK = {"관련", "관한", "및", "에", "의",
                "규정", "기준", "요령", "지침", "세칙", "방법", "규칙"}


def _toknorm(s):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s)   # 괄호·문장부호 제거(『』「」() 등)


def string_seeds(question):
    """규정명 핵심어가 질문에 들어있으면 문자열로 시드 선택 → LLM 생략.
    길이가중 커버리지 >=0.6, 또는 변별력 큰 토큰(>=6자) 단독 일치면 채택(부분명 대응)."""
    qn = _toknorm(question)
    scored = []
    for i, lab in REG_LABEL.items():
        core = [n for n in (_toknorm(t) for t in lab.split())
                if n and n not in _GENERIC_TOK and len(n) >= 2]
        if not core:
            continue
        matched = [t for t in core if t in qn]
        if not matched:
            continue
        cov = sum(len(t) for t in matched) / sum(len(t) for t in core)   # 길이가중 커버리지
        longest = max(len(t) for t in matched)
        if cov >= 0.6 or longest >= 6:        # 부분명이라도 변별 토큰이 들어오면 채택
            scored.append((cov, longest, i))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [i for _, _, i in scored[:3]]


def traverse(question, hops=HOPS):
    seeds = string_seeds(question)   # 1순위: 규정명이 질문에 거의 그대로면 LLM 생략
    method = "string"
    if not seeds:                    # 애매할 때만 LLM 의미 매칭 (왕복 1회)
        seeds = llm_seeds(question)
        method = "llm" if seeds else "none"
    visited, parent = {s: 0 for s in seeds}, {}
    q = deque(seeds)
    while q:
        cur = q.popleft()
        if visited[cur] >= hops:
            continue
        for nx in ADJ.get(cur, ()):
            if nx not in visited:
                visited[nx] = visited[cur] + 1; parent[nx] = cur; q.append(nx)
    rel_sorted = sorted(((i, d) for i, d in visited.items() if i not in seeds),
                        key=lambda x: x[1])
    related = [{"label": REG_LABEL.get(i, i), "distance": d}
               for i, d in rel_sorted[:MAX_RELATED]]              # UI·노이즈 방지 캡
    seed_labels = [REG_LABEL.get(s, s) for s in seeds]
    docs = seed_labels + [r["label"] for r in related]
    return {"method": method, "seeds": seed_labels,
            "seed_iris": list(seeds), "related": related,
            "docs": docs,
            # 벡터 스코프 = 그래프가 N홉(기본 3) 좁힌 집합 전체. (기존 1홉은 선언된 3홉 narrowing 과 불일치 —
            # 벡터는 이 스코프 안에서 TOP_K 만 고르므로 3홉으로 둬도 폭주하지 않음.)
            "scope": docs}


_REL_PRIO = {"위임받음": 0, "준용": 1, "개정": 2, "폐지": 3, "참조": 4}   # 고신호 관계 우선


def kg_facts(seed_iris):
    """시드 문서의 인용 관계(타입엣지)를 양방향 한국어 사실로. 위임/준용을 참조보다 우선(캡에 안 잘리게)."""
    scored, seen = [], set()
    for si in seed_iris:
        slab = LABEL.get(si, si)
        for ko, olab in REL_OUT.get(si, []):           # 현재문서 → 대상
            f = f"{slab} {ko} {olab}"
            if f not in seen:
                seen.add(f); scored.append((_REL_PRIO.get(ko, 9), f))
        for ko, src in REL_IN.get(si, []):             # 출처 → 현재문서 (피인용)
            f = f"{src} {ko} {slab}"
            if f not in seen:
                seen.add(f); scored.append((_REL_PRIO.get(ko, 9), f))
    scored.sort(key=lambda x: x[0])                    # 위임>준용>개정>폐지>참조
    return [f for _, f in scored[:20]]


_ANS_SYS = ("다음 <문서>와 <지식그래프 사실>만 근거로 한국어로 답하라. "
            "단답에 그치지 말고 **근거(관련 조항·별표·조건·예외·기준일 등)까지 포함해 완전하게** 답하라. "
            "단, 문서/사실에 실제로 있는 내용만 쓰고 추측·날조하지 마라. 없으면 모른다고 하라.")


def _retrieve(question):
    ensure_vecs()
    g = traverse(question)
    facts = kg_facts(g["seed_iris"])
    vg, vs = [], []
    rs = set(g.get("scope") or g["docs"])           # 벡터 스코프 = 그래프 N홉(3) 좁힌 문서집합
    if CHUNK_MAT is not None:                       # precomputed: numpy 고속 코사인
        import numpy as _np
        qv = _np.asarray(embed([question])[0], dtype=_np.float32)
        qv /= (_np.linalg.norm(qv) + 1e-9)
        sims = CHUNK_MAT @ qv
        order = _np.argsort(-sims)
        vg, seen = [], set()
        for i in order[:TOP_K]:
            i = int(i); vg.append({**CHUNKS[i], "score": float(sims[i])}); seen.add(CHUNKS[i]["text"])
        for i in order:                              # 그래프 스코프(전역 제외)
            if len(vs) >= TOP_K:
                break
            i = int(i); c = CHUNKS[i]
            if c["doc"] in rs and c["text"] not in seen:
                vs.append({**c, "score": float(sims[i])})
    elif CHUNK_VECS:
        qv = embed([question])[0]
        scored = sorted(({**c, "score": cos(qv, v)} for c, v in zip(CHUNKS, CHUNK_VECS)),
                        key=lambda r: -r["score"])
        vg = scored[:TOP_K]
        seen = {r["text"] for r in vg}
        vs = [r for r in scored if r["doc"] in rs and r["text"] not in seen][:TOP_K]
    return g, facts, vg, vs


def _answer_user(question, vg, vs, facts):
    ctx = {r["text"][:60]: r for r in (vs + vg)}.values()   # 그래프 스코프(vs) 우선 노출 후 전역(vg) 보강
    docs_block = "\n".join(f"[{i+1}] ({r['doc']}) {strip_markup(r['text'])[:1200]}" for i, r in enumerate(ctx))
    kgb = "\n".join("- " + f for f in facts) or "(없음)"
    return f"<문서>\n{docs_block}\n</문서>\n<지식그래프 사실>\n{kgb}\n</지식그래프 사실>\n\n질문: {question}"


_fmt = lambda rs: [{"doc": r["doc"], "score": round(r["score"], 3), "snippet": strip_markup(r["text"])[:90]} for r in rs]


def hybrid(question):
    g, facts, vg, vs = _retrieve(question)
    answer = chat(_ANS_SYS, _answer_user(question, vg, vs, facts))
    return {"question": question, "kg": {**g, "facts": facts},
            "vector_global": _fmt(vg), "vector_scoped": _fmt(vs), "answer": answer}


def chat_stream(system, user, max_tokens=600):
    """LLM 토큰 스트리밍(SSE). content delta 를 하나씩 yield."""
    base = os.environ["LLM_API_BASE"].rstrip("/")
    body = json.dumps({"model": os.environ.get("LLM_MODEL", "instruct"), "temperature": 0,
                       "max_tokens": max_tokens, "stream": True,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + os.environ.get("API_KEY", ""),
                                          "Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        except Exception:  # noqa: BLE001
            delta = ""
        if delta:
            yield delta


# ---------- API ----------
app = FastAPI(title="법령·매뉴얼 하이브리드 질의")


class Ask(BaseModel):
    question: str


@app.get("/healthz")
def healthz():
    return {"status": "ok", "regulations": len(REG_LABEL), "chunks": len(CHUNKS),
            "embedded": len(CHUNK_VECS), "kg_ttl": KG_TTL}


def _sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/ask")
def ask_post(body: Ask):
    """SSE 스트리밍: ① meta(kg+벡터) 즉시 → ② 답변 토큰 스트림 → ③ done."""
    def gen():
        g, facts, vg, vs = _retrieve(body.question)
        yield _sse({"type": "meta", "kg": {**g, "facts": facts},
                    "vector_global": _fmt(vg), "vector_scoped": _fmt(vs)})
        try:
            for tok in chat_stream(_ANS_SYS, _answer_user(body.question, vg, vs, facts)):
                yield _sse({"type": "tok", "t": tok})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "tok", "t": f"(생성 오류: {e})"})
        yield _sse({"type": "done"})
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/nodes")
def nodes_ep():
    """탐색기용 전체 노드 인덱스(라벨·유형·관계). 클라이언트가 검색·필터·페이지·그래프."""
    return NODES


_HTML = """<!doctype html><html lang=ko><meta charset=utf-8>
<title>법령·매뉴얼 하이브리드 질의</title>
<style>body{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:1440px;margin:0 auto;padding:20px;background:#fafafa;color:#222}
h1{font-size:20px}
.home{cursor:pointer;display:inline-flex;align-items:center;gap:7px;padding:5px 14px;border:1px solid #d4daf0;border-radius:10px;background:#fff;transition:background .15s,box-shadow .15s}
.home:hover{background:#eef2ff;box-shadow:0 1px 4px rgba(59,91,219,.18)}
.home .hi{font-size:16px;color:#3b5bdb}
.bar{display:flex;gap:8px;margin:14px 0}
.layout{display:flex;gap:16px;align-items:flex-start}
.main{flex:1;min-width:0}
.side{width:440px;flex:none;position:sticky;top:14px;max-height:calc(100vh - 28px);overflow:auto}
.card{background:#fff;border:1px solid #eaeaea;border-radius:12px;padding:16px 18px;margin:0 0 14px}
.card h2{font-size:15px;margin:0 0 10px;color:#333}
#q{flex:1;padding:12px;font-size:15px;border:1px solid #ccc;border-radius:8px}#nq{width:100%;padding:11px;font-size:14px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box;margin-top:6px}
button{padding:12px 20px;font-size:15px;border:0;border-radius:8px;background:#3b5bdb;color:#fff;cursor:pointer}button:disabled{opacity:.5}
.sec{margin-top:18px;border:1px solid #e3e3e3;border-radius:10px;padding:14px}
.sec h2{font-size:14px;margin:0 0 8px;color:#444}
.kg{background:#f3f1fb}.vec{background:#fff}.ans{white-space:pre-wrap;line-height:1.6}.ansbox{background:#eef6ff;border-color:#cfe2ff}.ansbox h2{color:#1c4fb3}
.row{font-size:13px;margin:3px 0}.sc{color:#0a7;font-variant-numeric:tabular-nums}
.dist{color:#888}small{color:#666}.tag{display:inline-block;border-radius:6px;padding:1px 8px;margin:2px;font-size:12px;color:#fff;cursor:pointer}
#nlist{max-height:56vh;overflow:auto;border:1px solid #eee;border-radius:8px;margin-top:8px}
#nlist .it{padding:6px 9px;cursor:pointer;border-bottom:1px solid #f0f0f0;font-size:13px}
#nlist .it:hover{background:#f3f7ff}
#ndetail{min-height:160px}#leg{margin:4px 0}
.lcount{padding:6px 9px;font-size:12px;color:#666;background:#fafafa;border-bottom:1px solid #eee}
.pager{display:flex;flex-wrap:wrap;gap:4px;padding:7px 9px;border-bottom:1px solid #eee}
.pg{min-width:22px;text-align:center;padding:2px 6px;border:1px solid #ccc;border-radius:6px;font-size:12px;cursor:pointer;background:#fff}
.pg.on{background:#36c;color:#fff;border-color:#36c;font-weight:bold}.pg.nav{color:#36c}.pg.dis{color:#ccc;cursor:default}.gap{color:#aaa}
.nhead{position:sticky;top:0;z-index:3;background:#fff}
.pred{color:#a05a00;font-size:12px;margin:6px 0 2px}.chip{display:inline-block;background:#e8eeff;border-radius:6px;padding:2px 8px;margin:2px;font-size:13px;cursor:pointer}.chip.in{background:#ffe8e8}
svg{border:1px solid #eee;border-radius:8px;background:#fcfcfd;max-width:100%;height:auto;display:block}svg text{font:10px system-ui;pointer-events:none}</style>
<h1 onclick=home() class=home title="처음으로 (클릭)"><span class=hi>⌂</span>법령·매뉴얼 하이브리드 질의</h1>
<div class=bar><input id=q placeholder="예) 할인율 적용 감면  ·  감사규정 관련 문서" autofocus><button id=go onclick=go()>질문</button></div>
<div class=layout>
 <div class=main>
  <div id=out></div>
  <div id=exp class=card><h2>지식그래프 탐색 — 노드 검색</h2>
   <div id=leg></div>
   <input id=nq placeholder="노드 이름 일부 입력 (예: 감사규정, 할인율, 운용요령, 위원회)">
   <div id=nlist><div class=lcount>불러오는 중…</div></div></div>
 </div>
 <div class=side><div class=card id=ndetail><div class=dist>왼쪽 목록·검색결과에서 노드를 누르면 여기에 상세(유형·관계·미니그래프)가 고정 표시됩니다.</div></div></div>
</div>
<script>
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function md(t){t=esc(t||'');
 t=t.replace(/\\$([^$]{0,40})\\$/g,function(m,inner){
   if(/right\\s*arrow|\\\\to\\b|→/.test(inner))return '→';
   if(/left\\s*arrow|←/.test(inner))return '←';
   return inner.replace(/\\\\/g,'').trim();});
 t=t.replace(/\\*\\*([^*]+?)\\*\\*/g,'<b>$1</b>');
 t=t.replace(/^(\\s*)[\\*\\-]\\s+/gm,(m,sp)=>sp.replace(/ /g,'\\u00a0')+'• ');
 return t.replace(/\\n/g,'<br>');}
function home(){document.getElementById('q').value='';document.getElementById('out').innerHTML='';
 document.getElementById('nq').value='';active.clear();drawLeg();hist=[];_page=0;_hits=IDS;nrender(true);
 const D=document.getElementById('ndetail');D.className='card';D.innerHTML='<div class=dist>왼쪽 목록·검색결과에서 노드를 누르면 여기에 상세(유형·관계·미니그래프)가 고정 표시됩니다.</div>';
 document.getElementById('exp').style.display='';
 window.scrollTo(0,0);document.getElementById('q').focus();}
var LABEL2IRI={};
function metaHtml(j){
 const kg=j.kg||{}; let h='';
 h+='<div class="card ansbox"><h2>최종 답변</h2><div class=ans id=ansdiv></div></div>';
 const vrow=rs=>(rs||[]).map(r=>'<div class=row><span class=sc>'+r.score+'</span> '+esc(r.doc)+' <span class=dist>| '+esc(r.snippet)+'</span></div>').join('')||'<div class=dist>(없음)</div>';
 const hasSeed=kg.seeds&&kg.seeds.length;
 h+='<div class="card kg"><h2>① 지식그래프 (시드: '+(hasSeed?kg.seeds.map(esc).join(", "):'없음 — 내용 기반 벡터 검색')+')</h2>';
 const relHtml=(kg.related||[]).map(x=>{const i=LABEL2IRI[x.label];return '<span class=tag style="background:#9a78d0"'+(i?' data-i="'+esc(i)+'"':'')+'>'+esc(x.label)+' '+x.distance+'홉</span>';}).join('');
 h+='<div class=row>3홉 관련 문서: '+(relHtml||'<span class=dist>'+(hasSeed?'없음 (이 문서는 다른 규정을 인용·피인용하지 않음)':'없음')+'</span>')+'</div>';
 h+=(kg.facts||[]).map(f=>'<div class=row>· '+esc(f)+'</div>').join('')+'</div>';
 h+='<div class="card vec"><h2>② 벡터 RAG — 전역</h2>'+vrow(j.vector_global)+'</div>';
 h+='<div class="card vec"><h2>② 벡터 RAG — 그래프 스코프(중복 제외)</h2>'+vrow(j.vector_scoped)+'</div>';
 return h;
}
async function go(){
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const btn=document.getElementById('go'); btn.disabled=true;
 const out=document.getElementById('out'); out.innerHTML='<div class=card>질의 중…</div>';
 document.getElementById('exp').style.display='none';
 try{
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const rd=r.body.getReader(); const dec=new TextDecoder(); let buf='',ans=null,full='';
  while(true){
   const {done,value}=await rd.read(); if(done)break;
   buf+=dec.decode(value,{stream:true});
   let p;
   while((p=buf.indexOf('\\n\\n'))>=0){
    const line=buf.slice(0,p); buf=buf.slice(p+2);
    if(!line.startsWith('data:'))continue;
    let ev; try{ev=JSON.parse(line.slice(5).trim());}catch(_){continue;}
    if(ev.type==='meta'){out.innerHTML=metaHtml(ev);ans=document.getElementById('ansdiv');full='';}
    else if(ev.type==='tok'&&ans){full+=ev.t;ans.textContent=full;}
    else if(ev.type==='done'&&ans){ans.innerHTML=md(full);}
   }
  }
  if(ans)ans.innerHTML=md(full);
 }catch(e){out.innerHTML='<div class=card>오류: '+esc(''+e)+'</div>';}
 btn.disabled=false;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')go();});

// ===== 노드 탐색기 (8810과 동일: 검색·유형필터·번호페이지·ego그래프·상세) =====
let DATA={}, IDS=[], TYPES=[], _hits=[], _page=0; const PAGE=100;
const PAL=['#1b6','#36c','#a63','#777','#b33','#609','#0a8','#c70','#558','#2a8','#946'];
const tcol=t=>PAL[TYPES.indexOf(t)%PAL.length];
const active=new Set(); let hist=[];
function tags(i,clk){return DATA[i].types.map(t=>'<span class="tag'+(active.size&&!active.has(t)?' off':'')+'" style="background:'+tcol(t)+(active.size&&!active.has(t)?';opacity:.35':'')+'"'+(clk?' data-t="'+esc(t)+'"':'')+'>'+esc(t)+'</span>').join('');}
function nrow(i){return '<div class=it data-i="'+esc(i)+'">'+esc(DATA[i].label)+tags(i)+'</div>';}
function drawLeg(){const c={};IDS.forEach(i=>DATA[i].types.forEach(t=>c[t]=(c[t]||0)+1));
 document.getElementById('leg').innerHTML=TYPES.slice(0,14).map(t=>{const off=active.size&&!active.has(t);return '<span class="tag'+(off?' off':'')+'" style="background:'+tcol(t)+(off?';opacity:.35':'')+'" data-t="'+esc(t)+'">'+esc(t)+' '+(c[t]||0)+'</span>';}).join(' ');}
function pgr(cur,tot){if(tot<=1)return '';const b=['<span class="pg nav'+(cur<=0?' dis':'')+'" data-page="'+(cur-1)+'">‹ 이전</span>'];
 const w=new Set([0,tot-1]);for(let p=cur-2;p<=cur+2;p++)if(p>=0&&p<tot)w.add(p);
 const ps=[...w].sort((a,b)=>a-b);let pv=-1;for(const p of ps){if(p-pv>1)b.push('<span class=gap>…</span>');b.push('<span class="pg'+(p===cur?' on':'')+'" data-page="'+p+'">'+(p+1)+'</span>');pv=p;}
 b.push('<span class="pg nav'+(cur>=tot-1?' dis':'')+'" data-page="'+(cur+1)+'">다음 ›</span>');return '<div class=pager>'+b.join('')+'</div>';}
function nrender(reset){if(reset)_page=0;const L=document.getElementById('nlist');
 if(!_hits.length){L.innerHTML='<div class=lcount>결과 없음</div>';return;}
 const tot=Math.ceil(_hits.length/PAGE);_page=Math.max(0,Math.min(_page,tot-1));
 const sl=_hits.slice(_page*PAGE,(_page+1)*PAGE);
 L.innerHTML='<div class=nhead><div class=lcount>'+_hits.length.toLocaleString()+'개 · '+(_page+1)+'/'+tot+' 페이지</div>'+pgr(_page,tot)+'</div>'+sl.map(nrow).join('');}
function nsearch(){const v=document.getElementById('nq').value.trim().toLowerCase();
 _hits=IDS.filter(i=>{const d=DATA[i];if(active.size&&!d.types.some(t=>active.has(t)))return false;return !v||(d.label||'').toLowerCase().includes(v);});
 nrender(true);}
function ego(i){const d=DATA[i],W=460,H=560,cx=W/2,cy=H/2,R=175;let nb=[...d.out.map(e=>[e[1],'o']),...d.in.map(e=>[e[1],'i'])].slice(0,16);
 let s='<svg width='+W+' height='+H+' viewBox="0 0 '+W+' '+H+'">';
 nb.forEach((x,k)=>{const a=2*Math.PI*k/nb.length-Math.PI/2,x2=cx+R*Math.cos(a),y2=cy+R*Math.sin(a);s+='<line x1='+cx+' y1='+cy+' x2='+x2+' y2='+y2+' stroke="'+(x[1]=='o'?'#9ab':'#d99')+'"/>';});
 nb.forEach((x,k)=>{const a=2*Math.PI*k/nb.length-Math.PI/2,x2=cx+R*Math.cos(a),y2=cy+R*Math.sin(a),lb=(DATA[x[0]]?DATA[x[0]].label:x[0]).slice(0,11);
  s+='<g data-i="'+esc(x[0])+'" style=cursor:pointer><circle cx='+x2+' cy='+y2+' r=6 fill="'+(x[1]=='o'?'#36c':'#b33')+'"/><text x='+x2+' y='+(y2+(y2<cy?-9:15))+' text-anchor=middle>'+esc(lb)+'</text></g>';});
 s+='<circle cx='+cx+' cy='+cy+' r=9 fill=#222 /><text x='+cx+' y='+(cy-13)+' text-anchor=middle style=font-weight:bold>'+esc(d.label.slice(0,16))+'</text></svg>';return s;}
function grp(es,cls){const by={};es.forEach(([p,o])=>{(by[p]=by[p]||[]).push(o);});
 return Object.keys(by).map(p=>'<div class=pred>'+esc(p)+' <span class=dist>'+by[p].length+'</span></div>'+by[p].slice(0,200).map(o=>'<span class="chip '+cls+'" data-i="'+esc(o)+'">'+esc(DATA[o]?DATA[o].label:o)+'</span>').join('')).join('');}
function nshow(i,push){if(push!==false)hist.push(i);const d=DATA[i];
 let h=(hist.length>1?'<span class="pg nav" id=nback>← 뒤로</span> ':'')+'<b style=font-size:15px>'+esc(d.label)+'</b> '+tags(i)+'<div class=dist style=font-size:11px>'+esc(i)+'</div>';
 const ak=Object.keys(d.attrs||{});if(ak.length)h+='<div style=margin-top:6px>'+ak.map(k=>'<div class=row><b style=color:#06a>'+esc(k.split(':').pop())+'</b>: '+(d.attrs[k]||[]).map(esc).join(', ')+'</div>').join('')+'</div>';
 if(d.out.length+d.in.length)h+=ego(i);
 h+='<div style=margin-top:8px><b>→ 나가는 관계 '+d.out.length+'</b>'+(grp(d.out,'')||'<div class=dist>없음</div>')+'</div>';
 h+='<div style=margin-top:8px><b>← 들어오는 관계 '+d.in.length+'</b>'+(grp(d.in,'in')||'<div class=dist>없음</div>')+'</div>';
 const D=document.getElementById('ndetail');D.className='card';D.innerHTML=h;
 const bk=document.getElementById('nback');if(bk)bk.onclick=()=>{hist.pop();nshow(hist[hist.length-1],false);};}
document.getElementById('nq').addEventListener('input',nsearch);
document.addEventListener('click',e=>{
 const pg=e.target.closest('.pg[data-page]');if(pg){if(!pg.classList.contains('dis')){_page=parseInt(pg.getAttribute('data-page'));nrender(false);const _nl=document.getElementById('nlist');if(_nl)_nl.scrollTop=0;}return;}
 const tg=e.target.closest('[data-t]');if(tg){const t=tg.getAttribute('data-t');active.has(t)?active.delete(t):active.add(t);drawLeg();nsearch();return;}
 const nd=e.target.closest('#nlist [data-i],#ndetail [data-i],#out [data-i]');if(nd&&DATA[nd.getAttribute('data-i')]){nshow(nd.getAttribute('data-i'));document.getElementById('ndetail').scrollIntoView({block:'nearest'});}});
fetch('/nodes').then(r=>r.json()).then(j=>{DATA=j;IDS=Object.keys(j);
 IDS.sort((a,b)=>(DATA[a].label||'').localeCompare(DATA[b].label||'','ko'));
 const c={};IDS.forEach(i=>{DATA[i].types.forEach(t=>c[t]=(c[t]||0)+1);LABEL2IRI[DATA[i].label]=i;});TYPES=Object.keys(c).sort((a,b)=>c[b]-c[a]);
 drawLeg();_hits=IDS;nrender(true);});
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return _HTML
