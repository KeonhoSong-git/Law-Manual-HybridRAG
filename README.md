# Law & Manual HybridRAG

**법령(Law)과 지침·매뉴얼(Manual)** 문서 코퍼스에 대한 **하이브리드 RAG**(벡터 검색 + 지식그래프) 데모.
"규칙으로 만들 수 있는 구조는 코드가 결정적으로, 의미 판단만 LLM이" 라는 원칙으로, 환각을 게이트로 거른다.

---

## 무엇인가 / 왜

규정·지침·매뉴얼은 양이 많고 서로 인용·준용으로 얽혀 있어, **단순 벡터 RAG만으로는 "문서 간 관계"를 못 잡는다.** 이 프로젝트는 두 검색을 합친다.

- **벡터 RAG** — 질문과 의미가 가까운 *본문 청크*를 찾음(세부 내용 답변).
- **지식그래프(KG)** — 문서·조문·정의·기관과 그 *관계망*(인용/준용/위임)을 찾음(구조·맥락).

→ **그래프로 관련 문서를 좁히고(N홉) 그 안에서 벡터검색**(GraphRAG)한 뒤, 답변은 *벡터 본문 + KG 사실*을 함께 근거로 생성한다.

## 핵심 기능

- **지식그래프 자동 생성** — 문서에서 조문·정의용어·기관 노드와 문서간 참조를 결정적으로 추출, 인용의 *종류*만 LLM이 분류.
- **하이브리드 질의응답(웹 데모)** — 시드 문서 탐색 → 3홉 관련 문서 → 벡터 검색 → 근거 기반 답변을 **SSE 스트리밍**.
- **지식그래프 탐색기** — 노드 타입별 색상 범례·이름검색·가나다순·페이지네이션, 노드 상세(관계 + 에고 그래프 시각화).

## 데모 화면

공개 데모: **https://construct-joyride-cartoon.ngrok-free.dev**

## 데이터 · 모델

| 구분 | 사용한 것 | 비고 |
|------|----------|------|
| **데이터 출처** | 국가법령정보센터(law.go.kr) **OPEN API**(DRF) | 모두 공개 데이터. 현재 코퍼스 **517문서**(법령 127 · 행정규칙 273 · 매뉴얼 117) |
| **① 법령·행정규칙(Law, 법규체)** | `target=law` + `target=admrul` 의 **조문 본문**을 API로 수신 → `data_tools/fetch_law.py` | 제N조 조문 구조 = 법규체. HTML 스크래핑은 law.go.kr이 **JS SPA라 본문 미수신** → **API 방식 채택**(OC 발급 + 호출 기기 IP 등록 필요) |
| **② 매뉴얼·지침서(Manual, 공문체)** | 행정규칙 중 **매뉴얼/안내서/가이드/운영요령** 류는 본문이 비어 있고 **PDF 첨부**로만 제공 → `data_tools/fetch_manual.py` 가 PDF를 받아 텍스트 추출·청킹 | 조문 구조 없는 서술형 = 공문체. **pymupdf**로 추출, 스캔(이미지)·글자깨짐 PDF는 한글 글자수 게이트로 자동 제외 |
| **HWPX 입력(선택)** | 로컬 HWPX 규정·공문은 **[hwpx_chunker](https://github.com/KeonhoSong-git/hwpx_chunker)**([rhwp](https://github.com/edwardkim/rhwp) 소스 사용)로 조문/절 청킹 | law.go.kr 외에 보유한 HWPX 문서를 적재할 때 사용 — 동일 `by_document` 형식으로 출력해 그대로 KG·벡터에 투입 |
| **KG 구축 LLM** | **Gemma 4 31B** | 인용 *종류분류*에만 사용. 구조 추출은 규칙(결정적), LLM은 evidence·방향·domain/range 게이트 안에서만 |
| **임베딩** | **BGE-M3 (1024-dim)** | 문서·질문 동일 모델. `data_tools/embed_corpus.py` |
| **데모 서빙 LLM** | **NVIDIA Nemotron 3 Nano 4B** (Jetson Orin Nano 8GB · GGUF Q4_K_M) | 온디바이스 로컬 추론. 4B라 응답 품질은 대형 모델보다 낮을 수 있음 |

> LLM·임베딩: **OpenAI 호환 엔드포인트**. `.env` 변경만으로 교체(llama.cpp·vLLM·클라우드 등).

## 개발 / 데모 환경

- NVIDIA Jetson Orin Nano 8GB
- Ubuntu 22.04 / JetPack 6.2.1 · CUDA 12.6

## 동작 원리

> KG 생성 방법론은 동반 하네스 **[law-regulation-kg-harness](https://github.com/KeonhoSong-git/law-regulation-kg-harness)** 와 동일하다.

| 원칙 | 내용 |
|------|------|
| **구조는 결정적, 의미만 LLM** | 조문·날짜·정의용어·「」참조·IRI(`uuid5`)는 정규식/규칙으로 추출. 인용의 *종류*와 기관 판정만 LLM이 제안 |
| **인용 종류분류** | 참조를 위임(`based_on`)·준용(`applies`)·개정(`amends`)·단순참조(`cites`)로 구분 |
| **제안 → 코드 게이트** | LLM 제안을 ① evidence∈원문 ② 방향(하위→상위 위계) ③ 신호어(위임/준용) ④ domain/range ⑤ 기관 named-body 판정 게이트로 검증, 통과만 채택 |
| **표준 어휘만, 날조 금지** | ELI(법령)·Dublin Core·SKOS(정의)·W3C ORG(기관). 매핑 없는 술어·타입은 드롭 |
| **Lean KG** | 문서·조문·정의·기관 노드까지만. 본문 세부는 벡터 청크에 위임(`dct:identifier`로 링크) |

문서 유형별 처리:
- **법령(법규체)**: `제N조(제목)` 구조가 정형 → 조문 노드 + 「」 인용을 정밀 추출.
- **매뉴얼/지침(공문체)**: 조문 구조가 없으면 *문서·관계 수준*만 KG로, 절차 등 세부는 벡터가 담당.

## 파이프라인

```
수집/청킹 → by_document/*.json
   │  ①-A 법령·행정규칙  fetch_law.py     (조문 단위 → 법규체)
   │  ①-B 매뉴얼·지침서  fetch_manual.py  (PDF 첨부 → 텍스트 추출 → 공문체 윈도우)
   │  ①-C (선택) 로컬 HWPX  ingest_hwpx.py / hwpx_chunker
   ▼
청크 JSON
   │  ② KG 구축   python -m kg.build_v3        → kg.ttl  (구조: 결정적)
   │  ②b 종류분류 python -m kg.classify_edges_v3 → 인용 엣지에 based_on/applies/amends/cites 부여(LLM+게이트)
   │  ③ 임베딩    BGE-M3                         → vectors.npy (+meta)
   ▼
demo/data/{kg.ttl, vectors.npy}
   │  ④ 데모 서버  demo/kg_api.py (:8800)
   ▼
하이브리드 질의응답 웹
```
- ①·③(청킹·임베딩) 같은 무거운 사전계산은 강한 기기에서 미리. 디바이스는 ④만 담당(가벼움).

> 📄 **단계별 청킹·데이터 처리 규칙 상세** → [`docs/data-processing.md`](docs/data-processing.md)
> (법령 vs 행정규칙(법규체) vs 매뉴얼(공문체) 청킹 차이, 정의·기관·참조 추출 규칙, 인용 종류분류 게이트, 임베딩 저장 형식)

## 검색이 이뤄지는 방법 (질의 1회 흐름)

질문이 들어오면 `kg_api.py`가 다음을 수행한다.

1. **시드 문서 선택** — 질문에서 규정명을 식별.
   - 1순위 `string_seeds`: 규정명 핵심 토큰이 질문에 충분히(길이가중 ≥60%) 또는 변별력 큰 토큰(≥6자)이 들어오면 채택(괄호·조사 정규화).
   - 실패 시 `llm_seeds`: 규정 목록을 주고 LLM이 의미로 매칭.
2. **그래프 탐색(3홉)** — 시드에서 참조 엣지(`dct:references`·`eli:*`)를 따라 **N홉(기본 3) BFS** → 관련 문서 집합.
3. **KG 사실 수집** — 관련 문서의 트리플을 한국어 사실로 정리(예: "A 준용 B", "위원회 설치근거 …").
4. **벡터 검색** — 질문을 BGE-M3로 임베딩 → 전 코퍼스 코사인 → **상위 TOP_K(기본 6)**.
   - `vector_global`: 전역 상위 · `vector_scoped`: 위 그래프 관련 문서 *안에서만* 추가 검색(중복 제외).
5. **답변 생성** — `<문서>(벡터 본문) + <지식그래프 사실>` *만* 근거로 LLM이 한국어 답변을 **토큰 스트리밍(SSE)**. 없으면 "모른다".

> 핵심: **그래프가 범위를 좁히고(관계), 벡터가 내용을 채우고(본문), LLM은 근거 안에서만 합성**한다.

## 데이터 형식

### (입력) 청킹 JSON — `by_document/<문서>.json`
문서 1개 = 청크 dict의 배열. **첫 청크가 문서 메타를 운반**한다.
```json
[
  {
    "doc_title": "○○감독규정 시행세칙",
    "source_file": "행정규칙_○○감독규정_시행세칙.json",
    "doc_family": "법규체",          // 법규체(조문 구조) | 공문체(매뉴얼·지침)
    "doc_type": "행정규칙",          // 법령 | 행정규칙 | 매뉴얼
    "enacted": "2010. 1. 1.",        // 제정일
    "last_amended": "2023. 5. 1.",   // 최종 개정일
    "unit": "조",                    // 조 | 본문 | 문서
    "article_label": "제1조",
    "article_title": "목적",
    "text": "제1조(목적) 이 세칙은 「○○감독규정」에서 위임된 사항을 정한다 …",
    "chunk_id": "d8f1a2…#0001",
    "content_hash": "9c3f…"
  }
]
```

| 필드 | 용도 |
|------|------|
| `doc_family` | 법규체(제N조 정형 → 조문 노드) / 공문체(서술형 → 문서 노드만) 분기 |
| `doc_type` · `unit` · `article_label`/`title` | 문서·조문 노드 라벨 |
| `enacted` · `last_amended` | 제정·개정일 리터럴(`xsd:date`) |
| `text` | 결정적 추출의 원천 — 모든 evidence가 여기 실재해야 채택 |
| `chunk_id` · `content_hash` | KG↔벡터 조인(`dct:identifier`) · 증분 재추출 |

### (KG) `kg.ttl` — RDF/Turtle
```turtle
doc:d8f… rdf:type eli:LegalResource ; rdfs:label "감사규정" ;
         eli:date_publication "2010-01-01"^^xsd:date .
doc:d8f…_제1조 rdf:type eli:LegalResourceSubdivision ; rdfs:label "제1조 목적" .
doc:d8f… eli:has_part doc:d8f…_제1조 .
doc:d8f… dct:references doc:d2a… .          # 다른 문서 인용
doc:d8f… eli:applies   doc:d2a… .           # 인용 종류분류(준용)
term:t1… rdf:type skos:Concept ; skos:definition "…" .
```
- 노드 타입: `eli:LegalResource`(법령) · `reg:Directive`/`reg:Manual`/`reg:ProvisionalMeasure`(공문) · `eli:LegalResourceSubdivision`(조문) · `skos:Concept`(정의용어) · `org:Organization`(기관)
- 관계: `eli:has_part` · `dct:references` · `eli:cites`/`based_on`/`applies`/`amends` · `skos:definition` · `reg:mentionedIn`/`establishedBy`

### (벡터) `vectors.npy` (float16, N×1024) + `vectors.meta.jsonl`
- `vectors.npy` — BGE-M3 1024차원 임베딩 행렬(float16). 부팅 시 정규화 numpy 행렬로 적재 → 쿼리당 코사인 1회.
- `vectors.meta.jsonl` — 행별 `{"id","doc","text"}` (벡터 행과 순서 일치).
```json
{"id":"<uuid5>","doc":"감사규정","text":"제1조(목적) …"}
```

---

## 시작하기 (Getting Started)

### 사전 준비
- **Python 3.12**, **Docker** (+ Docker Compose)
- **OpenAI 호환 LLM 엔드포인트** — 답변·인용 분류용 (예: vLLM/llama.cpp/클라우드)
- **OpenAI 호환 임베딩 엔드포인트** — BGE-M3, 1024-dim
- *(데이터를 직접 수집할 때만)* **law.go.kr OPEN API OC** — [open.law.go.kr](https://open.law.go.kr)에서 발급 + 호출 기기 IP 등록

### 1. 클론 & 환경변수
```bash
git clone https://github.com/KeonhoSong-git/Law-Manual-HybridRAG.git
cd Law-Manual-HybridRAG
cp .env.example .env          # ⚠️ .env 는 저장소 루트에 둔다
```
`.env` 최소 설정:
```ini
LLM_API_BASE=http://<your-llm>/v1          # /chat/completions 제공
API_KEY=<key>
LLM_MODEL=<model>
EMBEDDING_API_BASE=http://<your-embedder>/v1   # BGE-M3, /embeddings 제공
```

### 2. 데이터 준비 (`demo/data/`)
데모는 `demo/data/kg.ttl` + `demo/data/vectors.npy`(+ `.meta.jsonl`) 이 필요하다. **공개 코퍼스(law.go.kr 517문서)로 만든 KG·벡터가 저장소에 포함**되어 있어, `.env`에 LLM·임베딩 엔드포인트만 연결하면 바로 돈다. 직접 다시 만들려면 둘 중 하나:
- **A. 직접 생성** → 아래 [데이터 파이프라인](#데이터-파이프라인-직접-만들기)
- **B.** 이미 있는 `kg.ttl`·`vectors.npy`(+`.meta.jsonl`) 을 `demo/data/` 에 복사

### 3. 실행
```bash
python demo/preflight.py            # (선택) 연결 점검: 번들 데이터 정합 + LLM·임베딩 도달성·1024-dim
cd demo && docker compose up -d --build
```
→ 브라우저에서 **http://localhost:8800**

> **다른 환경 이식**: KG·벡터가 번들돼 있어 데이터는 옮길 필요 없다. 새 환경에선 `.env`에 LLM·임베딩(BGE-M3, 1024-dim) 엔드포인트만 연결하고 `preflight.py`가 **ALL PASS**면 그대로 굴러간다. 임베딩이 BGE-M3가 아니면 질문 벡터가 번들 벡터와 공간이 어긋나니 반드시 같은 모델로.

> 코드엔 **하드코딩된 주소·키가 없다**(전부 `.env`). 즉 클론 후 고칠 코드는 없고, **루트 `.env` + `demo/data/` 데이터** 두 가지만 준비하면 된다.

---

## 데이터 파이프라인 (직접 만들기)

입력은 **아래 경로 중 하나 이상**으로 `by_document/*.json`(청크)을 만든 뒤, 같은 ②③④를 탄다.

```bash
# 1-A) law.go.kr OPEN API — 법령·행정규칙 조문 본문(법규체)
#      (.env 에 LAW_OC=<oc> + 호출 기기 IP 등록: open.law.go.kr)
python -m data_tools.fetch_law --query 신용보증 --target law    --out datasets/by_document --max 20
python -m data_tools.fetch_law --query 신용정보 --target admrul --out datasets/by_document --max 20

# 1-B) law.go.kr 매뉴얼·지침서 — PDF 첨부에서 텍스트 추출(공문체). pip install pymupdf 필요
python -m data_tools.fetch_manual --out datasets/by_document --max 40
#      (기본 검색어: 매뉴얼,안내서,가이드,업무편람,지침서,운영요령 / 스캔·깨짐 PDF는 자동 제외)

# 1-C) (선택) 로컬 hwpx/txt 직접 ingest (OC·인터넷 불필요). 표·도형 정밀 추출은 hwpx_chunker 사용:
#      https://github.com/KeonhoSong-git/hwpx_chunker
python -m data_tools.ingest_hwpx --in <문서_디렉터리> --out datasets/by_document

# 2) 지식그래프
python -m kg.build_v3          datasets/by_document _out   # → _out/kg.ttl (구조: 결정적)
python -m kg.classify_edges_v3 datasets/by_document _out   # 인용 종류분류 (Gemma 4 31B + 게이트)

# 3) 임베딩 + 배치
python -m data_tools.embed_corpus --chunks datasets/by_document --out demo/data/vectors.npy
cat _out/kg.ttl _out/kg_relations.ttl > demo/data/kg.ttl

# 4) cd demo && docker compose up -d --build
```

> 임베더를 바꿔도 코드 수정 불필요 — 문서·질문을 **같은 모델**로만 맞추고 `vectors.npy` 만 재생성하면 된다(차원이 1024가 아니어도 numpy가 데이터 차원을 따른다).

---

## 디렉터리

```
data_tools/
  fetch_law.py         # law.go.kr OPEN API → by_document/*.json (법령·행정규칙 조문, 법규체)
  fetch_manual.py      # law.go.kr 매뉴얼·지침서 PDF 첨부 → by_document/*.json (공문체, pymupdf)
  ingest_hwpx.py       # 로컬 hwpx/txt → by_document/*.json (본문 추출 + 조문 청킹)
  embed_corpus.py      # 청크 → vectors.npy (+meta) (BGE-M3 임베딩)
kg/
  build_v3.py          # 문서/조문/정의/기관 노드 + dct:references 결정적 생성 → kg.ttl
  classify_edges_v3.py # 참조 엣지 종류분류(LLM 제안 + 키워드/방향 게이트)
  traverse.py          # 시드 문서 → N홉 관련 문서 탐색
  llm.py               # OpenAI 호환 LLM 호출 유틸
demo/
  kg_api.py            # FastAPI 단일 파일: KG 로드 + 벡터(numpy) + 하이브리드 Q&A 웹 UI(SSE)
  Dockerfile · docker-compose.yml
```

## Notice

- 원 데이터 셋은 타사의 내부 규정, 메뉴얼로 공개할 수 없어 비슷한 형태, 도메인 데이터를 국가법령센터에서 수집하여 VectorRAG, GraphRAG를 구축함.
- 오픈 소스 [rhwp](https://github.com/edwardkim/rhwp) 개발에 감사드림.
