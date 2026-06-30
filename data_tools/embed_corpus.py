"""data_tools.embed_corpus — by_document 청크를 임베딩해 demo용 vectors.jsonl 생성.

각 청크 텍스트를 BGE-M3(OpenAI 호환 /embeddings)로 임베딩 →
`{"id","doc","text","embedding"}` JSON Lines로 출력(데모 kg_api.py가 그대로 로드).

env: EMBEDDING_API_BASE (필수, 예: http://localhost:8081/v1), EMBEDDING_API_KEY(선택)
사용: python -m data_tools.embed_corpus --chunks datasets/by_document --out demo/data/vectors.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

NS = uuid.UUID("c1d2e3f4-5a6b-7c8d-9e0f-1a2b3c4d5e6f")  # build_v3와 동일 네임스페이스


def chunk_id(doc: str, text: str) -> str:
    return uuid.uuid5(NS, f"{doc}\x01{text}").hex


def _endpoints() -> list[dict]:
    """임베딩 엔드포인트 목록. 2개 이상이면 배치를 엔드포인트별로 나눠 병렬 처리.

    - EMBEDDING_API_BASE (필수): BGE-M3 native 서버. `/embeddings`, return_dense 플래그.
    - EMBEDDING_API_BASE_2 (선택): vLLM 호환 서버. `/v1/embeddings`, model 필드(+Bearer 키).
      같은 BGE-M3(1024-dim)여야 함(벡터 공간 일치). 응답은 둘 다 `data[].embedding`.
    """
    key = os.environ.get("EMBEDDING_API_KEY", "")
    b1 = os.environ.get("EMBEDDING_API_BASE")
    if not b1:
        raise SystemExit("EMBEDDING_API_BASE 환경변수가 필요합니다.")
    eps = [{"url": b1.rstrip("/") + "/embeddings", "key": key,
            "extra": {"return_dense": True, "return_sparse": False, "return_colbert": False}}]
    b2 = os.environ.get("EMBEDDING_API_BASE_2")
    if b2:
        eps.append({"url": b2.rstrip("/") + "/v1/embeddings",
                    "key": os.environ.get("EMBEDDING_API_KEY_2", key),
                    "extra": {"model": os.environ.get("EMBEDDING_MODEL_2", "/models/bge-m3")}})
    return eps


def _embed_batch(ep, texts, timeout=60, tries=4):
    """한 배치 임베딩 — 타임아웃/오류 시 재시도, 끝내 실패하면 배치 절반으로 분할(stall 격리)."""
    body = json.dumps({"input": texts, **ep["extra"]}).encode()
    headers = {"Content-Type": "application/json"}
    if ep["key"]:
        headers["Authorization"] = "Bearer " + ep["key"]
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(ep["url"], data=body, headers=headers)
            return [r["embedding"] for r in json.loads(
                urllib.request.urlopen(req, timeout=timeout).read())["data"]]
        except Exception as e:  # noqa: BLE001
            last = e
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = f" {e.code}: {e.read().decode('utf-8', 'replace')[:160]}"
                except Exception:  # noqa: BLE001
                    detail = f" {getattr(e, 'code', '')}"
            print(f"    retry {t + 1}/{tries} (batch {len(texts)} @ {ep['url']}): {type(e).__name__}{detail}", flush=True)
            time.sleep(2 * (t + 1))
    if len(texts) > 1:  # 분할해서 문제 텍스트 격리
        mid = len(texts) // 2
        return (_embed_batch(ep, texts[:mid], timeout, tries)
                + _embed_batch(ep, texts[mid:], timeout, tries))
    raise last


def embed(texts: list[str], batch: int = 32) -> list[list[float]]:
    """엔드포인트별 1요청씩(스레드 1개) 병렬 — 엔드포인트 과부하 없이 N배 throughput."""
    eps = _endpoints()
    batches = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    results: list = [None] * len(batches)
    counter = {"n": 0}
    lock = threading.Lock()

    def worker(eidx: int) -> None:
        ep = eps[eidx]
        for bi in range(eidx, len(batches), len(eps)):  # 엔드포인트별 배치 분담(인터리브)
            results[bi] = _embed_batch(ep, batches[bi])
            with lock:
                counter["n"] += len(batches[bi])
                print(f"  embedded ~{counter['n']}/{len(texts)}", flush=True)

    print(f"  엔드포인트 {len(eps)}개 병렬: {[e['url'] for e in eps]}", flush=True)
    threads = [threading.Thread(target=worker, args=(e,)) for e in range(len(eps))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out: list[list[float]] = []
    for r in results:
        out += r
    return out


def load_chunks(src_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        arr = json.load(open(p, encoding="utf-8"))
        title = arr[0].get("doc_title") if arr else None
        for c in arr:
            t = (c.get("text") or "").strip()
            if t:
                rows.append({"doc": title, "text": t})
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="청크 → vectors.jsonl 임베딩")
    ap.add_argument("--chunks", required=True, help="by_document 디렉터리")
    ap.add_argument("--out", required=True, help="vectors.jsonl 출력 경로")
    ap.add_argument("--max-chars", type=int, default=1000, help="청크당 임베딩 최대 길이")
    a = ap.parse_args()

    if not os.environ.get("EMBEDDING_API_BASE"):
        raise SystemExit("EMBEDDING_API_BASE 환경변수가 필요합니다.")
    rows = load_chunks(a.chunks)
    print(f"청크 {len(rows)}개 임베딩 시작…", flush=True)
    vecs = embed([r["text"][:a.max_chars] for r in rows])

    # 경량 저장: 벡터는 float16 .npy, 메타(id/doc/text)는 .meta.jsonl (JSON 풀정밀 대비 ~1/12).
    import numpy as np
    npy = a.out[:-6] + ".npy" if a.out.endswith(".jsonl") else (
        a.out if a.out.endswith(".npy") else a.out + ".npy")
    meta = npy[:-4] + ".meta.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(npy)), exist_ok=True)
    np.save(npy, np.asarray(vecs, dtype=np.float16))
    with open(meta, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": chunk_id(r["doc"] or "", r["text"]),
                                "doc": r["doc"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"완료: {len(rows)}개 → {npy} (+ {os.path.basename(meta)})")


if __name__ == "__main__":
    main()
