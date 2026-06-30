"""preflight — 새 환경에서 데모가 '굴러가는지' 한 번에 점검.

번들 데이터(kg.ttl + vectors)는 저장소에 포함돼 있으므로, 실제로 필요한 건 **연결**뿐이다:
OpenAI 호환 LLM 엔드포인트 + 같은 BGE-M3(1024-dim) 임베딩 엔드포인트. 이 스크립트는
① 번들 데이터 존재·정합 ② LLM 도달성 ③ 임베딩 도달성·차원(1024)을 검사해 PASS/FAIL로 보고한다.

사용(저장소 루트 또는 demo/에서):  python demo/preflight.py
env: `.env`(루트) 또는 환경변수 — LLM_API_BASE / API_KEY / EMBEDDING_API_BASE / EMBEDDING_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _load_env() -> None:
    for cand in (os.path.join(HERE, "..", ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(cand):
            for ln in open(cand, encoding="utf-8"):
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


def _post(base: str, path: str, body: dict, key: str, timeout: int = 30) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def check_data() -> bool:
    npy = os.path.join(DATA, "vectors.npy")
    meta = os.path.join(DATA, "vectors.meta.jsonl")
    ttl = os.path.join(DATA, "kg.ttl")
    for p in (ttl, npy, meta):
        if not os.path.exists(p):
            print(f"  [FAIL] 번들 데이터 없음: {os.path.relpath(p)}")
            return False
    try:
        import numpy as np
        rows = np.load(npy, mmap_mode="r").shape[0]
        n = sum(1 for _ in open(meta, encoding="utf-8"))
        if rows != n:
            print(f"  [FAIL] 벡터행({rows}) ≠ 메타({n}) — 데이터 불일치")
            return False
        print(f"  [PASS] 데이터: kg.ttl + vectors {rows}×{np.load(npy, mmap_mode='r').shape[1]} ↔ meta {n} 정합")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] 데이터 로드 오류: {type(e).__name__}: {e}")
        return False


def check_llm() -> bool:
    base = os.environ.get("LLM_API_BASE")
    if not base:
        print("  [FAIL] LLM_API_BASE 미설정 (.env)")
        return False
    try:
        d = _post(base, "/chat/completions",
                  {"model": os.environ.get("LLM_MODEL", "instruct"),
                   "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                  os.environ.get("API_KEY", ""))
        ok = bool(d.get("choices"))
        print(f"  [{'PASS' if ok else 'FAIL'}] LLM 도달: {base}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] LLM 연결 실패 {base}: {type(e).__name__}: {e}")
        return False


def check_embedding() -> bool:
    base = os.environ.get("EMBEDDING_API_BASE")
    if not base:
        print("  [FAIL] EMBEDDING_API_BASE 미설정 (.env)")
        return False
    key = os.environ.get("EMBEDDING_API_KEY", "")
    # native(/embeddings) → 실패 시 vLLM(/v1/embeddings) 순으로 시도
    for path, body in (("/embeddings", {"input": ["테스트"]}),
                       ("/v1/embeddings", {"input": ["테스트"], "model": os.environ.get("EMBEDDING_MODEL_2", "bge-m3")})):
        try:
            d = _post(base, path, body, key)
            vec = d["data"][0]["embedding"]
            dim = len(vec)
            ok = dim == 1024
            print(f"  [{'PASS' if ok else 'WARN'}] 임베딩 도달: {base}{path} (dim={dim}{'' if ok else ' — BGE-M3는 1024여야 번들 벡터와 호환'})")
            return ok
        except Exception:  # noqa: BLE001
            continue
    print(f"  [FAIL] 임베딩 연결 실패: {base} (/embeddings·/v1/embeddings 모두)")
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    _load_env()
    print("preflight — 데모 실행 전 연결 점검")
    results = [check_data(), check_llm(), check_embedding()]
    print("-" * 40)
    if all(results):
        print("ALL PASS → `cd demo && docker compose up -d --build` 로 실행 가능")
        sys.exit(0)
    print("일부 FAIL → 위 항목(데이터/엔드포인트) 조치 후 재실행. 데이터는 저장소에 번들돼 있으니 보통 .env 엔드포인트 문제다.")
    sys.exit(1)


if __name__ == "__main__":
    main()
