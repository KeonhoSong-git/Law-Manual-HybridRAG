"""공유 .env 로더 — kg/data_tools 모듈이 셸 환경에 의존하지 않도록 import 시 가까운 .env 를 읽어
환경변수에 채운다. 이미 설정된 값은 보존(셸 export 우선). 이게 없으면 셸에 .env 가 안 실린 채
실행될 때 LLM_API_BASE 등이 비어 available()=False 가 되어 분류가 조용히 결정적 폴백으로 떨어진다.
demo/kg_api._load_env 와 동일한 디렉터리 상향 탐색 방식."""
from __future__ import annotations

import os


def load_env() -> None:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.startswith("#") and "=" in ln:
                        k, v = ln.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            return
        d = os.path.dirname(d)
