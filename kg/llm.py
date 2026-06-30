"""kg.llm — 로컬 LLM(OpenAI 호환 vLLM) 호출 헬퍼. stdlib만 사용.

KG 의 의미 작업(시드 엔티티 판단, 관계 추출)에 쓰인다. 설정은 환경변수:
  LLM_API_BASE (예: http://host:port/v1), API_KEY, LLM_MODEL(기본 instruct).
설정이 없으면 ``available()`` 가 False → 호출부는 결정적 방식으로 폴백한다.
"""

from __future__ import annotations

import json
import os
import urllib.request

from ._env import load_env

load_env()   # import 시 .env 자동 로드 — 셸에 안 실려도 LLM 이 조용히 꺼지지 않게.


def available() -> bool:
    return bool(os.environ.get("LLM_API_BASE"))


def chat(system: str, user: str, *, max_tokens: int = 600, timeout: float = 60.0) -> str:
    base = os.environ["LLM_API_BASE"].rstrip("/")
    key = os.environ.get("API_KEY", "")
    model = os.environ.get("LLM_MODEL", "instruct")
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_json_array(text: str) -> list:
    """LLM 출력에서 JSON 배열만 안전 추출."""
    s = text.strip()
    a, b = s.find("["), s.rfind("]")
    if a >= 0 and b > a:
        try:
            v = json.loads(s[a:b + 1])
            return v if isinstance(v, list) else []
        except ValueError:
            return []
    return []
