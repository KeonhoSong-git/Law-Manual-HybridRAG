"""kg — 청크 산출물에서 규정 지식그래프(Turtle)를 결정적으로 생성.

- build_v3        : 문서/조문/정의/기관 노드 + 문서간 참조(dct:references) 결정적 생성
- classify_edges_v3 : 참조 엣지의 종류(based_on/applies/amends/cites)를 LLM 제안 + 키워드 게이트로 분류
- traverse        : 시드 문서에서 N홉 관련 문서 탐색
"""
