#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2. 리랭킹 + 3. 최종 청크 선정 — 자료구조

실제 재점수는 rerank_gpu.rerank_cross() 가 크로스인코더로 한다. 이 파일은
공용 자료구조만 들고 있다.

왜 리랭킹이 필요한가:
  임베딩 검색은 질의와 청크를 각각 따로 벡터로 만들어 비교한다(bi-encoder).
  빠르지만 둘을 같이 읽고 판단하지는 못한다. 그래서 "벌칙 조항을 나열한 목차
  청크"와 "그 형량이 실제로 적힌 조문 청크"를 잘 못 가른다. 법령·보고서는
  목차·색인·머리말이 본문과 같은 낱말을 그대로 반복하므로 이 구분이 특히
  중요하다.

최종 몇 개를 고르나 (FINAL_TOP_N = 5):
  한 조문이 512토큰 경계에 걸려 앞뒤 청크로 잘리는 일이 흔해서, 답이 한 청크에
  다 들어 있지 않은 경우를 덮으려면 3개는 위험하다. 10개까지 늘리면 다른 나라
  문서의 무관한 청크가 섞여 답이 흐려진다. 512토큰 x 5 = 2560토큰이라 LLM
  입력으로도 가볍다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search import Hit  # noqa: E402

# 최종적으로 답변 생성에 넘길 청크 수
FINAL_TOP_N = 5


@dataclass
class RankedHit:
    """리랭킹을 거친 후보 하나."""

    hit: Hit
    rank_before: int              # 리랭킹 전 등수 (dense 점수순)
    rank_after: int = 0           # 리랭킹 후 등수
    score: float | None = None    # 크로스인코더 raw logit. 정렬은 이 값으로 한다
    prob: float | None = None     # 위를 시그모이드에 넣은 0~1 확률 (표시용)
    reason: str = ""

    @property
    def moved(self) -> int:
        """등수가 몇 칸 올라갔는지. 양수면 상승."""
        return self.rank_before - self.rank_after

    @property
    def percent(self) -> str:
        """
        확률을 화면에 쓸 문자열로. 값이 아주 작아도 0 으로 뭉개지지 않게 한다.

        질문과 근거의 언어가 다르면 이 모델의 확률은 0.001 언저리까지 내려간다.
        절대값이 낮아도 후보끼리의 순서는 살아 있으므로 자릿수를 지켜 보여준다.
        """
        if self.prob is None:
            return "-"
        if self.prob >= 0.01:
            return f"{self.prob * 100:.1f}%"
        if self.prob >= 0.0001:
            return f"{self.prob * 100:.3f}%"
        return f"{self.prob * 100:.1e}%"


# 점수가 없는 후보를 정렬에서 맨 뒤로 보내는 값. raw logit 은 음수가 정상
# 범위(무관한 청크는 -10 근처)라 None 을 0 으로 대신 쓰면 채점에 실패한 후보가
# 정상적으로 낮게 나온 후보보다 위로 올라간다.
_NO_SCORE = float("-inf")


def sort_ranked(ranked: list[RankedHit]) -> list[RankedHit]:
    """
    score(raw logit) 내림차순으로 줄 세우고 rank_after 를 매긴다.

    정렬 기준을 이 한 곳에 둔다. 재점수는 rerank_gpu 가 하지만 "무엇으로
    줄 세우는가" 는 RankedHit 을 정의한 여기가 정한다.

    score 는 반올림하지 않은 실수라 사실상 동점이 없다. 크로스인코더가 죽어
    전부 None 이면 두 번째 키인 dense 점수만 남아 검색 순서가 그대로 유지된다.
    """
    ranked.sort(
        key=lambda x: (x.score if x.score is not None else _NO_SCORE,
                       x.hit.score),
        reverse=True,
    )
    for rank, item in enumerate(ranked, 1):
        item.rank_after = rank
    return ranked


@dataclass
class RerankResult:
    """2단계 + 3단계 결과."""

    question: str
    method: str                                             # "cross" | "dense"
    ranked: list[RankedHit] = field(default_factory=list)    # 후보 전체, 재정렬됨
    selected: list[Hit] = field(default_factory=list)        # 최종 선정 (top_n)
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None    # 크로스인코더 실패 시 사람이 읽을 메시지

    @property
    def ok(self) -> bool:
        return self.error is None
