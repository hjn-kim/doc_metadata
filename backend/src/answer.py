#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4. LLM 답변 생성 — 프롬프트 / 스키마 / 자료구조

실제 생성은 local_llm.generate_answer_local() 이 한다. 이 파일은 프롬프트와
결과 자료구조, 근거 블록 조립만 담당한다.

프롬프트에서 신경 쓴 것:
  1. 근거 밖으로 나가지 말 것. 법령·보고서는 모델이 사전지식으로 그럴듯하게
     지어내기 딱 좋은 소재다(조 번호, 형량, 통계). 근거가 모자라면 모자라다고
     답하게 하고 그 사실을 enough 필드로 따로 받는다 (답변 본문 파싱보다 안전).
  2. 근거가 7개 언어로 흩어져 있다. 질문과 답변은 한국어이므로 근거를 읽어
     한국어로 옮겨 답해야 한다. 청크마다 어느 문서 · 어느 언어인지 머리에
     붙여 준다.
  3. 숫자와 고유명사는 원문 표기를 그대로 옮기게 한다. 금액·연도·조 번호는
     번역하는 순간 틀리기 시작한다.
  4. 인용을 청크 id(ko#12)로 달게 해서 화면에서 근거로 되짚을 수 있게 한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search import Hit  # noqa: E402

# 답을 쓸 언어. 질문이 한국어로 들어오므로 답도 한국어로 받는다.
DEFAULT_ANSWER_LANGUAGE = "한국어"

SYSTEM_PROMPT = """당신은 각국의 법령·형사사법 보고서를 근거로 질문에 답하는 도우미입니다.
근거 청크는 한국어·영어·중국어·베트남어·필리핀어·러시아어·우즈베크어 중
어느 언어로든 올 수 있고, 답변은 {answer_language}로 씁니다.

지켜야 할 것:

1. **주어진 근거 안에서만 답하세요.** 근거에 없는 조문 번호·형량·금액·날짜·
   인명을 채워 넣지 마세요. 알고 있는 법 지식으로 근거를 보충하지도 마세요.
   답을 특정할 수 없으면 enough 를 false 로 두고 "관련 내용의 부재로 답변할 수
   없음"을 출력하세요. 절대로 지어내지 마세요.

2. **근거가 외국어면 읽어서 {answer_language}로 옮겨 답하세요.** 원문을 그대로
   붙여 넣지 마세요. 다만 숫자(금액·연도·인원·조문 번호)와 기관·인명·법령명
   같은 고유명사는 근거에 적힌 값을 그대로 옮기고, 필요하면 괄호에 원문 표기를
   덧붙이세요.

3. **여러 청크를 교차 확인하세요.** 같은 조문이 앞뒤 청크에 걸쳐 잘려 있을 수
   있습니다. 목차나 각주에만 나오는 제목을 본문 내용으로 착각하지 마세요.

4. **인용을 다세요.** 답의 근거가 된 청크 id(예: ko#12)를 citations 에 빠짐없이
   넣으세요. 쓰지 않은 청크는 넣지 마세요.

5. 답변은 두세 문장으로 짧게 쓰세요. 조문에 근거한 답이면 몇 조인지 함께
   적으면 좋습니다."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "질문에 대한 답. 근거에서 확인한 내용만 담은 한국어 두세 문장.",
        },
        "enough": {
            "type": "boolean",
            "description": "주어진 근거만으로 답을 특정할 수 있으면 true.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "답의 근거가 된 청크 id 목록 (예: ko#12).",
        },
        "note": {
            "type": "string",
            "description": "근거가 부족하거나 청크마다 값이 달라 판단이 갈린 지점. 없으면 빈 문자열.",
        },
    },
    "required": ["answer", "enough", "citations", "note"],
}


@dataclass
class AnswerResult:
    """4단계 결과. 실패해도 예외를 던지지 않고 error 에 담아 돌려준다."""

    question: str
    answer: str = ""
    enough: bool = False
    citations: list[str] = field(default_factory=list)
    note: str = ""
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_system_prompt(answer_language: str | None = None) -> str:
    """답변 언어를 끼운 시스템 프롬프트."""
    return SYSTEM_PROMPT.replace(
        "{answer_language}", answer_language or DEFAULT_ANSWER_LANGUAGE)


def build_context(chunks: list[Hit]) -> str:
    """
    청크들을 근거 블록으로 만든다.

    id 와 문서 제목·언어를 머리에 달아 준다. 전체 문서를 뒤진 경우 한 답변의
    근거가 여러 나라 문서에서 올 수 있어서, 모델이 어느 나라 제도 이야기인지
    구분하려면 출처가 붙어 있어야 한다.
    """
    blocks = []
    for hit in chunks:
        blocks.append(
            f"[{hit.key}] {hit.doc_title} ({hit.doc_lang})\n"
            f"{' '.join(hit.text.split())}"
        )
    return "\n\n---\n\n".join(blocks)
