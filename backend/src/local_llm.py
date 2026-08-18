#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로컬 LLM (Qwen3) — 4. 답변 생성

    answer.py 의 프롬프트/스키마  ->  generate_answer_local()

왜 Qwen3 인가:
  - 근거가 7개 언어로 들어오고 답은 한국어로 나가야 한다. Qwen3 는 다국어
    학습량이 많아 러시아어·베트남어 근거를 읽고 한국어로 옮기는 일을 4B 로도
    해낸다.
  - 4B 는 bf16 으로 약 8GB. bge-m3 2.3GB + bge-reranker 1.1GB 를 더해 약 11.4GB 라
    16GB GPU 에 들어간다. 여유가 있으면 LOCAL_LLM_MODEL 로 8B(약 16GB)로 올린다.
  - 답변 생성은 청크 5개(2500토큰)를 읽는 일이다. 4B 로도 감당하지만, 근거가
    없을 때 "모른다"고 말하는 능력은 모델을 키울수록 낫다.
    src/grade.py --run all 로 재보면 된다.

JSON 을 어떻게 보장하나 (여기가 제일 까다롭다):
  로컬 모델은 스키마 강제가 없다. 대개 맞는 JSON 이 나오지만 가끔 앞뒤에 설명이
  붙거나 코드펜스로 감싸져 나오고, 그러면 json.loads 가 통째로 실패한다.
  세 겹으로 막는다.
    1. 시스템 프롬프트 끝에 스키마와 "JSON 만 출력" 지시를 붙인다
    2. 코드펜스/군더더기를 벗기고 첫 { 부터 짝이 맞는 } 까지 잘라낸다
    3. 그래도 실패하면 온도를 0 으로 낮춰 한 번 더 시도한다
  더 확실한 방법은 vLLM·outlines 의 guided decoding 으로 디코딩 단계에서 문법을
  강제하는 것이다. 별도 서버나 추가 패키지가 필요해서 여기서는 넣지 않았다.

Qwen3 의 thinking 모드는 끈다: 기본값이 켜짐이라 <think>...</think> 를 먼저 뱉어
JSON 파싱이 깨지고 토큰도 몇 배로 늘어난다.

단독 실행:
    python src/local_llm.py "대마재배자는 누구에게 허가를 받나요?"   # 검색·리랭킹 후 답변
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import (  # noqa: E402
    RESPONSE_SCHEMA as ANSWER_SCHEMA,
    AnswerResult,
    build_context,
    build_system_prompt,
)
from search import Hit  # noqa: E402

# 모듈 import 시점에 한 번 읽는다. 바꾸려면 streamlit 을 띄우기 전에 설정할 것.
#   LOCAL_LLM_MODEL=Qwen/Qwen3-8B  더 큰 GPU 에서 품질을 올리고 싶을 때
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B")

# 답변은 두세 문장 + 인용이다. 크게 잡을수록 느려지기만 한다.
MAX_NEW_TOKENS_ANSWER = 768


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_llm(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    LLM 을 올린다. 프로세스당 한 번만. 캐시가 없으면 Streamlit 이 스크립트를 다시
    돌 때마다 수 GB 를 새로 올려 곧바로 OOM 이다.

    캐시 키가 (모델명, device) 라 maxsize=1 이다. 한 프로세스에서 모델을 바꿔
    부르면 순간 두 벌이 올라간다.

    device_map="auto" 는 GPU 에 안 들어갈 때 CPU 로 흘려보내며 버틴다. 그 상태로도
    돌긴 하지만 아주 느리므로 실제로 어디 올라갔는지 확인할 것.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto" if device.startswith("cuda") else None,
        dtype="bfloat16" if device.startswith("cuda") else "float32",
    )
    model.eval()
    return tokenizer, model, device


# --------------------------------------------------------------------------
# JSON 생성
# --------------------------------------------------------------------------

def _schema_hint(schema: dict) -> str:
    """스키마를 프롬프트에 붙일 짧은 안내로 바꾼다."""
    lines = []
    for key, spec in (schema.get("properties") or {}).items():
        kind = spec.get("type", "")
        if kind == "array":
            kind = f"array<{(spec.get('items') or {}).get('type', 'string')}>"
        lines.append(f'  "{key}": {kind}   // {spec.get("description", "")}')
    return "{\n" + "\n".join(lines) + "\n}"


def _extract_json(text: str) -> dict:
    """
    모델 출력에서 JSON 객체만 꺼낸다.

    첫 '{' 부터 괄호 짝이 맞는 '}' 까지를 잘라 쓴다. 문자열 안의 중괄호는 세지 않는다.
    """
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip(),
                  flags=re.MULTILINE)
    # thinking 을 못 끈 채 돌면 <think> 가 앞에 붙는다.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError(f"JSON 을 찾지 못했습니다: {text[:200]}")

    depth, in_str, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"JSON 괄호가 닫히지 않았습니다: {text[:200]}")


def generate_json(system: str, user: str, schema: dict,
                  max_new_tokens: int = 512, temperature: float = 0.7,
                  model_name: str = DEFAULT_MODEL,
                  device: str | None = None) -> dict:
    """
    JSON 하나를 받는다. 파싱에 실패하면 온도 0 으로 한 번 더 시도한다.

    재시도를 온도 0 으로 하는 이유: 형식이 깨지는 건 대개 생성이 흔들렸다는
    뜻이라, 같은 온도로 다시 굴리면 또 깨질 확률이 높다.
    """
    import torch

    tokenizer, model, _ = load_llm(model_name, device)

    system = (f"{system}\n\n"
              f"반드시 아래 형태의 JSON 하나만 출력하세요. 설명, 인사말, 코드펜스를\n"
              f"붙이지 마세요.\n\n{_schema_hint(schema)}")

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:       # enable_thinking 을 모르는 토크나이저
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    last_error: Exception | None = None
    for attempt, temp in enumerate((temperature, 0.0)):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=0.9 if temp > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        try:
            return _extract_json(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                continue
    raise ValueError(f"JSON 파싱 실패: {last_error}")


# --------------------------------------------------------------------------
# 4 단계 : 답변 생성
# --------------------------------------------------------------------------

def generate_answer_local(question: str, chunks: list[Hit],
                          model: str | None = None,
                          device: str | None = None,
                          answer_language: str | None = None) -> AnswerResult:
    """
    선정된 청크를 근거로 답을 만든다.

    answer_language  답을 쓸 언어 (기본: 한국어). 근거 청크의 언어는 블록마다
                     머리에 붙으므로 따로 넘기지 않는다.

    모델이 지어낸 청크 id 는 버린다. 로컬 모델은 이런 실수가 잦다.
    """
    question = (question or "").strip()
    if not question:
        return AnswerResult(question="", error="질문이 비어 있습니다.")
    if not chunks:
        return AnswerResult(question=question, error="근거로 쓸 청크가 없습니다.")

    model = model or DEFAULT_MODEL
    started = time.time()

    try:
        data = generate_json(
            build_system_prompt(answer_language),
            (f"질문: {question}\n\n"
             f"근거 청크 {len(chunks)}개:\n\n{build_context(chunks)}"),
            ANSWER_SCHEMA,
            max_new_tokens=MAX_NEW_TOKENS_ANSWER,
            # 근거에서 답을 뽑는 일이라 매번 흔들릴 이유가 없다.
            temperature=0.2,
            model_name=model,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        return AnswerResult(question=question, model=model,
                            elapsed=time.time() - started,
                            error=f"{type(exc).__name__}: {exc}")

    given = {hit.key for hit in chunks}
    citations = [c for c in (data.get("citations") or []) if c in given]

    return AnswerResult(
        question=question,
        answer=(data.get("answer") or "").strip(),
        enough=bool(data.get("enough")),
        citations=citations,
        note=(data.get("note") or "").strip(),
        model=model,
        elapsed=time.time() - started,
    )


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="검색·리랭킹한 청크를 근거로 로컬 Qwen3 가 답을 만든다.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--doc", default=None,
                        help="근거를 찾을 문서 (기본: 전체 문서)")
    parser.add_argument("--answer-lang", default=None,
                        help="답변을 쓸 언어 이름 (기본: 한국어)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    args = parser.parse_args()

    question = " ".join(args.question) or "대마재배자는 누구에게 허가를 받나요?"

    print(f"모델 로드 중... ({args.model}, 최초 1회는 다운로드에 오래 걸립니다)")
    _, model_obj, device = load_llm(args.model, args.device)
    print(f"장치: {getattr(model_obj, 'device', device)}")

    from rerank_gpu import rerank_cross
    from search import search

    sr = search(question, doc=args.doc)
    rr = rerank_cross(question, sr)
    print(f"\n[1-3] 후보 {len(sr.hits)}개 -> 선정 "
          f"{', '.join(h.key for h in rr.selected)}")

    ans = generate_answer_local(question, rr.selected,
                                model=args.model, device=args.device,
                                answer_language=args.answer_lang)
    if not ans.ok:
        sys.exit(f"\n[4] 답변 생성 실패: {ans.error}")
    print(f"\n[4] 답변 생성    ({ans.elapsed:.1f}초, 근거 충분: "
          f"{'예' if ans.enough else '아니오'})")
    print(f"    {ans.answer}")
    if ans.citations:
        print(f"    인용: {', '.join(ans.citations)}")


if __name__ == "__main__":
    main()
