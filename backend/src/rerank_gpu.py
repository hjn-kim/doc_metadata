#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2. 리랭킹 (GPU / 크로스인코더)

search.py 가 뽑은 상위 15개 안팎을 BAAI/bge-reranker-v2-m3 로 다시 줄 세운다.

bi-encoder(검색) 와 뭐가 다른가:
  검색 단계는 질의와 청크를 각각 따로 벡터로 만들어 비교한다. 빠르지만 둘을 같이
  읽지는 못한다. 크로스인코더는 (질의, 청크) 를 한 입력으로 붙여 넣고 통째로 읽어
  관련성 점수 하나를 낸다. 느린 대신 정확해서 후보가 20개로 줄어든 뒤에 쓴다.

왜 bge-reranker-v2-m3 인가:
  검색에 쓴 bge-m3 와 같은 XLM-RoBERTa-large(568M) 계열이라 이 코퍼스의 7개 언어를
  전부 커버하고, 한국어 질문 대 러시아어 청크처럼 언어가 엇갈린 쌍도 같은 기준으로
  채점한다. fp16 으로 1.1GB 남짓이고 임베딩 모델과 같이 올려도 16GB GPU 에
  여유롭다. 인코더 전용이라 쌍 하나당 forward 한 번이면 끝난다.

  판단 근거 문장은 나오지 않는다. 숫자 하나만 내므로 화면의 '판단 근거' 칸에는
  설명 대신 확률과 logit 을 적는다.

CPU 에서는 쓰지 말 것: 후보 15개 x 512토큰을 CPU 로 재점수하면 10~20초가 걸린다.
실패하면(모델 없음 / VRAM 부족) 검색 점수 순서(dense)를 그대로 쓴다.

단독 실행:
    python src/rerank_gpu.py "대마재배자는 누구에게 허가를 받나요?"
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank import FINAL_TOP_N, RankedHit, RerankResult, sort_ranked  # noqa: E402
from search import SearchResult, search  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# 청크가 bge-m3 토큰 512개이고 리랭커도 같은 XLM-R 토크나이저를 쓴다. 질의까지
# 붙으므로 1024 면 잘리지 않는다. 길게 잡을수록 느려지기만 한다.
MAX_LENGTH = 1024

# 후보가 20개 남짓이라 사실상 한 배치로 끝난다.
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def load_reranker(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    크로스인코더를 올린다. 프로세스당 한 번만.

    dtype 은 GPU 면 float16, CPU 면 float32. CPU 에서 half 는 느리거나 아예
    지원되지 않는 연산이 있다.
    """
    from sentence_transformers import CrossEncoder
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CrossEncoder(
        name,
        max_length=MAX_LENGTH,
        device=device,
        model_kwargs={"dtype": "float16" if device.startswith("cuda") else "float32"},
    )
    return model, device


def cross_scores(question: str, texts: list[str],
                 model_name: str = DEFAULT_MODEL,
                 device: str | None = None,
                 batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    (질의, 청크) 쌍마다 raw logit 을 돌려준다. (N,) float32

    activation_fn 에 Identity 를 넘겨 활성함수를 끈다. bge 계열은 num_labels=1
    이라 CrossEncoder 가 기본으로 시그모이드를 씌우는데, 그러면 0~1 로 눌려서
    상위권끼리의 차이가 안 보인다. 표시할 확률은 아래에서 직접 계산한다.
    """
    import torch

    model, _ = load_reranker(model_name, device)
    pairs = [(question, " ".join(t.split())) for t in texts]

    raw = model.predict(
        pairs,
        batch_size=batch_size,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def rerank_cross(question: str, result: SearchResult, top_n: int = FINAL_TOP_N,
                 model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = BATCH_SIZE) -> RerankResult:
    """
    검색 결과를 크로스인코더로 다시 줄 세우고 상위 top_n 을 고른다.

      RankedHit.score      raw logit. 정렬은 이 값으로 한다
      RankedHit.prob       위를 시그모이드에 넣은 0~1 확률 (표시용)
      RankedHit.reason     "관련성 0.985 (logit +4.21)"
      RerankResult.method  "cross", 실패하면 "dense"
    """
    started = time.time()

    candidates = result.hits
    if not candidates:
        return RerankResult(question=question, method="cross",
                            elapsed=time.time() - started)

    ranked = [RankedHit(hit=hit, rank_before=i)
              for i, hit in enumerate(candidates, 1)]

    error = None
    used = "cross"
    model_used = model_name

    try:
        logits = cross_scores(question, [h.hit.text for h in ranked],
                              model_name, device, batch_size)
        # 정렬은 raw logit 으로 한다. 확률은 화면 표시용으로만 쓴다.
        #
        # 예전에는 확률 x10 을 반올림한 0~10 정수로 줄을 세웠는데, 질문과 청크의
        # 언어가 다르면 이 모델의 확률이 0.001 언저리로 깔려서 후보가 전부 0 으로
        # 동점이 되고 결국 dense 순서가 그대로 복원됐다. 리랭킹이 무효가 된다.
        # logit 은 반올림하지 않으므로 그 구간에서도 순서가 살아 있다.
        #
        # 시그모이드는 float64 로 계산한다. float32 는 logit 이 -88 아래로
        # 내려가면 exp 가 넘쳐 inf 가 되고 경고가 뜬다.
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        for item, logit, prob in zip(ranked, logits, probs):
            item.score = float(logit)
            item.prob = float(prob)
            item.reason = f"관련성 {prob:.3f} (logit {logit:+.2f})"
    except Exception as exc:  # noqa: BLE001 - 모델 없음/VRAM 부족 등 모두 여기로
        error = f"{type(exc).__name__}: {exc}"
        used = "dense"
        model_used = ""
        for item in ranked:
            item.score = None
            item.prob = None
            item.reason = ""

    # 정렬 기준: score -> dense 점수. 규칙 자체는 rerank.sort_ranked 에 있다.
    sort_ranked(ranked)

    return RerankResult(
        question=question,
        method=used,
        ranked=ranked,
        selected=[item.hit for item in ranked[:top_n]],
        model=model_used,
        elapsed=time.time() - started,
        error=error,
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
        description="검색 후보를 크로스인코더로 리랭킹한다 (GPU 권장).")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--doc", default=None,
                        help="검색할 문서 (기본: 전체 문서)")
    parser.add_argument("--top-n", type=int, default=FINAL_TOP_N,
                        help=f"최종 선정 청크 수 (기본: {FINAL_TOP_N})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"리랭커 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 판별)")
    args = parser.parse_args()

    question = " ".join(args.question) or "대마재배자는 누구에게 허가를 받나요?"

    sr = search(question, doc=args.doc)
    print(f"\n후보 {len(sr.hits)}개  ({sr.doc_label} · 청크 {sr.n_indexed}개 중 상위)")

    print("\n리랭커 로드 중... (최초 1회만 느립니다)")
    rr = rerank_cross(question, sr, top_n=args.top_n,
                      model_name=args.model, device=args.device)
    _, device = load_reranker(args.model, args.device)
    print(f"장치 {device} · {args.model} · {rr.elapsed:.1f}초")
    if rr.error:
        print(f"[!] 크로스인코더 실패, 검색 순서로 대체: {rr.error}")

    print(f"\n{'후':>3} {'전':>3} {'이동':>4} {'관련성':>9} {'dense':>6}  청크")
    for item in rr.ranked:
        mark = "  <= 선정" if item.rank_after <= args.top_n else ""
        moved = f"{item.moved:+d}" if item.moved else "-"
        print(f"{item.rank_after:>3} {item.rank_before:>3} {moved:>4} "
              f"{item.percent:>9} {item.hit.score:.4f}  {item.hit.key}{mark}")
        if item.reason:
            print(f"                             {item.reason}")


if __name__ == "__main__":
    main()
