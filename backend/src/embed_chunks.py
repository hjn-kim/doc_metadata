#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0. 색인 만들기 — data/chunks/*.jsonl 를 그대로 임베딩한다.

    data/chunks/{키}.jsonl  ->  data/emb/{모델}/{키}_embeddings.npz

embed.py 와의 차이가 이 파일의 존재 이유다. embed.py 는 평문 .txt 를 512토큰
창으로 밀어 자른다 — 문서에는 맞지만 대화 로그에는 맞지 않는다. 대화는 이미
chunking.py / chunking_ko.py 가 발화 경계를 지켜 잘라 놨고, 그 경계를 다시
토큰 창으로 덮으면 한 통화가 중간에서 끊긴다. 그래서 여기서는 자르지 않는다.
청크 한 줄 = 벡터 한 개다.

npz 에 담는 것은 search.py 의 _load_one 이 읽는 것과 정확히 같다.

    embeddings   (N, dim) float32, L2 정규화 (내적을 코사인으로 그대로 쓴다)
    texts        (N,)  청크 본문
    chunk_index  (N,)  jsonl 의 chunk_index 를 그대로 (인용 표기 "ko#12" 의 뒤)
    token_start / token_end / token_count
                 n_tokens 를 누적한 값. 문서 안에서의 대략적인 위치 표시다.
                 창으로 자른 색인이 아니라 청크 경계가 곧 토큰 경계다.

모델은 corpus.EMB_MODELS 의 판본 이름으로 고른다. 판본마다 디렉터리가 따로라
같은 청크를 여러 모델로 임베딩해 두고 RAG_EMB_MODEL 로 갈아 끼울 수 있다.

단독 실행:
    python src/embed_chunks.py --doc ko_voice
    python src/embed_chunks.py --doc ko_voice --emb-model kurev1
    python src/embed_chunks.py --force            # 청크가 있는 코퍼스 전부
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import (CHUNKS_ROOT, DEFAULT_EMB_MODEL, EMB_MODELS,  # noqa: E402
                    EMB_ROOT, EMB_SUFFIX)

# 청크가 아닌 것들. 본문이 없거나(메타데이터 전용) 파생물이라 임베딩 대상이 아니다.
SKIP_KEYS = {"sessions", "chunk_meta"}

# 한 번에 인코딩할 청크 수. 1024토큰 x 8 이면 CPU 에서도 견딘다.
BATCH_SIZE = 8


def read_chunks(path: Path) -> list[dict]:
    """jsonl 을 읽는다. text 가 빈 줄은 벡터를 만들어도 쓸모가 없어 버린다."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not str(row.get("text", "")).strip():
                print(f"  [!] {path.name} {lineno}행: 본문이 비어 건너뜁니다",
                      file=sys.stderr)
                continue
            rows.append(row)
    return rows


def token_spans(rows: list[dict], tokenizer=None) -> np.ndarray:
    """청크별 토큰 수. jsonl 에 n_tokens 가 없으면 그때만 직접 센다."""
    if all(isinstance(r.get("n_tokens"), int) for r in rows):
        return np.array([r["n_tokens"] for r in rows], dtype=np.int32)
    if tokenizer is None:
        raise ValueError("n_tokens 가 없는 청크가 있습니다. 토크나이저가 필요합니다.")
    ids = tokenizer([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
    return np.array([len(x) for x in ids], dtype=np.int32)


def embed_chunks(path: Path, out_dir: Path, model, model_name: str,
                 batch_size: int = BATCH_SIZE, tokenizer=None,
                 max_tokens: int = 8192) -> dict:
    """코퍼스 하나(.jsonl)를 .npz 하나로 만든다."""
    started = time.time()

    rows = read_chunks(path)
    if not rows:
        raise ValueError(f"청크가 없습니다: {path.name}")

    texts = [str(r["text"]) for r in rows]
    counts = token_spans(rows, tokenizer)

    over = int((counts > max_tokens).sum())
    if over:
        print(f"  [!] {max_tokens}토큰을 넘는 청크 {over}개 — 모델이 뒤를 잘라 "
              f"버립니다. 청킹 단계에서 --max-tokens 를 줄이세요.", file=sys.stderr)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    # 검색이 내적을 그대로 코사인으로 쓰므로 여기서 한 번 더 확실히 맞춘다.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # 청크 경계가 곧 토큰 경계다. 앞에서부터 누적해 위치를 만든다.
    ends = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    starts = (ends - counts).astype(np.int32)

    chunk_index = np.array([int(r.get("chunk_index", i))
                            for i, r in enumerate(rows)], dtype=np.int32)

    info = {
        "source": path.name,
        "source_field": "chunking",
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "chunk_size": int(counts.max()),
        "overlap": None,
        "n_chunks": len(rows),
        "device": str(getattr(model, "device", "cpu")),
        "elapsed_sec": round(time.time() - started, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}{EMB_SUFFIX}"
    np.savez(
        out_path,
        embeddings=vectors,
        texts=np.array(texts),
        chunk_index=chunk_index,
        token_start=starts,
        token_end=ends,
        token_count=counts,
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )
    info["out"] = str(out_path)
    return info


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="data/chunks/*.jsonl 을 청크 그대로 임베딩한다.")
    parser.add_argument("--doc", default=None,
                        help="코퍼스 키 일부. 주면 그것만 (기본: 청크가 있는 전부)")
    parser.add_argument("--emb-model", default=DEFAULT_EMB_MODEL,
                        help=f"임베딩 판본 ({', '.join(EMB_MODELS)})")
    parser.add_argument("--force", action="store_true",
                        help="이미 색인이 있어도 다시 만든다")
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.emb_model not in EMB_MODELS:
        sys.exit(f"모르는 임베딩 판본입니다: {args.emb_model} "
                 f"(쓸 수 있는 것: {', '.join(EMB_MODELS)})")
    model_name = EMB_MODELS[args.emb_model][0]
    out_dir = EMB_ROOT / args.emb_model

    paths = [p for p in sorted(CHUNKS_ROOT.glob("*.jsonl"))
             if p.stem not in SKIP_KEYS]
    if args.doc:
        paths = [p for p in paths if args.doc in p.stem]
    if not paths:
        sys.exit(f"청크가 없습니다: {CHUNKS_ROOT}")

    todo = [p for p in paths
            if args.force or not (out_dir / f"{p.stem}{EMB_SUFFIX}").exists()]
    if not todo:
        print(f"색인이 이미 다 있습니다 ({len(paths)}개). 다시 만들려면 --force")
        return

    from sentence_transformers import SentenceTransformer
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"모델 로드 중... ({model_name}, {device})")
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"dtype": "float32"})

    for path in todo:
        print(f"\n{path.name}")
        info = embed_chunks(path, out_dir, model, model_name, args.batch_size)
        print(f"  청크 {info['n_chunks']}개 · {info['dim']}차원 · "
              f"{info['elapsed_sec']}초 -> {args.emb_model}/{Path(info['out']).name}")


if __name__ == "__main__":
    main()
