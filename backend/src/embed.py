#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0. 색인 만들기 — data/txt 를 청크로 잘라 bge-m3 로 임베딩한다.

    data/txt/{키}.txt  ->  data/emb/{키}_embeddings.npz

앱이 돌 때는 쓰이지 않는다. data/emb 가 이미 있으면 다시 돌릴 필요가 없고,
원문을 갈아 끼웠을 때만 돌린다. 색인이 어떤 규칙으로 만들어졌는지는 검색
품질을 좌우하므로 이 파일이 그 규칙의 유일한 기록이다.

청킹 (512 / 128):
  bge-m3 토크나이저로 512토큰씩 자르고 128토큰을 겹친다. 법령·보고서는 한
  조문이 문단 몇 개에 걸쳐 있어서 자른 자리에 답이 걸리기 쉬운데, 겹치는
  128토큰이 그 경계를 덮는다. bge-m3 자체는 8192토큰까지 받지만, 청크를
  키우면 한 청크 안에 여러 주제가 섞여 벡터가 뭉개진다.

원문 형식:
  .txt 가 그냥 평문일 수도 있고, PDF 를 뽑을 때 쓴 JSON 껍데기
  ({"source": ..., "text": "..."}) 일 수도 있다. JSON 이면 text 필드만 꺼낸다.

단독 실행:
    python src/embed.py                 # 색인이 없는 문서만
    python src/embed.py --force         # 전부 다시
    python src/embed.py --doc ko        # 하나만
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import EMB_ROOT, EMB_SUFFIX, TXT_ROOT  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 512
OVERLAP = 128

# 한 번에 인코딩할 청크 수. 512토큰 x 8 이면 CPU 에서도 견딘다.
BATCH_SIZE = 8


def read_source(path: Path) -> tuple[str, str]:
    """
    원문을 읽는다. (본문, 어디서 꺼냈는지)

    PDF 추출기가 남긴 JSON 껍데기면 text 필드를 꺼내고, 아니면 파일 전체가
    본문이다. 껍데기째 임베딩하면 메타데이터가 청크 0을 오염시킨다.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw, "(평문)"
        if isinstance(data, dict):
            for field in ("text", "content", "markdown"):
                if isinstance(data.get(field), str) and data[field].strip():
                    return data[field], field
    return raw, "(평문)"


def split_tokens(tokenizer, text: str,
                 chunk_size: int = CHUNK_SIZE,
                 overlap: int = OVERLAP) -> list[tuple[str, int, int, int]]:
    """
    본문을 (청크 원문, 시작 토큰, 끝 토큰, 토큰 수) 목록으로 자른다.

    특수 토큰을 빼고 인코딩한 뒤 창을 밀어 가며 자르고 다시 디코딩한다. 글자
    단위로 자르면 언어마다 토큰당 글자 수가 달라(중국어 1자 ≈ 1토큰, 한국어
    1자 ≈ 0.7토큰) 청크의 정보량이 언어별로 들쭉날쭉해진다.

    문서 전체를 한 번에 인코딩하므로 "sequence length is longer than 8192"
    경고가 뜬다. 모델에 넣는 것이 아니라 자르려고 세는 것이라 무시해도 된다.
    """
    ids = tokenizer.encode(" ".join(text.split()), add_special_tokens=False)
    step = max(1, chunk_size - overlap)

    chunks: list[tuple[str, int, int, int]] = []
    for start in range(0, max(1, len(ids)), step):
        window = ids[start:start + chunk_size]
        if not window:
            break
        body = tokenizer.decode(window, skip_special_tokens=True).strip()
        if body:
            chunks.append((body, start, start + len(window), len(window)))
        if start + chunk_size >= len(ids):
            break
    return chunks


def embed_file(path: Path, out_dir: Path, model, tokenizer,
               model_name: str = DEFAULT_MODEL) -> dict:
    """문서 하나를 청크로 잘라 .npz 하나로 저장한다."""
    started = time.time()

    text, source_field = read_source(path)
    chunks = split_tokens(tokenizer, text)
    if not chunks:
        raise ValueError(f"본문이 비어 있습니다: {path.name}")

    bodies = [c[0] for c in chunks]
    vectors = model.encode(
        bodies,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    # 검색이 내적을 그대로 코사인으로 쓰므로 여기서 한 번 더 확실히 맞춘다.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    device = str(getattr(model, "device", "cpu"))
    info = {
        "source": path.name,
        "source_field": source_field,
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "n_chunks": len(chunks),
        "device": device,
        "elapsed_sec": round(time.time() - started, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}{EMB_SUFFIX}"
    np.savez(
        out_path,
        embeddings=vectors,
        texts=np.array(bodies),
        chunk_index=np.arange(len(chunks), dtype=np.int32),
        token_start=np.array([c[1] for c in chunks], dtype=np.int32),
        token_end=np.array([c[2] for c in chunks], dtype=np.int32),
        token_count=np.array([c[3] for c in chunks], dtype=np.int32),
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
        description="data/txt 를 청크로 잘라 bge-m3 로 임베딩한다.")
    parser.add_argument("--doc", default=None,
                        help="파일 이름 일부. 주면 그 문서만 (기본: 전부)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true",
                        help="이미 색인이 있어도 다시 만든다")
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    args = parser.parse_args()

    paths = sorted(TXT_ROOT.glob("*.txt"))
    if args.doc:
        paths = [p for p in paths if args.doc in p.stem]
    if not paths:
        sys.exit(f"원문이 없습니다: {TXT_ROOT}")

    todo = [p for p in paths
            if args.force or not (EMB_ROOT / f"{p.stem}{EMB_SUFFIX}").exists()]
    if not todo:
        print(f"색인이 이미 다 있습니다 ({len(paths)}개). 다시 만들려면 --force")
        return

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"모델 로드 중... ({args.model}, {device})")
    model = SentenceTransformer(args.model, device=device,
                                model_kwargs={"dtype": "float32"})
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for path in todo:
        print(f"\n{path.name}")
        info = embed_file(path, EMB_ROOT, model, tokenizer, args.model)
        print(f"  청크 {info['n_chunks']}개 · {info['dim']}차원 · "
              f"{info['elapsed_sec']}초 -> {Path(info['out']).name}")


if __name__ == "__main__":
    main()
