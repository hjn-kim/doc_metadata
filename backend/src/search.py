#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1. 검색 랭킹 (질의 임베딩 + 색인 로드 + dense 검색)

data/emb/*.npz 를 읽어 질문을 같은 모델로 인코딩하고 코사인 유사도 상위 top_k 를
뽑는다. 이 목록이 2단계 리랭킹의 후보가 된다.

왜 BAAI/bge-m3 인가:
  원문이 한국어·영어·중국어·베트남어·필리핀어·러시아어·우즈베크어로 흩어져 있는데
  질문은 한국어로 들어온다. bge-m3 는 100개 넘는 언어를 한 벡터 공간에 넣도록
  학습돼 있어서 한국어 질문으로 러시아어 원문을 바로 찾는다. 번역을 한 겹 끼우지
  않아도 되고, 뒤에 붙는 리랭커(bge-reranker-v2-m3)와 같은 계열이라 두 단계가
  같은 언어 감각으로 움직인다. 8192토큰까지 받으므로 512토큰 청크는 잘리지 않는다.

문서 임베딩과 반드시 맞춰야 하는 것 (틀리면 조용히 정확도만 떨어진다):
  1. 같은 모델    BAAI/bge-m3, 1024차원
  2. 프리픽스 없음   bge-m3 는 질의에 지시문을 붙이지 않는 대칭 모델이다
  3. float32 재정규화   내적을 그대로 코사인으로 쓰기 위해

문서를 좁히지 않으면(doc=None) 7개 문서를 한 색인으로 합쳐 뒤진다. 전체 청크가
1,100개 남짓이라 전수 비교로 충분하다 (ANN 색인이 필요 없다).

단독 실행:
    python src/search.py "대마재배자는 누구에게 허가를 받나요?"
    python src/search.py --doc ko "마약 밀매의 형량은?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ALL_DOCS, Document, documents, find  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"

# 문서 하나가 82~243청크다. 후보를 15개 뽑아 리랭커에 넘긴다.
#
# 왜 15개인가: 크로스인코더가 다시 줄 세울 후보 풀이다. 여기서 빠진 청크는
# 리랭킹이 아무리 잘 돼도 되살아나지 못한다. 전체 검색(7개 문서 1,136청크)에서
# 답이 든 청크가 dense 점수 10위 밖으로 밀리는 일이 있어 여유를 뒀다. 리랭커
# 배치가 16이라 15개까지는 한 번에 채점돼 시간도 거의 그대로다.
DEFAULT_TOP_K = 15


# --------------------------------------------------------------------------
# 자료구조
# --------------------------------------------------------------------------

@dataclass
class Index:
    """검색 대상 청크 전체. 문서 하나일 수도 있고 전부일 수도 있다."""

    name: str                    # 'ko' 또는 'all'
    vectors: np.ndarray          # (N, dim) float32, L2 정규화됨
    texts: np.ndarray            # (N,) 청크 원문
    doc_keys: list[str]          # (N,) 청크가 나온 문서 키
    doc_codes: list[str]         # (N,) 짧은 코드 (인용 표기에 쓴다)
    chunk_indices: np.ndarray    # (N,) int32  문서 안에서 몇 번째 청크인지
    token_starts: np.ndarray     # (N,) int32
    token_ends: np.ndarray       # (N,) int32
    model: str = ""
    n_docs: int = 0

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass
class Hit:
    """검색 결과 한 건."""

    score: float
    text: str
    doc_key: str
    doc_code: str
    doc_title: str
    doc_lang: str                # 원문 언어 이름 ('러시아어'). 답변 프롬프트가 쓴다
    chunk_index: int
    token_start: int
    token_end: int

    @property
    def key(self) -> str:
        """청크를 가리키는 고유 이름. 인용 표기이자 중복 제거의 기준."""
        return f"{self.doc_code}#{self.chunk_index}"

    def preview(self, n: int = 200) -> str:
        one_line = " ".join(self.text.split())
        return one_line[:n] + ("..." if len(one_line) > n else "")


@dataclass
class SearchResult:
    """1단계 결과. 질문 하나로 뽑은 상위 top_k."""

    query: str
    doc: str                                        # 'ko' 또는 'all'
    doc_label: str                                  # 화면에 쓸 이름
    top_k: int
    hits: list[Hit] = field(default_factory=list)   # 점수 내림차순
    n_indexed: int = 0                              # 색인 전체 청크 수
    n_candidates: int = 0                           # 필터를 통과한 청크 수
    n_docs: int = 1                                 # 뒤진 문서 수
    keywords: list[str] = field(default_factory=list)   # 실제로 건 키워드
    narrowed: str = ""                              # 어떤 조건으로 좁혔는지
    elapsed: float = 0.0

    @property
    def filtered(self) -> bool:
        """필터가 실제로 범위를 좁혔는가."""
        return 0 < self.n_candidates < self.n_indexed


# --------------------------------------------------------------------------
# 색인 로드
# --------------------------------------------------------------------------

def _load_one(doc: Document) -> dict:
    """.npz 한 개를 dict 로 읽는다."""
    data = np.load(doc.emb_path)
    vectors = data["embeddings"]
    model = ""
    try:
        model = json.loads(str(data["info"])).get("model", "")
    except (ValueError, KeyError):
        pass
    return {
        "vectors": vectors,
        "texts": data["texts"],
        "chunk_index": data["chunk_index"],
        "token_start": data["token_start"],
        "token_end": data["token_end"],
        "n": int(vectors.shape[0]),
        "model": model,
    }


@lru_cache(maxsize=16)
def load_index(doc: str | None = None) -> Index:
    """
    색인을 만든다. doc 이 None 이거나 'all' 이면 문서 전체를 하나로 합친다.

    Streamlit 이 클릭마다 스크립트를 다시 돌리므로 반드시 캐시한다. 전부 합쳐도
    1,136 x 1024 float32 = 약 4.6MB 라 메모리는 문제가 되지 않는다.
    """
    target = find(doc)
    docs = [target] if target else documents()
    if not docs:
        raise FileNotFoundError("색인이 없습니다. python src/embed.py 를 먼저 돌리세요.")

    vec_parts, text_parts = [], []
    idx_parts, start_parts, end_parts = [], [], []
    doc_keys: list[str] = []
    doc_codes: list[str] = []
    model_name = ""

    for item in docs:
        loaded = _load_one(item)
        vec_parts.append(loaded["vectors"])
        text_parts.append(loaded["texts"])
        idx_parts.append(loaded["chunk_index"])
        start_parts.append(loaded["token_start"])
        end_parts.append(loaded["token_end"])
        doc_keys.extend([item.key] * loaded["n"])
        doc_codes.extend([item.code] * loaded["n"])
        model_name = model_name or loaded["model"]

    return Index(
        name=target.code if target else ALL_DOCS,
        vectors=np.vstack(vec_parts).astype(np.float32),
        texts=np.concatenate(text_parts),
        doc_keys=doc_keys,
        doc_codes=doc_codes,
        chunk_indices=np.concatenate(idx_parts),
        token_starts=np.concatenate(start_parts),
        token_ends=np.concatenate(end_parts),
        model=model_name,
        n_docs=len(docs),
    )


# --------------------------------------------------------------------------
# 질의 임베딩
# --------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    임베딩 모델을 올린다. 프로세스당 한 번만 (Streamlit 이 매 클릭마다 스크립트를
    다시 돌기 때문에 캐시가 없으면 클릭마다 모델을 새로 올린다).

    dtype 은 float32 로 고정한다. CPU 에서 bf16 은 AVX512-BF16 이 없으면
    에뮬레이션으로 떨어져 오히려 느리다.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(name, device=device,
                               model_kwargs={"dtype": "float32"})


def encode_queries(queries: list[str], model_name: str = DEFAULT_MODEL,
                   device: str | None = None) -> np.ndarray:
    """
    질의를 (Q, dim) float32 정규화 벡터로 만든다.

    bge-m3 는 질의와 문서를 같은 방식으로 인코딩한다(대칭). 프리픽스를 붙이면
    오히려 문서 쪽 분포와 어긋나므로 아무것도 붙이지 않는다.
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("질의가 비어 있습니다.")

    model = load_model(model_name, device)
    vectors = model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


# --------------------------------------------------------------------------
# 1 단계 : 검색 랭킹
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _doc_by_key() -> dict[str, Document]:
    return {d.key: d for d in documents()}


def make_hit(index: Index, row: int, score: float) -> Hit:
    """행 번호 하나를 Hit 으로 만든다."""
    doc = _doc_by_key()[index.doc_keys[row]]
    return Hit(
        score=score,
        text=str(index.texts[row]),
        doc_key=doc.key,
        doc_code=doc.code,
        doc_title=doc.title,
        doc_lang=doc.lang_name,
        chunk_index=int(index.chunk_indices[row]),
        token_start=int(index.token_starts[row]),
        token_end=int(index.token_ends[row]),
    )


# --------------------------------------------------------------------------
# 키워드 채널 — 본문에 글자 그대로 있는가
# --------------------------------------------------------------------------

# 소문자로 미리 접어 둔 본문. 질의마다 31,044개를 다시 접으면 아깝다.
_LOWER_TEXTS: dict[str, list[str]] = {}

# 리터럴로 맞춰 볼 만한 키워드인가. 한글이 섞이면 버린다 — 원문이 러시아어와
# 영어라 '서버' 같은 낱말은 본문에 그대로 나올 수 없고, 그런 뜻은 이미 질문
# 임베딩이 처리하고 있다.
KEYWORD_MIN_LEN = 3


def _lower_texts(index: Index) -> list[str]:
    got = _LOWER_TEXTS.get(index.name)
    if got is None or len(got) != index.size:
        got = [str(t).lower() for t in index.texts]
        _LOWER_TEXTS[index.name] = got
    return got


def usable_keywords(keywords) -> list[str]:
    """리터럴 대조에 쓸 수 있는 것만 남긴다 (ASCII, 3글자 이상)."""
    out: list[str] = []
    for kw in keywords or []:
        text = str(kw).strip()
        if len(text) >= KEYWORD_MIN_LEN and text.isascii() and text not in out:
            out.append(text)
    return out


# 키워드를 다 맞춘 청크에 얹어 주는 점수. 코사인 점수 차이가 상위권에서
# 0.01~0.05 라 이 정도면 순위를 바꾸되 뒤집지는 않는다.
KEYWORD_BONUS = 0.05


def keyword_signal(index: Index, keywords
                   ) -> tuple[np.ndarray | None, np.ndarray | None, str, list]:
    """
    키워드 채널. (후보 마스크, 점수 가산, 설명, 쓴 키워드)

    IP·도메인·파일명 같은 식별자는 임베딩이 제일 못하는 것이다. 토크나이저가
    68.224.217.72 를 조각으로 쪼개 버려서 dense 점수로는 비슷한 숫자 나열과
    구분되지 않는다. 반대로 리터럴 대조는 이런 것에 정확하다 — qa.json 의
    식별자 문항에서 31,044행이 2행으로 줄어든다.

    후보는 '하나라도 들어 있는' 행이다. '전부 들어 있는' 행으로 좁히면 안 된다:

        512토큰으로 자른 청크라 근거가 두 청크에 걸쳐 있는 일이 흔하다.
        "maze와 conti가 등장하는 기록에서 ... locker 를 설치하도록" 같은 질문은
        maze 가 든 청크와 locker 가 든 청크가 따로다. 전부 만족을 요구하면
        정답 5개가 통째로 걸러진다(qa.json jabber-044). 하나라도로 넓히면
        후보가 95개에서 485개로 늘 뿐이고, 100문항 전부 정답이 살아남는다.

    대신 많이 맞춘 행에 KEYWORD_BONUS 를 얹어 위로 올린다. 하드 필터의 절벽
    없이 '다 맞춘 것이 먼저' 를 표현하는 자리다.

    하나도 안 걸리면 키워드를 통째로 버린다. 추출기가 원문에 없는 낱말을 하나
    끼워 넣었다고 검색이 빈손이 되면 안 된다.
    """
    words = usable_keywords(keywords)
    if not words:
        return None, None, "", []

    texts = _lower_texts(index)
    hits = np.zeros(index.size, dtype=np.float32)
    matched: list[str] = []
    for word in words:
        needle = word.lower()
        found = np.fromiter((needle in t for t in texts), dtype=bool,
                            count=index.size)
        if found.any():
            matched.append(word)
        hits += found

    if not matched:
        return None, None, "키워드 무시(본문에 없음)", []

    keep = hits >= 1
    bonus = KEYWORD_BONUS * (hits / len(words))
    note = (f"키워드 {len(matched)}개" if len(matched) == len(words)
            else f"키워드 {len(matched)}/{len(words)}개")
    return keep, bonus, note, matched


def expand_mask(index: Index, mask: np.ndarray | None) -> np.ndarray | None:
    """
    청크 마스크를 색인 행 마스크로 편다.

    filter_metadata.py 가 주는 마스크는 청크 메타데이터 행(청크 번호 0..N-1)
    기준이다. 색인은 문서를 여러 개 이어 붙인 것이라 (jabber_ru 15,522행 +
    jabber_en 15,522행) 행 번호가 그대로 맞지 않는다. chunk_indices 로 되짚어
    같은 청크의 모든 언어 판본을 함께 살린다.

    길이가 색인 행 수와 같으면 이미 행 마스크로 보고 그대로 쓴다.
    """
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] == index.size:
        return mask

    top = int(index.chunk_indices.max()) + 1 if index.size else 0
    if mask.shape[0] < top:
        raise ValueError(f"마스크 길이가 맞지 않습니다: {mask.shape[0]} "
                         f"(청크 번호는 {top}까지 나옵니다)")
    return mask[index.chunk_indices]


@dataclass
class Narrowing:
    """
    0-3단계 결과 — 두 채널이 각각 몇 개를 남겼고 교집합이 몇 개인가.

    숫자는 전부 **청크** 단위다. 색인은 같은 청크의 러시아어·영어 두 판본을
    행으로 갖고 있어서 행 수로 세면 두 배로 보인다. 화면에 "청크 2개"라고
    적어야 사람이 아는 수와 맞는다.

    keep/bonus 는 계산에만 쓰는 배열이라 직렬화하지 않는다 (schema.py 참고).
    """

    n_chunks: int = 0                    # 전체 청크 수
    n_meta: int | None = None            # 메타 조건을 통과한 청크 (없으면 None)
    n_keyword: int | None = None         # 키워드가 든 청크
    n_both: int | None = None            # 교집합
    n_used: int = 0                      # 실제로 검색에 쓴 후보
    step: str = "전체"                   # 무엇을 썼는지
    keywords: list[str] = field(default_factory=list)   # 본문에서 찾은 것만

    keep: np.ndarray | None = field(default=None, repr=False)
    bonus: np.ndarray | None = field(default=None, repr=False)

    @property
    def ratio(self) -> float:
        return self.n_used / self.n_chunks if self.n_chunks else 1.0


def chunk_count(index: Index, mask: np.ndarray | None) -> int:
    """행 마스크가 가리키는 서로 다른 청크 수."""
    if mask is None:
        return int(np.unique(index.chunk_indices).shape[0])
    return int(np.unique(index.chunk_indices[mask]).shape[0])


def narrow(index: Index, mask: np.ndarray | None, keywords=None) -> Narrowing:
    """
    메타 마스크와 키워드 채널을 합친다.

    둘 다 통과하는 청크가 없으면 키워드를 먼저 푼다. 사람·기간은 명부와 날짜
    파싱을 통과한 것이라 근거가 단단하지만, 키워드는 추출기가 질문에서 주워
    온 문자열이라 원문 표기와 어긋날 수 있기 때문이다.

    어느 채널이 몇 개를 남겼는지는 버리지 않고 다 담아 돌려준다. 교집합만
    보여 주면 "왜 이만큼 줄었는지" 를 화면에서 되짚을 수 없다.
    """
    meta_keep = expand_mask(index, mask)
    kw_keep, bonus, kw_note, used = keyword_signal(index, keywords)

    if meta_keep is not None and not meta_keep.any():   # 필터가 다 지웠다
        meta_keep = None

    out = Narrowing(
        n_chunks=chunk_count(index, None),
        n_meta=None if meta_keep is None else chunk_count(index, meta_keep),
        n_keyword=None if kw_keep is None else chunk_count(index, kw_keep),
        keywords=used,
    )

    if meta_keep is not None and kw_keep is not None:
        both = meta_keep & kw_keep
        out.n_both = chunk_count(index, both)
        if both.any():
            out.keep, out.bonus, out.step = both, bonus, f"메타 + {kw_note}"
        else:
            # 교집합이 비었다. 키워드를 푼다 (메타 조건이 더 단단하다).
            out.keep, out.step = meta_keep, f"{kw_note} -> 메타만"
            out.keywords = []
    elif kw_keep is not None:
        out.keep, out.bonus, out.step = kw_keep, bonus, kw_note
    elif meta_keep is not None:
        out.keep, out.step = meta_keep, "메타"
    else:
        out.step = kw_note or "전체"

    out.n_used = chunk_count(index, out.keep)
    return out


def search(query: str, doc: str | None = None, top_k: int = DEFAULT_TOP_K,
           model_name: str = DEFAULT_MODEL,
           device: str | None = None,
           mask: np.ndarray | None = None,
           keywords: list[str] | None = None,
           narrowing: "Narrowing | None" = None) -> SearchResult:
    """
    질문 하나를 임베딩해 색인 전체와 견주고 상위 top_k 를 돌려준다.

    doc 에 문서 키나 짧은 코드를 주면 그 문서 안에서만 찾고, 주지 않으면 전체
    문서를 뒤진다. 벡터가 전부 L2 정규화돼 있으므로 내적이 곧 코사인 유사도다.

    mask 는 메타데이터 필터(filter_metadata.py)가 남긴 청크만 True 인 불리언
    배열이고, keywords 는 본문에 글자 그대로 있어야 하는 문자열이다. 걸러진
    행은 점수를 -inf 로 눌러 순위에서 뺀다. 내적은 어차피 전수라 비용이
    그대로고, 후보가 줄어든 만큼 top_k 를 올려도 리랭커만 조금 더 일한다.

    조건이 한 행도 남기지 않으면 그 조건을 버리고 넓힌다. 후보 0개는 답변이
    반드시 틀리는 상태라, 조용히 빈 결과를 내는 것보다 필터를 포기하는 편이
    낫다 (filter_metadata.py 의 사다리와 같은 이유다).
    """
    started = time.time()

    clean = (query or "").strip()
    if not clean:
        raise ValueError("질문이 비어 있습니다.")

    index = load_index(doc)

    scores = encode_queries([clean], model_name, device)[0] @ index.vectors.T

    # 파이프라인은 좁히기를 먼저 끝내고(화면에 숫자를 먼저 띄우려고) 그 결과를
    # 그대로 넘긴다. 단독으로 부르면 여기서 만든다.
    if narrowing is None:
        narrowing = narrow(index, mask, keywords)

    n_candidates = index.size
    if narrowing.bonus is not None:
        # 키워드를 많이 맞춘 청크를 위로. 후보에서 빼지는 않는다.
        scores = scores + narrowing.bonus
    if narrowing.keep is not None:
        n_candidates = int(narrowing.keep.sum())
        scores = np.where(narrowing.keep, scores, -np.inf)

    k = min(top_k, n_candidates)
    rows = [int(r) for r in np.argsort(-scores)[:k]]

    target = find(doc)
    return SearchResult(
        query=clean,
        doc=target.code if target else ALL_DOCS,
        doc_label=target.label if target else f"전체 문서 {index.n_docs}종",
        top_k=k,
        hits=[make_hit(index, r, float(scores[r])) for r in rows],
        n_indexed=index.size,
        n_candidates=n_candidates,
        n_docs=index.n_docs,
        keywords=used,
        narrowed=note,
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

    parser = argparse.ArgumentParser(description="질문 하나로 색인을 검색한다.")
    parser.add_argument("question", nargs="*", help="검색할 질문")
    parser.add_argument("--doc", default=None,
                        help="문서 키나 짧은 코드 (기본: 전체 문서)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"뽑을 청크 수 (기본: {DEFAULT_TOP_K})")
    parser.add_argument("--people", help="메타 필터: 사람. 'poll,stern'")
    parser.add_argument("--since", help="메타 필터: 시작 (2020-09-29)")
    parser.add_argument("--until", help="메타 필터: 끝 (2020-09-29)")
    parser.add_argument("--keywords", help="본문에 그대로 있어야 할 문자열")
    parser.add_argument("--extract", choices=("llm", "rule", "off"),
                        help="질문에서 조건을 뽑아 자동으로 건다")
    args = parser.parse_args()

    question = " ".join(args.question) or "대마재배자는 누구에게 허가를 받나요?"

    mask, keywords = None, None
    if args.extract:
        from extract_metadata import extract
        got = extract(question, mode=args.extract)
        raw, keywords = got.query, got.query.keywords
        print(f"\n추출 : {got.label()}  ({got.source}, {got.elapsed:.1f}초)")
    else:
        raw = {"people": args.people, "since": args.since, "until": args.until}
        keywords = [k.strip() for k in (args.keywords or "").split(",")
                    if k.strip()]

    if args.extract or args.people or args.since or args.until:
        from filter_metadata import build_mask
        picked = build_mask(raw)
        mask = picked.mask
        print(f"메타 : {picked.summary()}")

    sr = search(question, doc=args.doc, top_k=args.top_k, mask=mask,
                keywords=keywords)

    print(f"\n질문 : {sr.query}")
    if sr.narrowed:
        print(f"좁힘 : {sr.narrowed}"
              f"{'  [' + ', '.join(sr.keywords) + ']' if sr.keywords else ''}")
    print(f"{sr.doc_label} · 청크 {sr.n_indexed}개"
          f"{f' 중 후보 {sr.n_candidates:,}개' if 