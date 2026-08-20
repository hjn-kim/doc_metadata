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

from corpus import (  # noqa: E402
    ALL_DOCS,
    Document,
    documents,
    find,
    script_of,
)

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

    # 청크를 세는 단위. 메타데이터를 함께 쓰는 문서는 같은 값을 갖는다
    # (jabber_ru 와 jabber_en 은 같은 대화의 두 판본이므로 한 묶음).
    group_names: tuple[str, ...] = ()       # 묶음 이름
    group_codes: np.ndarray = field(        # (N,) group_names 의 첨자
        default_factory=lambda: np.empty(0, np.int32), repr=False)

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
    n_candidates: int = 0                           # 필터를 통과한 색인 행 수
    n_candidate_chunks: int = 0                     # 같은 것을 청크로 센 수
    n_docs: int = 1                                 # 뒤진 문서 수
    keywords: list[str] = field(default_factory=list)   # 실제로 건 키워드
    narrowed: str = ""                              # 어떤 조건으로 좁혔는지
    elapsed: float = 0.0

    @property
    def filtered(self) -> bool:
        """필터가 실제로 범위를 좁혔는가."""
        return 0 < self.n_candidates < self.n_indexed

    @property
    def pool(self) -> str:
        """
        상위 몇 개를 '무엇 중에서' 골랐는지. 화면과 CLI 가 같은 문구를 쓴다.

        색인은 같은 청크의 러시아어·영어 두 판본을 각각 행으로 갖고 있어서 행
        수로 적으면 두 배로 보인다. 0단계 카드가 청크로 세므로 여기도 청크로
        맞춘다 (두 카드의 숫자가 다르면 화면에서 이어지지 않는다).
        """
        if self.filtered and self.n_candidate_chunks:
            return f"후보 {self.n_candidate_chunks:,}청크"
        return f"청크 {self.n_indexed:,}개"


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
    group_names: list[str] = []
    group_of: list[int] = []
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
        if item.group not in group_names:
            group_names.append(item.group)
        group_of.extend([group_names.index(item.group)] * loaded["n"])
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
        group_names=tuple(group_names),
        group_codes=np.asarray(group_of, dtype=np.int32),
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

# 리터럴로 맞춰 볼 만한 키워드인가. 최소 길이는 문자 종류마다 다르다.
#
#     ascii    3글자. 'id' 'ip' 같은 두 글자는 아무 데나 들어 있다
#     hangul   2글자. 한글은 글자당 정보가 많아 '통장' '검찰' 이면 충분하고,
#              3글자를 요구하면 쓸 만한 낱말이 거의 다 걸러진다
KEYWORD_MIN_LEN = {"ascii": 3, "hangul": 2}

def index_scripts(index: Index) -> tuple[str, ...]:
    """
    이 색인에서 리터럴로 맞춰 볼 만한 문자 종류. 문서들의 합집합이다.

    문서마다 다른 이유가 여기 있다. jabber 는 원문이 러시아어와 그 영어
    번역이라 '대포통장' 같은 한글이 본문에 나올 수가 없다 — 걸어 봐야 0건이고,
    그 뜻은 이미 질문 임베딩이 처리하고 있다. 반대로 ko_voice 는 원문이
    한국어라 한글이야말로 제일 정확히 맞는 키워드다. 색인을 안 보고 하나로
    정하면 어느 한쪽이 반드시 손해를 본다 (corpus.py DOC_SPECS 참고).
    """
    by_key = _doc_by_key()
    out: list[str] = []
    for key in dict.fromkeys(index.doc_keys):
        doc = by_key.get(key)
        for name in (doc.kw_scripts if doc else ("ascii",)):
            if name not in out:
                out.append(name)
    return tuple(out)


def _lower_texts(index: Index) -> list[str]:
    got = _LOWER_TEXTS.get(index.name)
    if got is None or len(got) != index.size:
        got = [str(t).lower() for t in index.texts]
        _LOWER_TEXTS[index.name] = got
    return got


def usable_keywords(keywords, scripts: tuple[str, ...] = ("ascii",)
                    ) -> list[str]:
    """
    리터럴 대조에 쓸 수 있는 것만 남긴다.

    scripts 는 색인에 든 문서들이 받는 문자 종류다 (index_scripts 참고).
    """
    out: list[str] = []
    for kw in keywords or []:
        text = str(kw).strip()
        kind = script_of(text)
        if kind not in scripts:
            continue
        if len(text) >= KEYWORD_MIN_LEN.get(kind, 3) and text not in out:
            out.append(text)
    return out


# 키워드를 다 맞춘 청크에 얹어 주는 점수. 코사인 점수 차이가 상위권에서
# 0.01~0.05 라 이 정도면 순위를 바꾸되 뒤집지는 않는다.
KEYWORD_BONUS = 0.05

# 너무 흔한 낱말은 키워드에서 버린다.
#
# 후보를 '하나라도 든 청크' 로 잡기 때문에(OR), 전 청크에 걸리는 낱말이 하나
# 섞이면 나머지 키워드의 축소 효과가 통째로 사라진다. ko_voice 에서 '통화' 는
# 70/70, '본인' 은 45/70 이라 이런 말이 하나만 들어와도 채널이 무력해진다.
#
# 한국어 질문에서 특히 문제가 된다. 원문이 러시아어·영어일 때는 키워드가
# ASCII 식별자뿐이라 이런 일이 없었지만, 한국어 원문에서는 질문에 쓰인 흔한
# 명사가 그대로 키워드가 된다. 뽑는 자리(extract_metadata.KO_STOP)에도 그물이
# 있지만, 목록으로 다 적을 수 없는 종류라 세어 보고 버리는 쪽이 확실하다.
COMMON_RATIO = 0.5          # 이 비율을 넘게 걸리면 버린다
COMMON_MIN_ROWS = 20        # 색인이 이보다 작으면 비율을 못 믿는다


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
    words = usable_keywords(keywords, index_scripts(index))
    if not words:
        return None, None, "", []

    texts = _lower_texts(index)
    hits = np.zeros(index.size, dtype=np.float32)
    matched: list[str] = []
    common: list[str] = []
    for word in words:
        needle = word.lower()
        found = np.fromiter((needle in t for t in texts), dtype=bool,
                            count=index.size)
        n_found = int(found.sum())
        if not n_found:
            continue
        if (index.size >= COMMON_MIN_ROWS
                and n_found > index.size * COMMON_RATIO):
            common.append(word)          # 너무 흔하다. 세지도 않는다
            continue
        matched.append(word)
        hits += found

    if not matched:
        why = "너무 흔함" if common else "본문에 없음"
        return None, None, f"키워드 무시({why})", []

    keep = hits >= 1
    bonus = KEYWORD_BONUS * (hits / len(matched))
    note = (f"키워드 {len(matched)}개" if len(matched) == len(words)
            else f"키워드 {len(matched)}/{len(words)}개")
    if common:
        note += f" (흔해서 뺌: {', '.join(common[:3])})"
    return keep, bonus, note, matched


def expand_mask(index: Index, mask) -> np.ndarray | None:
    """
    청크 마스크를 색인 행 마스크로 편다.

    받는 것이 두 가지다.

        사전 {문서키: 마스크}   filter_metadata.build_doc_masks 가 주는 것
                                값이 None 이면 그 문서는 조건 없음(전부 통과),
                                키가 아예 없으면 그 문서는 후보에서 제외
        배열 하나              색인이 문서 하나일 때만 쓰는 옛 경로

    **사전을 받는 것이 이 함수의 핵심이다.** 청크 번호는 문서 안에서만 유일해서
    ko_voice 에도 5번 청크가 있고 jabber 에도 5번 청크가 있다. 마스크 하나를
    번호로 갖다 붙이면 (mask[chunk_indices]) ko_voice 5번이 jabber 5번의 조건을
    그대로 물려받는다 — "stern 과 poll 의 대화" 를 물었는데 보이스피싱 통화가
    딸려 나오는 식이다. 문서별로 나눠 채워야 그 일이 없다.

    같은 문서의 여러 언어 판본은 함께 살아난다. jabber_ru 와 jabber_en 은 각각
    자기 키로 같은 마스크를 받으므로 청크 하나가 두 행 다 켜진다.
    """
    if mask is None:
        return None

    if isinstance(mask, dict):
        keys = np.asarray(index.doc_keys)
        out = np.zeros(index.size, dtype=bool)
        for key in dict.fromkeys(index.doc_keys):
            if key not in mask:                  # 후보에서 뺀 문서
                continue
            rows = keys == key
            got = mask[key]
            if got is None:                      # 조건 없음 -> 전부 통과
                out[rows] = True
                continue
            got = np.asarray(got, dtype=bool)
            here = index.chunk_indices[rows]
            top = int(here.max()) + 1 if here.shape[0] else 0
            if got.shape[0] < top:
                raise ValueError(f"{key}: 마스크 길이가 맞지 않습니다 "
                                 f"({got.shape[0]}, 청크 번호는 {top}까지)")
            out[rows] = got[here]
        return out

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
    """
    행 마스크가 가리키는 서로 다른 청크 수.

    청크 번호만으로 세면 안 된다. 번호가 문서마다 0부터 다시 시작해서 ko_voice
    70개가 jabber 0~69번에 통째로 흡수된다 (합쳐도 15,522개로 보인다).
    (묶음, 번호) 쌍으로 센다.

    묶음으로 세는 것이지 문서로 세는 것이 아니다. jabber_ru 와 jabber_en 은
    같은 대화의 두 판본이라 행이 두 벌 있을 뿐 청크는 한 벌이고, 문서로 세면
    화면 숫자가 두 배로 보인다.
    """
    codes, idx = index.group_codes, index.chunk_indices
    if mask is not None:
        codes, idx = codes[mask], idx[mask]
    if idx.shape[0] == 0:
        return 0
    # (묶음, 번호) 를 정수 하나로 접는다. 번호 최대값+1 을 자릿수로 쓴다.
    span = int(idx.max()) + 1
    return int(np.unique(codes.astype(np.int64) * span + idx).shape[0])


def narrow(index: Index, mask=None, keywords=None) -> Narrowing:
    """
    메타 마스크와 키워드 채널을 합친다.

    mask 는 배열 하나가 아니라 {문서키: 마스크} 사전일 수 있다
    (filter_metadata.build_doc_masks). expand_mask 가 알아서 편다.

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
           mask=None,
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
        n_candidate_chunks=narrowing.n_used,
        n_docs=index.n_docs,
        keywords=list(narrowing.keywords),
        narrowed=narrowing.step,
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
        from filter_metadata import build_doc_masks
        picked = build_doc_masks(raw, args.doc)
        mask = picked.masks
        print(f"메타 : {picked.summary()}")
        for note in picked.notes:
            print(f"       {note}")

    nr = narrow(load_index(args.doc), mask, keywords)
    print(f"좁힘 : 메타 {nr.n_meta if nr.n_meta is not None else '-'} · "
          f"키워드 {nr.n_keyword if nr.n_keyword is not None else '-'} · "
          f"교집합 {nr.n_both if nr.n_both is not None else '-'} "
          f"-> {nr.n_used:,}개 ({nr.step})")

    sr = search(question, doc=args.doc, top_k=args.top_k, narrowing=nr)

    print(f"\n질문 : {sr.query}")
    if sr.keywords:
        print(f"키워드: {', '.join(sr.keywords)}")
    print(f"{sr.doc_label} · {sr.pool} 중 상위 {sr.top_k}개 "
          f"({sr.elapsed:.1f}초)\n")
    for rank, hit in enumerate(sr.hits, 1):
        print(f"  {rank:2d} {hit.score:.4f} {hit.key:<8s} {hit.preview(70)}")


if __name__ == "__main__":
    main()
