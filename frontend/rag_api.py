#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 서버와 이야기하는 부분 — CPU 서버는 이 파일과 app.py 두 개로 끝난다.

문서도 색인도 모델도 여기에는 없다. 문서 목록·질문 세트·모델 이름은 /rag/meta
에서 받아 오고, 검색은 /rag/search 로 넘긴다. 그래서 CPU 쪽에 필요한 것은
streamlit 과 requests 뿐이다 (numpy·torch 없음).

    from rag_api import run_pipeline, documents, load_questions

자료구조를 여기에 다시 적어 둔 이유:
    backend/src 의 dataclass 와 같은 모양을 쓴다. app.py 의 렌더 함수가
    hit.preview(220), item.percent, gr.gold_display 같은 property 를 그대로
    부르기 때문이다. 필드 이름이 어긋나면 SCHEMA_VERSION 이 달라지므로
    화면 위쪽에 경고가 뜬다.

    backend/src/{search,rerank,answer,grade,main}.py 의 dataclass 를 고치면
    이 파일도 같이 고쳐야 한다. 두 서버가 지키기로 한 계약이다.

    python frontend/rag_api.py            # GPU 서버와 통신 점검
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

# 계약 버전. backend/src/schema.py 와 같아야 한다.
# 2 = 0단계(조건 추출 · 후보 좁히기)가 들어간 판. 서버가 1을 돌려주면 GPU 쪽이
# 옛 배포라는 뜻이고, 그때는 문서 목록도 질문 세트도 옛것이 온다.
SCHEMA_VERSION = 2

# GPU 서버
API_BASE = os.getenv("RAG_API_BASE", "http://147.46.15.89:58567").rstrip("/")
SEARCH_PATH = os.getenv("RAG_SEARCH_PATH", "/rag/search")
META_PATH = os.getenv("RAG_META_PATH", "/rag/meta")
API_KEY = os.getenv("RAG_API_KEY", "").strip()

# (연결, 읽기) 초. 답변 생성만 1분이 넘고 앞 요청이 줄 서 있을 수도 있어서
# 읽기 쪽을 넉넉히 준다.
CONNECT_TIMEOUT = float(os.getenv("RAG_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("RAG_READ_TIMEOUT", "600"))
META_TIMEOUT = float(os.getenv("RAG_META_TIMEOUT", "10"))

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# 문서를 좁히지 않고 전부 뒤질 때 쓰는 값 (backend/src/corpus.py 와 같다).
ALL_DOCS = "all"


# ==========================================================================
# 자료구조 — backend/src 의 dataclass 와 같은 모양
# ==========================================================================

@dataclass(frozen=True)
class Document:
    """문서 하나. 화면의 선택 상자와 데이터셋 표가 쓴다."""

    key: str
    code: str
    title: str
    lang: str
    lang_name: str
    note: str
    has_text: bool = False
    chars: int = 0          # 원문 글자 수
    chunks: int = 0         # 색인 청크 수
    tokens: int = 0
    dim: int = 0

    @property
    def label(self) -> str:
        return f"{self.title} · {self.lang_name}"


@dataclass(frozen=True)
class Question:
    """질문-정답 세트 한 항목."""

    id: int
    doc: str
    question: str
    answer: str = ""
    keywords: tuple[str, ...] = ()
    # 정답 청크 번호. 5단계는 최종 선정 청크에 이 중 하나가 들었는지로 본다.
    answer_chunks: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.id}. {self.question}"


@dataclass
class Hit:
    """검색 결과 한 건."""

    score: float
    text: str
    doc_key: str
    doc_code: str
    doc_title: str
    doc_lang: str
    chunk_index: int
    token_start: int
    token_end: int

    @property
    def key(self) -> str:
        return f"{self.doc_code}#{self.chunk_index}"

    def preview(self, n: int = 200) -> str:
        one_line = " ".join(self.text.split())
        return one_line[:n] + ("..." if len(one_line) > n else "")


@dataclass
class MetaQuery:
    """0단계가 뽑은 조건. 사람·기간·키워드뿐이다 (방향은 두지 않는다)."""

    people: list[str] = field(default_factory=list)     # 아래 셋을 합친 것
    sender: list[str] = field(default_factory=list)
    receiver: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)    # 명부에 없어 버린 이름
    since: str | None = None
    until: str | None = None
    keywords: list[str] = field(default_factory=list)

    def label(self) -> str:
        bits = []
        if len(self.people) >= 2:
            bits.append(" ~ ".join(self.people))
        elif self.people:
            bits.append(self.people[0])
        if self.since or self.until:
            bits.append(f"{self.since or '...'} ~ {self.until or '...'}")
        return " · ".join(bits) or "조건 없음"


@dataclass
class ExtractResult:
    """0-1단계 결과. 질문에서 뽑은 조건."""

    question: str
    query: MetaQuery
    raw: dict = field(default_factory=dict)
    rule: dict = field(default_factory=dict)
    llm: dict | None = None
    source: str = "rule"          # rule / llm+rule / off
    elapsed: float = 0.0
    error: str = ""

    def label(self) -> str:
        return self.query.label()


@dataclass
class FilterResult:
    """0-2단계 결과. mask 는 서버에만 있고 여기에는 숫자만 온다."""

    query: MetaQuery
    n_total: int = 0
    n_kept: int = 0
    step: str = ""                # 사다리의 어느 단계로 좁혔는지
    relaxed: list[str] = field(default_factory=list)   # 비어서 푼 조건
    summary: str = ""

    @property
    def ratio(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 1.0


@dataclass
class Narrowing:
    """0-3단계 결과 — 두 채널이 각각 몇 개를 남겼고 교집합이 몇 개인가.

    숫자는 전부 청크 단위다 (색인은 같은 청크의 러시아어·영어 두 판본을 행으로
    갖고 있어서 행으로 세면 두 배로 보인다).
    """

    n_chunks: int = 0             # 전체 청크
    n_meta: int | None = None     # 메타 조건을 통과한 청크 (조건 없으면 None)
    n_keyword: int | None = None  # 키워드가 든 청크
    n_both: int | None = None     # 교집합
    n_used: int = 0               # 실제로 검색에 쓴 후보
    step: str = ""                # 무엇을 썼는지
    keywords: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.n_used / self.n_chunks if self.n_chunks else 1.0


@dataclass
class SearchResult:
    """1단계 결과."""

    query: str
    doc: str
    doc_label: str
    top_k: int
    hits: list[Hit] = field(default_factory=list)
    n_indexed: int = 0
    n_candidates: int = 0         # 0단계를 통과한 색인 행 수
    n_candidate_chunks: int = 0   # 같은 것을 청크로 센 수
    n_docs: int = 1
    keywords: list[str] = field(default_factory=list)
    narrowed: str = ""            # 어떤 조건으로 좁혔는지
    elapsed: float = 0.0

    @property
    def filtered(self) -> bool:
        return 0 < self.n_candidates < self.n_indexed

    @property
    def pool(self) -> str:
        """상위 몇 개를 '무엇 중에서' 골랐는지 (색인 행이 아니라 청크로)."""
        if self.filtered and self.n_candidate_chunks:
            return f"후보 {self.n_candidate_chunks:,}청크"
        return f"청크 {self.n_indexed:,}개"


@dataclass
class RankedHit:
    """리랭킹을 거친 후보 하나."""

    hit: Hit
    rank_before: int
    rank_after: int = 0
    score: float | None = None
    prob: float | None = None
    reason: str = ""

    @property
    def moved(self) -> int:
        return self.rank_before - self.rank_after

    @property
    def percent(self) -> str:
        """확률 표기. 값이 아주 작아도 0 으로 뭉개지지 않게 한다."""
        if self.prob is None:
            return "-"
        if self.prob >= 0.01:
            return f"{self.prob * 100:.1f}%"
        if self.prob >= 0.0001:
            return f"{self.prob * 100:.3f}%"
        return f"{self.prob * 100:.1e}%"


@dataclass
class RerankResult:
    """2단계 + 3단계 결과."""

    question: str
    method: str
    ranked: list[RankedHit] = field(default_factory=list)
    selected: list[Hit] = field(default_factory=list)
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class AnswerResult:
    """4단계 결과."""

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


@dataclass
class GradeResult:
    """5단계 결과."""

    question: str
    llm_answer: str = ""
    candidates: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    verdict: str = ""
    reason: str = ""
    gold_answer: str = ""
    method: str = "문자열 포함"   # 무엇으로 판정했는지 (화면 표시용)
    elapsed: float = 0.0
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.verdict == "정답"

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def gold_display(self) -> str:
        if self.gold_answer:
            return self.gold_answer
        return ", ".join(self.matched if self.correct else self.candidates)


# 최종 선정 청크 수. 리랭킹을 건너뛰었을 때 검색 상위 몇 개로 대신할지.
FINAL_TOP_N = 5


@dataclass
class PipelineResult:
    """전체 결과. 화면은 이것만 보고 그린다."""

    question: str
    raw_question: str
    doc: str | None
    doc_name: str

    extract: ExtractResult | None = None
    filter: FilterResult | None = None
    narrow: Narrowing | None = None
    search: SearchResult | None = None
    rerank: RerankResult | None = None
    answer: AnswerResult | None = None
    grade: GradeResult | None = None
    elapsed: float = 0.0

    @property
    def selected(self):
        if self.rerank and self.rerank.selected:
            return self.rerank.selected
        if self.search:
            return self.search.hits[:FINAL_TOP_N]
        return []

    def errors(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.extract and self.extract.error:
            out["조건 추출"] = self.extract.error
        if self.rerank and self.rerank.error:
            out["리랭킹"] = self.rerank.error
        if self.answer and self.answer.error:
            out["답변 생성"] = self.answer.error
        if self.grade and self.grade.error:
            out["정답 비교"] = self.grade.error
        return out


# ==========================================================================
# JSON -> 자료구조
# ==========================================================================

def _hit(d: dict) -> Hit:
    return Hit(
        score=float(d["score"]), text=d["text"],
        doc_key=d["doc_key"], doc_code=d["doc_code"],
        doc_title=d["doc_title"], doc_lang=d["doc_lang"],
        chunk_index=int(d["chunk_index"]),
        token_start=int(d["token_start"]), token_end=int(d["token_end"]),
    )


def _query(d: dict | None) -> MetaQuery:
    d = d or {}
    return MetaQuery(
        people=list(d.get("people", [])), sender=list(d.get("sender", [])),
        receiver=list(d.get("receiver", [])),
        participants=list(d.get("participants", [])),
        unknown=list(d.get("unknown", [])),
        since=d.get("since"), until=d.get("until"),
        keywords=list(d.get("keywords", [])),
    )


def _extract(d: dict | None) -> ExtractResult | None:
    if d is None:
        return None
    return ExtractResult(
        question=d.get("question", ""), query=_query(d.get("query")),
        raw=d.get("raw") or {}, rule=d.get("rule") or {}, llm=d.get("llm"),
        source=d.get("source", "rule"), elapsed=float(d.get("elapsed", 0.0)),
        error=d.get("error", ""),
    )


def _filter(d: dict | None) -> FilterResult | None:
    if d is None:
        return None
    return FilterResult(
        query=_query(d.get("query")), n_total=int(d.get("n_total", 0)),
        n_kept=int(d.get("n_kept", 0)), step=d.get("step", ""),
        relaxed=list(d.get("relaxed", [])), summary=d.get("summary", ""),
    )


def _narrow(d: dict | None) -> Narrowing | None:
    if d is None:
        return None
    return Narrowing(
        n_chunks=int(d.get("n_chunks", 0)), n_meta=d.get("n_meta"),
        n_keyword=d.get("n_keyword"), n_both=d.get("n_both"),
        n_used=int(d.get("n_used", 0)), step=d.get("step", ""),
        keywords=list(d.get("keywords", [])),
    )


def _search(d: dict | None) -> SearchResult | None:
    if d is None:
        return None
    return SearchResult(
        query=d["query"], doc=d["doc"], doc_label=d["doc_label"],
        top_k=int(d["top_k"]), hits=[_hit(h) for h in d["hits"]],
        n_indexed=int(d["n_indexed"]),
        # 0단계를 붙이기 전 서버가 보낸 응답에는 없다. 없으면 안 좁힌 것으로 본다.
        n_candidates=int(d.get("n_candidates") or d["n_indexed"]),
        n_candidate_chunks=int(d.get("n_candidate_chunks", 0)),
        n_docs=int(d["n_docs"]), keywords=list(d.get("keywords", [])),
        narrowed=d.get("narrowed", ""),
        elapsed=float(d["elapsed"]),
    )


def _rerank(d: dict | None) -> RerankResult | None:
    if d is None:
        return None
    return RerankResult(
        question=d["question"], method=d["method"],
        ranked=[
            RankedHit(
                hit=_hit(r["hit"]),
                rank_before=int(r["rank_before"]),
                rank_after=int(r["rank_after"]),
                score=None if r["score"] is None else float(r["score"]),
                prob=None if r["prob"] is None else float(r["prob"]),
                reason=r["reason"],
            )
            for r in d["ranked"]
        ],
        selected=[_hit(h) for h in d["selected"]],
        model=d["model"], elapsed=float(d["elapsed"]), error=d["error"],
    )


def _answer(d: dict | None) -> AnswerResult | None:
    if d is None:
        return None
    return AnswerResult(
        question=d["question"], answer=d["answer"], enough=bool(d["enough"]),
        citations=list(d["citations"]), note=d["note"], model=d["model"],
        elapsed=float(d["elapsed"]), error=d["error"],
    )


def _grade(d: dict | None) -> GradeResult | None:
    if d is None:
        return None
    return GradeResult(
        question=d["question"], llm_answer=d["llm_answer"],
        candidates=list(d["candidates"]), matched=list(d["matched"]),
        verdict=d["verdict"], reason=d["reason"],
        gold_answer=d["gold_answer"], method=d.get("method", "문자열 포함"),
        elapsed=float(d["elapsed"]),
        error=d["error"],
    )


def result_from_dict(d: dict) -> PipelineResult:
    return PipelineResult(
        question=d["question"], raw_question=d["raw_question"],
        doc=d["doc"], doc_name=d["doc_name"],
        extract=_extract(d.get("extract")),
        filter=_filter(d.get("filter")),
        narrow=_narrow(d.get("narrow")),
        search=_search(d.get("search")),
        rerank=_rerank(d.get("rerank")),
        answer=_answer(d.get("answer")),
        grade=_grade(d.get("grade")),
        elapsed=float(d.get("elapsed", 0.0)),
    )


STAGE_FROM_DICT = {
    "extract": _extract,
    "filter": _filter,
    "narrow": _narrow,
    "search": _search,
    "rerank": _rerank,
    "answer": _answer,
    "grade": _grade,
}


# ==========================================================================
# 메타 — 문서 목록 · 질문 세트 · 모델 이름
# ==========================================================================

# GPU 서버에 한 번만 물어본다. Streamlit 은 위젯을 건드릴 때마다 app.py 를
# 처음부터 다시 도므로 캐시가 없으면 매번 왕복한다. 서버가 죽어 있을 때도
# 매번 연결을 기다리지 않도록 실패를 잠깐(RETRY_AFTER 초) 기억한다.
_META: dict = {}
RETRY_AFTER = 30.0

# GPU 서버에 못 붙었을 때 쓰는 값. 화면이 비어도 앱은 뜬다.
_FALLBACK_MODELS = {
    "embed": "BAAI/bge-m3",
    "rerank": "BAAI/bge-reranker-v2-m3",
    "llm": "Qwen/Qwen3-4B",
}
_FALLBACK_CHUNK = {"size": 512, "overlap": 128}


def meta(force: bool = False) -> dict:
    """/rag/meta 를 읽어 캐시한다. 실패하면 빈 값을 돌려준다."""
    if not force and _META.get("data"):
        return _META["data"]
    if not force and time.time() - _META.get("failed_at", 0) < RETRY_AFTER:
        return _META.get("stale") or {}

    try:
        import requests
        res = requests.get(API_BASE + META_PATH, headers=HEADERS,
                           timeout=(CONNECT_TIMEOUT, META_TIMEOUT))
        res.raise_for_status()
        data = res.json()
        _META["data"] = data
        _META["stale"] = data
        _META.pop("failed_at", None)
        _META["error"] = None
        return data
    except Exception as exc:  # noqa: BLE001
        _META["failed_at"] = time.time()
        _META["error"] = f"{type(exc).__name__}: {exc}"
        return _META.get("stale") or {}


def meta_error() -> str | None:
    """마지막 메타 조회 실패 메시지. 화면 위쪽 배너에 쓴다."""
    return _META.get("error")


def schema_mismatch() -> str | None:
    """서버와 계약 버전이 다르면 사람이 읽을 경고를 돌려준다."""
    got = (meta() or {}).get("schema_version")
    if got is None or got == SCHEMA_VERSION:
        return None
    return (f"GPU 서버가 옛 판본입니다 (서버 v{got}, 화면 v{SCHEMA_VERSION}). "
            f"문서 목록·질문 세트도 그 서버가 들고 있는 옛것이 그대로 옵니다. "
            f"{API_BASE} 에서 지금 backend/ 로 서버를 다시 띄우세요.")


def documents() -> list[Document]:
    """문서 목록. GPU 서버에 못 붙으면 빈 목록."""
    return [
        Document(
            key=d["key"], code=d["code"], title=d["title"],
            lang=d.get("lang", ""), lang_name=d.get("lang_name", ""),
            note=d.get("note", ""), has_text=bool(d.get("has_text")),
            chars=int(d.get("chars", 0)), chunks=int(d.get("chunks", 0)),
            tokens=int(d.get("tokens", 0)), dim=int(d.get("dim", 0)),
        )
        for d in (meta().get("documents") or [])
    ]


def load_questions() -> list[Question]:
    """질문-정답 세트."""
    return [
        Question(
            id=int(q["id"]), doc=q.get("doc", ""), question=q["question"],
            answer=q.get("answer", ""),
            keywords=tuple(q.get("keywords") or []),
            answer_chunks=tuple(int(c) for c in (q.get("answer_chunks") or [])),
        )
        for q in (meta().get("questions") or [])
    ]


def find_question(text: str) -> Question | None:
    """
    화면에서 고른 문자열로 질문을 되찾는다.

    선택 상자에는 "3. 이 법에서 말하는..." 처럼 번호가 붙어 있으므로 label 과
    본문 양쪽으로 견준다. 직접 입력한 질문이면 None 이고 5단계를 건너뛴다.
    """
    clean = (text or "").strip()
    if not clean:
        return None
    for q in load_questions():
        if clean in (q.label, q.question):
            return q
    return None


def doc_label(name: str | None) -> str:
    """화면 문구용. 전체 검색이면 문서 수를 적어 준다."""
    docs = documents()
    if name and name != ALL_DOCS:
        for d in docs:
            if name in (d.key, d.code, d.title, d.label):
                return d.label
    return f"전체 문서 {len(docs)}종"


def model_names() -> dict:
    """화면에 적을 모델 이름 3개."""
    return {**_FALLBACK_MODELS, **((meta().get("models") or {}))}


def chunk_params() -> tuple[int, int]:
    """청킹 설정 (크기, 겹침)."""
    chunk = {**_FALLBACK_CHUNK, **((meta().get("chunk") or {}))}
    return int(chunk["size"]), int(chunk["overlap"])


def health() -> dict:
    """GPU 서버 상태. 못 붙으면 ok=False."""
    try:
        import requests
        res = requests.get(API_BASE + "/health",
                           timeout=(CONNECT_TIMEOUT, META_TIMEOUT))
        res.raise_for_status()
        return res.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ==========================================================================
# 검색
# ==========================================================================

def _strip_number(question: str) -> str:
    """선택 항목 앞의 "1. " 은 화면 표시용 번호다. 질문 내용이 아니다."""
    import re
    return re.sub(r"^\d+\.\s*", "", question or "").strip()


def _failed(question: str, doc: str | None, message: str,
            elapsed: float = 0.0) -> PipelineResult:
    """
    GPU 서버에 못 붙었을 때 돌려줄 결과.

    예외를 그대로 올리면 Streamlit 이 빨간 트레이스백을 뱉고 화면이 끝난다.
    답변 단계 실패로 감싸 두면 app.py 가 원래 쓰던 경고 배너로 설명한다.
    """
    clean = _strip_number(question)
    return PipelineResult(
        question=clean, raw_question=question, doc=doc,
        doc_name=doc_label(doc),
        answer=AnswerResult(question=clean, error=message),
        elapsed=elapsed,
    )


def run_pipeline(question: str, doc: str | None = None,
                 top_k: int | None = None,
                 final_n: int | None = None,
                 gold: list[str] | None = None,
                 gold_chunks: list[int] | None = None,
                 answer_language: str | None = None,
                 on_stage=None,
                 meta_mode: str | None = None) -> PipelineResult:
    """
    질문 하나를 GPU 서버의 파이프라인에 태운다.

    backend/src/main.py 의 run_pipeline 과 계약이 같다. on_stage(단계이름, 결과)
    는 "extract" / "filter" / "narrow" / "search" / "rerank" / "answer" /
    "grade" 로 그대로 불린다. 전체가 1분을
    훌쩍 넘기므로, on_stage 를 주면 단계별 스트리밍(NDJSON)으로 받아 끝난
    단계부터 화면에 그린다.

    meta_mode 는 0단계(조건 추출)를 어떻게 돌릴지다. "llm"(4B) / "rule"(규칙만)
    / "off"(안 씀). 주지 않으면 GPU 서버 쪽 기본값(RAG_EXTRACT, 보통 llm)을
    따른다.

    실패해도 예외를 던지지 않는다. 답변 단계에 error 를 담아 돌려준다.
    """
    started = time.time()
    payload = {
        "question": question,
        "doc": doc,
        "gold": list(gold) if gold else None,
        "gold_chunks": [int(c) for c in gold_chunks] if gold_chunks else None,
        "answer_language": answer_language,
        "stream": on_stage is not None,
    }
    if top_k:
        payload["top_k"] = top_k
    if final_n:
        payload["final_n"] = final_n
    if meta_mode:
        payload["meta"] = meta_mode

    try:
        import requests
    except ImportError:
        return _failed(question, doc,
                       "requests 가 설치돼 있지 않습니다. pip install requests")

    url = API_BASE + SEARCH_PATH

    try:
        if on_stage is None:
            res = requests.post(url, json=payload, headers=HEADERS,
                                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            res.raise_for_status()
            return result_from_dict(res.json()["result"])

        # 단계별 스트리밍. 한 줄이 한 단계다.
        headers = {**HEADERS, "Accept": "application/x-ndjson"}
        with requests.post(url, json=payload, headers=headers, stream=True,
                           timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as res:
            res.raise_for_status()
            final = None
            for raw in res.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                event = json.loads(raw)
                stage = event.get("stage")

                if stage == "done":
                    final = result_from_dict(event["response"]["result"])
                    break
                if stage == "error":
                    return _failed(question, doc,
                                   "GPU 서버에서 실패했습니다. "
                                   f"{event.get('message', '')}",
                                   time.time() - started)
                if stage == "queued":
                    continue          # 대기 알림. 화면에 따로 그리지 않는다

                from_dict = STAGE_FROM_DICT.get(stage)
                if from_dict is None:
                    continue
                on_stage(stage, from_dict(event["payload"]))

            if final is None:
                return _failed(question, doc,
                               "GPU 서버가 결과를 끝까지 보내지 않고 끊었습니다.",
                               time.time() - started)
            return final

    except requests.exceptions.ConnectionError:
        return _failed(question, doc,
                       f"GPU 서버에 연결하지 못했습니다 ({API_BASE}). "
                       "서버가 떠 있는지, 포트가 열려 있는지 확인하세요.",
                       time.time() - started)
    except requests.exceptions.ReadTimeout:
        return _failed(question, doc,
                       f"GPU 서버가 {READ_TIMEOUT:.0f}초 안에 답하지 않았습니다. "
                       "앞선 요청이 밀려 있거나 모델을 올리는 중일 수 있습니다.",
                       time.time() - started)
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:  # noqa: BLE001
            detail = (exc.response.text or "")[:200]
        return _failed(question, doc,
                       f"GPU 서버가 {exc.response.status_code} 를 돌려줬습니다. "
                       f"{detail}", time.time() - started)
    except Exception as exc:  # noqa: BLE001
        return _failed(question, doc, f"{type(exc).__name__}: {exc}",
                       time.time() - started)


# --------------------------------------------------------------------------
# 단독 실행 — 통신 점검
# --------------------------------------------------------------------------

def main() -> None:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    question = " ".join(sys.argv[1:]) or "대마재배자는 누구에게 허가를 받나요?"

    print(f"GPU 서버 : {API_BASE}")
    print(f"상태     : {health()}")
    print(f"문서     : {len(documents())}종  질문 {len(load_questions())}개")
    print(f"모델     : {model_names()}")
    if meta_error():
        print(f"[!] 메타를 읽지 못했습니다: {meta_error()}")
        sys.exit(1)
    print(f"\n질문     : {question}")

    t0 = time.time()
    result = run_pipeline(
        question,
        gold=list(find_question(question).keywords) if find_question(question) else None,
        on_stage=lambda s, p: print(f"  [{s:<6}] 도착 (+{time.time() - t0:.1f}초)"),
    )

    ans = result.answer
    if ans and ans.error:
        print(f"\n실패: {ans.error}")
        sys.exit(1)

    print(f"\n검색 상위: {[h.key for h in result.search.hits[:5]]}")
    print(f"최종 선정: {[h.key for h in result.selected]}")
    print(f"답변     : {ans.answer}")
    print(f"인용     : {ans.citations}")
    if result.grade:
        print(f"판정     : {result.grade.verdict}")
    print(f"전체     : {result.elapsed:.1f}초 (왕복 {time.time() - t0:.1f}초)")


if __name__ == "__main__":
    main()
