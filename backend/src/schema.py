#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파이프라인 결과 <-> JSON — CPU 서버와 GPU 서버가 함께 쓰는 계약

GPU 서버(server.py)가 result_to_dict 로 내보내고, CPU 서버(rag_client.py)가
result_from_dict 로 되살린다. 되살린 객체는 원래와 같은 dataclass 라서
app.py 의 렌더 함수를 한 줄도 고치지 않아도 된다.

이 파일에서 지켜야 할 것:

  1. property 는 보내지 않는다. Hit.key, RankedHit.moved/percent,
     GradeResult.gold_display 는 필드에서 다시 계산되므로 필드만 넘기면 된다.
  2. numpy 스칼라를 float/int 로 캐스팅한다. 검색 점수는 np.float32 로 올 수
     있고 json 이 그대로는 직렬화하지 못한다.
  3. 필드를 빠뜨리면 화면이 조용히 빈다. 새 필드를 dataclass 에 추가하면
     여기도 같이 고쳐야 한다 (아래 round_trip 자체 점검이 잡아 준다).

자체 점검:
    python src/schema.py        # 더미 결과로 왕복 검사
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import AnswerResult  # noqa: E402
from extract_metadata import ExtractResult  # noqa: E402
from filter_metadata import FilterResult, MetaQuery  # noqa: E402
from grade import GradeResult  # noqa: E402
from main import PipelineResult  # noqa: E402
from rerank import RankedHit, RerankResult  # noqa: E402
from search import Hit, Narrowing, SearchResult  # noqa: E402

# 계약 버전. 필드를 바꾸면 올린다. 서버와 클라이언트가 다르면 경고만 남기고
# 계속 돈다 (양쪽 배포 시점이 어긋나도 데모가 멈추지는 않게).
SCHEMA_VERSION = 2      # 0단계(extract/filter) 추가


# --------------------------------------------------------------------------
# 작은 헬퍼
# --------------------------------------------------------------------------

def _f(value) -> float:
    """np.float32 를 파이썬 float 으로."""
    return float(value)


def _fo(value) -> float | None:
    """None 을 허용하는 실수."""
    return None if value is None else float(value)


# --------------------------------------------------------------------------
# 0 단계 : 조건 추출 + 후보 좁히기
# --------------------------------------------------------------------------

def query_to_dict(q: MetaQuery) -> dict:
    """추출기가 뽑고 명부 대조를 마친 조건. 0단계 두 결과가 함께 쓴다."""
    return {"people": list(q.people),
            "sender": list(q.sender), "receiver": list(q.receiver),
            "participants": list(q.participants),
            "unknown": list(q.unknown),
            "since": q.since, "until": q.until, "keywords": list(q.keywords)}


def query_from_dict(d: dict | None) -> MetaQuery:
    d = d or {}
    return MetaQuery(people=list(d.get("people", [])),
                     sender=list(d.get("sender", [])),
                     receiver=list(d.get("receiver", [])),
                     participants=list(d.get("participants", [])),
                     unknown=list(d.get("unknown", [])),
                     since=d.get("since"), until=d.get("until"),
                     keywords=list(d.get("keywords", [])))


def extract_to_dict(ex: ExtractResult | None) -> dict | None:
    if ex is None:
        return None
    return {
        "question": ex.question,
        "query": query_to_dict(ex.query),
        "raw": ex.raw,
        "rule": ex.rule,
        "llm": ex.llm,
        "source": ex.source,
        "elapsed": _f(ex.elapsed),
        "error": ex.error,
    }


def extract_from_dict(d: dict | None) -> ExtractResult | None:
    if d is None:
        return None
    return ExtractResult(
        question=d["question"],
        query=query_from_dict(d.get("query")),
        raw=d.get("raw") or {},
        rule=d.get("rule") or {},
        llm=d.get("llm"),
        source=d.get("source", "rule"),
        elapsed=_f(d.get("elapsed", 0.0)),
        error=d.get("error", ""),
    )


def filter_to_dict(fr: FilterResult | None) -> dict | None:
    """
    mask 는 보내지 않는다.

    15,522칸짜리 불리언 배열이라 JSON 으로 옮기면 응답이 통째로 커지는데,
    화면이 쓰는 것은 몇 개로 좁혔는지와 어느 단계에서 좁혔는지뿐이다. 실제
    필터링은 이미 GPU 서버에서 끝나 검색 결과에 반영돼 있다.
    """
    if fr is None:
        return None
    return {
        "query": query_to_dict(fr.query),
        "n_total": int(fr.n_total),
        "n_kept": int(fr.n_kept),
        "step": fr.step,
        "relaxed": list(fr.relaxed),
        "summary": fr.summary(),
        # 문서마다 조건이 다르게 걸린다. 어느 문서가 왜 빠졌는지를 화면이
        # 보여줄 수 있어야 "왜 이 문서에서 안 나왔지" 를 되짚을 수 있다.
        # masks 는 계산용 배열이라 여전히 안 보낸다.
        "excluded": list(fr.excluded),
        "notes": list(fr.notes),
    }


def filter_from_dict(d: dict | None) -> FilterResult | None:
    if d is None:
        return None
    return FilterResult(
        query=query_from_dict(d.get("query")),
        mask=None,                      # 보내지 않는다 (위 설명 참고)
        n_total=int(d.get("n_total", 0)),
        n_kept=int(d.get("n_kept", 0)),
        step=d.get("step", ""),
        relaxed=list(d.get("relaxed", [])),
        excluded=list(d.get("excluded", [])),
        notes=list(d.get("notes", [])),
    )


def narrow_to_dict(nr: Narrowing | None) -> dict | None:
    """
    두 채널의 후보 수. keep/bonus 배열은 보내지 않는다.

    31,044칸짜리 배열이고, 화면이 쓰는 것은 숫자 네 개뿐이다. 좁히기는 이미
    GPU 서버에서 끝나 검색 결과에 반영돼 있다.
    """
    if nr is None:
        return None
    return {
        "n_chunks": int(nr.n_chunks),
        "n_meta": None if nr.n_meta is None else int(nr.n_meta),
        "n_keyword": None if nr.n_keyword is None else int(nr.n_keyword),
        "n_both": None if nr.n_both is None else int(nr.n_both),
        "n_used": int(nr.n_used),
        "step": nr.step,
        "keywords": list(nr.keywords),
    }


def narrow_from_dict(d: dict | None) -> Narrowing | None:
    if d is None:
        return None
    return Narrowing(
        n_chunks=int(d.get("n_chunks", 0)),
        n_meta=d.get("n_meta"),
        n_keyword=d.get("n_keyword"),
        n_both=d.get("n_both"),
        n_used=int(d.get("n_used", 0)),
        step=d.get("step", ""),
        keywords=list(d.get("keywords", [])),
    )


# --------------------------------------------------------------------------
# 1 단계 : 검색
# --------------------------------------------------------------------------

def hit_to_dict(h: Hit) -> dict:
    return {
        "score": _f(h.score),
        "text": str(h.text),
        "doc_key": h.doc_key,
        "doc_code": h.doc_code,
        "doc_title": h.doc_title,
        "doc_lang": h.doc_lang,
        "chunk_index": int(h.chunk_index),
        "token_start": int(h.token_start),
        "token_end": int(h.token_end),
    }


def hit_from_dict(d: dict) -> Hit:
    return Hit(
        score=_f(d["score"]),
        text=d["text"],
        doc_key=d["doc_key"],
        doc_code=d["doc_code"],
        doc_title=d["doc_title"],
        doc_lang=d["doc_lang"],
        chunk_index=int(d["chunk_index"]),
        token_start=int(d["token_start"]),
        token_end=int(d["token_end"]),
    )


def search_to_dict(sr: SearchResult | None) -> dict | None:
    if sr is None:
        return None
    return {
        "query": sr.query,
        "doc": sr.doc,
        "doc_label": sr.doc_label,
        "top_k": int(sr.top_k),
        "hits": [hit_to_dict(h) for h in sr.hits],
        "n_indexed": int(sr.n_indexed),
        "n_candidates": int(sr.n_candidates),
        "n_candidate_chunks": int(sr.n_candidate_chunks),
        "n_docs": int(sr.n_docs),
        "keywords": list(sr.keywords),
        "narrowed": sr.narrowed,
        "elapsed": _f(sr.elapsed),
    }


def search_from_dict(d: dict | None) -> SearchResult | None:
    if d is None:
        return None
    return SearchResult(
        query=d["query"],
        doc=d["doc"],
        doc_label=d["doc_label"],
        top_k=int(d["top_k"]),
        hits=[hit_from_dict(h) for h in d["hits"]],
        n_indexed=int(d["n_indexed"]),
        # 필터를 붙이기 전 판본이 보낸 응답에는 없다. 없으면 안 좁힌 것으로 본다.
        n_candidates=int(d.get("n_candidates") or d["n_indexed"]),
        n_candidate_chunks=int(d.get("n_candidate_chunks", 0)),
        n_docs=int(d["n_docs"]),
        keywords=list(d.get("keywords", [])),
        narrowed=d.get("narrowed", ""),
        elapsed=_f(d["elapsed"]),
    )


# --------------------------------------------------------------------------
# 2, 3 단계 : 리랭킹 + 최종 선정
# --------------------------------------------------------------------------

def ranked_to_dict(r: RankedHit) -> dict:
    return {
        "hit": hit_to_dict(r.hit),
        "rank_before": int(r.rank_before),
        "rank_after": int(r.rank_after),
        "score": _fo(r.score),
        "prob": _fo(r.prob),
        "reason": r.reason,
    }


def ranked_from_dict(d: dict) -> RankedHit:
    return RankedHit(
        hit=hit_from_dict(d["hit"]),
        rank_before=int(d["rank_before"]),
        rank_after=int(d["rank_after"]),
        score=_fo(d["score"]),
        prob=_fo(d["prob"]),
        reason=d["reason"],
    )


def rerank_to_dict(rr: RerankResult | None) -> dict | None:
    if rr is None:
        return None
    return {
        "question": rr.question,
        "method": rr.method,
        "ranked": [ranked_to_dict(r) for r in rr.ranked],
        "selected": [hit_to_dict(h) for h in rr.selected],
        "model": rr.model,
        "elapsed": _f(rr.elapsed),
        "error": rr.error,
    }


def rerank_from_dict(d: dict | None) -> RerankResult | None:
    if d is None:
        return None
    return RerankResult(
        question=d["question"],
        method=d["method"],
        ranked=[ranked_from_dict(r) for r in d["ranked"]],
        selected=[hit_from_dict(h) for h in d["selected"]],
        model=d["model"],
        elapsed=_f(d["elapsed"]),
        error=d["error"],
    )


# --------------------------------------------------------------------------
# 4 단계 : 답변
# --------------------------------------------------------------------------

def answer_to_dict(a: AnswerResult | None) -> dict | None:
    if a is None:
        return None
    return {
        "question": a.question,
        "answer": a.answer,
        "enough": bool(a.enough),
        "citations": list(a.citations),
        "note": a.note,
        "model": a.model,
        "elapsed": _f(a.elapsed),
        "error": a.error,
    }


def answer_from_dict(d: dict | None) -> AnswerResult | None:
    if d is None:
        return None
    return AnswerResult(
        question=d["question"],
        answer=d["answer"],
        enough=bool(d["enough"]),
        citations=list(d["citations"]),
        note=d["note"],
        model=d["model"],
        elapsed=_f(d["elapsed"]),
        error=d["error"],
    )


# --------------------------------------------------------------------------
# 5 단계 : 정답 비교
# --------------------------------------------------------------------------

def grade_to_dict(g: GradeResult | None) -> dict | None:
    if g is None:
        return None
    return {
        "question": g.question,
        "llm_answer": g.llm_answer,
        "candidates": list(g.candidates),
        "matched": list(g.matched),
        "verdict": g.verdict,
        "reason": g.reason,
        "gold_answer": g.gold_answer,
        "method": g.method,
        "elapsed": _f(g.elapsed),
        "error": g.error,
    }


def grade_from_dict(d: dict | None) -> GradeResult | None:
    if d is None:
        return None
    return GradeResult(
        question=d["question"],
        llm_answer=d["llm_answer"],
        candidates=list(d["candidates"]),
        matched=list(d["matched"]),
        verdict=d["verdict"],
        reason=d["reason"],
        gold_answer=d["gold_answer"],
        method=d.get("method", "문자열 포함"),
        elapsed=_f(d["elapsed"]),
        error=d["error"],
    )


# --------------------------------------------------------------------------
# 전체
# --------------------------------------------------------------------------

def result_to_dict(r: PipelineResult) -> dict:
    return {
        "question": r.question,
        "raw_question": r.raw_question,
        "doc": r.doc,
        "doc_name": r.doc_name,
        "extract": extract_to_dict(r.extract),
        "filter": filter_to_dict(r.filter),
        "narrow": narrow_to_dict(r.narrow),
        "search": search_to_dict(r.search),
        "rerank": rerank_to_dict(r.rerank),
        "answer": answer_to_dict(r.answer),
        "grade": grade_to_dict(r.grade),
        "elapsed": _f(r.elapsed),
    }


def result_from_dict(d: dict) -> PipelineResult:
    return PipelineResult(
        question=d["question"],
        raw_question=d["raw_question"],
        doc=d["doc"],
        doc_name=d["doc_name"],
        extract=extract_from_dict(d.get("extract")),
        filter=filter_from_dict(d.get("filter")),
        narrow=narrow_from_dict(d.get("narrow")),
        search=search_from_dict(d.get("search")),
        rerank=rerank_from_dict(d.get("rerank")),
        answer=answer_from_dict(d.get("answer")),
        grade=grade_from_dict(d.get("grade")),
        elapsed=_f(d.get("elapsed", 0.0)),
    )


# 단계 이름 -> 변환 함수. 스트리밍에서 단계 하나만 주고받을 때 쓴다.
STAGE_TO_DICT = {
    "extract": extract_to_dict,
    "filter": filter_to_dict,
    "narrow": narrow_to_dict,
    "search": search_to_dict,
    "rerank": rerank_to_dict,
    "answer": answer_to_dict,
    "grade": grade_to_dict,
}

STAGE_FROM_DICT = {
    "extract": extract_from_dict,
    "filter": filter_from_dict,
    "narrow": narrow_from_dict,
    "search": search_from_dict,
    "rerank": rerank_from_dict,
    "answer": answer_from_dict,
    "grade": grade_from_dict,
}


def response_envelope(r: PipelineResult) -> dict:
    """
    /rag/search 의 응답 본문.

    result 안에 전체가 들어 있지만, 답만 필요한 호출자(is-web 등)가 매번
    5단계 구조를 파고들지 않도록 자주 쓰는 값을 위로 한 번 더 꺼내 둔다.
    """
    ans = r.answer
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "question": r.question,
        "doc": r.doc,
        "doc_name": r.doc_name,
        "answer": ans.answer if ans else "",
        "enough": bool(ans.enough) if ans else False,
        "citations": list(ans.citations) if ans else [],
        "note": ans.note if ans else "",
        "meta": {
            "condition": r.extract.label() if r.extract else "",
            "keywords": list(r.extract.query.keywords) if r.extract else [],
            "step": r.filter.step if r.filter else "",
            "narrow": narrow_to_dict(r.narrow),
        },
        "elapsed": _f(r.elapsed),
        "errors": r.errors(),
        "result": result_to_dict(r),
    }


# --------------------------------------------------------------------------
# 단독 실행 — 왕복 검사
# --------------------------------------------------------------------------

def _dummy() -> PipelineResult:
    """모델 없이 만들 수 있는 가짜 결과. 필드 누락을 잡기 위한 것."""
    hit = Hit(score=0.7123, text="가나다 " * 30, doc_key="ko법률", doc_code="ko",
              doc_title="마약류 관리에 관한 법률", doc_lang="한국어",
              chunk_index=12, token_start=6144, token_end=6656)
    sr = SearchResult(query="질문", doc="all", doc_label="전체 문서 7종",
                      top_k=10, hits=[hit], n_indexed=1100, n_candidates=20,
                      n_candidate_chunks=10, n_docs=7,
                      keywords=["68.224.217.72"],
                      narrowed="메타 + 키워드 1개 모두", elapsed=1.5)
    rr = RerankResult(question="질문", method="cross",
                      ranked=[RankedHit(hit=hit, rank_before=3, rank_after=1,
                                        score=4.21, prob=0.985,
                                        reason="관련성 0.985 (logit +4.21)")],
                      selected=[hit], model="BAAI/bge-reranker-v2-m3",
                      elapsed=2.0)
    ans = AnswerResult(question="질문", answer="답입니다.", enough=True,
                       citations=["ko#12"], note="", model="Qwen/Qwen3-4B",
                       elapsed=40.0)
    q = MetaQuery(people=["stern", "tom"], sender=["stern"],
                  receiver=["tom"], participants=[], unknown=["없는닉"],
                  since="2020-09-29", until="2020-09-29",
                  keywords=["68.224.217.72"])
    ex = ExtractResult(question="질문", query=q,
                       raw={"people": ["stern", "tom"]},
                       rule={"people": ["stern", "tom"]},
                       llm={"people": ["stern", "tom"]},
                       source="llm+rule", elapsed=1.2, error="")
    fr = FilterResult(query=q, mask=None, n_total=15522, n_kept=20,
                      step="쌍", relaxed=["쌍 + 기간"])
    nr = Narrowing(n_chunks=15522, n_meta=20, n_keyword=9, n_both=2, n_used=2,
                   step="메타 + 키워드 1개", keywords=["68.224.217.72"])
    gr = GradeResult(question="질문", llm_answer="답입니다.",
                     candidates=["답"], matched=["답"], verdict="정답",
                     reason="후보 포함", gold_answer="답입니다.", elapsed=0.01)
    return PipelineResult(question="질문", raw_question="1. 질문", doc=None,
                          doc_name="전체 문서 7종", extract=ex, filter=fr,
                          narrow=nr, search=sr, rerank=rr,
                          answer=ans, grade=gr, elapsed=43.5)


def main() -> None:
    import json

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    before = _dummy()
    payload = json.loads(json.dumps(result_to_dict(before), ensure_ascii=False))
    after = result_from_dict(payload)

    # 화면이 실제로 읽는 값들을 그대로 견준다 (property 포함).
    checks = [
        ("질문", before.question, after.question),
        ("문서 이름", before.doc_name, after.doc_name),
        ("추출 조건", before.extract.label(), after.extract.label()),
        ("추출 키워드", before.extract.query.keywords,
         after.extract.query.keywords),
        ("추출 sender", before.extract.query.sender,
         after.extract.query.sender),
        ("추출 receiver", before.extract.query.receiver,
         after.extract.query.receiver),
        ("추출 participants", before.extract.query.participants,
         after.extract.query.participants),
        ("모르는 이름", before.extract.query.unknown,
         after.extract.query.unknown),
        ("필터 요약", before.filter.summary(), after.filter.summary()),
        ("필터 축소율", round(before.filter.ratio, 6),
         round(after.filter.ratio, 6)),
        ("메타 후보", before.narrow.n_meta, after.narrow.n_meta),
        ("키워드 후보", before.narrow.n_keyword, after.narrow.n_keyword),
        ("교집합", before.narrow.n_both, after.narrow.n_both),
        ("최종 후보", before.narrow.n_used, after.narrow.n_used),
        ("검색 청크 수", before.search.n_indexed, after.search.n_indexed),
        ("검색 후보 수", before.search.n_candidates, after.search.n_candidates),
        ("좁힌 방법", before.search.narrowed, after.search.narrowed),
        ("후보 표기", before.search.pool, after.search.pool),
        ("1위 청크 id", before.search.hits[0].key, after.search.hits[0].key),
        ("1위 미리보기", before.search.hits[0].preview(50),
         after.search.hits[0].preview(50)),
        ("리랭킹 방식", before.rerank.method, after.rerank.method),
        ("이동 칸수", before.rerank.ranked[0].moved, after.rerank.ranked[0].moved),
        ("관련성 표기", before.rerank.ranked[0].percent,
         after.rerank.ranked[0].percent),
        ("선정 수", len(before.rerank.selected), len(after.rerank.selected)),
        ("답변", before.answer.answer, after.answer.answer),
        ("인용", before.answer.citations, after.answer.citations),
        ("판정", before.grade.verdict, after.grade.verdict),
        ("실제 정답", before.grade.gold_display, after.grade.gold_display),
        ("최종 청크", [h.key for h in before.selected],
         [h.key for h in after.selected]),
    ]

    bad = [name for name, a, b in checks if a != b]
    for name, a, b in checks:
        print(f"    {'OK ' if a == b else '!! '} {name:<12} {a!r}")
    print(f"\n왕복 검사: {'통과' if not bad else '실패 ' + ', '.join(bad)}")
    print(f"JSON 크기: {len(json.dumps(result_to_dict(before))):,} 바이트")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
