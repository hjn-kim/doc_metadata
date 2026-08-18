"""
화면 전용 데모 앱 (CPU 서버) - 계산은 GPU 서버가 한다.

    [이 서버 : CPU]                      [GPU 서버]
    frontend/app.py   --- HTTP --->      backend/src/server.py
    frontend/rag_api.py                    1 질의 임베딩  BAAI/bge-m3
                      <--- 단계별 ----      2 리랭킹      bge-reranker-v2-m3
                           NDJSON           4 답변 생성   Qwen/Qwen3-4B
                                          + data/ (원문·색인·정답표)

이 서버에는 모델도 색인도 없다. 문서 목록·질문 세트·모델 이름은 /rag/meta 에서
받아 오고, 검색은 /rag/search 로 넘긴다. 필요한 패키지는 streamlit 과 requests
뿐이다.

    RAG_API_BASE   GPU 서버 주소 (기본 http://147.46.15.89:58566)
    RAG_API_KEY    설정돼 있으면 X-API-Key 로 보낸다

    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

import re
import sys
from html import escape
from pathlib import Path

import streamlit as st

# rag_api.py 는 한 단계 위의 document_lab/ 에 있다 (어느 경로에서 띄우든 찾도록).
LIB_DIR = Path(__file__).resolve().parent.parent / "document_lab"
sys.path.insert(0, str(LIB_DIR))
from rag_api import (  # noqa: E402
    ALL_DOCS,
    API_BASE,
    PipelineResult,
    chunk_params,
    documents,
    find_question,
    load_questions,
    meta_error,
    model_names,
    run_pipeline,
    schema_mismatch,
)

# 모델 이름과 청킹 설정은 GPU 서버가 알려 준다 (LOCAL_LLM_MODEL 은 그쪽
# 환경변수라 여기서는 알 수 없다). 못 물어보면 기본값으로 적는다.
_MODELS = model_names()
EMBED_MODEL = _MODELS["embed"]
RERANKER_MODEL = _MODELS["rerank"]
LLM_MODEL = _MODELS["llm"]
CHUNK_SIZE, OVERLAP = chunk_params()

# 화면에 쓸 짧은 이름과 어림 크기. LLM 을 바꿔도 표시가 따라오게 하드코딩하지
# 않고 모델 이름에서 뽑는다.
#   "Qwen/Qwen3-4B" -> "Qwen3-4B", "약 8GB"  (bf16 = 파라미터 수 x 2바이트)
LLM_SHORT = LLM_MODEL.split("/")[-1]
_params = re.search(r"(\d+(?:\.\d+)?)B", LLM_SHORT)
LLM_SIZE = f"약 {float(_params.group(1)) * 2:.0f}GB" if _params else "크기 미상"

EMBED_SHORT = EMBED_MODEL.split("/")[-1]
RERANKER_SHORT = RERANKER_MODEL.split("/")[-1]


# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
# set_page_config 는 다른 st.* 호출보다 반드시 먼저 와야 한다.
# (제목은 스타일이 주입된 뒤 아래 "화면" 절에서 그린다)
st.set_page_config(
    page_title="문서 AI 모델 데모 (GPU)",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# 선택 항목
#
# 질문은 data/qa.json, 문서는 data/emb 에서 읽는다. 앱에는 목록을 적어 두지
# 않는다. 문서나 질문을 늘려도 코드를 고칠 일이 없게 하려는 것이다.
# ---------------------------------------------------------
DOCUMENTS = documents()

# 화면에 보일 이름 -> 파이프라인에 넘길 문서 키. 첫 항목이 기본값이다.
# 전체 검색이 기본인 이유: 질문만 던지면 7개 문서 중 어디에 답이 있는지 모델이
# 찾아내는 편이 데모로서 정직하다.
DOCUMENT_OPTIONS = {f"전체 문서 {len(DOCUMENTS)}종": ALL_DOCS}
DOCUMENT_OPTIONS.update({doc.label: doc.key for doc in DOCUMENTS})

QUESTIONS = load_questions()

# 목록 맨 앞에 두는 항목. 이걸 고르면 아래에 입력칸이 열리고, 나머지 질문은
# 한 칸씩 밀린다. 정답표에 없는 질문이라 5단계(정답 비교)는 건너뛰게 된다.
CUSTOM_QUESTION = "직접 질문"
QUESTION_OPTIONS = [CUSTOM_QUESTION] + [q.label for q in QUESTIONS]

PIPELINE_STEPS = [
    "조건 추출 · 후보 좁히기",
    "검색 랭킹",
    "리랭킹",
    "최종 청크 선정",
    "LLM 답변",
    "정답 비교",
]


# ---------------------------------------------------------
# 카드 그리기
#
# 파이프라인이 단계를 끝낼 때마다 하나씩 불린다. 각 함수는 st.markdown 한 번으로
# 카드 하나를 그리고 끝낸다. 계산은 하지 않는다.
# ---------------------------------------------------------

def render_meta(ex, fr) -> None:
    """0. 조건 추출 + 후보 좁히기 — 무엇으로 얼마나 좁혔는지."""
    if ex is None:
        return

    q = ex.query
    rows = []
    if q.people:
        rows.append(("사람", " ~ ".join(q.people) if len(q.people) >= 2
                     else q.people[0]))
    if q.since or q.until:
        rows.append(("기간", f"{q.since or '...'} ~ {q.until or '...'}"))
    if q.keywords:
        rows.append(("키워드", ", ".join(q.keywords)))
    if q.unknown:
        rows.append(("버린 이름", ", ".join(q.unknown) + " (명부에 없음)"))
    if not rows:
        rows.append(("조건", "없음 — 전체 청크를 그대로 뒤집니다"))

    items = "".join(
        f'<div class="hit-line">'
        f'<span class="hit-src">{escape(name)}</span>'
        f'<span class="hit-oneline">{escape(value)}</span>'
        f'</div>'
        for name, value in rows
    )

    tags = [f'<span class="tag">{escape(ex.source)}</span>',
            f'<span class="tag">{ex.elapsed:.1f}초</span>']
    if fr is not None:
        tags.insert(0, f'<span class="tag">청크 {fr.n_total:,}개 → '
                       f'{fr.n_kept:,}개</span>')
    relaxed = ""
    if fr is not None and fr.relaxed:
        relaxed = (f'<div class="query-note">조건이 너무 좁아 '
                   f'{escape(" → ".join(fr.relaxed))} 을(를) 풀었습니다. '
                   f'후보가 0개가 되면 답변이 반드시 틀리기 때문에, 좁은 조건부터 '
                   f'걸어 보고 처음으로 남는 것을 씁니다.</div>')

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">0. 조건 추출 {"".join(tags)}</div>
            <div class="query-origin">질의 · {escape(ex.question)}</div>
            {items}
            {relaxed}
            <div class="query-note">질문에서 사람·기간·키워드를 뽑아 그 조건에
                맞는 청크만 남긴 뒤, 그 안에서 1단계 검색을 합니다. 사람은
                닉네임 명부에 있는 것만 씁니다(없는 이름은 버립니다). 보낸 사람과
                받는 사람은 나누지 않습니다 — 찾는 것은 그 사람이 낀 대화라
                대화쌍(dyad) 안에 있는지만 봅니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search(sr) -> None:
    """1. 검색 랭킹 — 질문을 임베딩해 뽑은 상위 청크. 한 항목이 정확히 한 줄."""
    if sr.hits:
        items = "".join(
            f'<div class="hit-line">'
            f'<span class="hit-rank">{rank}</span>'
            f'<span class="hit-score">{hit.score:.3f}</span>'
            f'<span class="hit-src">{hit.key}</span>'
            f'<span class="hit-oneline">{escape(hit.preview(220))}</span>'
            f'</div>'
            for rank, hit in enumerate(sr.hits, 1)
        )
    else:
        items = '<div class="query-note">검색된 청크가 없습니다.</div>'

    # 0단계가 좁혔으면 전체 청크 수 대신 후보 수를 보여 준다. 상위 몇 개를
    # 무엇 중에서 골랐는지가 이 줄의 뜻이다.
    pool = (f"후보 {sr.n_candidates:,}개" if sr.filtered
            else f"청크 {sr.n_indexed}개")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">1. 검색 랭킹
                <span class="tag">{escape(sr.doc_label)}</span>
                <span class="tag">{pool} → 상위 {sr.top_k}개</span>
                <span class="tag">{sr.elapsed:.1f}초</span></div>
            <div class="query-origin">질의 · {escape(sr.query)}</div>
            {items}
            <div class="query-note">점수는 질문 벡터와 청크 벡터의 코사인
                유사도입니다. 질문과 청크를 따로 임베딩해 비교하므로(bi-encoder)
                빠르지만 둘을 같이 읽고 판단하지는 못합니다. bge-m3 는 여러 언어를
                한 벡터 공간에 넣으므로 한국어 질문으로 외국어 원문이 바로 걸립니다.
                이 목록이 2단계 리랭킹의 후보가 됩니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rerank(rr) -> None:
    """2. 리랭킹 — 등수가 어떻게 바뀌었는지"""
    method = {
        "cross": f"크로스인코더 {rr.model or RERANKER_MODEL}",
        "dense": "검색 점수 순서 (크로스인코더 실패)",
    }.get(rr.method, rr.method)
    rows = ""
    for item in rr.ranked:
        selected = item.rank_after <= len(rr.selected)
        if item.moved > 0:
            move = f'<span class="rr-up">▲{item.moved}</span>'
        elif item.moved < 0:
            move = f'<span class="rr-down">▼{-item.moved}</span>'
        else:
            move = '<span class="rr-same">-</span>'
        score = (f'<span class="rr-llm">{item.percent}</span>'
                 if item.prob is not None else "-")
        rows += (
            f'<tr class="{"rr-picked" if selected else ""}">'
            f'<td class="qrank">{item.rank_after}</td>'
            f'<td class="qrank">{item.rank_before}</td>'
            f'<td>{move}</td>'
            f'<td>{score}</td>'
            f'<td><span class="qchunk">{item.hit.key}</span></td>'
            f'<td class="rr-reason">{escape(item.reason)}</td></tr>'
        )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">2. 리랭킹
                <span class="tag">{escape(method)}</span>
                <span class="tag">{rr.elapsed:.1f}초</span></div>
            <div class="qtable-wrap">
                <table class="qtable">
                    <thead><tr>
                        <th class="qrank">후</th><th class="qrank">전</th>
                        <th>이동</th><th>관련성</th><th>청크</th><th>판단 근거</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            <div class="query-note">관련성은 cross-Encoder가 질의와 청크를 한 입력으로
                붙여 읽고 낸 점수입니다. 순서는 반올림하지 않은 raw logit 으로
                정하고 표에는 그것을 확률로 바꿔 적었습니다. 질문과 청크의 언어가
                다르면 확률 자체는 1% 아래로 깔리지만 후보끼리의 순서는 그대로
                유효합니다. 질의와 청크를 같이 읽으므로 "낱말만 겹치는 목차 청크"와
                "답이 실제로 든 조문 청크"를 더 잘 가릅니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected(rr) -> None:
    """3. 최종 청크 선정"""
    items = "".join(
        f'<div class="hit">'
        f'  <div class="hit-head">'
        f'    <span class="hit-rank">{rank}</span>'
        f'    <span class="hit-src">{hit.key}</span>'
        f'    <span>{escape(hit.doc_title)} · {escape(hit.doc_lang)} · '
        f'{hit.token_start}~{hit.token_end}토큰</span>'
        f'  </div>'
        f'  <div class="hit-text">{escape(hit.preview(260))}</div>'
        f'</div>'
        for rank, hit in enumerate(rr.selected, 1)
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">3. 최종 청크 선정
                <span class="tag">{len(rr.ranked)}개 → {len(rr.selected)}개</span>
                <span class="tag">약 {len(rr.selected) * CHUNK_SIZE}토큰</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(ans) -> None:
    """4. LLM 답변"""
    if ans is None:
        return
    if not ans.ok:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">4. LLM 답변
                    <span class="tag">실패</span></div>
                <div class="query-note">{escape(ans.error or "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    badge = ('<span class="tag tag-changed">근거 존재</span>' if ans.enough
             else '<span class="tag">근거 부족</span>')
    cites = ("".join(f'<span class="cite">{escape(c)}</span>'
                     for c in ans.citations)
             if ans.citations else '<span class="cite-none">없음</span>')
    note = (f'<div class="query-note">{escape(ans.note)}</div>'
            if ans.note else "")
    st.markdown(
        f"""
        <div class="result-card answer-card">
            <div class="card-title">4. LLM 답변 {badge}
                <span class="tag">{ans.elapsed:.1f}초</span></div>
            <div class="answer-text">{escape(ans.answer)}</div>
            <div class="cite-row">근거 {cites}</div>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grade(gr) -> None:
    """5. 정답 비교 — data/qa.json 의 정답과 4단계 답변을 나란히 놓는다."""
    if gr is None:
        return

    if gr.correct:
        style, mark = "grade-ok", "O"
    elif gr.verdict == "오답":
        style, mark = "grade-no", "X"
    else:
        style, mark = "grade-none", "?"

    hit_words = ", ".join(gr.matched) if gr.matched else "없음"
    st.markdown(
        f"""
        <div class="result-card grade-card {style}">
            <div class="card-title">5. 정답 비교
                <span class="tag tag-verdict {style}">{mark} {gr.verdict}</span>
                <span class="tag">문자열 포함 판정</span></div>
            <div class="grade-row">
                <span class="grade-label">LLM 답변</span>
                <span class="grade-value">{escape(gr.llm_answer) or "—"}</span>
            </div>
            <div class="grade-row">
                <span class="grade-label">실제 정답</span>
                <span class="grade-value grade-gold">
                    {escape(gr.gold_display) or "—"}</span>
            </div>
            <div class="grade-row">
                <span class="grade-label">정답 후보</span>
                <span class="grade-value">{escape(", ".join(gr.candidates))}</span>
            </div>
            <div class="grade-row">
                <span class="grade-label">겹친 후보</span>
                <span class="grade-value">{mark} {escape(hit_words)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_all(result: PipelineResult) -> None:
    """
    완성된 결과에서 카드 5개를 한꺼번에 그린다.

    파이프라인을 돌리지 않는 경로 전용이다(캐시 히트, 지난 결과 되살리기).
    on_stage 콜백이 불리지 않아 단계별로 그려 줄 사람이 없다. 순서는 on_stage
    와 같아야 한다.
    """
    if result.extract:
        render_meta(result.extract, result.filter)          # 0
    if result.search:
        render_search(result.search)                        # 1
    if result.rerank:
        render_rerank(result.rerank)                        # 2
        render_selected(result.rerank)                      # 3
    render_answer(result.answer)                            # 4
    render_grade(result.grade)                              # 5


def render_tail(result: PipelineResult, gold: list, note: str = "") -> None:
    """
    카드 5개 아래에 붙는 것들 - 실패 경고, 안내 문구, 개발용 데이터.

    새로 돌렸을 때와 지난 결과를 되살렸을 때가 같아야 해서 함수로 뺐다.
    note 는 결과의 출처를 알리는 한 줄이고, 갓 돌린 결과면 비운다.
    """
    for stage_name, message in result.errors().items():
        st.warning(f"{stage_name} 단계가 실패했습니다. {message}")

    if not gold:
        st.caption("정답표에 없는 질문이라 5단계(정답 비교)는 건너뛰었습니다.")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    if note:
        st.caption(note)

    with st.expander("개발용 데이터 보기"):
        st.json(dev_payload(result))


def dev_payload(result: PipelineResult) -> dict:
    """개발용 데이터 보기에 넣을 것들."""
    sr, rr, ans = result.search, result.rerank, result.answer
    return {
        "question": result.question,
        "raw_question": result.raw_question,
        "doc": result.doc,
        "doc_name": result.doc_name,
        "elapsed_sec": round(result.elapsed, 2),
        "0_meta": None if result.extract is None else {
            "source": result.extract.source,
            "rule": result.extract.rule,
            "llm": result.extract.llm,
            "people": result.extract.query.people,
            "unknown": result.extract.query.unknown,
            "since": result.extract.query.since,
            "until": result.extract.query.until,
            "keywords": result.extract.query.keywords,
            "error": result.extract.error,
            "filter": None if result.filter is None else {
                "step": result.filter.step,
                "relaxed": result.filter.relaxed,
                "n_kept": result.filter.n_kept,
                "n_total": result.filter.n_total,
            },
        },
        "1_search": {
            "query": sr.query,
            "n_indexed": sr.n_indexed,
            "n_candidates": sr.n_candidates,
            "narrowed": sr.narrowed,
            "keywords": sr.keywords,
            "n_docs": sr.n_docs,
            "top_k": sr.top_k,
            "hits": [f"{h.key} {h.score:.4f}" for h in sr.hits],
        },
        "2_3_rerank": {
            "method": rr.method,
            "error": rr.error,
            "ranked": [
                {
                    "rank_after": x.rank_after,
                    "rank_before": x.rank_before,
                    "score": None if x.score is None else round(x.score, 4),
                    "prob": None if x.prob is None else round(x.prob, 6),
                    "dense": round(x.hit.score, 4),
                    "chunk": x.hit.key,
                    "reason": x.reason,
                }
                for x in rr.ranked
            ],
            "selected": [h.key for h in rr.selected],
        },
        "4_answer": None if ans is None else {
            "answer": ans.answer,
            "enough": ans.enough,
            "citations": ans.citations,
            "note": ans.note,
            "model": ans.model,
            "error": ans.error,
        },
        "5_grade": None if result.grade is None else {
            "verdict": result.grade.verdict,
            "correct": result.grade.correct,
            "llm_answer": result.grade.llm_answer,
            "gold_answer": result.grade.gold_answer,
            "candidates": result.grade.candidates,
            "matched": result.grade.matched,
            "reason": result.grade.reason,
        },
        "selected_chunks": [
            {"chunk": h.key, "doc": h.doc_title,
             "tokens": [h.token_start, h.token_end], "text": h.text}
            for h in result.selected
        ],
    }


# ---------------------------------------------------------
# 데이터셋 & 모델 탭
#
# 하드코딩하지 않는다. 원문 글자 수·청크 수·토큰 수는 GPU 서버가 실제 파일을
# 세어 /rag/meta 에 실어 보낸 값이다 (이 서버에는 data 가 없다).
# ---------------------------------------------------------

# 파이프라인이 도는 순서 그대로.
MODELS = [
    ("임베딩", EMBED_SHORT,
     f"{EMBED_MODEL} · 1024차원 · 100개 넘는 언어를 한 벡터 공간에 넣는다"),
    ("검색", "Cosine similarity",
     "벡터가 L2 정규화돼 있어 내적이 곧 코사인 유사도"),
    ("리랭킹", RERANKER_SHORT,
     f"{RERANKER_MODEL} · 질문-청크 쌍을 직접 채점하는 크로스 인코더 (GPU)"),
    ("답변", LLM_SHORT,
     f"{LLM_MODEL} · 선정된 청크만 근거로 (GPU, bf16 {LLM_SIZE})"),
]


def corpus_stats() -> list[dict]:
    """문서마다 원문 크기와 색인 규모. GPU 서버가 세어 준 값을 옮겨 담는다."""
    return [
        {
            "코드": doc.code,
            "제목": doc.title,
            "언어": doc.lang_name,
            "글자": doc.chars,
            "청크": doc.chunks,
            "토큰": doc.tokens,
            "차원": doc.dim,
            "설명": doc.note,
        }
        for doc in DOCUMENTS
    ]


def _kv_table(pairs: list[tuple[str, str]]) -> str:
    """항목 | 값 두 칸짜리 표."""
    rows = "".join(
        f'<tr><td class="dkey">{k}</td><td>{v}</td></tr>' for k, v in pairs
    )
    return f'<table class="dtable"><tbody>{rows}</tbody></table>'


def render_dataset_tab() -> None:
    rows = corpus_stats()

    # ---- 데이터셋 -------------------------------------------------------
    st.markdown('<div class="dhead">데이터셋</div>', unsafe_allow_html=True)

    if rows:
        head = ("<tr><th>코드</th><th>문서</th><th>언어</th>"
                "<th class='num'>원문 글자</th><th class='num'>청크</th>"
                "<th class='num'>토큰</th></tr>")
        body = "".join(
            f"<tr><td class='dkey'>{escape(r['코드'])}</td>"
            f"<td class='dmodel'>{escape(r['제목'])}"
            f"<div class='dnote'>{escape(r['설명'])}</div></td>"
            f"<td>{escape(r['언어'])}</td>"
            f"<td class='num'>{r['글자']:,}</td>"
            f"<td class='num'>{r['청크']}</td>"
            f"<td class='num'>{r['토큰']:,}</td></tr>"
            for r in rows
        )
        total_chunks = sum(r["청크"] for r in rows)
        total_tokens = sum(r["토큰"] for r in rows)
        st.markdown(
            f"""
            <div class="result-card">
                <div class="dtitle">문서 {len(rows)}종</div>
                <div class="ddesc">각국의 마약·부패·자금세탁·고문방지 관련 법령과
                    문서입니다. 원문 언어는 7가지 언어이고 질문은 한국어로 번역 없이 교차 검색합니다.</div>
                <table class="dtable dtable-docs">
                    <thead>{head}</thead><tbody>{body}</tbody>
                </table>
                <div class="dsource"><b>합계</b> · 청크 {total_chunks:,}개 ·
                    {total_tokens:,}토큰 · {rows[0]['차원']}차원
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "문서 목록을 받지 못했습니다. GPU 서버가 떠 있는지, 색인"
            "(`backend/data/emb`)이 있는지 확인하세요."
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- 전처리 ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <div class="dtitle">전처리</div>
            {_kv_table([
                ("본문 추출",
                 "PDF 에서 뽑은 파일은 JSON 에 담겨 있어 text 필드만 "
                 "꺼내 씁니다."),
                ("청킹",
                 f"{CHUNK_SIZE} / {OVERLAP}토큰 — bge-m3 토크나이저로 "
                 f"{CHUNK_SIZE}토큰마다 자르고 {OVERLAP}토큰을 겹칩니다. "
                 ),
                ("겹치는 이유",
                 "한 조문이 청크 경계에 걸려 잘리면 답의 앞뒤가 나뉩니다. "
                 f"{OVERLAP}토큰을 겹쳐 그 경계를 덮습니다."),
                ("정규화",
                 "벡터를 L2 정규화해 저장합니다. 검색이 내적을 그대로 코사인 "
                 "유사도로 쓸 수 있습니다."),
                ("메타데이터",
                 "청크마다 문서코드·청크번호·토큰 위치를 함께 저장합니다."
                 "벡터에는 들어가지 않습니다."),
            ])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- 질문-정답 세트 --------------------------------------------------
    if QUESTIONS:
        by_doc: dict[str, int] = {}
        for q in QUESTIONS:
            by_doc[q.doc] = by_doc.get(q.doc, 0) + 1
        body = "".join(
            f'<tr><td class="dkey">{q.id}</td>'
            f'<td>{escape(q.question)}'
            f'<div class="dnote">정답 · {escape(q.answer)}</div></td>'
            f'<td class="dnote">{escape(" / ".join(q.keywords))}</td></tr>'
            for q in QUESTIONS
        )
        st.markdown(
            f"""
            <div class="result-card">
                <div class="dtitle">질문-정답 세트 {len(QUESTIONS)}문항</div>
                <div class="ddesc">원문을 읽고 만든 세트입니다
                    (<code>backend/data/qa.json</code>). 질문과 정답은 한국어이고 근거는
                    7개 언어이므로, 그대로 돌리면 교차 언어 검색 평가가 됩니다.
                    5단계는 정답 후보가 답변 안에 하나라도 들어 있는지만 봅니다.</div>
                <table class="dtable">
                    <thead><tr><th>번호</th><th>질문 · 정답</th>
                        <th>정답 후보</th></tr></thead>
                    <tbody>{body}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- 모델 -----------------------------------------------------------
    body = "".join(
        f'<tr><td class="dkey">{escape(role)}</td>'
        f'<td class="dmodel">{escape(name)}</td>'
        f'<td class="dnote">{escape(note)}</td></tr>'
        for role, name, note in MODELS
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="dtitle">모델</div>
            <table class="dtable">
                <thead><tr><th>단계</th><th>모델</th><th>비고</th></tr></thead>
                <tbody>{body}</tbody>
            </table>
            <div class="ddesc" style="margin-top:.9rem">임베딩과 리랭킹이 같은
                m3 계열입니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 스타일
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        /* Outfit / Inter 는 시스템에 없어 웹폰트로 가져온다 */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@800&family=Inter:wght@400;500;700&display=swap');

        .block-container {
            max-width: 1080px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #4A90E2, #8E2DE2);
            -webkit-background-clip: text;
            background-clip: text;              /* 웹킷 아닌 브라우저용 */
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-family: 'Inter', sans-serif;
            color: #7f8c8d;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-label {
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .pipeline-card {
            border: 1px solid #E4E7EC;
            border-radius: 10px;
            padding: 0.5rem 0.6rem;
            text-align: center;
            min-height: 58px;
            background: #FFFFFF;
        }

        .pipeline-number {
            font-size: 0.7rem;
            color: #98A2B3;
            margin-bottom: 0.1rem;
        }

        .pipeline-name {
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.3;
        }

        /* 처리 단계와 검색 폼 사이 간격 */
        .section-gap { height: 2.2rem; }

        .result-card {
            border: 1px solid #D0D5DD;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            background: #F9FAFB;
            /* 카드마다 st.markdown 이 따로 나가므로 카드 사이 여백은 이것 하나로 */
            margin-top: 1.2rem;
        }

        /* 질문·문서를 고르는 박스. 예전 st.form 이 그려 주던 테두리를 대신하고
           결과 카드와 같은 모양으로 맞춘다. st.container(key="search_box") 가
           붙여 주는 클래스이고, 위젯이 놓이는 세로 블록 자체에 붙으므로 여백도
           여기에 준다. div 를 앞에 붙인 건 컨테이너가 기본으로 들고 있는
           테두리·여백보다 우선순위를 높이기 위해서다. */
        div.st-key-search_box {
            border: 1px solid #D0D5DD;
            border-radius: 14px;
            background: #F9FAFB;
            padding: 1.25rem 1.35rem;
        }

        .card-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        /* 검색 랭킹 카드 */
        .query-origin {
            font-size: 0.8rem;
            color: #98A2B3;
            margin-bottom: 0.6rem;
        }
        .query-note {
            font-size: 0.82rem;
            color: #667085;
            margin-top: 0.55rem;
        }

        /* 2. 리랭킹 표 */
        .qtable-wrap {
            overflow-x: auto;          /* 칸이 늘어도 페이지가 밀리지 않게 */
            margin-top: 0.3rem;
        }
        .qtable {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.8rem;
        }
        .qtable th, .qtable td {
            padding: 0.34rem 0.5rem;
            border-bottom: 1px solid #EAECF0;
            text-align: left;
            white-space: nowrap;
        }
        .qtable th {
            font-weight: 700;
            color: #475467;
            border-bottom: 1px solid #D0D5DD;
        }
        .qrank {
            color: #98A2B3;
            font-variant-numeric: tabular-nums;
            width: 2rem;
        }
        .qscore {
            font-variant-numeric: tabular-nums;
            color: #344054;
            margin-right: 0.35rem;
        }
        .qchunk {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.74rem;
            color: #667085;
        }
        /* 3. 최종 청크 선정 카드 */
        .hit {
            padding: 0.6rem 0;
            border-top: 1px solid #EAECF0;
        }
        .hit:first-of-type { border-top: none; padding-top: 0; }

        .hit-head {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #667085;
            margin-bottom: 0.25rem;
        }
        .hit-rank {
            font-weight: 700;
            color: #3B5BDB;
            min-width: 1.4rem;
        }
        .hit-score {
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            color: #344054;
        }
        .hit-src {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.76rem;
            background: #EAECF0;
            border-radius: 4px;
            padding: 0.05rem 0.35rem;
        }
        .hit-text {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        /* 1. 검색 랭킹 목록 : 한 항목이 정확히 한 줄 */
        .hit-line {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            padding: 0.3rem 0;
            border-top: 1px solid #EAECF0;
            font-size: 0.82rem;
        }
        .hit-line:first-of-type { border-top: none; }

        /* 넘치는 만큼만 '...' 로 자른다. min-width:0 이 없으면 flex 항목이 안 줄어든다 */
        .hit-oneline {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #475467;
        }

        /* 2. 리랭킹 표의 이동 표시 */
        .rr-up   { color: #2F9E44; font-weight: 700; }
        .rr-down { color: #E03131; font-weight: 700; }
        .rr-same { color: #C1C7D0; }
        .rr-llm  { font-weight: 700; color: #344054; }
        .rr-reason {
            color: #667085;
            white-space: normal;      /* 근거 문장만 줄바꿈 허용 */
            min-width: 18rem;
        }
        .qtable tr.rr-picked td { background: rgba(76, 110, 245, 0.09); }

        /* 4. LLM 답변 */
        .answer-card {
            background: #FFFFFF;
            border-color: #B9C6FF;
        }
        .answer-text {
            font-size: 1.05rem;
            line-height: 1.65;
            font-weight: 500;
        }
        /* color 를 지정하지 않아야 .answer-text 와 같이 테마 글자색을 물려받는다 */
        .cite-row {
            margin-top: 0.7rem;
            font-size: 1.05rem;
            line-height: 1.65;
            font-weight: 500;
        }
        .cite {
            font-weight: 700;
            margin-left: 0.3rem;
        }
        .cite-none { margin-left: 0.3rem; }

        /* 5. 정답 비교 */
        .grade-card.grade-ok   { border-color: #8CE99A; background: #F4FCF5; }
        .grade-card.grade-no   { border-color: #FFA8A8; background: #FFF5F5; }
        .grade-card.grade-none { border-color: #D0D5DD; }

        .tag-verdict { font-weight: 700; }
        .tag-verdict.grade-ok   { background: rgba(47,158,68,.16);  color: #2B8A3E; }
        .tag-verdict.grade-no   { background: rgba(224,49,49,.14);  color: #C92A2A; }
        .tag-verdict.grade-none { background: #EAECF0; color: #667085; }

        .grade-row {
            display: flex;
            gap: 0.75rem;
            padding: 0.4rem 0;
            border-top: 1px solid rgba(0, 0, 0, 0.06);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .grade-row:first-of-type { border-top: none; }
        .grade-label {
            flex: 0 0 5.5rem;
            color: #667085;
            font-size: 0.85rem;
            font-weight: 600;
            padding-top: 0.1rem;
        }
        .grade-value { flex: 1; min-width: 0; }
        .grade-gold { font-weight: 700; }

        .tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.1rem 0.45rem;
            border-radius: 5px;
            background: #EAECF0;
            color: #475467;
            margin-left: 0.4rem;
            vertical-align: middle;
        }
        .tag-changed {
            background: rgba(76,110,245,.14);
            color: #3B5BDB;
        }

        /* ---- 데이터셋 & 모델 탭 (읽는 화면이라 데모 탭보다 한 단계 크게) ---- */
        .dhead {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 0.6rem;
        }
        .dtitle {
            font-size: 1.35rem;
            font-weight: 750;
            margin-bottom: 0.5rem;
        }
        .ddesc {
            font-size: 1.02rem;
            line-height: 1.7;
            color: #475467;
            margin-bottom: 1.1rem;
        }

        .dtable {
            width: 100%;
            border-collapse: collapse;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .dtable th {
            background: #F2F4F7;
            color: #475467;
            font-size: 0.95rem;
            font-weight: 700;
            text-align: left;
            padding: 0.6rem 0.9rem;
            border-bottom: 1px solid #D0D5DD;
        }
        .dtable td {
            padding: 0.72rem 0.9rem;
            border-bottom: 1px solid #EAECF0;
            vertical-align: top;
        }
        .dtable tr:last-child td { border-bottom: none; }
        .dtable td.dkey {
            width: 190px;
            font-weight: 700;
            color: #344054;
            background: #FCFCFD;
            white-space: nowrap;
        }
        .dtable td.dmodel {
            font-weight: 700;
            font-size: 1.05rem;
        }
        .dtable td.dnote, .dtable .dnote {
            color: #667085;
            font-size: 0.95rem;
            font-weight: 400;
        }
        .dtable .num, .dtable th.num {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        /* 문서 표만 열 너비를 따로 잡는다. .dkey 는 다른 표 네 곳에서도 쓰므로
           공용 값(190px)은 건드리지 않는다.

           코드는 'ko', 'en' 처럼 두 글자뿐이라 190px 이 크게 남는다. 줄인 만큼을
           제목과 설명이 함께 들어가는 문서 열이 가져가고, 원문 글자·청크 열은
           머리글이 두 줄로 접히지 않을 만큼 넓힌다. 언어·토큰 열은 내용에 맞춰
           그대로 둔다. */
        .dtable-docs td.dkey { width: 64px; }
        .dtable-docs th:nth-child(4),
        .dtable-docs td:nth-child(4) { width: 120px; }   /* 원문 글자 */
        .dtable-docs th:nth-child(5),
        .dtable-docs td:nth-child(5) { width: 92px; }    /* 청크 */
        .dtable-docs th.num { white-space: nowrap; }

        /* 원문 미리보기 라벨 */
        .dpreview-label {
            font-size: 0.98rem;
            color: #475467;
            margin-bottom: 0.45rem;
        }
        .dpreview-label code {
            font-size: 0.92rem;
            color: #344054;
            background: #F2F4F7;
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
        }

        /* 표 바로 아래 붙는 출처 */
        .dsource {
            margin-top: 1rem;
            padding-top: 0.85rem;
            border-top: 1px solid #EAECF0;
            font-size: 0.98rem;
            color: #475467;
        }
        .dsource-sub {
            margin-top: 0.25rem;
            font-size: 0.92rem;
            color: #98A2B3;
        }
        .dsource a { color: #3B5BDB; }

        div.stButton > button {
            height: 3rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 10px;
        }
        /* primary 버튼 색을 직접 박는다.
           .streamlit/config.toml 의 primaryColor 는 앱을 띄운 폴더 기준이라,
           공용 앱에 페이지로 얹히면 그쪽 테마가 이겨서 안 먹는다. */
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background-color: #4C6EF5 !important;
            border-color: #4C6EF5 !important;
            color: #FFFFFF !important;
        }
        div.stButton > button[kind="primary"]:hover,
        div.stButton > button[kind="primary"]:focus,
        div.stButton > button[kind="primary"]:active,
        div.stButton > button[data-testid="stBaseButton-primary"]:hover,
        div.stButton > button[data-testid="stBaseButton-primary"]:focus,
        div.stButton > button[data-testid="stBaseButton-primary"]:active {
            background-color: #3B5BDB !important;
            border-color: #3B5BDB !important;
            color: #FFFFFF !important;
            box-shadow: none !important;
        }
        .st-key-dataset_tab [data-stale="true"],
        .st-key-dataset_tab[data-stale="true"] {
            opacity: 1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 화면
# ---------------------------------------------------------
st.markdown('<h1 class="main-title">문서 AI 모델 데모</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">{escape(EMBED_SHORT)}, {escape(RERANKER_SHORT)}, '
    f'{escape(LLM_SHORT)} · 문서 {len(DOCUMENTS)}종 · 7개 언어</p>',
    unsafe_allow_html=True,
)

# GPU 서버에 못 붙으면 문서 목록도 질문 목록도 비어 화면이 텅 빈 채로 뜬다.
# 왜 비었는지 여기서 한 줄로 알려 준다.
if meta_error():
    st.error(f"GPU 서버({API_BASE})에 연결하지 못했습니다. 검색을 눌러도 "
             f"동작하지 않습니다. — {meta_error()}")
elif schema_mismatch():
    st.warning(schema_mismatch())

tab_demo, tab_data = st.tabs(["🔎  데모", "📚  데이터셋 & 모델"])

with tab_data:
    with st.container(key="dataset_tab"):
        render_dataset_tab()

with tab_demo:
    st.markdown("#### 처리 단계")

    pipeline_columns = st.columns(len(PIPELINE_STEPS), gap="small")
    for index, (column, step_name) in enumerate(
        zip(pipeline_columns, PIPELINE_STEPS),
        start=1,
    ):
        with column:
            st.markdown(
                f"""
                <div class="pipeline-card">
                    <div class="pipeline-number">STEP {index}</div>
                    <div class="pipeline-name">{step_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # st.form 을 쓰지 않는다. 폼 안에서는 위젯을 바꿔도 스크립트가 다시 돌지
    # 않으므로, 질문을 "직접 질문" 으로 고른 그 자리에서 입력칸을 띄울 수 없다.
    # 대신 위젯을 건드릴 때마다 화면이 다시 그려지므로, 마지막 결과를
    # session_state 에 남겨 두고 아래에서 다시 그린다.
    #
    # 테두리는 폼이 그려 주던 것을 컨테이너로 대신한다. key 를 주면 그 이름으로
    # .st-key-search_box 클래스가 붙어 결과 카드와 같은 모양으로 맞출 수 있다.
    # 직접 질문 입력칸이 열리면 박스가 그만큼 늘어난다.
    with st.container(border=True, key="search_box"):
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown(
                '<div class="section-label">1. 질문 선택</div>',
                unsafe_allow_html=True,
            )
            selected_question = st.selectbox(
                label="질문",
                options=QUESTION_OPTIONS,
                label_visibility="collapsed",
            )

        with right:
            st.markdown(
                '<div class="section-label">2. 검색 문서 선택</div>',
                unsafe_allow_html=True,
            )
            selected_document = st.selectbox(
                label="검색 문서",
                options=list(DOCUMENT_OPTIONS.keys()),
                label_visibility="collapsed",
            )

        # 목록에서 고른 질문이 곧 질의다. "직접 질문" 을 골랐을 때만 입력칸을 연다.
        if selected_question == CUSTOM_QUESTION:
            st.markdown(
                '<div class="section-label" style="margin-top:.8rem">'
                '직접 질문 입력</div>',
                unsafe_allow_html=True,
            )
            question = st.text_input(
                label="직접 질문",
                placeholder="예: 대마재배자는 누구에게 허가를 받나요?",
                label_visibility="collapsed",
            ).strip()
        else:
            question = selected_question

        st.write("")
        search_clicked = st.button(
            "검색",
            type="primary",
            use_container_width=True,
            disabled=not question,
        )

    if search_clicked:
        doc_key = DOCUMENT_OPTIONS[selected_document]
        doc_arg = None if doc_key == ALL_DOCS else doc_key

        # 5단계용 실제 정답. 직접 입력한 질문은 정답표에 없으므로 못 찾고,
        # 그러면 gold 가 비어 5단계를 건너뛴다.
        found = find_question(question)
        gold = list(found.keywords) if found else []

        # 이미 돌려 본 조합이면 GPU 를 다시 태우지 않는다. 같은 질문을 반복해
        # 보여주는 시연에서 매번 답변 생성을 다시 도는 것을 막는다.
        # 세션 단위라 새 탭이나 서버 재시작에는 남지 않는다.
        cache: dict = st.session_state.setdefault("results", {})
        cache_key = (question, doc_key)

        # 진행 상태 한 줄. 단계가 끝날 때마다 문구를 갈아 끼우고 마지막에 지운다.
        progress = st.empty()
        progress.info(
            f"질의를 {EMBED_SHORT} 로 임베딩해 {selected_document}에서 청크를 "
            "찾고 있습니다. (예상 시간: 20초)"
        )

        # 0단계는 카드 하나를 둘이 나눠 채운다. extract 결과를 들고 있다가
        # filter 가 끝나면 함께 그린다.
        stage0: dict = {}

        def on_stage(stage: str, payload) -> None:
            """단계가 끝날 때마다 불린다. 끝난 단계부터 바로 그린다."""
            if stage == "extract":
                stage0["extract"] = payload
                progress.info("뽑은 조건으로 후보 청크를 좁히고 있습니다.")

            elif stage == "filter":
                render_meta(stage0.get("extract"), payload)   # 0번
                progress.info(
                    f"후보 {payload.n_kept:,}개 안에서 질의를 {EMBED_SHORT} 로 "
                    "임베딩해 청크를 찾고 있습니다."
                )

            elif stage == "search":
                render_search(payload)           # 1번
                progress.info(
                    f"후보 {len(payload.hits)}개를 {RERANKER_MODEL} 로 재점수하고 "
                    "있습니다."
                )

            elif stage == "rerank":
                render_rerank(payload)           # 2번
                render_selected(payload)         # 3번
                progress.info(
                    f"{LLM_MODEL} 이 근거 청크를 읽고 답변을 만들고 있습니다. "
                    "(예상 시간: 1분)"
                )

            elif stage == "answer":
                render_answer(payload)           # 4번
                if gold:
                    progress.info("실제 정답과 견주고 있습니다.")
                else:
                    progress.empty()

            elif stage == "grade":
                progress.empty()
                render_grade(payload)            # 5번

        if cache_key in cache:
            # 캐시 히트. 단계별로 그려 줄 콜백이 안 불리므로 한꺼번에 그린다.
            progress.empty()
            result = cache[cache_key]
            render_all(result)
            note = (f"이전 실행 결과를 재사용했습니다 "
                    f"(원래 {result.elapsed:.1f}초). "
                    f"다시 계산하려면 페이지를 새로 고칩니다.")
        else:
            result = run_pipeline(question, doc=doc_arg, gold=gold,
                                  on_stage=on_stage)
            note = ""
            # 실패한 결과는 캐시하지 않는다. 일시적인 OOM 이나 파싱 실패가
            # 캐시에 박히면 다시 눌러도 계속 그 결과만 나온다.
            if not result.errors():
                cache[cache_key] = result

        # 질문을 바꾸거나 입력칸에 타자를 치면 스크립트가 처음부터 다시 도는데,
        # 결과 카드는 이 블록 안에서만 그려지므로 그때 화면에서 사라진다.
        # 마지막 결과를 남겨 두었다가 아래 elif 에서 되살린다.
        st.session_state["last_run"] = (result, gold)
        render_tail(result, gold, note)

    elif st.session_state.get("last_run"):
        # 검색을 누른 게 아니라 위젯을 건드려 다시 그려진 경우. 직전 결과를
        # 그대로 되살린다. 파이프라인은 다시 돌지 않는다.
        result, gold = st.session_state["last_run"]
        render_all(result)
        render_tail(result, gold,
                    f"지난 검색 결과입니다 · {result.question} "
                    f"({result.doc_name}). 검색을 누르면 새로 돌립니다.")
