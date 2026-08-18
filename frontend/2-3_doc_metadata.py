"""
화면 전용 데모 앱 (CPU 서버) - 계산은 GPU 서버가 한다.

    [이 서버 : CPU]                          [GPU 서버]
    frontend/2-3_doc_metadata.py --- HTTP --> backend/src/server.py
    frontend/rag_api.py                        0 조건 추출  Qwen3-4B + 규칙
                      <--- 단계별 ------        1 질의 임베딩 BAAI/bge-m3
                           NDJSON               2 리랭킹     bge-reranker-v2-m3
                                                4 답변 생성  Qwen/Qwen3-4B
                                              + data/ (대화 로그·색인·정답표)

이 서버에는 모델도 색인도 없다. 코퍼스 목록·질문 세트·모델 이름은 /rag/meta 에서
받아 오고, 검색은 /rag/search 로 넘긴다. 필요한 패키지는 streamlit 과 requests
뿐이다.

    RAG_API_BASE   GPU 서버 주소 (기본 http://147.46.15.89:58567)
    RAG_API_KEY    설정돼 있으면 X-API-Key 로 보낸다

    streamlit run 2-3_doc_metadata.py --server.address 0.0.0.0 --server.port 8501
"""

import re
import sys
from html import escape
from pathlib import Path

import streamlit as st

# rag_api.py 는 이 파일 옆에 있다. 어느 경로에서 띄우든 찾도록 직접 넣는다
# (streamlit 이 스크립트 폴더를 넣어 주긴 하지만, 옛 판본이 다른 폴더에 남아
# 있으면 그쪽이 먼저 잡혀 화면만 옛것으로 도는 일이 있었다).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
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
    page_title="문서 메타데이터 테스트 페이지",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# 선택 항목
#
# 질문은 data/qa/qa.json, 코퍼스는 data/emb 에서 읽는다. 앱에는 목록을 적어
# 두지 않는다. 코퍼스나 질문을 늘려도 코드를 고칠 일이 없게 하려는 것이다.
# ---------------------------------------------------------
DOCUMENTS = documents()

# 화면에 보일 이름 -> 파이프라인에 넘길 코퍼스 키. 첫 항목이 기본값이다.
# 전체 검색이 기본인 이유: 같은 대화가 러시아어 원문과 영어 번역 두 벌로 들어
# 있어, 어느 벌에서 걸리는지까지 보여 주는 편이 데모로서 정직하다.
DOCUMENT_OPTIONS = {f"전체 문서 {len(DOCUMENTS)}종": ALL_DOCS}
DOCUMENT_OPTIONS.update({doc.label: doc.key for doc in DOCUMENTS})


QUESTIONS = load_questions()

# 목록 맨 앞에 두는 항목. 이걸 고르면 아래에 입력칸이 열리고, 나머지 질문은
# 한 칸씩 밀린다. 정답표에 없는 질문이라 5단계(정답 비교)는 건너뛰게 된다.
CUSTOM_QUESTION = "직접 질문"
QUESTION_OPTIONS = [CUSTOM_QUESTION] + [q.label for q in QUESTIONS]

PIPELINE_STEPS = [
    "메타데이터 추출",
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

# 0단계 카드에 적는 칸. LLM 이 뽑아 주는 JSON 의 키 그대로이고, 값이 비어도
# 줄은 남긴다 — 무엇을 못 뽑았는지가 뽑은 것만큼 중요하다.
META_FIELDS = (
    ("sender", "sender"),
    ("receiver", "receiver"),
    ("participants", "participants"),
    ("since", "since"),
    ("until", "until"),
    ("keywords", "keywords"),
)



def _kv_text(value) -> str:
    """JSON 값 하나를 한 줄 문자열로. 빈 값은 빈 문자열."""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def _kv_map(data: dict | None) -> dict:
    """JSON(또는 MetaQuery 를 옮긴 dict) 을 칸 이름 -> 한 줄 로 만든다."""
    data = data or {}
    return {key: _kv_text(data.get(key)) for key, _ in META_FIELDS}


def _kv_block(values: dict, title: str, hints: bool = False) -> str:
    """key : value 를 한 줄씩. 네 칸을 늘 다 그린다."""
    lines = "".join(
        f'<div class="kv-line">'
        f'<span class="kv-key">{escape(label)}</span>'
        f'<span class="kv-val{"" if values.get(key) else " kv-empty"}">'
        f'{escape(values.get(key) or "—")}</span>'
        + '</div>'
        for key, label in META_FIELDS
    )
    return (f'<div class="kv-block"><div class="kv-head">{escape(title)}</div>'
            f'{lines}</div>')


def render_meta(ex, fr, nr=None) -> None:
    """
    0. 조건 추출 + 후보 좁히기 — LLM 이 뱉은 JSON 을 그대로 네 줄로 보여 준다.

    화면에 두는 것은 두 가지뿐이다.
        1) 조건 여섯 칸  sender / receiver / participants / since / until /
                        keywords — 검증을 마치고 실제로 검색에 쓴 값이다
        2) 몇 개로 좁혔는지  '필터링' 한 줄
    4B 가 적어 보낸 원문 JSON 과 나머지 설명은 카드 밑 접이칸으로 내렸다.
    버려진 이름이 있으면 그것만 한 줄로 따로 알린다.
    """
    if ex is None:
        return

    q = ex.query
    used = _kv_map({"sender": q.sender, "receiver": q.receiver,
                    "participants": q.participants, "since": q.since,
                    "until": q.until, "keywords": q.keywords})

    if ex.source == "off":
        model_out = _kv_map(None)
        title = "조건 추출을 껐습니다 (전체 청크를 그대로 뒤집니다)"
    elif ex.llm:
        model_out = _kv_map(ex.llm)
        title = f"{LLM_SHORT} 가 출력한 JSON"
    else:
        model_out = _kv_map(ex.raw)
        title = "규칙이 뽑은 값 (4B 를 못 썼습니다)"

    # 블록은 늘 한 벌이다. 실제로 검색에 쓴 값(명부 대조와 날짜 펴기를 마친
    # 것)을 보여 준다. 4B 가 적어 보낸 것과 갈린 부분은 아래 '버린 이름' 줄과
    # 카드 밑 '원본 JSON' 접이칸이 말해 준다. 같은 여섯 줄을 두 벌 그리면
    # 카드만 두 배로 길어지는데 정작 다른 칸은 보통 하나다.
    blocks = _kv_block(model_out if ex.source == "off" else used, title,
                       hints=True)

    dropped = ""
    if q.unknown:
        dropped = (f'<div class="query-note">명부(nicks.json)에 없어 버린 이름 · '
                   f'{escape(", ".join(q.unknown))}</div>')

    funnel = ""
    if nr is not None:
        def count(value):
            return "-" if value is None else f"{value:,}"

        both = nr.n_both if nr.n_both is not None else nr.n_used
        line = (f"청크 {nr.n_chunks:,} → 메타 {count(nr.n_meta)}"
                f" ∩ 키워드 {count(nr.n_keyword)} → {count(both)}")
        # 교집합이 비면 좁히기가 한쪽 조건을 풀고 넓힌다. 그때 마지막 숫자가
        # 0 이면 "검색할 게 없었다" 로 읽히는데, 실제로는 푼 쪽으로 검색했다.
        # 무엇으로 몇 개를 뒤졌는지까지 적는다.
        if nr.n_both is not None and nr.n_both != nr.n_used:
            line += f" → 실제 후보 {nr.n_used:,} ({escape(nr.step)})"
        funnel = (
            f'<div class="kv-line">'
            f'<span class="kv-key">필터링</span>'
            f'<span class="kv-val">{line}</span>'
            f'</div>'
        )

    tags = [f'<span class="tag">{escape(ex.source)}</span>',
            f'<span class="tag">{ex.elapsed:.1f}초</span>']
    if nr is not None:
        tags.insert(0, f'<span class="tag">청크 {nr.n_chunks:,}개 → '
                       f'{nr.n_used:,}개</span>')
    elif fr is not None:
        tags.insert(0, f'<span class="tag">청크 {fr.n_total:,}개 → '
                       f'{fr.n_kept:,}개</span>')

    relaxed = ""
    if fr is not None and fr.relaxed:
        relaxed = (f'<div class="query-note">조건이 너무 좁아 '
                   f'{escape(" → ".join(fr.relaxed))} 을(를) 풀었습니다.</div>')

    error = ""
    if ex.error:
        error = (f'<div class="query-note">추출 중 오류 · '
                 f'{escape(ex.error)}</div>')

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">1. 필터링
                {"".join(tags)}</div>
            {blocks}
            {funnel}
            {dropped}
            {relaxed}
            {error}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("원본 JSON"):
        st.json({"llm": ex.llm, "rule": ex.rule, "merged": ex.raw,
                 "used": {"people": q.people, "unknown": q.unknown,
                          "since": q.since, "until": q.until,
                          "keywords": q.keywords}})
        st.caption(
            "4B 가 질문에서 사람·기간·키워드를 뽑고, 두 채널(메타·키워드)이 각각 "
            "청크를 고른 뒤 교집합 안에서만 검색합니다. 사람은 닉네임 명부에 있는 "
            "것만 씁니다(없는 이름은 버립니다). 보낸 사람과 받는 사람은 나누지 "
            "않습니다 — 찾는 것은 그 사람이 낀 대화라 대화쌍(dyad) 안에 있는지만 "
            "봅니다. 키워드는 본문에 글자 그대로 있는지를 봅니다(IP·파일명처럼 "
            "임베딩이 제일 못하는 것). 한쪽이 0개가 되면 그 조건을 풀고 넓힙니다."
        )


def render_search(sr) -> None:
    """2. 검색 랭킹 — 질문을 임베딩해 뽑은 상위 청크. 한 항목이 정확히 한 줄."""
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

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">2. 검색 랭킹
                <span class="tag">{escape(sr.doc_label)}</span>
                <span class="tag">{sr.pool} → 상위 {sr.top_k}개</span>
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
    """3. 리랭킹 — 등수가 어떻게 바뀌었는지"""
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
            <div class="card-title">3. 리랭킹
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
                유효합니다. 질의와 청크를 같이 읽으므로 "낱말만 겹치는 잡담 청크"와
                "증거가 실제로 든 대화 청크"를 더 잘 가릅니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected(rr) -> None:
    """4. 최종 청크 선정"""
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
            <div class="card-title">4. 최종 청크 선정
                <span class="tag">{len(rr.ranked)}개 → {len(rr.selected)}개</span>
                <span class="tag">약 {len(rr.selected) * CHUNK_SIZE}토큰</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(ans) -> None:
    """5. LLM 답변"""
    if ans is None:
        return
    if not ans.ok:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">5. LLM 답변
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
            <div class="card-title">5. LLM 답변 {badge}
                <span class="tag">{ans.elapsed:.1f}초</span></div>
            <div class="answer-text">{escape(ans.answer)}</div>
            <div class="cite-row">근거 {cites}</div>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grade(gr, selected=None) -> None:
    """
    6. 정답 비교 — 최종 청크에 정답 청크가 들었는지.
    """
    if gr is None:
        return

    if gr.verdict == "오답":
        style, mark = "grade-no", "X"
    else:
        style, mark = "grade-none", "O" if gr.correct else "?"

    by_chunk = gr.method.startswith("최종 청크")
    if by_chunk:
        picked = ", ".join(dict.fromkeys(
            f"#{h.chunk_index}" for h in (selected or []))) or "—"
        rows = [
            ("정답 청크", ", ".join(gr.candidates) or "—"),
            ("최종 청크", picked),
            ("겹친 청크", f"{mark} " + (", ".join(gr.matched) or "없음")),
            ("LLM 답변", gr.llm_answer or "—"),
        ]
    else:
        rows = [
            ("LLM 답변", gr.llm_answer or "—"),
            ("실제 정답", gr.gold_display or "—"),
            ("정답 후보", ", ".join(gr.candidates) or "—"),
            ("겹친 후보", f"{mark} " + (", ".join(gr.matched) or "없음")),
        ]

    body = "".join(
        f'<div class="grade-row">'
        f'<span class="grade-label">{escape(label)}</span>'
        f'<span class="grade-value">{escape(value)}</span>'
        f'</div>'
        for label, value in rows
    )

    st.markdown(
        f"""
        <div class="result-card grade-card {style}">
            <div class="card-title">6. 정답 비교
                <span class="tag tag-verdict {style}">{mark} {gr.verdict}</span>
                <span class="tag">{escape(gr.method)} 판정</span></div>
            {body}
            <div class="query-note">{escape(gr.reason)}</div>
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
        render_meta(result.extract, result.filter, result.narrow)   # 0
    if result.search:
        render_search(result.search)                        # 1
    if result.rerank:
        render_rerank(result.rerank)                        # 2
        render_selected(result.rerank)                      # 3
    render_answer(result.answer)                            # 4
    render_grade(result.grade, result.selected)             # 5


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
            "narrow": None if result.narrow is None else {
                "n_chunks": result.narrow.n_chunks,
                "n_meta": result.narrow.n_meta,
                "n_keyword": result.narrow.n_keyword,
                "n_both": result.narrow.n_both,
                "n_used": result.narrow.n_used,
                "step": result.narrow.step,
                "keywords": result.narrow.keywords,
            },
        },
        "1_search": {
            "query": sr.query,
            "n_indexed": sr.n_indexed,
            "n_candidates": sr.n_candidates,
            "n_candidate_chunks": sr.n_candidate_chunks,
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
                <div class="dtitle">Conti Jabber 로그</div>
                <div class="ddesc">Conti 랜섬웨어 조직의 Jabber 1:1 대화 로그입니다
                    (2020-06-21 ~ 11-16 · 106,566행 · 닉네임 289명 · 대화쌍 1,114개).
                    같은 대화가 러시아어 원문과 영어 번역 두 벌로 들어 있고, 청크 경계를
                    영어 기준으로 한 번만 계산했기 때문에 두 벌의 청크 번호가 서로
                    같습니다. 질문은 한국어로 던지고 번역 없이 교차 검색합니다.</div>
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
            "코퍼스 목록을 받지 못했습니다. GPU 서버가 떠 있는지, 청크"
            "(`backend/data/chunks`)와 색인(`backend/data/emb`)이 있는지 "
            "확인하세요."
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- 전처리 ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <div class="dtitle">전처리</div>
            {_kv_table([
                ("정제",
                 "원본 CSV(107,967행)에서 HTML 엔티티·앞뒤 공백·시스템 메시지·"
                 "그룹채팅방(conference) 22행·화자 중복 270행을 걷어내 "
                 "106,566행을 남깁니다. 짧다고 버리지 않습니다 — 세션 200자 "
                 "필터를 걸면 BTC 주소의 43%, IP 의 26% 가 사라집니다."),
                ("세션 분리",
                 "메시지 1건은 중앙값 20자라 그대로 임베딩하면 'Ok' 3,118개가 "
                 "같은 벡터가 됩니다. 대화쌍(dyad)별로 모은 뒤 1시간 gap 으로 "
                 "끊어 12,478 세션을 만듭니다."),
                ("청킹",
                 f"{CHUNK_SIZE} / {OVERLAP}토큰 — 세션 안에서만 bge-m3 "
                 f"토크나이저로 {CHUNK_SIZE}토큰마다 자르고 {OVERLAP}토큰을 "
                 f"겹칩니다. 자르는 자리는 발화 경계로 스냅합니다. "
                 f"512토큰을 넘는 세션은 7% 뿐입니다."),
                ("두 벌의 청크 번호가 같은 이유",
                 "청크 경계를 영어 기준으로 한 번만 계산해 두 언어에 그대로 "
                 "적용합니다(--split-basis en). 그래서 정답(청크 위치)도, "
                 "메타데이터도 한 벌만 있으면 됩니다."),
                ("청크 메타데이터",
                 "청크마다 대화쌍(dyad)과 걸친 구간(ts_start~ts_end)을 따로 "
                 "저장합니다(chunk_meta.npz). 0단계 필터가 읽는 것이 이것이고, "
                 "벡터에는 들어가지 않습니다. 보낸 사람과 받는 사람은 나누지 "
                 "않습니다 — 찾는 것은 그 사람이 낀 대화입니다."),
                ("닉네임 명부",
                 "289명을 nicks.json 에 모아 둡니다. 0단계가 뽑은 이름은 이 "
                 "명부에 있는 것만 씁니다(4B 가 지어낸 이름을 여기서 버립니다)."),
                ("정규화",
                 "벡터를 L2 정규화해 저장합니다. 검색이 내적을 그대로 코사인 "
                 "유사도로 쓸 수 있습니다."),
            ])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- 질문-정답 세트 --------------------------------------------------
    if QUESTIONS:
        # 정답 후보(keywords)가 있어야 5단계가 돈다. 지금 Jabber 세트는 정답을
        # 근거 청크 번호(answer_chunk_indices)로만 들고 있어 비어 있고, 그때는
        # 빈 칸 두 줄을 그리는 대신 표를 줄이고 왜 비었는지 한 줄로 적는다.
        gradable = [q for q in QUESTIONS if q.keywords or q.answer]

        if gradable:
            head = ("<tr><th>번호</th><th>질문 · 정답</th>"
                    "<th>정답 후보</th></tr>")
            body = "".join(
                f'<tr><td class="dkey">{q.id}</td>'
                f'<td>{escape(q.question)}'
                + (f'<div class="dnote">정답 · {escape(q.answer)}</div>'
                   if q.answer else "")
                + f'</td>'
                f'<td class="dnote">{escape(" / ".join(q.keywords))}</td></tr>'
                for q in QUESTIONS
            )
            note = ("5단계는 정답 후보가 답변 안에 하나라도 들어 있는지만 "
                    "봅니다.")
        else:
            head = "<tr><th>번호</th><th>질문</th></tr>"
            body = "".join(
                f'<tr><td class="dkey">{q.id}</td>'
                f'<td>{escape(q.question)}</td></tr>'
                for q in QUESTIONS
            )
            note = ("정답이 근거 청크 번호(<code>answer_chunk_indices</code>)로만 "
                    "적혀 있고 답변에서 맞춰 볼 문자열이 없어, 지금은 "
                    "5단계(정답 비교)를 건너뜁니다.")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="dtitle">질문-정답 세트 {len(QUESTIONS)}문항</div>
                <div class="ddesc">대화 로그를 읽고 만든 세트입니다
                    (<code>backend/data/qa/qa.json</code>). 세 유형을 5문항씩
                    두었습니다 — 날짜+키워드 · 화자+키워드 · 대화쌍+키워드.
                    앞의 조건이 0단계가 뽑아야 할 것이고, 키워드는 본문에 글자
                    그대로 나오는 것(파일명·랜섬웨어 이름·IP)입니다. 질문은
                    한국어이고 근거는 러시아어·영어이므로 그대로 돌리면 교차 언어
                    검색 평가가 됩니다. {note}</div>
                <table class="dtable">
                    <thead>{head}</thead>
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

        /* 0. 조건 추출 : LLM 이 뱉은 JSON 을 key : value 한 줄씩 */
        .kv-block { margin: 0.35rem 0 0.15rem; }
        .kv-head {
            font-size: 0.86rem;
            color: #475467;
            margin-bottom: 0.25rem;
        }
        .kv-line {
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
            padding: 0.26rem 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.98rem;
        }
        /* 키와 값의 농도를 같이 간다. 한쪽만 연하면 읽는 사람이 그 줄을
           덜 중요한 것으로 읽는데, 여기서는 못 뽑은 칸도 뽑은 칸만큼
           중요하다 (비어 있다는 사실이 곧 정보다). 굵기는 주지 않는다 —
           여섯 줄이 전부 굵으면 강조가 아니라 그냥 읽기 힘든 덩어리가 된다. */
        .kv-key {
            min-width: 7rem;
            color: #1D2939;
            font-weight: 400;
        }
        .kv-val {
            color: #1D2939;
            font-weight: 400;
            overflow-wrap: anywhere;
        }
        .kv-val.kv-empty { color: #1D2939; font-weight: 400; }
        /* 칸 뜻풀이. 자리가 모자라면 잘린다 (값이 먼저다) */
        .kv-hint {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: inherit;
            font-size: 0.75rem;
            color: #98A2B3;
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
        /* 정답을 초록으로 칠하지 않는다 (render_grade 설명 참고).
           틀렸을 때만 색을 준다. */
        .grade-card.grade-no   { border-color: #FFA8A8; background: #FFF5F5; }
        .grade-card.grade-none { border-color: #D0D5DD; }

        .tag-verdict { font-weight: 700; }
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
st.markdown('<h1 class="main-title">문서 메타데이터 테스트 페이지</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">{escape(EMBED_SHORT)}, {escape(RERANKER_SHORT)}, </p>',
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
                placeholder="2020-08-20 부터 2020-08-22 사이에 rdpscanDll에 대한 증거를 찾아주세요",
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

        # 5단계용 정답. 지금 정답표는 모범 답안이 아니라 정답 청크 번호를
        # 들고 있어서, 최종 선정 청크에 그 번호가 들었는지로 채점한다.
        # 직접 입력한 질문은 정답표에 없으므로 못 찾고 5단계를 건너뛴다.
        found = find_question(question)
        gold = list(found.keywords) if found else []
        gold_chunks = list(found.answer_chunks) if found else []

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

        # 0단계는 카드 하나를 셋이 나눠 채운다(추출 -> 메타 -> 키워드·교집합).
        # 앞 둘을 들고 있다가 narrow 가 끝나면 한 번에 그린다.
        stage0: dict = {}

        def on_stage(stage: str, payload) -> None:
            """단계가 끝날 때마다 불린다. 끝난 단계부터 바로 그린다."""
            if stage == "extract":
                stage0["extract"] = payload
                progress.info("뽑은 조건에 해당하는 청크를 세고 있습니다.")

            elif stage == "filter":
                stage0["filter"] = payload

            elif stage == "narrow":
                # 0번 카드는 추출·메타·키워드 세 결과가 다 모여야 그릴 수 있다.
                render_meta(stage0.get("extract"), stage0.get("filter"),
                            payload)
                progress.info(
                    f"후보 {payload.n_used:,}개 안에서 질의를 {EMBED_SHORT} 로 "
                    "임베딩해 청크를 찾고 있습니다."
                )

            elif stage == "search":
                render_search(payload)           # 1번
                progress.info(
                    f"후보 {len(payload.hits)}개를 {RERANKER_MODEL} 로 재점수하고 "
                    "있습니다."
                )

            elif stage == "rerank":
                stage0["selected"] = payload.selected
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
                # 채점 카드에 '최종 청크' 줄을 그리려면 3번 결과가 필요하다.
                render_grade(payload, stage0.get("selected"))   # 5번

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
                                  gold_chunks=gold_chunks, on_stage=on_stage)
            note = ""
            # 실패한 결과는 캐시하지 않는다. 일시적인 OOM 이나 파싱 실패가
            # 캐시에 박히면 다시 눌러도 계속 그 결과만 나온다.
            if not result.errors():
                cache[cache_key] = result

        # 질문을 바꾸거나 입력칸에 타자를 치면 스크립트가 처음부터 다시 도는데,
        # 결과 카드는 이 블록 안에서만 그려지므로 그때 화면에서 사라진다.
        # 마지막 결과를 남겨 두었다가 아래 elif 에서 되살린다.
        st.session_state["last_run"] = (result, gold or gold_chunks)
        render_tail(result, gold or gold_chunks, note)

    elif st.session_state.get("last_run"):
        # 검색을 누른 게 아니라 위젯을 건드려 다시 그려진 경우. 직전 결과를
        # 그대로 되살린다. 파이프라인은 다시 돌지 않는다.
        result, gold = st.session_state["last_run"]
        render_all(result)
        render_tail(result, gold,
                    f"지난 검색 결과입니다 · {result.question} "
                    f"({result.doc_name}). 검색을 누르면 새로 돌립니다.")
