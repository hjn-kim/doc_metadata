#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5. 정답 비교 (+ 질문-정답 세트 로더)

data/qa.json 에 적어 둔 질문-정답 세트를 읽고, 4단계 답변이 정답 후보를 담고
있는지 본다. 문자열 포함 여부만 보므로 모델 호출이 없다.

    data/qa.json
        {"questions": [
            {"id": 1,
             "doc": "ko마약류관리에관한법률",     # 답이 실제로 들어 있는 문서
             "question": "마약을 수출입하거나 ...",
             "answer": "제58조 제1항에 따라 무기 또는 5년 이상의 징역에 처한다.",
             "keywords": ["5년 이상의 징역", "무기징역"]},
            ...
        ]}

질문과 정답은 한국어이고 원문은 7개 언어다. 그래서 이 세트를 그대로 돌리면
교차 언어 검색 평가가 된다 (한국어 질문 -> 러시아어 원문 -> 한국어 답변).

판정 규칙 (any-include):
    keywords 중 하나라도 답변 안에 들어 있으면 정답. 비교 전에 양쪽을 정규화한다
    (NFKC, 소문자, 공백/문장부호 제거). "84,076" = "84076", "Power BI" = "PowerBI".

왜 후보를 여러 개 두나:
    답이 한 낱말로 고정되지 않는다. 같은 사실을 모델이 다르게 옮겨 적는다.
        정답 "시날로아 카르텔"  <-> 답변 "...Sinaloa Cartel과 CJNG입니다."
        정답 "무기 또는 5년"    <-> 답변 "...무기징역 또는 5년 이상의 징역."
    낱말 하나만 두면 맞은 답이 전부 오답이 되고, 너무 짧게 잡으면 엉뚱한 것이
    통과한다. 그래서 표기가 갈릴 만한 지점마다 후보를 적어 둔다.

단독 실행:
    python src/grade.py                 # 질문-정답 세트를 훑는다
    python src/grade.py --run 1         # 1번 질문을 파이프라인에 태우고 채점
    python src/grade.py --run all       # 21개 전부 + 정답률
    python src/grade.py --run all --doc all   # 정답 문서로 좁혀서도 한 번 더
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# 질문-정답 세트. 예전에는 data/qa.json 하나였고 지금은 data/qa/ 밑에 있다.
# 둘 다 찾아 본다 — 파일을 못 찾으면 앱의 질문 목록이 조용히 비어 버린다.
QA_CANDIDATES = (ROOT / "data" / "qa" / "qa.json", ROOT / "data" / "qa.json")
QA_PATH = next((p for p in QA_CANDIDATES if p.is_file()), QA_CANDIDATES[0])


@dataclass
class Question:
    """질문-정답 세트 한 항목."""

    id: int
    doc: str                                              # 정답이 든 문서 키
    question: str
    answer: str = ""                                      # 사람이 쓴 모범 답안
    keywords: list[str] = field(default_factory=list)     # 문자열 판정용 후보
    # 정답 청크 번호. qa.json 의 answer_chunk_indices 다. 같은 근거가
    # 512/128 overlap 으로 여러 청크에 걸쳐 있어 목록으로 온다.
    answer_chunks: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        """화면 선택 상자에 넣을 문자열. 앞의 번호는 파이프라인이 떼고 쓴다."""
        return f"{self.id}. {self.question}"


@dataclass
class GradeResult:
    """5단계 결과."""

    question: str
    llm_answer: str = ""
    candidates: list[str] = field(default_factory=list)   # 정답 후보 전체
    matched: list[str] = field(default_factory=list)      # 답변에 실제로 있던 후보
    verdict: str = ""            # "정답" | "오답" | "판정 불가"
    reason: str = ""
    gold_answer: str = ""        # 사람이 쓴 모범 답안 (화면에 같이 보여준다)
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
        """
        화면의 '실제 정답' 칸.

        모범 답안이 있으면 그것을, 없으면 후보 목록을 보여준다.
        """
        if self.gold_answer:
            return self.gold_answer
        return ", ".join(self.matched if self.correct else self.candidates)


# --------------------------------------------------------------------------
# 질문-정답 세트
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_questions() -> tuple[Question, ...]:
    """
    data/qa.json 을 읽는다.

    파일이 없거나 깨졌으면 빈 목록을 돌려준다 (정답 세트가 없어도 1~4 단계는
    굴러가야 한다. 앱에서 질문을 직접 입력하는 경로가 그 경우다).
    """
    try:
        with QA_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return ()

    # 판본에 따라 키가 다르다. 옛 법률 세트는 questions, 지금 Jabber 세트는
    # qa_pairs 다. 둘 다 받는다.
    rows = data.get("qa_pairs") or data.get("questions") or []

    items: list[Question] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not row.get("question"):
            continue
        words = row.get("keywords") or []
        if isinstance(words, str):
            words = [words]
        items.append(Question(
            # id 가 'jabber-007' 처럼 문자열일 수 있다. 숫자만 뽑아 쓰고,
            # 숫자가 없으면 순번을 준다 (앱이 번호로 질문을 되찾는다).
            id=_question_id(row.get("id"), i),
            doc=str(row.get("doc") or ""),
            question=str(row["question"]).strip(),
            answer=str(row.get("answer") or "").strip(),
            keywords=[str(w).strip() for w in words if str(w).strip()],
            answer_chunks=[int(c) for c in
                           (row.get("answer_chunk_indices") or [])
                           if str(c).lstrip("-").isdigit()],
        ))
    return tuple(items)


def _question_id(value, fallback: int) -> int:
    if isinstance(value, int):
        return value
    digits = re.findall(r"\d+", str(value or ""))
    return int(digits[-1]) if digits else fallback


def questions_for(doc: str | None = None) -> list[Question]:
    """문서 하나에 딸린 질문만. doc 이 없으면 전부."""
    items = load_questions()
    if not doc:
        return list(items)
    return [q for q in items if q.doc == doc]


def question_by_id(index: int) -> Question | None:
    for q in load_questions():
        if q.id == index:
            return q
    return None


def find_question(text: str) -> Question | None:
    """
    화면에서 고른 문자열로 질문을 되찾는다.

    선택 상자에는 "3. 이 법에서 말하는..." 처럼 번호가 붙어 있으므로 label 과
    본문 양쪽으로 견준다. 직접 입력한 질문이면 아무것도 못 찾고 None 이 되며,
    그때는 5단계를 건너뛴다.
    """
    clean = (text or "").strip()
    if not clean:
        return None
    for q in load_questions():
        if clean in (q.label, q.question):
            return q
    return None


def gold_for(text_or_id: str | int) -> list[str]:
    """질문(문자열 또는 번호)의 정답 후보 목록. 없으면 빈 목록."""
    found = (question_by_id(text_or_id) if isinstance(text_or_id, int)
             else find_question(text_or_id))
    return list(found.keywords) if found else []


# --------------------------------------------------------------------------
# 판정
# --------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    비교용으로 다듬는다. NFKC 로 전각/반각을 통일하고, 소문자로 내리고, 공백과
    문장부호를 지운다. \\W 는 유니코드 모드에서 한글/한자/키릴을 글자로 본다.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


@lru_cache(maxsize=32)
def _group_of(doc_key: str) -> str:
    """
    문서 키 -> 청크 번호가 유일한 묶음 이름. 모르는 문서면 빈 문자열.

    jabber_ru 와 jabber_en 은 같은 대화의 두 판본이라 한 묶음("jabber")이고,
    ko_voice 는 자기 혼자 한 묶음이다 (corpus.py 의 Document.group).
    """
    if not doc_key:
        return ""
    try:
        from corpus import find
        doc = find(doc_key)
    except (KeyError, ImportError):
        return ""
    return doc.group if doc else ""


def grade_chunks(question: str, selected, gold_chunks,
                 llm_answer: str = "", gold_doc: str = "") -> GradeResult:
    """
    최종 선정된 청크에 정답 청크가 하나라도 들었으면 정답으로 본다.

    답변 문장을 대조하지 않는 이유:

        지금 정답표(data/qa/qa.json)가 들고 있는 것은 모범 답안이 아니라
        answer_chunk_indices 다. 근거를 제대로 찾아왔는지를 재는 세트이지
        문장을 얼마나 잘 옮겨 적었는지를 재는 세트가 아니다. 검색·리랭킹까지가
        이 파이프라인에서 고칠 수 있는 부분이기도 하다.

    하나만 들어도 정답으로 치는 이유:

        같은 근거가 512/128 overlap 으로 여러 청크에 겹쳐 들어가 있어서
        정답 청크가 2~5개씩이다. 그중 하나만 근거로 들어가도 답은 나온다.

    청크 번호는 언어와 무관하다. ru#93 과 en#93 은 같은 대화의 두 판본이라
    둘 중 무엇이 뽑혀도 93 으로 센다.

    다만 **문서가 다르면 같은 번호라도 다른 청크다.** 청크 번호는 문서 안에서만
    유일해서 ko_voice 에도 93번이 있고 jabber 에도 93번이 있다. 번호만 보고
    채점하면 jabber 문항(정답 [93, 94])에 ko#93 이 뽑혀 와도 정답이 된다.
    gold_doc 을 주면 그 문서와 같은 묶음에서 나온 청크만 센다.

    gold_doc 을 안 주면 예전처럼 번호만 본다 (문서가 하나뿐인 옛 호출부).
    """
    started = time.time()
    gold = [int(c) for c in (gold_chunks or [])]
    gold_group = _group_of(gold_doc)

    picked, dropped = [], 0
    for hit in selected or []:
        index = getattr(hit, "chunk_index", hit)
        if gold_group:
            key = getattr(hit, "doc_key", "")
            # doc_key 가 없으면 그냥 번호만 온 것이다. 판단할 근거가 없으니
            # 예전대로 센다 (여기서 버리면 옛 호출부가 전부 오답이 된다).
            if key and _group_of(key) != gold_group:
                dropped += 1
                continue
        picked.append(int(index))

    hit_chunks = [c for c in gold if c in picked]
    result = GradeResult(
        question=question,
        llm_answer=llm_answer or "",
        candidates=[f"#{c}" for c in gold],
        matched=[f"#{c}" for c in hit_chunks],
        gold_answer="",
        method="최종 청크 포함",
    )

    if not gold:
        result.verdict = "판정 불가"
        result.reason = "이 질문에는 정답 청크가 적혀 있지 않습니다."
    elif hit_chunks:
        result.verdict = "정답"
        result.reason = (f"최종 청크에 정답 청크 "
                         f"{', '.join(f'#{c}' for c in hit_chunks)} 가 있습니다.")
    else:
        result.verdict = "오답"
        result.reason = (f"최종 청크 {', '.join(f'#{c}' for c in sorted(set(picked)))} "
                         f"에 정답 청크가 없습니다.")
        if dropped:
            result.reason += f" (다른 문서에서 온 청크 {dropped}개는 뺐습니다)"

    result.elapsed = time.time() - started
    return result


def grade_answer(question: str, llm_answer: str,
                 candidates: list[str] | str,
                 gold_answer: str = "") -> GradeResult:
    """
    후보 중 하나라도 답변에 들어 있으면 정답으로 본다.

    겹친 후보는 candidates 에 적힌 순서대로 matched 에 담는다.
    """
    started = time.time()
    if isinstance(candidates, str):
        candidates = [candidates]
    candidates = [c for c in (candidates or []) if c and c.strip()]

    # 모범 답안을 안 넘겼으면 질문으로 되찾아 본다 (CLI 에서 --gold 만 준 경우).
    if not gold_answer:
        found = find_question(question)
        gold_answer = found.answer if found else ""

    result = GradeResult(question=question, llm_answer=llm_answer or "",
                         candidates=candidates, gold_answer=gold_answer)

    if not candidates:
        result.verdict = "판정 불가"
        result.reason = "data/qa.json 에 이 질문의 정답 후보가 없습니다."
    elif not (llm_answer or "").strip():
        result.verdict = "오답"
        result.reason = "답변이 비어 있습니다."
    else:
        haystack = normalize(llm_answer)
        result.matched = [c for c in candidates
                          if normalize(c) and normalize(c) in haystack]
        if result.matched:
            result.verdict = "정답"
            result.reason = (f"후보 {len(candidates)}개 중 "
                             f"{len(result.matched)}개가 답변에 들어 있습니다.")
        else:
            result.verdict = "오답"
            result.reason = f"후보 {len(candidates)}개 중 답변에 들어 있는 것이 없습니다."

    result.elapsed = time.time() - started
    return result


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="정답 후보와 파이프라인 답변을 견준다 (문자열 포함 판정).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--run", default=None,
                        help="채점할 질문 번호 또는 all.\n없으면 세트만 훑는다")
    parser.add_argument("--doc", default=None,
                        help="검색 대상 문서.\n"
                             "없으면 전체 문서에서 찾는다 (기본, 실제 앱과 같다)\n"
                             "all 을 주면 질문마다 정답 문서로 좁혀 한 번 더 돈다\n"
                             "그 밖의 값은 그 문서 하나로 고정한다")
    args = parser.parse_args()

    items = load_questions()
    if not items:
        sys.exit(f"질문-정답 세트를 읽지 못했습니다: {QA_PATH}")

    # --- 세트만 훑기 --------------------------------------------------------
    if not args.run:
        from corpus import find as find_doc

        print(f"질문 {len(items)}개  ({QA_PATH})\n")
        for q in items:
            try:
                code = (find_doc(q.doc).code if q.doc else "-")
            except KeyError:
                code = "??"
            print(f"{q.id:>2}  [{code:<3}] {q.question}")
            # 이 세트가 들고 있는 정답은 청크 번호다. 모범 답안·후보 낱말은
            # 옛 법률 세트에만 있으므로 있을 때만 적는다.
            if q.answer_chunks:
                nums = " ".join(f"#{c}" for c in q.answer_chunks)
                print(f"      정답 청크 {nums}")
            if q.answer:
                print(f"      정답 {q.answer}")
            if q.keywords:
                print(f"      후보 {' / '.join(q.keywords)}")

        # 채점할 근거가 아예 없는 문항. 청크도 후보도 없으면 '판정 불가'가 된다.
        missing = [q.id for q in items if not q.keywords and not q.answer_chunks]
        if missing:
            print(f"\n[!] 채점 근거(정답 청크·후보)가 없는 번호: {missing}")

        # doc 이 비면 어느 문서의 청크 번호인지 알 수 없다. 문서가 여럿인
        # 지금은 번호만으로 채점하면 다른 문서의 같은 번호가 정답이 된다.
        no_doc = [q.id for q in items if not q.doc]
        if no_doc:
            print(f"[!] doc 이 없는 번호: {no_doc}")

        # 번호는 id 문자열의 마지막 숫자다 (_question_id). 문서가 여럿이면
        # 'jabber-001' 과 'ko-001' 이 똑같이 1번이 되어 --run 1 이 어느 쪽을
        # 고르는지 알 수 없다. qa.json 에서 번호대를 나눠 적어야 한다.
        seen: dict[int, list[str]] = {}
        for row, q in zip(load_questions(), items):
            seen.setdefault(q.id, []).append(q.question[:28])
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        if dupes:
            print(f"[!] 번호가 겹치는 문항 {sorted(dupes)} — qa.json 의 id 에서 "
                  f"문서마다 번호대를 나누세요 (jabber 1~99, ko 101~)")

        unknown = []
        for q in items:
            try:
                find_doc(q.doc)
            except KeyError:
                unknown.append(q.id)
        if unknown:
            print(f"[!] 문서를 찾을 수 없는 번호: {unknown}")
        return

    # --- 파이프라인에 태워 채점 ---------------------------------------------
    from main import run_pipeline

    targets = list(items) if args.run == "all" else [
        q for q in items if str(q.id) == str(args.run)]
    if not targets:
        sys.exit(f"그런 번호가 없습니다: {args.run}")

    # 돌릴 문서 목록.
    #   지정 없음   전체 문서 (앱 기본값과 같다)
    #   all         전체 문서 한 번 + 질문마다 정답 문서로 좁혀 한 번
    #   그 밖       그 문서 하나로 고정
    if args.doc == "all":
        modes: list[str | None] = [None, "gold"]
    else:
        modes = [args.doc]

    n_runs = len(targets) * len(modes)
    print(f"채점 {n_runs}회 (문항 {len(targets)} x 조건 {len(modes)})")

    started_all = time.time()
    rows = []       # (질문, 검색 문서, GradeResult)
    for mode in modes:
        if len(modes) > 1:
            print(f"\n{'=' * 60}")
            print("검색 범위: " + ("정답 문서로 좁힘" if mode == "gold" else "전체 문서"))
            print("=" * 60)
        for q in targets:
            doc = q.doc if mode == "gold" else mode
            # 정답 청크가 있으면 그것으로 채점한다. 이 세트가 들고 있는 것이
            # answer_chunk_indices 라, 예전처럼 gold(=keywords)만 넘기면
            # 판정 근거가 없어 GradeResult 가 아예 안 만들어진다.
            result = run_pipeline(q.question, doc=doc, gold=q.keywords,
                                  gold_chunks=q.answer_chunks,
                                  gold_doc=q.doc)
            gr = result.grade or GradeResult(
                question=q.question, verdict="판정 불가",
                reason="정답 청크도 정답 후보도 없습니다.")
            rows.append((q, doc, gr))

            mark = "O" if gr.correct else ("?" if gr.verdict == "판정 불가" else "X")
            print(f"\n[{mark}] {q.id:>2}번 ({result.doc_name})  "
                  f"{gr.verdict}   {result.elapsed:.1f}초")
            print(f"     질문      : {q.question}")
            print(f"     LLM 답변  : {gr.llm_answer[:110]}")
            print(f"     정답 청크 : {' '.join(gr.candidates) or '(없음)'}"
                  f"   -> 맞힌 것 {' '.join(gr.matched) or '(없음)'}")
            # 단계별 실패는 조용히 넘어가면 정답률만 보고 원인을 못 찾는다.
            for stage, message in result.errors().items():
                print(f"     [!] {stage} 실패: {message[:90]}")

    if len(rows) > 1:
        n_ok = sum(1 for *_, g in rows if g.correct)
        print(f"\n정답 {n_ok}/{len(rows)} · 총 {time.time() - started_all:.0f}초")
        wrong = [str(q.id) for q, _, g in rows if not g.correct]
        if wrong:
            print(f"틀린 문항: {', '.join(wrong)}")

        # 문서별로 나눠 보면 특정 언어의 검색이 약한 것인지 구분된다.
        by_doc: dict[str, list[bool]] = {}
        for q, _, g in rows:
            by_doc.setdefault(q.doc, []).append(g.correct)
        print("정답 문서별: " + "  ".join(
            f"{key[:2]} {sum(v)}/{len(v)}" for key, v in by_doc.items()))


if __name__ == "__main__":
    main()
