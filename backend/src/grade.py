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
QA_PATH = ROOT / "data" / "qa.json"


@dataclass
class Question:
    """질문-정답 세트 한 항목."""

    id: int
    doc: str                                              # 정답이 든 문서 키
    question: str
    answer: str = ""                                      # 사람이 쓴 모범 답안
    keywords: list[str] = field(default_factory=list)     # 판정에 쓰는 후보

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

    items: list[Question] = []
    for i, row in enumerate(data.get("questions") or [], 1):
        if not isinstance(row, dict) or not row.get("question"):
            continue
        words = row.get("keywords") or []
        if isinstance(words, str):
            words = [words]
        items.append(Question(
            id=int(row.get("id") or i),
            doc=str(row.get("doc") or ""),
            question=str(row["question"]).strip(),
            answer=str(row.get("answer") or "").strip(),
            keywords=[str(w).strip() for w in words if str(w).strip()],
        ))
    return tuple(items)


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
            print(f"      정답 {q.answer}")
            print(f"      후보 {' / '.join(q.keywords) or '(없음)'}")

        missing = [q.id for q in items if not q.keywords]
        if missing:
            print(f"\n[!] 정답 후보가 없는 번호: {missing}")

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
            result = run_pipeline(q.question, doc=doc, gold=q.keywords)
            gr = result.grade
            rows.append((q, doc, gr))

            mark = "O" if gr.correct else ("?" if gr.verdict == "판정 불가" else "X")
            print(f"\n[{mark}] {q.id:>2}번 ({result.doc_name})  "
                  f"{gr.verdict}   {result.elapsed:.1f}초")
            print(f"     질문      : {q.question}")
            print(f"     LLM 답변  : {gr.llm_answer[:110]}")
            print(f"     실제 정답 : {q.answer}")
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
