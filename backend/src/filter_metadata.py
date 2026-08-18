#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0-2. 메타데이터 필터 — 추출된 조건을 행 마스크로

    extract_metadata.py (4B LLM)
        {sender, receiver, participants, since, until, keywords}
                                          |
                                      여기서 정규화 + 마스크
                                          |
                                  search.py(mask=...) 가 후보를 좁힌다

두 가지 일을 한다:

    1. 정규화   추출기가 뭐라고 적어 보내든 사람 목록과 기간 두 개로 접는다.
    2. 사다리   그 조건으로 후보가 0개가 되면 한 겹씩 풀어 준다.

왜 sender/receiver/participants 를 한 목록으로 접나:

    청크 메타데이터에는 방향이 없다 (chunk_meta.py 참고). 4B 에게는
    "sender: tom" 이 자연스러운 출력이라 칸을 나눠 받되, 필터에서는 셋을
    합쳐 "이 사람들이 다 대화쌍 안에 있어야 한다" 로 건다.

        sender: [tom]                     -> tom 이 낀 대화
        sender: [tom], receiver: [stern]  -> tom~stern 대화쌍
        participants: [tom, stern]        -> 같은 것

    칸별 목록은 MetaQuery 에 그대로 남겨 화면이 4B 의 판단을 보여 줄 수 있게
    한다. 판정에 쓰는 것은 합친 people 하나다.

왜 사다리가 필요한가 (이 파일에서 제일 중요한 부분):

    4B 가 닉을 하나 잘못 짚거나 날짜를 헛짚으면 후보가 0개가 되고, 그 순간
    답변은 반드시 틀린다. 근거가 아예 없으니 리랭커도 LLM 도 할 수 있는 일이
    없다. 필터는 '틀리면 안 되는' 장치가 아니라 '틀려도 죽지 않아야 하는'
    장치다. 그래서 좁은 순서대로 걸어 보고 처음으로 살아남는 조건을 쓴다.

        1. 쌍 + 기간         두 사람이 그 기간에 나눈 대화
        2. 쌍               (기간을 푼다)
        3. 사람(OR) + 기간   (쌍을 푼다 — 둘 중 하나만 맞아도)
        4. 사람(OR)
        5. 기간
        6. 전체             (필터를 포기한다)

    기간을 먼저 푸는 이유: 사람 이름은 명부(nicks.json) 대조를 통과한 것이라
    실재가 보장되지만, 날짜는 질문에 적힌 것을 그대로 믿는 수밖에 없다.
    "9월 29일 대화" 라고 했는데 그날 그 둘이 말한 적이 없으면, 날짜 쪽을
    의심하는 편이 맞다.

    몇 단계에서 몇 개로 좁혔는지는 FilterResult 가 들고 나간다. 화면에
    보여 주지 않으면 필터가 조용히 포기한 것을 아무도 모른다.

keywords 는 여기서 쓰지 않는다. 본문을 봐야 하는 조건이라 임베딩·희소검색
쪽(search.py)에서 처리한다. 추출 결과를 그대로 실어 나르기만 한다.

단독 실행:
    python src/filter_metadata.py --people poll,stern
    python src/filter_metadata.py --people stern --since 2020-09-29
    python src/filter_metadata.py --json '{"sender":"tom","date":"2020-09-29"}'
    python src/filter_metadata.py --qa      # qa.json 문항별 축소율/누락 표
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_meta import ChunkMeta, from_epoch, load_chunk_meta  # noqa: E402

# 추출기가 사람을 적어 보낼 수 있는 이름들. 역할별로 받되, 필터에서는 셋을
# 합쳐 쓴다 — 청크 메타데이터에 방향이 없기 때문이다 (chunk_meta.py 참고).
SENDER_KEYS = ("sender", "senders", "from_", "speaker", "speakers")
RECEIVER_KEYS = ("receiver", "receivers", "recipient", "recipients", "to_")
PARTICIPANT_KEYS = ("participants", "participant", "people", "persons",
                    "names", "who", "dyad", "nick", "nicks")

# 기간의 시작/끝. 추출기가 흔들리는 자리라 넉넉히 받는다.
SINCE_KEYS = ("since", "timefrom", "time_from", "date_from", "start",
              "start_date", "ts_start", "after")
UNTIL_KEYS = ("until", "timeuntill", "timeuntil", "time_until", "date_to",
              "end", "end_date", "ts_end", "before")

# 하루 한 날짜만 온 경우. since = until 로 편다.
DATE_KEYS = ("date", "day", "on")

KEYWORD_KEYS = ("keywords", "keyword", "terms", "query_terms")

DATE_RE = re.compile(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?")


# --------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------

def _as_list(value) -> list[str]:
    """문자열 하나든 목록이든 목록으로. 'a, b' 처럼 붙여 보내는 것도 받는다."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;|]| 와 | 과 ", value)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(_as_list(v))
        return out
    return [str(value).strip()]


def _first(raw: dict, keys) -> str | None:
    for k in keys:
        v = raw.get(k)
        if v not in (None, "", [], {}):
            return str(v).strip()
    return None


def parse_span(value: str | None, end: bool = False) -> str | None:
    """
    '2020-09-29' / '2020/9/29' / '2020-09' -> ISO 날짜.

    달만 준 경우 end 면 그달 말일로, 아니면 1일로 편다. 추출기가 "9월쯤" 을
    '2020-09' 로 적어 보내는 일이 있어서다. 못 알아보면 None (조건을 안 건다).
    """
    if not value:
        return None
    m = DATE_RE.search(str(value))
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), m.group(3)
    if day:
        return f"{year:04d}-{month:02d}-{int(day):02d}"
    if not end:
        return f"{year:04d}-{month:02d}-01"
    nxt = (year + 1, 1) if month == 12 else (year, month + 1)
    import datetime as _dt
    last = _dt.date(*nxt, 1) - _dt.timedelta(days=1)
    return last.isoformat()


@dataclass
class MetaQuery:
    """추출 결과를 필터가 쓸 형태로 접은 것."""

    # 필터가 실제로 쓰는 것. 아래 세 칸을 합친 것이다 (중복 제거).
    people: list[str] = field(default_factory=list)    # 명부에 있는 실제 닉
    # 추출기가 어느 칸에 적어 보냈는지. 화면 표시와 검수용이다.
    # 방향은 청크 메타데이터에 없으므로 필터에서는 셋 다 같은 뜻이 된다:
    # "그 사람이 대화쌍 안에 있어야 한다".
    sender: list[str] = field(default_factory=list)
    receiver: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)   # 명부에 없어 버린 이름
    since: str | None = None
    until: str | None = None
    keywords: list[str] = field(default_factory=list)  # 4단계로 넘길 뿐

    @property
    def empty(self) -> bool:
        return not self.people and not self.since and not self.until

    def label(self) -> str:
        bits = []
        if len(self.people) >= 2:
            bits.append(" ~ ".join(self.people))
        elif self.people:
            bits.append(self.people[0])
        if self.since or self.until:
            bits.append(f"{self.since or '...'} ~ {self.until or '...'}")
        return " · ".join(bits) or "조건 없음"


def normalize(raw: dict, meta: ChunkMeta | None = None) -> MetaQuery:
    """
    추출기 출력 dict -> MetaQuery.

    이름은 명부에 대조해서 있는 것만 남긴다. 없는 이름을 조건으로 걸면 후보가
    0개가 되므로, 버리되 unknown 에 적어 화면에서 볼 수 있게 한다 (추출기가
    무엇을 지어냈는지가 여기 남는다).
    """
    meta = meta or load_chunk_meta()
    raw = raw or {}

    def resolved(keys) -> tuple[list[str], list[str]]:
        """이름 목록 -> (명부에 있는 것, 없어서 버린 것)."""
        good, bad = [], []
        wanted: list[str] = []
        for key in keys:
            wanted.extend(_as_list(raw.get(key)))
        for name in dict.fromkeys(wanted):        # 중복 제거, 순서 유지
            real = meta.resolve(name)
            if real is None:
                bad.append(name)
            elif real not in good:
                good.append(real)
        return good, bad

    sender, bad_s = resolved(SENDER_KEYS)
    receiver, bad_r = resolved(RECEIVER_KEYS)
    participants, bad_p = resolved(PARTICIPANT_KEYS)

    # 세 칸은 배타적이다. 방향을 말한 칸이 이기고, 같은 이름이 participants
    # 에도 있으면 그쪽에서 뺀다. 추출기가 이미 정리해서 보내지만, 다른 경로
    # (CLI --json, 옛 판본 응답)로 들어온 것도 화면에 두 번 뜨지 않게 한다.
    receiver = [n for n in receiver if n not in sender]
    directed = set(sender) | set(receiver)
    participants = [n for n in participants if n not in directed]

    # 필터는 방향을 모른다. 셋을 합쳐 "이 사람들이 다 낀 대화" 로 건다.
    # sender 하나면 그 사람이 낀 대화, sender + receiver 면 그 둘의 대화쌍,
    # participants 두 명이면 마찬가지로 그 둘의 대화쌍이다.
    people: list[str] = []
    for name in (*sender, *receiver, *participants):
        if name not in people:
            people.append(name)
    unknown = list(dict.fromkeys([*bad_s, *bad_r, *bad_p]))

    date = _first(raw, DATE_KEYS)
    since = parse_span(_first(raw, SINCE_KEYS) or date, end=False)
    until = parse_span(_first(raw, UNTIL_KEYS) or date, end=True)

    keywords: list[str] = []
    for key in KEYWORD_KEYS:
        keywords.extend(_as_list(raw.get(key)))

    return MetaQuery(people=people, sender=sender, receiver=receiver,
                     participants=participants, unknown=unknown,
                     since=since, until=until,
                     keywords=list(dict.fromkeys(keywords)))


# --------------------------------------------------------------------------
# 마스크
# --------------------------------------------------------------------------

@dataclass
class FilterResult:
    """필터 결과. search.py 에 mask 를, 화면에 나머지를 준다."""

    query: MetaQuery
    mask: np.ndarray | None = None      # None = 조건 없음 (전체를 뒤진다)
    n_total: int = 0
    n_kept: int = 0
    step: str = "전체"                  # 실제로 걸린 사다리 단계
    relaxed: list[str] = field(default_factory=list)   # 시도했다 비어서 푼 것

    @property
    def ratio(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 1.0

    @property
    def filtered(self) -> bool:
        return self.mask is not None

    def summary(self) -> str:
        head = (f"{self.step}: {self.n_kept:,}/{self.n_total:,}개 "
                f"({100 * self.ratio:.1f}%)")
        if self.relaxed:
            head += f"  [완화: {' -> '.join(self.relaxed)}]"
        if self.query.unknown:
            head += f"  [모르는 이름: {', '.join(self.query.unknown)}]"
        return head


def build_mask(raw: dict | MetaQuery,
               meta: ChunkMeta | None = None) -> FilterResult:
    """
    조건 dict -> 행 마스크. 후보가 0개가 되면 한 겹씩 풀어 준다.

    돌려주는 mask 는 청크 메타데이터 행(0..N-1) 기준이다. 검색 색인이 문서를
    여러 개 합쳐 놓았다면 search.py 가 chunk_index 로 다시 펴서 쓴다.
    """
    meta = meta or load_chunk_meta()
    q = raw if isinstance(raw, MetaQuery) else normalize(raw, meta)
    n = meta.size

    if q.empty:
        return FilterResult(query=q, mask=None, n_total=n, n_kept=n,
                            step="조건 없음")

    time_mask = meta.mask_time(q.since, q.until)
    pair_mask = meta.mask_people(q.people, how="auto") if q.people else None
    any_mask = (meta.mask_people(q.people, how="any")
                if len(q.people) >= 2 else pair_mask)

    def both(a, b):
        if a is None:
            return b
        return a if b is None else a & b

    # 좁은 것부터. 처음으로 하나라도 남는 단계를 쓴다.
    ladder = [
        ("쌍 + 기간", both(pair_mask, time_mask)),
        ("쌍", pair_mask),
        ("사람(OR) + 기간", both(any_mask, time_mask)),
        ("사람(OR)", any_mask),
        ("기간", time_mask),
    ]

    relaxed: list[str] = []
    for name, mask in ladder:
        if mask is None:
            continue
        kept = int(mask.sum())
        if kept:
            return FilterResult(query=q, mask=mask, n_total=n, n_kept=kept,
                                step=name, relaxed=relaxed)
        relaxed.append(name)

    # 전부 0건. 필터를 포기하고 전체를 넘긴다 — 근거 없는 답변보다 낫다.
    return FilterResult(query=q, mask=None, n_total=n, n_kept=n,
                        step="전체(포기)", relaxed=relaxed)


def filter_rows(raw: dict, meta: ChunkMeta | None = None) -> np.ndarray:
    """마스크가 아니라 행 번호가 필요할 때. 조건이 없으면 전체 행."""
    meta = meta or load_chunk_meta()
    result = build_mask(raw, meta)
    if result.mask is None:
        return np.arange(meta.size, dtype=np.int64)
    return np.flatnonzero(result.mask)


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def _qa_selfcheck(meta: ChunkMeta) -> None:
    """
    qa.json 문항을 규칙 추출 -> 필터에 통과시켜 축소율과 누락을 본다.

    추출은 extract_metadata.rule_extract 를 그대로 쓴다. 여기서 정규식을 또
    쓰면 두 벌이 서로 어긋나고, 그러면 이 표가 필터를 재는지 정규식을 재는지
    알 수 없게 된다.

    키워드 채널은 빼고 메타 조건(사람·기간)만 건 결과다. 키워드까지 얹은 표는
    extract_metadata.py --qa 에 있다. 두 표의 차이가 키워드 채널의 몫이다.
    """
    from extract_metadata import rule_extract
    qa_path = Path(__file__).resolve().parent.parent / "data" / "qa" / "qa.json"
    if not qa_path.is_file():
        print(f"[!] qa.json 이 없습니다: {qa_path}")
        return
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))["qa_pairs"]

    row_of = {int(c): i for i, c in enumerate(meta.chunk_index)}

    tally: dict[int, list] = {}
    for p in pairs:
        raw = dict(rule_extract(p["question"]))
        raw.pop("keywords", None)          # 여기서는 메타 조건만 잰다
        res = build_mask(raw, meta)
        gold = [row_of[c] for c in p["answer_chunk_indices"] if c in row_of]
        kept = (int(res.mask[gold].sum()) if res.mask is not None
                else len(gold))

        row = tally.setdefault(p["q_type"], [0, 0, 0, [], 0])
        row[0] += 1                                   # 문항 수
        row[1] += 1 if kept == len(gold) else 0       # 정답 온전히 통과
        row[2] += res.n_kept                          # 후보 합
        if kept != len(gold):
            row[3].append(f"{p['id']} {kept}/{len(gold)}")
        row[4] += 1 if res.relaxed else 0             # 완화가 걸린 문항

    names = {p["q_type"]: p.get("q_type_name", f"유형 {p['q_type']}")
             for p in pairs}
    print(f"\n  {'유형':<14}{'문항':>5}{'정답통과':>9}{'평균후보':>10}"
          f"{'축소율':>9}{'완화':>6}")
    for qt in sorted(tally):
        n, ok, pool, miss, relax = tally[qt]
        avg = pool / n
        print(f"  {str(names.get(qt, qt)):<14}{n:>5}{ok:>9}{avg:>10,.0f}"
              f"{100 * avg / meta.size:>8.1f}%{relax:>6}")
        if miss:
            print(f"      [!] 정답이 걸러진 문항: {', '.join(miss)}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(
        description="추출된 메타데이터로 청크 후보를 좁혀 본다.")
    ap.add_argument("--json", help="추출기 출력 그대로 (JSON 문자열)")
    ap.add_argument("--people", help="사람. 'poll,stern'")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--qa", action="store_true", help="qa.json 문항 표")
    args = ap.parse_args()

    meta = load_chunk_meta()
    print(f"청크 {meta.size:,}개 · 닉 {len(meta.nicks)}명")

    if args.qa:
        _qa_selfcheck(meta)
        return

    raw = json.loads(args.json) if args.json else {}
    if args.people:
        raw["people"] = args.people
    if args.since:
        raw["since"] = args.since
    if args.until:
        raw["until"] = args.until

    res = build_mask(raw, meta)
    print(f"\n조건 : {res.query.label()}")
    if res.query.keywords:
        print(f"키워드: {', '.join(res.query.keywords)}  (검색 단계에서 씀)")
    print(f"결과 : {res.summary()}\n")

    rows = (np.flatnonzero(res.mask) if res.mask is not None
            else np.arange(min(meta.size, 10)))
    for r in rows[:10]:
        print(f"  {meta.describe(int(r))}")
    if len(rows) > 10:
        print(f"  ... 외 {len(rows) - 10:,}개")


if __name__ == "__main__":
    main()
