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

from chunk_meta import (  # noqa: E402
    ChunkMeta,
    from_epoch,
    load_chunk_meta,
    meta_for,
)
from corpus import Document, documents_for, script_of  # noqa: E402

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

    # --- 문서별 결과 (build_doc_masks 가 채운다) -------------------------
    #
    # 문서마다 메타데이터가 다르므로 마스크도 문서마다 다르다. 예전처럼 마스크
    # 하나를 색인 전체에 갖다 붙이면, 청크 번호가 겹치는 다른 문서가 남의
    # 조건을 물려받는다. search.expand_mask 가 이 사전을 문서별로 편다.
    #
    #     masks[문서키] is None   조건 없음 -> 그 문서는 전부 통과
    #     masks[문서키] = 배열    그 문서의 청크 마스크 (0..N-1)
    #     excluded 에 있는 문서   후보에서 통째로 뺐다
    masks: dict[str, np.ndarray | None] | None = None
    excluded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)     # 문서별 한 줄 설명

    @property
    def ratio(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 1.0

    @property
    def filtered(self) -> bool:
        if self.masks is not None:
            return bool(self.excluded) or any(m is not None
                                              for m in self.masks.values())
        return self.mask is not None

    def summary(self) -> str:
        head = (f"{self.step}: {self.n_kept:,}/{self.n_total:,}개 "
                f"({100 * self.ratio:.1f}%)")
        if self.relaxed:
            head += f"  [완화: {' -> '.join(self.relaxed)}]"
        if self.excluded:
            head += f"  [제외한 문서: {', '.join(self.excluded)}]"
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


# --------------------------------------------------------------------------
# 문서별 마스크 — 파이프라인이 실제로 쓰는 입구
# --------------------------------------------------------------------------

def normalize_across(raw: dict, metas: list[ChunkMeta]) -> MetaQuery:
    """
    문서 여럿의 명부를 한꺼번에 대조해서 MetaQuery 를 만든다.

    명부가 문서마다 다르므로 "모르는 이름" 도 문서마다 다르다. 한 문서 기준으로
    정규화하면, 다른 문서에만 있는 이름이 조용히 버려진다. 어느 문서든 아는
    이름이면 살리고, **아무 문서도 모르는 이름만** unknown 으로 내린다.

    조건을 실제로 거는 것은 문서별 mask_people 이라, 자기가 모르는 이름은
    그 문서에서 알아서 빠진다. 여기서는 넓게 살려 두기만 하면 된다.
    """
    if not metas:
        return normalize(raw, _empty_meta())

    parts = [normalize(raw, m) for m in metas]
    merged = parts[0]

    def union(attr: str) -> list[str]:
        out: list[str] = []
        for q in parts:
            for name in getattr(q, attr):
                if name not in out:
                    out.append(name)
        return out

    known = set(union("people"))
    unknown: list[str] = []
    for q in parts:
        for name in q.unknown:
            if name not in known and name not in unknown:
                unknown.append(name)

    return MetaQuery(
        people=union("people"),
        sender=union("sender"),
        receiver=union("receiver"),
        participants=union("participants"),
        unknown=unknown,
        since=merged.since, until=merged.until,
        keywords=merged.keywords,
    )


class Roster:
    """
    스코프 안 문서들의 이름 명부. "이 이름이 실재하는가" 한 가지를 답한다.

    명부가 문서마다 다르기 때문에 필요한 물건이다. 예전에는 명부가 한 벌
    (jabber 289명)이라 아무 데서나 load_chunk_meta().resolve() 를 불렀는데,
    그러면 '가해자17' 이 jabber 명부에 없다는 이유로 버려진다 — ko_voice.npz
    안에 그 이름이 멀쩡히 있는데도.

    대조 순서:

        1. 스코프 안 문서들의 명부를 차례로 본다. 아는 문서가 있으면 그 문서가
           쓰는 정식 표기로 돌려준다 ('Stern' -> 'stern').
        2. 아무도 모르는 이름은
             - 명부 파일(_nicks.json)이 없는 문서가 스코프에 있으면 그대로 통과
             - 아니면 None (조건에서 버린다)

    2번의 통과 규칙은 '명부를 두지 않는 문서' 를 위한 자리다. 그런 문서에서는
    질문에 적힌 이름을 확인해 줄 화이트리스트가 없으므로, 버리는 대신 그대로
    넘기고 실제 판정은 mask_people 에 맡긴다. 없는 이름이면 거기서 0건이 되고,
    filter_metadata 의 사다리가 조건을 풀어 준다.

    ChunkMeta 와 같은 resolve() 를 갖고 있어서 normalize() 에 그대로 넣을 수
    있다 (normalize 는 meta 에서 resolve 하나만 쓴다).
    """

    def __init__(self, docs: list[Document]):
        self.docs = list(docs)
        self.metas = [m for m in (meta_for(d) for d in self.docs)
                      if m is not None]
        # 명부를 두지 않는 문서가 스코프에 있는가. 있으면 모르는 이름을 버리지
        # 않는다. 이 한 줄을 False 로 고정하면 예전처럼 전부 화이트리스트다.
        self.passthrough = any(not d.has_nicks for d in self.docs)

    @property
    def hangul(self) -> bool:
        """
        이름에 한글을 쓰는 문서가 스코프에 있는가. 규칙 패턴이 이걸 본다.

        '비ASCII' 로 갈음하면 안 된다. jabber 닉 289개 중 'стов' 하나가
        키릴이라, 그것 때문에 jabber 만 뒤지는 질문에서도 한글 이름 패턴이
        켜지고 "지난달의 대화" 가 사람 조건으로 잡힌다.
        """
        return any(script_of(n) == "hangul"
                   for m in self.metas for n in m.nicks)

    @property
    def names(self) -> list[str]:
        """스코프 안 모든 이름."""
        out: list[str] = []
        for meta in self.metas:
            for name in meta.nicks:
                if name not in out:
                    out.append(str(name))
        return out

    @property
    def has_time(self) -> bool:
        """기간으로 물어볼 수 있는 문서가 하나라도 있는가."""
        return any(d.has_time and d.has_meta for d in self.docs)

    @property
    def name_scripts(self) -> tuple[str, ...]:
        """명부에 실제로 있는 이름의 문자 종류. 프롬프트 보기를 고르는 데 쓴다."""
        out: list[str] = []
        for meta in self.metas:
            for name in meta.nicks:
                kind = script_of(name)
                if kind not in out:
                    out.append(kind)
        return tuple(out)

    @property
    def kw_scripts(self) -> tuple[str, ...]:
        """본문에 그대로 나올 수 있는 문자 종류. 문서들의 합집합."""
        out: list[str] = []
        for doc in self.docs:
            for name in doc.kw_scripts:
                if name not in out:
                    out.append(name)
        return tuple(out)

    def groups(self) -> list[tuple[str, list[Document], ChunkMeta | None]]:
        """메타데이터를 함께 쓰는 문서끼리 묶어서. 화면·프롬프트 설명에 쓴다."""
        by: dict[str, list[Document]] = {}
        for doc in self.docs:
            by.setdefault(doc.group, []).append(doc)
        return [(g, ds, meta_for(ds[0])) for g, ds in by.items()]

    def prompt_names(self, cap: int = 320) -> list[str]:
        """
        프롬프트에 실을 명부. 문서마다 몫을 나눠 뽑는다.

        앞에서부터 자르면 안 된다. 전체 검색이면 jabber 289 + ko_voice 138 =
        427명이라 320에서 자르는 순간 ko_voice 이름이 통째로 날아가고, 4B 는
        '가해자17' 을 "명부에 없는 이름" 으로 보게 된다. 문서 수로 나눠 각자
        몫만큼 실으면 어느 문서도 통째로 빠지지 않는다.
        """
        buckets = [[str(n) for n in m.nicks] for m in self.metas if m.nicks]
        if not buckets:
            return []

        share = max(1, cap // len(buckets))
        out: list[str] = []
        for names in buckets:
            for name in names[:share]:
                if name not in out:
                    out.append(name)

        # 몫을 다 못 쓴 문서가 있으면 남은 자리를 앞 문서부터 채운다.
        if len(out) < cap:
            for names in buckets:
                for name in names:
                    if len(out) >= cap:
                        break
                    if name not in out:
                        out.append(name)
        return out

    def resolve(self, nick: str | None) -> str | None:
        if not nick:
            return None
        name = str(nick).strip()
        if not name:
            return None
        for meta in self.metas:
            got = meta.resolve(name)
            if got is not None:
                return got
        return name if self.passthrough else None


def normalize_across(raw: dict, docs: list[Document]) -> MetaQuery:
    """
    스코프 안 문서들의 명부를 한꺼번에 보고 MetaQuery 를 만든다.

    Roster 가 문서별 대조를 대신하므로 normalize 를 한 번만 부르면 된다.
    """
    return normalize(raw, Roster(docs))


def _why_excluded(doc: Document, q: MetaQuery, meta: ChunkMeta | None,
                  people: list[str] | None = None) -> str:
    """
    이 문서를 후보에서 빼야 하는가. 빼야 하면 그 이유, 아니면 빈 문자열.

    규칙은 하나다 — **조건을 만족시킬 방법이 아예 없는 문서는 뺀다.**

        "2020-09 대화"      ko_voice 에는 시각이 없다 -> 뺀다
        "stern 과 poll"     ko_voice 명부에 그런 이름이 없다 -> 뺀다

    통과시켜 두고 dense 점수에 맡길 수도 있지만, 그러면 조건을 건 질문일수록
    엉뚱한 문서가 상위에 섞인다. 조건이 잡혔다는 것은 이미 그 문서 얘기라는
    뜻이다.

    빼는 것이 늘 옳지는 않아서, 문서가 전부 빠지면 build_doc_masks 가 통째로
    되돌린다. 최악이 '안 좁힌 검색' 이어야 한다는 원칙은 그대로다.

    people 은 **어느 문서든 명부로 확인된 이름만** 추린 것이다. q.people 을
    그대로 쓰면 안 된다: Roster 의 통과 규칙 때문에 아무도 모르는 이름이 섞여
    들어올 수 있고, 그 이름 하나 때문에 모든 문서가 빠지면 멀쩡한 기간 조건까지
    같이 버려진다 ("2020-09-29 zzz999가 말한..." 이 날짜 필터를 잃는다).
    """
    people = q.people if people is None else people
    if q.since or q.until:
        if not doc.has_time:
            return "시각 없음"
        if meta is None:
            return "메타데이터 없음"

    if people:
        if meta is None or not doc.has_people:
            return "메타데이터 없음"
        if not any(meta.resolve(n) for n in people):
            return "명부에 없는 이름"

    return ""


def build_doc_masks(raw: dict | MetaQuery,
                    docs: list[Document] | str | None = None) -> FilterResult:
    """
    조건 -> **문서마다 하나씩** 마스크. 파이프라인은 build_mask 대신 이것을 쓴다.

    build_mask 는 메타데이터 한 벌 안에서만 도는 함수다. 문서가 여럿이면 그
    한 벌이 어느 문서 것인지가 문제가 되고, 잘못 고르면 청크 번호가 겹치는
    다른 문서가 남의 조건을 물려받는다. 여기서 문서를 먼저 갈라 준다.

        1. 같은 메타데이터를 쓰는 문서끼리 묶는다 (jabber_ru + jabber_en)
        2. 묶음마다 자기 .npz 로 build_mask 를 돈다
        3. 조건을 만족시킬 수 없는 묶음은 통째로 뺀다 (_why_excluded)
        4. 다 빠졌으면 전부 되돌린다

    docs 에 문자열을 주면 그 문서 하나로 좁힌다 (search 의 --doc 과 같은 값).
    """
    targets = documents_for(docs) if isinstance(docs, (str, type(None)))         else list(docs)

    # 1. 메타데이터를 함께 쓰는 문서끼리 묶는다. jabber_ru 와 jabber_en 은 같은
    #    대화의 두 판본이라 마스크가 같다 — 두 번 계산할 이유가 없다.
    groups: dict[str, list[Document]] = {}
    for doc in targets:
        groups.setdefault(doc.group, []).append(doc)

    metas = {g: meta_for(members[0]) for g, members in groups.items()}
    known = [m for m in metas.values() if m is not None]

    q = raw if isinstance(raw, MetaQuery) else normalize_across(raw, targets)

    if q.empty:
        n = sum(m.size for m in known)
        return FilterResult(query=q, mask=None, masks=None,
                            n_total=n, n_kept=n, step="조건 없음")

    # 어느 문서든 명부로 확인된 이름만. 아무도 모르는 이름뿐이면 사람 조건은
    # 없는 것으로 본다 (그 이름 하나로 전 문서를 빼면 기간 조건까지 날아간다).
    real_people = [n for n in q.people
                   if any(m is not None and m.resolve(n) for m in metas.values())]

    masks: dict[str, np.ndarray | None] = {}
    excluded: list[str] = []
    notes: list[str] = []
    relaxed: list[str] = []
    steps: list[str] = []
    n_total = n_kept = 0

    for group, members in groups.items():
        meta = metas[group]
        keys = [d.key for d in members]
        why = _why_excluded(members[0], q, meta, real_people)

        if why:
            excluded.extend(keys)
            notes.append(f"{group}: 제외 ({why})")
            # 제외한 문서의 청크는 분모에도 넣지 않는다. 후보가 될 수 없는
            # 것을 세면 "15,592개 중 20개" 가 "몇 개 중 몇 개" 인지 흐려진다.
            continue

        n_total += meta.size
        got = build_mask(q, meta)
        masks.update({k: got.mask for k in keys})
        n_kept += got.n_kept if got.mask is not None else meta.size
        steps.append(f"{group}: {got.step}")
        relaxed.extend(f"{group} {r}" for r in got.relaxed)
        notes.append(f"{group}: {got.step} {got.n_kept:,}/{meta.size:,}개")

    # 4. 전부 빠졌다. 필터를 포기하고 전체를 넘긴다 — 0건짜리 검색보다 낫다.
    if not masks:
        n = sum(m.size for m in known)
        return FilterResult(query=q, mask=None, masks=None,
                            n_total=n, n_kept=n, step="전체(전 문서 제외)",
                            relaxed=relaxed, notes=notes)

    # 아무 문서도 안 빠졌고 어느 문서도 좁혀지지 않았다. 마스크를 통째로
    # None 으로 돌려준다 — 전부 True 인 마스크를 넘기면 search.narrow 가
    # "메타 15,592청크" 라고 적어 조건이 걸린 것처럼 보인다.
    if not excluded and all(m is None for m in masks.values()):
        n = sum(m.size for m in known)
        return FilterResult(query=q, mask=None, masks=None, notes=notes,
                            n_total=n, n_kept=n,
                            step=" · ".join(steps) or "조건 없음",
                            relaxed=relaxed)

    return FilterResult(
        query=q, mask=None, masks=masks, excluded=excluded, notes=notes,
        n_total=n_total, n_kept=n_kept,
        step=" · ".join(steps) or "조건 없음", relaxed=relaxed,
    )


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
