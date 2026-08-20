#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
청크 메타데이터 — 검색이 읽는 쪽

    data/build_chunk_meta.py  ->  data/meta/{메타키}.npz  ->  여기

메타데이터는 **문서마다 한 벌씩**이다. 파이프라인은 meta_for(문서) 로 받고,
load_chunk_meta() 를 직접 부르면 jabber 것이 온다 (옛 호출부·단독 실행용).

하는 일은 하나다. "누가 / 언제" 를 받아 **행 마스크 (N,) bool** 로 바꾼다.
그 마스크를 filter_metadata.py 가 모아 search.py 에 넘긴다.

물어볼 수 있는 것 세 가지:

    사람      mask_person("stern")            stern 이 낀 대화 전부
    대화쌍    mask_dyad("poll", "stern")      둘이 나눈 대화 (무순)
    기간      mask_time("2020-09-29")         그날 걸친 세션

방향(sender/receiver)은 두지 않는다:

    "stern 이 보낸" 이라는 질문이 들어와도 결국 찾는 것은 stern 이 낀 대화다.
    저장하는 것이 무순 쌍뿐이라 여기서도 방향을 물어볼 수 없고, 물어볼 필요도
    없다. 2단계 추출기가 sender/receiver 를 뽑아 오더라도 filter_metadata.py
    가 사람 목록으로 접어서 넘긴다.

    참여자 필터도 따로 없다. participant(A) == dyad 한쪽이 A 인 청크다.

dyad 는 왜 교집합이 아닌가:

    mask_person(a) & mask_person(b) 로 대신하면 안 된다. 수신자 196명짜리
    공지에서 서로 말한 적 없는 두 수신자가 한 쌍으로 잡힌다. 쌍 목록을 그대로
    들고 있다가 (a, b) 가 실제로 있는 청크만 고른다.

기간은 '포함'이 아니라 '겹침'으로 본다:

    세션은 점이 아니라 구간(ts_start~ts_end)이다. 자정을 넘기는 청크가 있어서
    세션이 통째로 그날 안에 들어야 한다고 보면 그것들을 놓친다.

        걸린다  ts_end >= 시작  AND  ts_start <= 끝

닉네임은 반드시 resolve() 를 거친다. 명부(nicks.json)에 없는 이름은 None 이
되고, 그러면 그 조건은 아예 걸지 않는다. LLM 이 지어낸 이름 하나에 검색 결과가
0건이 되는 것을 막는 자리다.

단독 실행:
    python src/chunk_meta.py --dyad poll,stern
    python src/chunk_meta.py --who stern --since 2020-09-29 --until 2020-09-29
    python src/chunk_meta.py --qa            # qa.json 화자·날짜 문항 자가 검증
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import (  # noqa: E402
    CHUNK_META_PATH,
    NICKS_PATH,
    Document,
    find,
)

# build_chunk_meta.py 가 dyad 를 적을 때 쓴 구분자. 화면 표시에만 쓴다.
DYAD_SEP = " | "


# --------------------------------------------------------------------------
# 시각 파싱
# --------------------------------------------------------------------------

def to_epoch(value, end_of_day: bool = False) -> float | None:
    """
    '2020-09-29' / ISO 문자열 / datetime / epoch 초 -> epoch 초.

    날짜만 준 경우 end_of_day 면 그날 끝으로 민다. 안 그러면 "2020-09-29 까지"
    가 그날 00:00 까지가 되어 그날 대화를 통째로 놓친다.

    시간대 없는 시각은 전부 UTC 로 본다. 로그 시각에 시간대가 붙어 있지 않고,
    build_chunk_meta.py 도 pandas 로 같은 규칙(naive = UTC)을 써서 epoch 을
    만들었다. 여기서 로컬 시간대로 읽으면 KST 기준 9시간이 통째로 어긋나
    "9월 29일" 필터가 9월 28일 15시부터 걸린다.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip()
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            stamp = datetime.fromisoformat(text.replace("/", "-"))
        if len(text) == 10 and end_of_day:      # 날짜만 준 경우
            stamp = stamp + timedelta(days=1) - timedelta(microseconds=1)

    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def from_epoch(value: float) -> str:
    """epoch 초 -> 원본 로그에 적혀 있던 그대로의 문자열."""
    return (datetime.fromtimestamp(float(value), tz=timezone.utc)
            .replace(tzinfo=None).isoformat(sep=" ", timespec="seconds"))


# --------------------------------------------------------------------------
# 자료구조
# --------------------------------------------------------------------------

@dataclass
class ChunkMeta:
    """청크 N개의 메타데이터. 전부 행 번호(0..N-1) 기준이다."""

    chunk_index: np.ndarray
    session_id: np.ndarray
    ts_start: np.ndarray         # (N,) epoch 초 (초 단위 버림)
    ts_end: np.ndarray
    n_messages: np.ndarray
    nicks: list[str]
    info: dict = field(default_factory=dict)

    # 이 문서에 시각이 있는가. ko_voice 처럼 원문에 시각이 없는 문서는 ts 가
    # 통째로 NO_TS 라 기간으로 물어볼 것이 없다 (build_chunk_meta.py 참고).
    has_time: bool = True

    # 어느 메타데이터 파일에서 왔는지. 오류 문구와 검수에만 쓴다.
    meta_key: str = ""

    # 쌍 목록 (M,). dyad_row[k] 번 청크가 (dyad_a[k], dyad_b[k]) 를 갖는다.
    dyad_row: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32),
                                 repr=False)
    dyad_a: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32),
                               repr=False)
    dyad_b: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32),
                               repr=False)

    _by_nick: list[np.ndarray] = field(default_factory=list, repr=False)
    _by_pair: dict[tuple[int, int], np.ndarray] = field(default_factory=dict,
                                                        repr=False)
    _nick_id: dict[str, int] = field(default_factory=dict, repr=False)
    _lower: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- 기본
    @property
    def size(self) -> int:
        return int(self.chunk_index.shape[0])

    def none(self) -> np.ndarray:
        return np.zeros(self.size, dtype=bool)

    def all(self) -> np.ndarray:
        return np.ones(self.size, dtype=bool)

    # ------------------------------------------------------------ 닉네임
    def resolve(self, nick: str | None) -> str | None:
        """
        명부에 있는 실제 닉으로 맞춘다. 없으면 None.

        대소문자만 다른 경우까지 받아 준다. 그 이상(오타 교정)은 여기서 하지
        않는다 — 편집거리로 붙이면 'poll' 과 'polk' 처럼 실제로 둘 다 존재하는
        닉을 조용히 바꿔치기한다. 오타 교정은 2단계 추출기가 명부를 보고 하되,
        고른 결과를 다시 이 함수로 확인하는 순서가 맞다.
        """
        if not nick:
            return None
        name = str(nick).strip()
        if name in self._nick_id:
            return name
        return self._lower.get(name.lower())

    def _mask(self, rows: np.ndarray) -> np.ndarray:
        m = self.none()
        m[rows] = True
        return m

    # -------------------------------------------------------------- 마스크
    def mask_person(self, nick: str | None) -> np.ndarray | None:
        """nick 이 낀 청크 (dyad 한쪽이 nick). 모르는 닉이면 None."""
        real = self.resolve(nick)
        if real is None:
            return None
        return self._mask(self._by_nick[self._nick_id[real]])

    def mask_dyad(self, a: str | None, b: str | None) -> np.ndarray | None:
        """
        a 와 b 가 나눈 대화 (무순). 둘 중 하나라도 모르는 닉이면 None.

        쌍 목록을 그대로 보므로 공지에서 수신자끼리 엮이는 일이 없다.
        self-dyad(a == b) 도 그대로 걸린다.
        """
        ra, rb = self.resolve(a), self.resolve(b)
        if ra is None or rb is None:
            return None
        key = tuple(sorted((self._nick_id[ra], self._nick_id[rb])))
        rows = self._by_pair.get(key)
        return self.none() if rows is None else self._mask(rows)

    def mask_people(self, names, how: str = "auto") -> np.ndarray | None:
        """
        사람 여럿을 한 번에. 아는 이름이 하나도 없으면 None.

        how="auto" 는 두 명일 때만 dyad 로 본다. 세 명 이상을 AND 로 걸면
        1:1 세션에는 두 명뿐이라 반드시 0건이 되므로, 그때는 OR 로 넓힌다
        ("A, B, C 중 누군가가 낀 대화"). 좁히려다 통째로 날리는 것보다 낫다.
        """
        real = [r for r in (self.resolve(n) for n in names or []) if r]
        real = list(dict.fromkeys(real))          # 중복 제거, 순서 유지
        if not real:
            return None
        if len(real) == 1:
            return self.mask_person(real[0])
        if len(real) == 2 and how in ("auto", "dyad"):
            return self.mask_dyad(*real)

        masks = [self.mask_person(n) for n in real]
        out = masks[0].copy()
        for m in masks[1:]:
            if how == "all":
                out &= m
            else:
                out |= m
        return out

    def mask_time(self, since=None, until=None) -> np.ndarray | None:
        """
        [since, until] 과 겹치는 세션. 한쪽만 줘도 된다.

        until 에 날짜만 주면 그날 끝(23:59:59.999999)까지로 본다.

        **조건을 안 걸었는지 먼저 본다.** 시각이 없는 문서라고 해서 무조건
        0건을 돌려주면, 날짜를 묻지도 않은 질문에서 "쌍 + 기간" 이 통째로
        비어 사다리가 매번 한 겹씩 풀린다 (실제로 좁혀지기는 하지만 화면에는
        "조건이 너무 좁아 풀었다" 고 뜬다).

        조건이 있는데 시각이 없는 문서면 그때는 0건이다. 조건을 무시하고 전부
        통과시키면 "2020-09-29 대화" 라는 질문에 시각을 모르는 청크가 섞여
        들어온다. 모르는 것은 걸리지 않는 편이 낫다 — 문서를 통째로 후보에서
        뺄지는 filter_metadata.py 가 한 단계 위에서 정한다.
        """
        lo = to_epoch(since)
        hi = to_epoch(until, end_of_day=True)
        if lo is None and hi is None:
            return None                     # 기간 조건 자체가 없다
        if not self.has_time:
            return self.none()              # 물었는데 답할 수가 없다
        m = self.all()
        if lo is not None:
            m &= self.ts_end >= lo
        if hi is not None:
            m &= self.ts_start <= hi
        return m

    # ------------------------------------------------------------- 도움말
    def dyads(self, row: int) -> list[str]:
        """행 하나가 가진 쌍을 'a | b' 문자열로. 공지는 여러 개다."""
        at = self.dyad_row == row
        return [f"{self.nicks[a]}{DYAD_SEP}{self.nicks[b]}"
                for a, b in zip(self.dyad_a[at], self.dyad_b[at])]

    def describe(self, row: int) -> str:
        """행 하나를 한 줄로. 검수와 CLI 출력에 쓴다."""
        pairs = self.dyads(row)
        who = pairs[0] if len(pairs) == 1 else f"공지 {len(pairs)}명"
        when = ("시각 없음" if not self.has_time else
                f"{from_epoch(self.ts_start[row])} ~ "
                f"{from_epoch(self.ts_end[row])[11:]}")
        return (f"#{self.chunk_index[row]:<6} {who:<28} {when} "
                f"({self.n_messages[row]}건)")

    def counts(self, nick: str) -> dict:
        real = self.resolve(nick)
        if real is None:
            return {}
        return {"nick": real, "chunks": len(self._by_nick[self._nick_id[real]])}


# --------------------------------------------------------------------------
# 로드
# --------------------------------------------------------------------------

@lru_cache(maxsize=8)
def load_chunk_meta(path: str | None = None) -> ChunkMeta:
    """
    data/meta/{메타키}.npz 를 읽는다. 파일 하나당 프로세스당 한 번.

    경로를 주지 않으면 jabber 것을 읽는다 (옛 호출부와 단독 실행용). 파이프라인
    은 meta_for(문서) 를 거치므로 문서마다 다른 파일이 온다.

    캐시를 파일 경로로 잡는다. 문서가 여럿이면 메타데이터도 여럿이고, 한 벌만
    들고 있으면 문서를 옮겨 다닐 때마다 .npz 를 다시 읽고 역색인을 다시 짓는다.
    jabber 가 0.9MB, ko_voice 가 4KB 라 몇 벌을 들고 있어도 메모리는 문제가
    되지 않는다.
    """
    target = Path(path) if path else CHUNK_META_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"청크 메타데이터가 없습니다: {target}\n"
            f"    python data/build_chunk_meta.py 를 먼저 돌리세요.")

    z = np.load(target, allow_pickle=False)
    info = json.loads(str(z["info"]))
    nicks = [str(n) for n in z["nicks"]]
    n_rows = int(z["chunk_index"].shape[0])

    indptr = z["dyad_indptr"]
    dyad_a, dyad_b = z["dyad_a"], z["dyad_b"]
    # 쌍 -> 그 쌍을 가진 청크. 공지 하나가 쌍 196개를 갖기도 해서 M > N 이다.
    dyad_row = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(indptr))

    meta = ChunkMeta(
        chunk_index=z["chunk_index"],
        session_id=z["session_id"],
        ts_start=z["ts_start"],
        ts_end=z["ts_end"],
        n_messages=z["n_messages"],
        nicks=nicks,
        info=info,
        # 옛 판본 .npz 에는 has_time 이 없다. 그때는 문서가 jabber 하나뿐이고
        # 시각이 반드시 있었으므로 True 가 맞는 기본값이다.
        has_time=bool(info.get("has_time", True)),
        meta_key=str(info.get("meta_key", "")),
        dyad_row=dyad_row,
        dyad_a=dyad_a,
        dyad_b=dyad_b,
    )
    meta._nick_id = {n: i for i, n in enumerate(nicks)}

    # 닉 id -> 그 닉이 낀 청크 행. 쌍의 양쪽을 모두 훑는다.
    both_ids = np.concatenate([dyad_a, dyad_b])
    both_rows = np.concatenate([dyad_row, dyad_row])
    order = np.argsort(both_ids, kind="stable")
    sorted_ids, sorted_rows = both_ids[order], both_rows[order]
    bounds = np.searchsorted(sorted_ids, np.arange(len(nicks) + 1))
    meta._by_nick = [np.unique(sorted_rows[bounds[i]:bounds[i + 1]])
                     for i in range(len(nicks))]

    # (a, b) -> 그 쌍이 있는 청크 행. 서로 다른 쌍이 1,300개 남짓이라
    # 사전 하나로 충분하다.
    by_pair: dict[tuple[int, int], list[int]] = {}
    for a, b, r in zip(dyad_a.tolist(), dyad_b.tolist(), dyad_row.tolist()):
        by_pair.setdefault((a, b), []).append(r)
    meta._by_pair = {k: np.array(v, dtype=np.int32) for k, v in by_pair.items()}

    # 소문자 -> 실제 닉. 겹치는 것이 있으면 그 키는 빼 버린다 (임의로 하나를
    # 고르면 조용히 다른 사람을 찾게 된다).
    lower: dict[str, str | None] = {}
    for n in nicks:
        key = n.lower()
        lower[key] = None if key in lower else n
    meta._lower = {k: v for k, v in lower.items() if v is not None}
    return meta


EMPTY_NICKS = {"n_nicks": 0, "nicks": [], "by_lower": {}, "counts": {}}


@lru_cache(maxsize=8)
def load_nicks(path: str | None = None) -> dict:
    """
    닉네임 명부. 2단계 추출기가 화이트리스트로 쓴다.

    명부가 없는 문서도 있다 (ko_voice). 그런 문서는 빈 명부를 돌려준다 —
    파일이 없다고 예외를 던지면, 명부가 필요 없는 문서 하나 때문에 파이프라인이
    통째로 멈춘다.
    """
    target = Path(path) if path else NICKS_PATH
    if not target.is_file():
        return dict(EMPTY_NICKS)
    return json.loads(target.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 문서 -> 메타데이터
# --------------------------------------------------------------------------

def meta_for(doc: "Document | str | None") -> ChunkMeta | None:
    """
    문서 하나의 메타데이터. 없는 문서면 None.

    **파이프라인은 이 함수만 쓴다.** load_chunk_meta() 를 직접 부르면 문서와
    상관없이 jabber 것이 오고, 그러면 ko_voice 5번 청크가 jabber 5번 청크의
    대화쌍을 물려받는다 (번호로만 찾기 때문이다). 여기를 거치면 문서마다
    자기 파일이 온다.

    None 을 돌려주는 경우가 둘이다.
        1. 메타데이터를 아예 두지 않는 문서 (meta_key 가 None)
        2. 등록은 됐는데 .npz 를 아직 안 만든 문서
    둘 다 "사람·기간으로는 못 좁히는 문서" 라는 뜻이라 부르는 쪽에서는 같다.
    """
    target = doc if isinstance(doc, Document) else find(doc)
    if target is None or not target.has_meta:
        return None
    return load_chunk_meta(str(target.meta_path))


def nicks_for(doc: "Document | str | None") -> dict:
    """문서 하나의 명부. 없으면 빈 명부."""
    target = doc if isinstance(doc, Document) else find(doc)
    if target is None or not target.has_nicks:
        return dict(EMPTY_NICKS)
    return load_nicks(str(target.nicks_path))


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def _preview(doc: str, rows: list[int], meta: ChunkMeta, n: int = 5) -> None:
    """청크 본문을 몇 줄 보여 준다. 메타데이터가 맞는지 눈으로 확인하는 용도."""
    from corpus import find
    target = find(doc)
    if target is None:
        return
    with np.load(target.emb_path, allow_pickle=False) as z:
        texts = z["texts"]
    for r in rows[:n]:
        body = " ".join(str(texts[r]).split())
        print(f"      {body[:150]}{'...' if len(body) > 150 else ''}")


def _report(name: str, mask, meta: ChunkMeta, doc: str | None) -> np.ndarray:
    if mask is None:
        print(f"  {name:<28} (조건 없음 / 모르는 닉)")
        return meta.all()
    rows = np.flatnonzero(mask)
    print(f"  {name:<28} {len(rows):>7,}개")
    for r in rows[:3]:
        print(f"      {meta.describe(int(r))}")
        if doc:
            _preview(doc, [int(r)], meta, n=1)
    return mask


def _qa_selfcheck(meta: ChunkMeta | None = None) -> None:
    """
    qa.json 문항으로 **메타데이터 자체**를 검증한다.

    질문에서 규칙만으로 뽑을 수 있는 것(사람, 날짜)을 뽑아 필터를 걸고, 정답
    청크가 살아남는지 본다. 여기서 떨어지면 추출기를 아무리 고쳐도 소용없다.
    extract_metadata.py --qa 와 다른 점은 4B 를 아예 안 쓴다는 것이다.

    두 가지가 고쳐졌다.

        문서별   문항마다 doc 이 다르므로 메타데이터도 그 문서 것을 읽는다.
                 예전에는 jabber 하나로 고정이라, ko 문항을 넣으면 청크 번호가
                 jabber 것으로 읽혀 엉뚱한 청크를 정답으로 셌다.
        패턴     "A와 B 사이의 대화" 만 보던 정규식을 extract_metadata 의 것과
                 같은 것으로 바꿨다. qa.json 은 "A와 B의 대화속에" 라고 적혀
                 있어서 옛 정규식은 한 문항도 못 잡았고, 그래서 이 표가 늘
                 0/0 이었다 (검사를 하고 있다고 착각하기 딱 좋다).
    """
    from corpus import find
    from extract_metadata import NAME_ASCII, NAME_HANGUL, _dates_of, _people_res
    from filter_metadata import parse_span

    qa_path = CHUNK_META_PATH.parent.parent / "qa" / "qa.json"
    if not qa_path.is_file():
        print(f"[!] qa.json 이 없습니다: {qa_path}")
        return
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))["qa_pairs"]

    names = {p["q_type"]: p.get("q_type_name", f"유형 {p['q_type']}")
             for p in pairs}
    tally: dict[tuple, dict] = {}
    skipped: list[str] = []

    for p in pairs:
        doc_key = p.get("doc") or ""
        target = None
        try:
            target = find(doc_key) if doc_key else None
        except KeyError:
            target = None
        here = meta_for(target) if target is not None else meta
        if here is None:
            skipped.append(f"{p['id']}({doc_key or 'doc 없음'})")
            continue

        question = p["question"]
        hangul = any(str(n) and not str(n).isascii() for n in here.nicks)
        res = _people_res(NAME_HANGUL if hangul else NAME_ASCII)

        # 사람 조건: 규칙과 같은 순서로 본다 (쌍 -> 방향 -> 한 명)
        people: list[str] = []
        m = res["dyad"].search(question)
        if m:
            people = [m.group(1), m.group(2)]
        else:
            m = res["directed"].search(question) or res["solo"].search(question)
            if m:
                people = [m.group(1)]

        since_list, until_list = _dates_of(question)
        since = parse_span(min(since_list), end=False) if since_list else None
        until = parse_span(max(until_list), end=True) if until_list else None

        mask, bits = None, []
        if people:
            got = here.mask_people(people)
            if got is not None:
                mask, _ = got, bits.append("+".join(people))
        if since or until:
            got = here.mask_time(since, until)
            if got is not None:
                mask = got if mask is None else (mask & got)
                bits.append(f"{since or '...'}~{until or '...'}")

        key = (doc_key, p["q_type"])
        row = tally.setdefault(key, {"n": 0, "ok": 0, "miss": [], "none": [],
                                     "pool": 0, "size": here.size})
        row["n"] += 1
        if mask is None:
            row["none"].append(str(p["id"]))
            row["pool"] += here.size
            continue

        row_of = {int(c): i for i, c in enumerate(here.chunk_index)}
        gold = [row_of[c] for c in p["answer_chunk_indices"] if c in row_of]
        kept = int(mask[gold].sum()) if gold else 0
        row["pool"] += int(mask.sum())
        if gold and kept == len(gold):
            row["ok"] += 1
        else:
            row["miss"].append(f"{p['id']}({','.join(bits)}) {kept}/{len(gold)}")

    print(f"\n  {'문서':<11}{'유형':<14}{'문항':>5}{'정답전부':>9}"
          f"{'평균후보':>10}{'축소율':>9}{'조건없음':>9}")
    for key in sorted(tally):
        row = tally[key]
        avg = row["pool"] / row["n"]
        print(f"  {key[0][:10]:<11}{str(names.get(key[1], key[1])):<14}"
              f"{row['n']:>5}{row['ok']:>9}{avg:>10,.0f}"
              f"{100 * avg / row['size']:>8.1f}%{len(row['none']):>9}")
        if row["miss"]:
            print(f"      [!] 정답이 걸러진 문항: {', '.join(row['miss'])}")
        if row["none"]:
            print(f"      규칙으로 조건을 못 뽑은 문항: "
                  f"{', '.join(row['none'])} (전체를 뒤진 것으로 셌다)")

    if skipped:
        print(f"\n  [!] 건너뛴 문항 (메타데이터 없음): {', '.join(skipped)}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="청크 메타데이터로 행을 좁혀 본다.")
    ap.add_argument("--dyad", help="무순 쌍. 'poll,stern'")
    ap.add_argument("--who", help="사람 (여럿이면 쉼표로)")
    ap.add_argument("--since", help="시작 (2020-09-29)")
    ap.add_argument("--until", help="끝 (2020-09-29)")
    ap.add_argument("--doc", help="코퍼스 (ru / en / ko). 그 문서의 "
                                  "메타데이터를 읽고 본문도 그것으로 보여 준다")
    ap.add_argument("--qa", action="store_true", help="qa.json 자가 검증")
    args = ap.parse_args()

    # 문서를 고르면 그 문서의 메타데이터를 읽는다. 안 고르면 jabber.
    # 여기서 문서를 안 보면 --doc ko --who 가해자3 이 jabber 명부를 뒤지고,
    # 보이스피싱 본문 옆에 Conti 대화쌍과 2020년 시각이 붙어 나온다.
    meta = meta_for(args.doc) if args.doc else load_chunk_meta()
    if meta is None:
        sys.exit(f"{args.doc}: 메타데이터가 없습니다. "
                 f"python data/build_chunk_meta.py --doc all 를 돌리세요.")

    when = (f" · {from_epoch(meta.ts_start.min())[:10]} ~ "
            f"{from_epoch(meta.ts_end.max())[:10]}" if meta.has_time
            else " · 시각 없음")
    print(f"[{meta.meta_key or 'jabber'}] 청크 {meta.size:,}개 · "
          f"이름 {len(meta.nicks)}개 · 쌍 {meta.dyad_row.shape[0]:,}개{when}")

    if args.qa:
        _qa_selfcheck(meta)
        return

    if not any((args.dyad, args.who, args.since, args.until)):
        print("\n대화가 많은 닉 10명")
        top = sorted(meta.nicks,
                     key=lambda n: -len(meta._by_nick[meta._nick_id[n]]))[:10]
        for n in top:
            print(f"  {n:<14} {meta.counts(n)['chunks']:>6,}개 청크")
        print("\n예:  python src/chunk_meta.py --dyad poll,stern --doc en")
        return

    print()
    mask = meta.all()
    if args.dyad:
        a, b = (x.strip() for x in args.dyad.split(",", 1))
        mask &= _report(f"dyad {a} ~ {b}", meta.mask_dyad(a, b), meta, args.doc)
    if args.who:
        names = [x.strip() for x in args.who.split(",") if x.strip()]
        mask &= _report(f"사람 {', '.join(names)}", meta.mask_people(names),
                        meta, args.doc)
    if args.since or args.until:
        mask &= _report(f"기간 {args.since or '...'} ~ {args.until or '...'}",
                        meta.mask_time(args.since, args.until), meta, args.doc)

    rows = np.flatnonzero(mask)
    print(f"\n조건을 모두 만족: {len(rows):,}개 "
          f"(전체 {meta.size:,}개의 {100 * len(rows) / meta.size:.2f}%)")
    for r in rows[:10]:
        print(f"  {meta.describe(int(r))}")
        if args.doc:
            _preview(args.doc, [int(r)], meta, n=1)
    if len(rows) > 10:
        print(f"  ... 외 {len(rows) - 10:,}개")


if __name__ == "__main__":
    main()
