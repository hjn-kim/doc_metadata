#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
청크 메타데이터 — 검색이 읽는 쪽

    data/build_chunk_meta.py  ->  data/chunks/chunk_meta.npz  ->  여기

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

from corpus import CHUNK_META_PATH, NICKS_PATH  # noqa: E402

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
        """
        lo = to_epoch(since)
        hi = to_epoch(until, end_of_day=True)
        if lo is None and hi is None:
            return None
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
        return (f"#{self.chunk_index[row]:<6} {who:<28} "
                f"{from_epoch(self.ts_start[row])} ~ "
                f"{from_epoch(self.ts_end[row])[11:]} "
                f"({self.n_messages[row]}건)")

    def counts(self, nick: str) -> dict:
        real = self.resolve(nick)
        if real is None:
            return {}
        return {"nick": real, "chunks": len(self._by_nick[self._nick_id[real]])}


# --------------------------------------------------------------------------
# 로드
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_chunk_meta(path: str | None = None) -> ChunkMeta:
    """
    chunk_meta.npz 를 읽는다. 프로세스당 한 번.

    0.9MB 라 메모리는 문제가 되지 않는다. 역색인까지 만들어 두면 마스크 하나는
    불리언 배열 채우기 한 번(15,522칸)이라 사실상 공짜다.
    """
    target = Path(path) if path else CHUNK_META_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"청크 메타데이터가 없습니다: {target}\n"
            f"    python data/build_chunk_meta.py 를 먼저 돌리세요.")

    z = np.load(target, allow_pickle=False)
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
        info=json.loads(str(z["info"])),
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


@lru_cache(maxsize=1)
def load_nicks() -> dict:
    """닉네임 명부. 2단계 추출기가 화이트리스트로 쓴다."""
    if not NICKS_PATH.is_file():
        return {"n_nicks": 0, "nicks": [], "by_lower": {}, "counts": {}}
    return json.loads(NICKS_PATH.read_text(encoding="utf-8"))


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


def _qa_selfcheck(meta: ChunkMeta) -> None:
    """
    qa.json 의 화자기반·날짜기반 문항으로 메타데이터 자체를 검증한다.

    질문에서 규칙만으로 뽑을 수 있는 것(닉 쌍, 날짜)을 뽑아 필터를 걸고, 정답
    청크가 살아남는지 본다. 여기서 떨어지면 추출기를 아무리 고쳐도 소용없다.
    """
    import re

    qa_path = CHUNK_META_PATH.parent.parent / "qa" / "qa.json"
    if not qa_path.is_file():
        print(f"[!] qa.json 이 없습니다: {qa_path}")
        return
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))["qa_pairs"]

    pair_re = re.compile(r"([A-Za-z0-9_.\-]+)\s*(?:와|과)\s+"
                         r"([A-Za-z0-9_.\-]+)\s*사이의\s*대화")
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")

    row_of = {int(c): i for i, c in enumerate(meta.chunk_index)}
    tally = {3: [0, 0, [], []], 4: [0, 0, [], []]}

    for p in pairs:
        qt = p["q_type"]
        if qt not in tally:
            continue
        q = p["question"]

        if qt == 3:
            m = pair_re.search(q)
            if not m:
                tally[qt][3].append(p["id"])
                continue
            mask = meta.mask_dyad(m.group(1), m.group(2))
            label = f"{m.group(1)}~{m.group(2)}"
        else:
            m = date_re.search(q)
            if not m:
                tally[qt][3].append(p["id"])
                continue
            mask = meta.mask_time(m.group(1), m.group(1))
            label = m.group(1)

        if mask is None:
            tally[qt][3].append(f"{p['id']}({label})")
            continue

        gold = [row_of[c] for c in p["answer_chunk_indices"] if c in row_of]
        kept = int(mask[gold].sum()) if gold else 0
        tally[qt][0] += 1
        if gold and kept == len(gold):
            tally[qt][1] += 1
        else:
            tally[qt][2].append(f"{p['id']}({label}) {kept}/{len(gold)}")
        # 후보를 얼마나 줄였는지도 같이 본다
        tally[qt].append(int(mask.sum()))

    for qt, name in ((3, "화자기반 (dyad)"), (4, "날짜기반 (기간)")):
        n, ok, miss, unparsed, *pools = tally[qt]
        pool = f"평균 후보 {sum(pools) / len(pools):,.0f}개" if pools else ""
        print(f"\n  {name}: 정답 청크 온전히 통과 {ok}/{n}   {pool} "
              f"(전체 {meta.size:,}개 중)")
        if miss:
            print(f"    [!] 정답이 걸러진 문항: {', '.join(miss)}")
        if unparsed:
            print(f"    [!] 규칙으로 못 뽑은 문항: {', '.join(map(str, unparsed))}")


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
    ap.add_argument("--doc", help="본문 미리보기에 쓸 코퍼스 (ru / en)")
    ap.add_argument("--qa", action="store_true", help="qa.json 자가 검증")
    args = ap.parse_args()

    meta = load_chunk_meta()
    print(f"청크 {meta.size:,}개 · 닉 {len(meta.nicks)}명 · "
          f"쌍 {meta.dyad_row.shape[0]:,}개 · "
          f"{from_epoch(meta.ts_start.min())[:10]} ~ "
          f"{from_epoch(meta.ts_end.max())[:10]}")

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
