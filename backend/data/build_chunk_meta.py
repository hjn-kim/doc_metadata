# -*- coding: utf-8 -*-
"""청크 .jsonl -> data/meta/{메타키}.npz (+ _nicks.json)

메타데이터 필터가 읽을 청크별 메타데이터를 굳힌다. 검색 때 다시 계산하지
않도록 numpy 배열로 떨어뜨리는 것이 목적이다.

**문서마다 한 벌씩 만든다.** 예전에는 문서가 jabber 하나뿐이라 chunk_meta.npz
한 개면 됐지만, 문서가 늘면 그 한 개가 조용히 틀린 답을 준다. 파일은 청크
번호(0..N-1)로만 찾으므로, ko_voice 5번 청크가 jabber 5번 청크의 메타데이터를
그대로 물려받아 버리기 때문이다.

    data/meta/jabber.npz          Conti Jabber 로그 (ru/en 공용)
    data/meta/jabber.jsonl        같은 내용을 사람이 읽는 형태로 (검수용)
    data/meta/jabber_nicks.json   닉네임 명부. 2단계 추출기의 화이트리스트
    data/meta/ko_voice.npz        보이스피싱 통화록
    data/meta/ko_voice.jsonl

'문서키' 가 아니라 '메타키' 로 나누는 이유는 jabber_ru 와 jabber_en 때문이다.
둘은 같은 대화를 번역만 한 것이라 청크 번호가 완전히 일치한다. 메타데이터
한 벌이 두 언어에 그대로 들어맞으므로 문서별로 쪼개면 똑같은 파일이 두 벌
생긴다. corpus.py 의 DOC_SPECS 에서 두 문서가 같은 meta_key 를 가리킨다.

시각이 없는 문서도 있다:

    ko_voice 는 원문에 통화 시각이 적혀 있지 않아 ts 가 전부 null 이다.
    그런 문서는 ts 자리에 NO_TS 를 적고 info["has_time"] 을 false 로 둔다.
    기간 조건은 애초에 이 문서에 걸리지 않는다 (corpus.py DOC_SPECS 참고).

담는 것은 dyad 와 ts 둘뿐이다:

    dyad(A,B)   A 와 B 가 그 청크에서 대화했다 (무순 쌍)
    ts_start/ts_end   그 청크가 걸친 구간

    sender / receiver 는 두지 않는다. 이 로그에서 방향을 물어봐야 할 일이
    없기 때문이다. "sender: tom" 이라는 질문이 들어와도 결국 찾는 것은 tom 이
    낀 대화이고, 그것은 dyad 안에 tom 이 있는지만 보면 된다. 방향을 따로
    저장하면 (1) 원본 CSV 를 되짚어 from/to 를 모아야 하고 (2) 필터가
    sender·receiver·pair·participant 네 갈래로 늘어나는데, 얻는 것은 한 사람만
    말한 청크에서의 미세한 차이뿐이다.

    참여자 필터도 따로 두지 않는다. participant(A) == dyad 한쪽이 A 인 청크다.

    그래서 입력이 sessions.jsonl 하나로 줄었다. 예전에는 본문에만 있는 화자를
    되짚으려고 jabberchat2020process.csv 를 다시 읽었지만, dyad 는 세션이 이미
    들고 있다.

청크 하나에 dyad 가 여럿일 수 있다:

    conversation (15,491개)  참여자 두 명 -> dyad 1개
    broadcast    (31개)      발신자 1명 x 수신자 5~196명 -> dyad 그만큼

    공지에서 dyad 를 '참여자 교집합'으로 잡으면 안 된다. 수신자 196명짜리
    공지에서 서로 말한 적 없는 수신자 두 명이 한 쌍으로 묶여 버린다
    (한 청크에서 19,110쌍). (발신자, 수신자) 쌍만 만든다.

    'conversation / broadcast' 같은 종류 이름표도 두지 않는다. 이 로그에서는
    수신자가 많으면 공지인데, 그건 dyad 개수를 보면 아는 것이지 따로 이름표를
    붙일 일이 아니다. 이름표를 두면 필터마다 "그런데 broadcast 는..." 하는
    예외가 붙고, 그 예외는 이 데이터에만 있는 것이다.

까다로운 것 하나:

    self-dyad 68개. from == to 인 행이 있어 participants 가
    ['alarm2','alarm2'] 로 나온다. 자기 자신에게 보낸 메모다. 지우지 않고
    그대로 둔다 — dyad(a,a) 로 찾을 수 있어야 한다.

사용:
    python data/build_chunk_meta.py
    python data/build_chunk_meta.py --check      # 저장하지 않고 검증만
    python data/build_chunk_meta.py --no-jsonl   # npz 만
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_DIR = os.path.join(ROOT, "data", "chunks")
META_DIR = os.path.join(ROOT, "data", "meta")

# preprocess 가 dyad 를 만든 규칙. 여기서도 똑같이 맞춰야 필터가 문자열로
# 비교했을 때 어긋나지 않는다.
DYAD_SEP = " | "

# 시각이 없는 청크에 적어 두는 값. epoch 로는 있을 수 없는 수라 덤프를 눈으로
# 봐도 "시각 없음" 이 바로 보이고, 기간 마스크에 넣으면 반드시 False 가 된다
# (ts_end >= since 가 성립할 수 없다). 0 이나 -1 을 쓰면 1970년으로 읽혀서
# "1969년 이후" 같은 질문에 조용히 걸린다.
NO_TS = int(np.iinfo(np.int64).min)

# 메타키 -> 무엇을 읽어 무엇을 만들지.
#
#     input     청크 메타데이터가 들어 있는 .jsonl
#     has_time  ts_start/ts_end 가 있는가
#     nicks     명부(_nicks.json)를 만들 것인가
#     docs      이 메타데이터를 함께 쓰는 문서들 (임베딩 행 수 검증에 쓴다)
#
# jabber 의 입력이 sessions.jsonl 인 이유는 본문이 없는 메타데이터 전용 파일이라
# 두 언어 어느 쪽에도 치우치지 않기 때문이다. 다른 문서는 그런 중간 파일 없이
# 자기 청크 .jsonl 이 메타데이터를 이미 들고 있다.
SOURCES: dict[str, dict] = {
    "jabber": {
        "input": os.path.join(CHUNKS_DIR, "sessions.jsonl"),
        "has_time": True,
        "nicks": True,
        "docs": ("jabber_ru", "jabber_en"),
    },
    "ko_voice": {
        "input": os.path.join(CHUNKS_DIR, "ko_voice.jsonl"),
        "has_time": False,
        "nicks": False,
        "docs": ("ko_voice",),
    },
}


# --------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------
def load_sessions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"세션이 비어 있습니다: {path}")

    # chunk_index 가 0..N-1 로 빠짐없이 늘어서야 .npz 의 행 번호와 그대로 맞는다.
    got = [r["chunk_index"] for r in rows]
    if got != list(range(len(rows))):
        raise ValueError("chunk_index 가 0..N-1 이 아닙니다. "
                         "chunking.py 출력을 확인하세요.")
    return rows


# --------------------------------------------------------------------------
# 청크 하나의 메타데이터
# --------------------------------------------------------------------------
def trim_seconds(value: str) -> str:
    """'2020-07-09T21:57:20.123456' -> '2020-07-09T21:57:20' (내림).

    소수점 이하는 버린다. 필터가 보는 것은 날짜나 시각대이지 마이크로초가
    아니고, 남겨 둬 봐야 사람이 읽는 chunk_meta.jsonl 만 지저분해진다.

    반올림이 아니라 버림이다. ts_start 는 뒤로 가지 않으므로 구간이 앞으로
    늘어나고, ts_end 만 최대 1초 짧아진다. 질의가 날짜 단위(2020-09-29)라
    이 1초가 경계에 걸리는 일은 없다. 초 단위로 자른 구간을 다시 초 미만으로
    물어볼 일이 생기면 그때는 여기가 아니라 질의 쪽을 초 단위로 맞춰야 한다.
    """
    text = str(value)
    return text.split(".", 1)[0]


def dyads_of(session: dict) -> list[tuple[str, str]]:
    """
    세션 한 줄 -> 무순 쌍 목록. 쌍 안은 항상 정렬해 둔다 (a <= b).

    참여자가 둘이면 그 둘이 곧 dyad 다. 공지면 참여자(발신자) 한 명과
    recipients 명부를 곱한다.
    """
    people = [str(p) for p in session["participants"]]
    recipients = [str(x) for x in session.get("recipients", [])]

    if recipients:
        pairs = {tuple(sorted((s, r))) for s in people for r in recipients}
    elif len(people) == 2:
        pairs = {tuple(sorted(people))}
    elif len(people) == 1:                    # 명부가 지워진 공지 (있으면)
        pairs = {(people[0], people[0])}
    else:
        pairs = {tuple(sorted((a, b)))
                 for i, a in enumerate(people) for b in people[i + 1:]}

    return sorted(pairs)


def chunk_meta(session: dict, has_time: bool = True) -> dict:
    """
    세션 한 줄 -> 필터가 쓸 메타데이터 한 줄.

    has_time 이 False 면 ts 를 아예 읽지 않는다. 원문에 시각이 없는 문서에서
    억지로 값을 지어내면(예: 파일 순서를 시각으로) 기간 필터가 그럴듯하게
    틀린 답을 낸다. 없는 것은 없다고 적는 편이 낫다.
    """
    pairs = dyads_of(session)
    if not pairs:
        raise ValueError(f"청크 {session['chunk_index']}: 참여자가 없습니다.")

    return {
        "chunk_index": int(session["chunk_index"]),
        "session_id": int(session["session_id"]),
        "dyads": [DYAD_SEP.join(p) for p in pairs],
        "ts_start": (trim_seconds(session["ts_start"])
                     if has_time and session.get("ts_start") else None),
        "ts_end": (trim_seconds(session["ts_end"])
                   if has_time and session.get("ts_end") else None),
        "n_messages": int(session["n_messages"]),
        "n_tokens": int(session.get("n_tokens", 0)),
        "part": int(session.get("part", 1)),
        "n_parts": int(session.get("n_parts", 1)),
        # 저장하지 않는다. 검증에만 쓴다.
        "_pairs": pairs,
    }


def build(sessions_path: str, has_time: bool = True) -> tuple[list[dict], dict]:
    sessions = load_sessions(sessions_path)
    rows = [chunk_meta(s, has_time) for s in sessions]

    n_dyad = Counter(len(r["_pairs"]) for r in rows)
    n_self = sum(1 for r in rows if any(a == b for a, b in r["_pairs"]))
    nicks = {n for r in rows for p in r["_pairs"] for n in p}
    stat = {
        "청크": len(rows),
        "이름": len(nicks),
        "dyad 1개 (1:1)": n_dyad.get(1, 0),
        "dyad 여러 개 (공지)": sum(v for k, v in n_dyad.items() if k > 1),
        "dyad 총 개수": sum(len(r["_pairs"]) for r in rows),
        "서로 다른 dyad": len({p for r in rows for p in r["_pairs"]}),
        "자기 자신에게": n_self,
    }
    return rows, stat


# --------------------------------------------------------------------------
# 저장 — 검색 때 numpy 로 바로 마스킹할 수 있는 형태
# --------------------------------------------------------------------------
def to_epoch(values: list[str | None]) -> np.ndarray:
    """
    ISO 문자열 -> epoch 초(int64). 시간 비교를 정수 연산으로 만든다.

    None 은 NO_TS 로 간다. 시각이 없는 문서는 전부 None 이라 배열이 통째로
    NO_TS 가 되고, 그러면 기간 마스크가 그 문서에서 아무것도 고르지 못한다 —
    "모르는 것은 걸리지 않는다" 가 "모르는 것은 다 걸린다" 보다 안전하다.
    """
    if all(v is None for v in values):
        return np.full(len(values), NO_TS, dtype=np.int64)

    ts = pd.to_datetime(pd.Series(values)).dt.floor("s")
    out = (ts.astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)
    out[pd.isna(ts).to_numpy()] = NO_TS
    return out


def to_indptr(lists: list[list]) -> np.ndarray:
    """가변 길이 목록 -> 경계 배열. .npz 는 들쭉날쭉한 배열을 못 담는다."""
    indptr = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in lists], out=indptr[1:])
    return indptr


def save_npz(rows: list[dict], out_path: str, src_path: str,
             meta_key: str, has_time: bool, docs: tuple[str, ...]) -> dict:
    """
    저장하는 것은 dyad 쌍 CSR 과 시각뿐이다.

    쌍은 문자열이 아니라 닉 id 두 개로 적는다. dyad 문자열만 해도 1.5MB 로
    파일의 절반이었고, 공지처럼 쌍이 여럿인 청크는 애초에 한 문자열로 적을
    수도 없다. 이름은 nicks 배열에서 되찾는다.

        dyad_indptr[i]:dyad_indptr[i+1]   i번 청크가 가진 쌍의 범위
        dyad_a[k], dyad_b[k]              k번 쌍의 두 닉 id (a <= b)

    "tom 이 낀 청크" = dyad_a 나 dyad_b 가 tom 인 쌍을 가진 청크.
    "tom 과 stern" = 그 쌍이 (tom, stern) 인 청크.
    """
    # 닉네임 명부. 정렬해 두면 id 가 판본마다 바뀌지 않는다.
    nicks = sorted({n for r in rows for p in r["_pairs"] for n in p})
    nick_id = {n: i for i, n in enumerate(nicks)}

    dyad_ptr = to_indptr([r["_pairs"] for r in rows])
    flat = [p for r in rows for p in r["_pairs"]]
    dyad_a = np.fromiter((nick_id[a] for a, _ in flat), dtype=np.int32,
                         count=len(flat))
    dyad_b = np.fromiter((nick_id[b] for _, b in flat), dtype=np.int32,
                         count=len(flat))

    info = {
        "meta_key": meta_key,
        "docs": list(docs),
        "sessions": os.path.basename(src_path),
        "n_chunks": len(rows),
        "n_nicks": len(nicks),
        "n_dyads": len(flat),
        "has_time": bool(has_time),
        "model": "청크 = 무순 쌍(dyad)의 묶음. participant(A) 는 "
                 "dyad 한쪽이 A 인 것. 방향(sender/receiver)은 두지 않는다",
        "ts_unit": ("epoch seconds (naive = UTC 로 고정. 원본 로그 시각 그대로)"
                    if has_time else f"없음 (ts 는 전부 NO_TS={NO_TS})"),
        "note": f"chunk_index 는 {' / '.join(docs)} 임베딩 .npz 의 행 번호와 같다",
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        chunk_index=np.array([r["chunk_index"] for r in rows], dtype=np.int32),
        session_id=np.array([r["session_id"] for r in rows], dtype=np.int32),
        ts_start=to_epoch([r["ts_start"] for r in rows]),
        ts_end=to_epoch([r["ts_end"] for r in rows]),
        n_messages=np.array([r["n_messages"] for r in rows], dtype=np.int32),
        n_tokens=np.array([r["n_tokens"] for r in rows], dtype=np.int32),
        part=np.array([r["part"] for r in rows], dtype=np.int32),
        n_parts=np.array([r["n_parts"] for r in rows], dtype=np.int32),
        nicks=np.array(nicks),
        dyad_indptr=dyad_ptr, dyad_a=dyad_a, dyad_b=dyad_b,
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )
    return info


def save_nicks(rows: list[dict], path: str) -> dict:
    """
    닉네임 명부 — 2단계 추출기의 화이트리스트.

    LLM 이 뽑아 온 이름을 여기에 대조해서, 없으면 null 로 강등한다. 소문자 키를
    같이 실어 두는 이유는 질문에 'Stern' 처럼 대문자로 적혀 오기 때문이다.
    """
    seen = Counter(n for r in rows for n in {x for p in r["_pairs"] for x in p})

    nicks = sorted(seen)
    lower: dict[str, list[str]] = {}
    for n in nicks:
        lower.setdefault(n.lower(), []).append(n)

    payload = {
        "n_nicks": len(nicks),
        "nicks": nicks,
        # 소문자 -> 실제 닉. 소문자가 겹치는 닉이 있으면 그대로 여러 개 남긴다
        # (임의로 하나를 고르면 조용히 틀린 사람을 찾게 된다).
        "by_lower": {k: v for k, v in sorted(lower.items())},
        "ambiguous_lower": {k: v for k, v in sorted(lower.items())
                            if len(v) > 1},
        # 그 닉이 낀 청크 수. 방향은 따지지 않는다.
        "counts": {n: int(seen[n]) for n in nicks},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


def save_jsonl(rows: list[dict], path: str) -> None:
    """
    검수용. npz 는 눈으로 못 보니 같은 내용을 한 벌 더 둔다.

    검증용 밑줄 필드는 뺀다 (원본은 sessions.jsonl 에 있다).
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(
                {k: v for k, v in r.items() if not k.startswith("_")},
                ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 검증
# --------------------------------------------------------------------------
def check(rows: list[dict], sessions_path: str,
          docs: tuple[str, ...] = ()) -> list[str]:
    """조용히 틀리면 검색이 빈 결과를 내므로, 틀릴 수 있는 곳을 다 짚는다."""
    bad: list[str] = []

    # 1. 쌍 안이 정렬되어 있는가. dyad 를 문자열로 비교하는 쪽이 있어서
    #    (a,b) 와 (b,a) 가 섞이면 조용히 절반을 놓친다.
    unsorted = sum(1 for r in rows for a, b in r["_pairs"] if a > b)
    if unsorted:
        bad.append(f"쌍 안이 뒤집힌 dyad {unsorted}건")

    # 2. 빈 이름이 섞여 있는가 (닉 명부에 '' 가 들어가면 필터가 전부 걸린다)
    empty = sum(1 for r in rows for p in r["_pairs"] for n in p if not n.strip())
    if empty:
        bad.append(f"이름이 빈 dyad {empty}건")

    # 3. 세션 참여자가 dyad 밖으로 새지 않는가
    with open(sessions_path, encoding="utf-8") as f:
        sessions = [json.loads(line) for line in f if line.strip()]
    for s, r in zip(sessions, rows):
        people = {str(p) for p in s["participants"]}
        got = {n for p in r["_pairs"] for n in p}
        if not people <= got:
            bad.append(f"청크 {r['chunk_index']}: 참여자 "
                       f"{sorted(people - got)} 가 dyad 에 없음")

    # 4. 시각이 뒤집힌 청크. 시각이 없는 문서는 볼 것이 없다.
    timed = [r for r in rows if r["ts_start"] and r["ts_end"]]
    flipped = sum(1 for r in timed if r["ts_start"] > r["ts_end"])
    if flipped:
        bad.append(f"ts_start > ts_end 인 청크 {flipped}건")
    if timed and len(timed) != len(rows):
        bad.append(f"시각이 있는 청크와 없는 청크가 섞여 있습니다 "
                   f"({len(timed)}/{len(rows)}건). 기간 필터가 절반만 걸립니다")

    # 5. dyad 가 하나도 없는 청크 (있으면 닉 필터가 그 청크를 영영 못 찾는다)
    silent = sum(1 for r in rows if not r["_pairs"])
    if silent:
        bad.append(f"dyad 가 없는 청크 {silent}건")

    # 6. 임베딩 .npz 와 청크 수가 같은가.
    #
    #    이 메타데이터를 쓰는 문서만 본다. 예전에는 emb 디렉터리를 통째로
    #    훑었는데, 문서가 늘면 관계없는 문서의 행 수까지 견주게 되어 멀쩡한
    #    빌드가 매번 경고를 뱉는다. 행 수가 맞아야 하는 것은 같은 청크를
    #    가리키는 문서들뿐이다.
    emb_dir = os.path.join(ROOT, "data", "emb")
    if docs and os.path.isdir(emb_dir):
        for model in sorted(os.listdir(emb_dir)):
            d = os.path.join(emb_dir, model)
            if not os.path.isdir(d):
                continue
            for key in docs:
                path = os.path.join(d, f"{key}_embeddings.npz")
                if not os.path.isfile(path):
                    continue
                with np.load(path, allow_pickle=False) as z:
                    n = int(z["embeddings"].shape[0])
                if n != len(rows):
                    bad.append(f"{model}/{key}: 청크 {n:,}개 "
                               f"(메타데이터는 {len(rows):,}개)")
    return bad


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------
def build_one(meta_key: str, save: bool, want_jsonl: bool) -> list[str]:
    """
    메타키 하나를 굳힌다. 돌려주는 것은 검증에서 걸린 문제 목록이다.

    문서별로 따로 도는 이유는 실패를 가두기 위해서다. ko_voice 가 깨졌다고
    jabber 메타데이터까지 안 만들어지면, 멀쩡하던 검색이 새 문서 하나 때문에
    통째로 멈춘다.
    """
    src = SOURCES[meta_key]
    print(f"\n[{meta_key}]  {os.path.relpath(src['input'], ROOT)}")

    if not os.path.isfile(src["input"]):
        print(f"  [!] 입력이 없습니다: {src['input']}")
        return [f"{meta_key}: 입력 없음"]

    rows, stat = build(src["input"], src["has_time"])
    for k, v in stat.items():
        print(f"  {k:<20}{v:>9,}")
    if not src["has_time"]:
        print(f"  {'시각':<20}{'없음':>9}")

    problems = check(rows, src["input"], src["docs"])
    for bad in problems:
        print(f"  [!] {bad}")
    if not problems:
        print("  검증 이상 없음")

    if not save:
        return problems

    os.makedirs(META_DIR, exist_ok=True)
    out = os.path.join(META_DIR, f"{meta_key}.npz")
    info = save_npz(rows, out, src["input"], meta_key,
                    src["has_time"], src["docs"])
    print(f"  저장: {os.path.relpath(out, ROOT)}  "
          f"({os.path.getsize(out) / 1e6:.1f}MB · 이름 {info['n_nicks']}개 · "
          f"쌍 {info['n_dyads']:,}개)")

    if src["nicks"]:
        nick_path = os.path.join(META_DIR, f"{meta_key}_nicks.json")
        nicks = save_nicks(rows, nick_path)
        print(f"  저장: {os.path.relpath(nick_path, ROOT)}")
        if nicks["ambiguous_lower"]:
            print(f"  [!] 대소문자만 다른 닉 {len(nicks['ambiguous_lower'])}쌍: "
                  f"{list(nicks['ambiguous_lower'].values())[:3]}")
    else:
        # 명부를 안 만드는 문서. 이름이 '가해자1' 처럼 역할 이름뿐이라
        # 질문에 그렇게 적혀 올 일이 없고, 화이트리스트가 할 일이 없다.
        # dyad 를 적기 위한 이름표는 .npz 안에 그대로 들어 있다.
        print("  명부: 만들지 않음 (질문에 나올 이름이 아니다)")

    if want_jsonl:
        js = os.path.join(META_DIR, f"{meta_key}.jsonl")
        save_jsonl(rows, js)
        print(f"  저장: {os.path.relpath(js, ROOT)}  "
              f"({os.path.getsize(js) / 1e6:.1f}MB)")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(
        description="청크별 메타데이터(대화쌍·시각)를 문서마다 한 벌씩 굳힌다.")
    ap.add_argument("--doc", default="all",
                    choices=sorted(SOURCES) + ["all"],
                    help="굳힐 메타키 (기본 all)")
    ap.add_argument("--no-jsonl", action="store_true", help="검수용 jsonl 생략")
    ap.add_argument("--check", action="store_true", help="저장하지 않고 검증만")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    keys = sorted(SOURCES) if args.doc == "all" else [args.doc]
    problems: list[str] = []
    for key in keys:
        problems += build_one(key, save=not args.check,
                              want_jsonl=not args.no_jsonl)

    if args.check:
        print(f"\n--check 라 저장하지 않았습니다.")
    if problems:
        print(f"\n[!] 문제 {len(problems)}건")
        sys.exit(1)


if __name__ == "__main__":
    main()
