# -*- coding: utf-8 -*-
"""jabberchat2020process.csv -> 검색용 청크 (원문/영문 2벌)

    ① 세션 분리  dyad(대화 상대 쌍) + 시간 gap
    ② 청킹      세션 내부에서만 512/128 토큰, 발화 경계 스냅

메시지 1건은 중앙값 20자라 그대로 임베딩하면 'Ok' 3,118개가 동일 벡터가 된다.
그래서 대화 세션으로 묶은 뒤 긴 세션만 토큰 기준으로 다시 쪼갠다.

세션 분리에서 dyad 로 먼저 정렬하는 것이 핵심이다. cumsum 은 전역 누적이라
시간순 상태에서 돌리면 동시에 진행 중인 다른 대화가 한 세션에 섞인다.

gap 1시간 기준 12,478 세션이 나오고, 그중 512 토큰을 넘어 쪼개지는 것은 7%
뿐이다. gap 을 6시간까지 늘려도 초과 세션은 867 -> 931 로 거의 늘지 않는다.
작은 세션끼리 합쳐질 뿐이라 1~2시간이면 충분하다.

청크 경계는 영어를 축으로 한 번만 계산해 모든 언어에 똑같이 적용한다
(--split-basis en). 언어를 나중에 추가해도 같은 발화 경계를 그대로 쓰므로 청크
인덱스가 전 언어에서 일치하고, QA 정답(청크 위치)을 1벌만 만들면 된다. 언어별로
따로 계산하면(--split-basis each) 청크 수가 어긋나 정답을 언어 수만큼 만들어야
한다.

축이 영어라 토큰을 더 먹는 언어(러시아어 원문)는 512 를 조금 넘길 수 있다.
embed.py 의 --slack 이 그 여유를 흡수하고, 넘치면 경고를 찍는다.

사용:
    python data/chunking.py
    python data/chunking.py --gap-hours 2 --chunk-size 1024 --overlap 256
    python data/chunking.py --split-basis each     # 원문/영문 따로 분할
    python data/chunking.py --stats-only           # 저장 없이 통계만
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "jabberchat2020process.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "chunks")

TOKENIZER = "BAAI/bge-m3"
HDR_BUDGET = 32           # 청크 헤더 몫으로 떼어두는 토큰
FIELDS = ("ru", "en")     # ru=원문(body), en=영문(body_en)


# --------------------------------------------------------------------------
# ① 세션 분리
# --------------------------------------------------------------------------
def load(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.rename(columns={"body": "ru", "body_en": "en"})
    conv = df[df.kind == "conversation"].copy()
    bcast = df[df.kind == "broadcast"].copy()
    return conv, bcast


def split_sessions(conv: pd.DataFrame, gap_hours: float) -> pd.DataFrame:
    """dyad + 시간 gap 으로 세션 번호를 매긴다."""
    c = conv.sort_values(["dyad", "ts"], kind="mergesort").reset_index(drop=True)
    gap = c.groupby("dyad")["ts"].diff().dt.total_seconds()
    c["sid"] = (gap.isna() | (gap > gap_hours * 3600)).cumsum()

    if (c.groupby("sid").dyad.nunique() > 1).any():
        raise RuntimeError("세션에 여러 dyad 가 섞였습니다. 정렬 순서를 확인하세요.")
    return c


# --------------------------------------------------------------------------
# ② 청킹
# --------------------------------------------------------------------------
def count_tokens(tok, texts: list[str], batch: int = 2000,
                 label: str = "", verbose: bool = True) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int32)
    for i in range(0, len(texts), batch):
        ids = tok(texts[i:i + batch], add_special_tokens=False)["input_ids"]
        out[i:i + len(ids)] = [len(x) for x in ids]
        if verbose:
            print(f"\r  토큰 계산 {label} {min(i + batch, len(texts)):,}/{len(texts):,}",
                  end="", file=sys.stderr)
    if verbose:
        print(file=sys.stderr)
    return out


def header_of(dyad: str, ts_start, ts_end, part: int, n_parts: int) -> str:
    """청크마다 다시 붙는 대화 헤더.

    발화 줄에는 시:분만 있어 날짜가 없고, 한쪽이 연달아 말한 구간은 화자가
    1명뿐이라 상대방도 알 수 없다(전체 청크의 25.6%). 헤더가 그 둘을 담는다.
    줄마다 날짜를 넣는 대안은 토큰을 6.6% 더 쓰면서도 상대방 문제를 못 푼다.
    """
    d0, d1 = ts_start.strftime("%Y-%m-%d"), ts_end.strftime("%Y-%m-%d")
    when = d0 if d0 == d1 else f"{d0}~{d1[5:]}"
    tail = f" ({part}/{n_parts})" if n_parts > 1 else ""
    return f"[대화] {dyad} | {when} {ts_start:%H:%M}~{ts_end:%H:%M}{tail}"


def plan_splits(ntok: np.ndarray, budget: int, overlap: int) -> list[tuple[int, int]]:
    """토큰 수 배열 -> [start, end) 구간 목록. 발화 경계에서만 자른다."""
    n = len(ntok)
    spans, start = [], 0
    while start < n:
        end, acc = start, 0
        while end < n and (acc + ntok[end] <= budget or end == start):
            acc += ntok[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        # overlap 만큼 되감되 최소 한 칸은 전진해야 무한루프가 나지 않는다
        back, acc2 = end, 0
        while back > start + 1 and acc2 < overlap:
            back -= 1
            acc2 += ntok[back]
        start = max(back, start + 1)
    return spans


def cut_long_message(tok, prefix: str, body: str, n_pieces: int) -> list[str]:
    """budget 을 혼자 넘는 발화를 토큰 기준 n_pieces 조각으로 나눈다.

    조각마다 '시:분 화자:' 프리픽스를 다시 붙여 화자 귀속을 유지한다. n_pieces
    는 원문·영문에 같은 값을 쓰므로 조각 수가 어긋나지 않는다. 이 처리가 없으면
    Kerberoast 덤프(5,566자) 같은 가장 가치 있는 증거가 임베딩 시 잘려나간다.
    """
    ids = tok(body, add_special_tokens=False)["input_ids"]
    if n_pieces <= 1 or not ids:
        return [prefix + body]
    step = -(-len(ids) // n_pieces)
    return [prefix + (tok.decode(ids[i * step:(i + 1) * step]).strip() or "…")
            for i in range(n_pieces)]


def expand(tok, prefix: list[str], bodies: list[str], ntok: np.ndarray,
           pieces: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """긴 발화를 조각으로 펼친다 -> (원본 발화 인덱스, 텍스트, 토큰 수)."""
    e_msg, e_txt, e_tok = [], [], []
    for i, k in enumerate(pieces):
        if k <= 1:
            e_msg.append(i)
            e_txt.append(prefix[i] + bodies[i])
            e_tok.append(int(ntok[i]))
            continue
        parts = cut_long_message(tok, prefix[i], bodies[i], int(k))
        e_msg.extend([i] * len(parts))
        e_txt.extend(parts)
        e_tok.extend(len(x) for x in tok(parts, add_special_tokens=False)["input_ids"])
    return np.asarray(e_msg), e_txt, np.asarray(e_tok)


def build_chunks(sess: pd.DataFrame, tok, chunk_size: int, overlap: int,
                 basis: str, verbose: bool = True) -> dict[str, list[dict]]:
    budget = chunk_size - HDR_BUDGET

    prefix = [f"{ts:%H:%M} {nk}: " for ts, nk in zip(sess.ts, sess.nick_from)]
    bodies = {f: sess[f].astype(str).tolist() for f in FIELDS}
    ntok = {f: count_tokens(tok, [p + b for p, b in zip(prefix, bodies[f])],
                            label=f, verbose=verbose) for f in FIELDS}

    # 조각 수는 두 언어에 같은 값을 써야 청크 인덱스가 일치한다.
    if basis == "max":
        base = np.maximum(ntok["ru"], ntok["en"])
        plan_src = {f: base for f in FIELDS}
    elif basis in FIELDS:
        plan_src = {f: ntok[basis] for f in FIELDS}
    else:                                    # each - 언어별 (인덱스 어긋남)
        plan_src = ntok

    exp = {f: expand(tok, prefix, bodies[f], ntok[f],
                     np.maximum(1, np.ceil(plan_src[f] / budget).astype(int)))
           for f in FIELDS}

    sid_arr = sess.sid.to_numpy()
    ts_arr = sess.ts.to_numpy()
    row_arr = sess.src_row.to_numpy()
    dyad_arr = sess.dyad.to_numpy()

    out: dict[str, list[dict]] = {f: [] for f in FIELDS}
    for field in FIELDS:
        e_msg, e_txt, e_tok = exp[field]
        e_plan = (np.maximum(exp["ru"][2], exp["en"][2]) if basis == "max"
                  else exp[basis][2] if basis in FIELDS else e_tok)
        e_sid = sid_arr[e_msg]
        for sid in pd.unique(e_sid):
            idx = np.flatnonzero(e_sid == sid)
            spans = plan_splits(e_plan[idx], budget, overlap)
            for part, (a, b) in enumerate(spans, start=1):
                sel = idx[a:b]
                msgs = pd.unique(e_msg[sel])
                dyad = dyad_arr[msgs[0]]
                t0 = pd.Timestamp(ts_arr[msgs[0]])
                t1 = pd.Timestamp(ts_arr[msgs[-1]])
                out[field].append({
                    "chunk_index": len(out[field]),
                    "session_id": int(sid),
                    "kind": "conversation",
                    "participants": dyad.split(" | "),
                    "ts_start": t0.isoformat(), "ts_end": t1.isoformat(),
                    "n_messages": int(len(msgs)),
                    "part": part, "n_parts": len(spans),
                    "n_tokens": int(e_tok[sel].sum()) + HDR_BUDGET,
                    "src_rows": [int(row_arr[m]) for m in msgs],
                    "text": (header_of(dyad, t0, t1, part, len(spans)) + "\n"
                             + "\n".join(e_txt[i] for i in sel)),
                })
    return out


def broadcast_chunks(bcast: pd.DataFrame, start: dict[str, int],
                     tok=None) -> dict[str, list[dict]]:
    """공지 -> 청크. 수신자 목록을 본문에 넣어 그 시점 명부를 검색 가능하게 한다.

    수신자가 197명인 공지는 명부만으로 500토큰을 넘길 수 있어 n_tokens 를 실제로
    센다. embed.py 가 이 값으로 잘림 여부를 경고한다.
    """
    out: dict[str, list[dict]] = {f: [] for f in FIELDS}
    for i, (_, r) in enumerate(bcast.iterrows()):
        recips = [x for x in str(r.recipients).split(";") if x]
        rows = [int(x) for x in str(r.merged_rows).split(";") if x]
        head = (f"[공지] {r.nick_from} -> {len(recips)}명 | "
                f"{r.ts:%Y-%m-%d %H:%M}")
        for field in FIELDS:
            text = f"{head}\n수신: {', '.join(recips)}\n{r.nick_from}: {r[field]}"
            out[field].append({
                "chunk_index": start[field] + i,
                "session_id": -(i + 1),
                "kind": "broadcast",
                "participants": [r.nick_from],
                "recipients": recips,
                "ts_start": r.ts.isoformat(), "ts_end": r.ts.isoformat(),
                "n_messages": len(rows),
                "part": 1, "n_parts": 1,
                "n_tokens": (len(tok(text, add_special_tokens=False)["input_ids"])
                             if tok is not None else None),
                "src_rows": rows,
                "text": text,
            })
    return out


# --------------------------------------------------------------------------
def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(tag: str, rows: list[dict], chunk_size: int) -> None:
    conv = [r for r in rows if r["kind"] == "conversation"]
    n = np.array([r["n_tokens"] for r in conv])
    print(f"  {tag:<10} 청크 {len(rows):>7,}  (대화 {len(conv):,} + 공지 "
          f"{len(rows) - len(conv):,})  토큰 중앙 {int(np.median(n)):>4} "
          f"최대 {int(n.max()):>5}  초과 {int((n > chunk_size).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="정제 CSV -> 검색용 청크")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--gap-hours", type=float, default=1.0,
                    help="이 시간 이상 끊기면 다른 주제로 보고 세션 분리 (기본 1)")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--split-basis", choices=("en", "ru", "max", "each"), default="en",
                    help="분할 경계를 정하는 기준 언어 (기본 en). 영어를 축으로 두면 "
                         "나중에 언어를 추가해도 모두 같은 발화 경계를 공유한다. "
                         "each 는 각각 따로 잘라 인덱스가 어긋난다.")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--stats-only", action="store_true", help="저장하지 않는다")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.overlap >= args.chunk_size:
        raise SystemExit("--overlap 은 --chunk-size 보다 작아야 합니다.")
    verbose = not args.quiet

    conv, bcast = load(args.csv)
    sess = split_sessions(conv, args.gap_hours)
    size = sess.groupby("sid").size()
    print(f"[세션] gap {args.gap_hours}시간 -> {len(size):,}개 "
          f"(발화 중앙 {int(size.median())}, 최대 {int(size.max())})")

    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out = build_chunks(sess, tok, args.chunk_size, args.overlap,
                       args.split_basis, verbose=verbose)
    extra = broadcast_chunks(bcast, {f: len(out[f]) for f in FIELDS}, tok)
    for f in FIELDS:
        out[f] += extra[f]

    print(f"\n[청크] {args.chunk_size}/{args.overlap} 토큰, 경계기준={args.split_basis}")
    for f in FIELDS:
        report(f"jabber_{f}", out[f], args.chunk_size)
    if args.split_basis != "each" and len(out["ru"]) != len(out["en"]):
        print("  [!] 청크 수가 다릅니다. 정답 공유가 불가능합니다.")

    if args.stats_only:
        print("\nSTATS-ONLY: 저장하지 않았습니다.")
        return

    for f in FIELDS:
        write_jsonl(os.path.join(args.out, f"jabber_{f}.jsonl"), out[f])
    write_jsonl(os.path.join(args.out, "sessions.jsonl"),
                [{k: v for k, v in r.items() if k != "text"} for r in out["ru"]])
    print(f"\n저장 위치: {args.out}")


if __name__ == "__main__":
    main()
