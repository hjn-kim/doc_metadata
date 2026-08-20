# -*- coding: utf-8 -*-
"""ko보이스피싱.txt -> 검색용 청크 (대화 1건 = 청크 1개)

원문은 "N번 대화" 헤더 아래에 '가해자N: ...' / '피해자N: ...' 발화가 이어지는
평문이다. 통화 1건이 곧 하나의 사건이라 대화 경계가 그대로 의미 경계다.
embed.py 처럼 512토큰 창을 밀어 자르면 한 통화가 중간에서 끊겨 "돈을 어디로
보내라" 와 "왜 보내야 하는지" 가 다른 청크로 갈라진다. 그래서 여기서는 대화를
먼저 통째로 잡고, 그것만으로 너무 긴 통화만 발화 경계에서 다시 쪼갠다.

    통화 70건, 글자 수 중앙 502 / 최대 4,411 (bge-m3 로 대략 1토큰 = 1.4글자)

--max-tokens (기본 1024) 를 넘는 통화만 part 로 나뉜다. bge-m3 는 8192토큰까지
받으므로 잘림 걱정은 없지만, 한 벡터에 담기는 주제가 많아질수록 벡터가 뭉개져서
상한을 둔다. 0 을 주면 아무리 길어도 나누지 않는다.

'※ 7번은 원문에 피해자 발화가 확인되지 않습니다' 같은 편집자 주석은 발화가
아니라 그 통화에 대한 메모다. note 필드에 따로 담고 본문(text)에는 넣지 않는다
— 임베딩 대상은 통화 내용이어야 한다.

시각 정보는 원문에 없다. ts_start/ts_end 를 null 로 두므로 이 코퍼스에는
chunk_meta.py 의 기간 필터가 걸리지 않는다.

사용:
    python data/chunking_ko.py
    python data/chunking_ko.py --max-tokens 512 --overlap 128
    python data/chunking_ko.py --stats-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TXT = os.path.join(ROOT, "data", "ko보이스피싱.txt")
DEFAULT_OUT = os.path.join(ROOT, "data", "chunks")
DEFAULT_KEY = "ko_voice"

TOKENIZER = "BAAI/bge-m3"
HDR_BUDGET = 32           # 청크 헤더 몫으로 떼어두는 토큰

RE_HEADER = re.compile(r"^(\d+)\s*번\s*대화$")
RE_UTTER = re.compile(r"^([^:：]{1,20})\s*[:：]\s*(.*)$")


# --------------------------------------------------------------------------
# ① 파싱  — 평문 -> 통화 목록
# --------------------------------------------------------------------------
def parse(path: str) -> list[dict]:
    """N번 대화 단위로 끊어 발화 목록을 만든다.

    src_row 는 원문 줄 번호(1부터)다. 검색 결과에서 원문을 되짚을 때 쓰는 유일한
    끈이라 발화마다 들고 다닌다.
    """
    talks: list[dict] = []
    cur: dict | None = None

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            m = RE_HEADER.match(line)
            if m:
                cur = {"no": int(m.group(1)), "utters": [], "note": None,
                       "line": lineno}
                talks.append(cur)
                continue

            if cur is None:                      # 헤더보다 앞선 줄은 버린다
                continue

            if line.startswith("※"):             # 편집자 주석
                cur["note"] = (cur["note"] + " " if cur["note"] else "") + line
                continue

            m = RE_UTTER.match(line)
            if m:
                cur["utters"].append({"speaker": m.group(1).strip(),
                                      "text": m.group(2).strip(),
                                      "row": lineno})
            elif cur["utters"]:                  # 줄바꿈으로 끊긴 발화의 뒷부분
                cur["utters"][-1]["text"] += " " + line
            else:
                print(f"  [!] {lineno}행: 화자를 못 찾아 버립니다: {line[:40]}",
                      file=sys.stderr)

    empty = [t["no"] for t in talks if not t["utters"]]
    if empty:
        print(f"  [!] 발화가 없는 대화: {empty}", file=sys.stderr)
    return [t for t in talks if t["utters"]]


def participants_of(talk: dict) -> list[str]:
    """등장 순서를 지키면서 중복만 제거한 화자 목록."""
    seen: list[str] = []
    for u in talk["utters"]:
        if u["speaker"] not in seen:
            seen.append(u["speaker"])
    return seen


def header_of(no: int, parts: list[str], part: int, n_parts: int) -> str:
    """청크마다 다시 붙는 대화 헤더.

    part 로 쪼개진 청크는 앞부분이 없어 몇 번 통화인지, 누가 나오는지를 본문만
    보고는 알 수 없다. 헤더가 그것을 담는다 (chunking.py 와 같은 규칙).
    """
    tail = f" ({part}/{n_parts})" if n_parts > 1 else ""
    return f"[통화] 보이스피싱 {no}번 | {' | '.join(parts)}{tail}"


# --------------------------------------------------------------------------
# ② 청킹  — 긴 통화만 발화 경계에서 다시 쪼갠다
# --------------------------------------------------------------------------
def plan_splits(ntok: list[int], budget: int, overlap: int) -> list[tuple[int, int]]:
    """토큰 수 배열 -> [start, end) 구간 목록. 발화 중간에서는 자르지 않는다."""
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


def build_chunks(talks: list[dict], tok, max_tokens: int,
                 overlap: int) -> list[dict]:
    budget = (max_tokens - HDR_BUDGET) if max_tokens else 0

    out: list[dict] = []
    for talk in talks:
        parts = participants_of(talk)
        lines = [f"{u['speaker']}: {u['text']}" for u in talk["utters"]]
        ntok = [len(x) for x in tok(lines, add_special_tokens=False)["input_ids"]]

        spans = ([(0, len(lines))] if not budget or sum(ntok) <= budget
                 else plan_splits(ntok, budget, overlap))

        for part, (a, b) in enumerate(spans, start=1):
            sel = talk["utters"][a:b]
            out.append({
                "chunk_index": len(out),
                "session_id": talk["no"],
                "kind": "call",
                "participants": parts,
                "ts_start": None, "ts_end": None,
                "n_messages": len(sel),
                "part": part, "n_parts": len(spans),
                "n_tokens": sum(ntok[a:b]) + HDR_BUDGET,
                "src_rows": [u["row"] for u in sel],
                "note": talk["note"],
                "text": header_of(talk["no"], parts, part, len(spans))
                        + "\n" + "\n".join(lines[a:b]),
            })
    return out


# --------------------------------------------------------------------------
def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="보이스피싱 통화 원문 -> 대화 단위 청크")
    ap.add_argument("--txt", default=DEFAULT_TXT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--key", default=DEFAULT_KEY, help="코퍼스 키 (파일 이름)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="이 토큰을 넘는 통화만 발화 경계에서 쪼갠다 (0=무제한)")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--stats-only", action="store_true", help="저장하지 않는다")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.max_tokens and args.overlap >= args.max_tokens:
        raise SystemExit("--overlap 은 --max-tokens 보다 작아야 합니다.")

    talks = parse(args.txt)
    n_utt = sum(len(t["utters"]) for t in talks)
    print(f"[파싱] 통화 {len(talks)}건 · 발화 {n_utt:,}개 · "
          f"주석 {sum(1 for t in talks if t['note'])}건")

    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = build_chunks(talks, tok, args.max_tokens, args.overlap)
    n = sorted(r["n_tokens"] for r in rows)
    split = sum(1 for r in rows if r["n_parts"] > 1)
    print(f"[청크] {len(rows)}개 (쪼개진 통화에서 나온 것 {split}개) · "
          f"토큰 중앙 {n[len(n) // 2]} 최대 {n[-1]}")

    if args.stats_only:
        print("\nSTATS-ONLY: 저장하지 않았습니다.")
        return

    path = os.path.join(args.out, f"{args.key}.jsonl")
    write_jsonl(path, rows)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
