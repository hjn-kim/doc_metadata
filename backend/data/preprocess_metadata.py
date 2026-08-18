# -*- coding: utf-8 -*-
"""jabberchat2020.csv 정제 -> jabberchat2020process.csv

Conti 조직의 Jabber 1:1 대화 로그에서 실제 결함만 걷어낸다. 세션 분리와 청킹은
chunking.py 가 맡는다. 이 단계에서는 길이 필터를 걸지 않는다.

행 순서는 원본 그대로 둔다(src_row 오름차순). 원본은 149개 날짜 블록이 뒤섞인
순서라 시간순이 아니지만, 정렬해 버리면 src_row 로 원본을 되짚기 어려워진다.
시간 순서가 필요한 곳은 각자 그때그때 정렬한다.

원본에서 확인된 결함 (107,967행 기준):

    HTML 엔티티        body_en 10,521행 (body 는 1행)
    앞뒤 공백          body 1,657행 / body_en 598행
    시스템 메시지      'Your message was not sent...' 6행에 혼입
    conference 도메인  22행 (그룹채팅방, dyad 개념 불성립)
    화자 중복 오류     270행 (a->b 와 b->a 가 같은 본문·같은 시각)
    발화 중복 나열     1,103행 (공지 1건을 최대 197명에게 개별 발송)

하지 않는 것 (중요):

    <target> 같은 꺾쇠는 HTML 태그가 아니라 인용문의 화자명이다. 1,839행에
    있고 진짜 HTML 태그는 1행뿐이라 태그 제거를 돌리면 화자 정보만 날아간다.

    body_language 로 언어를 거르지 않는다. sr/bg/uk/ky 로 찍힌 것이 전부
    실제로는 러시아어라(짧은 문장 오탐) ru 만 남기면 1.5만 행을 잃는다.
    body 컬럼 전체가 곧 원문 코퍼스다.

    길이 필터를 걸지 않는다. 세션 200자 필터를 걸면 BTC 주소의 43%, IP 의
    26% 가 사라진다. 짧은 세션이 증거를 담고 있다.

사용:
    python data/preprocess.py
    python data/preprocess.py --csv data/jabberchat2020.csv --out data/x.csv

출력 컬럼:
    src_row       원본 CSV 행 번호 (추적용)
    ts            타임스탬프 (마이크로초 정밀도, 전 행 유일)
    nick_from     발신 닉네임 (도메인 제거)
    nick_to       수신 닉네임
    dyad          정렬된 대화 상대 쌍 'a | b'
    body          정제된 원문
    body_en       정제된 영문
    kind          conversation | broadcast
    recipients    공지의 수신자 목록 ';' 구분 (대화는 빈 값)
    merged_rows   공지로 병합된 원본 행 번호 ';' 구분 (대화는 빈 값)
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "jabberchat2020.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "jabberchat2020process.csv")

# 조직 내부 도메인. '@' 가 빠진 주소(10행)에서 접미사로 떼어내야 한다.
#   twinq3mcco35auwcstmt.onion  -> twin
#   vjud.q3mcco35auwcstmt.onion -> vjud
DOMAIN_SUFFIX = re.compile(r"\.?q?3?mcco35auwcstmt\.(?:onion|oinon)$", re.I)

# OTR 시스템 메시지. 순수 시스템 행(2행)도 있지만 실제 대화에 섞인 행(4행)이
# 더 많아서 행째로 지우면 backdoor.js 배포 URL 같은 증거까지 사라진다. 줄만 뺀다.
SYSTEM_LINE = re.compile(
    r"(?im)^[ \t]*(?:\[\d{2}:\d{2}:\d{2}\][ \t]*)?\*\*\*.*?"
    r"(?:message was not sent|сообщение не было отправлено).*?$\n?")

MIRROR_BUCKET = "10s"     # 화자 중복 오류 판정 시각 해상도
BCAST_GAP = 600           # 같은 공지의 연속 발송 간격 상한(초)
BCAST_MIN_RECIPIENTS = 5  # 이 인원 이상이면 브로드캐스트로 본다


def norm_text(s) -> str:
    """HTML 엔티티 디코딩 + 공백 정규화. 줄바꿈은 공백 1개로 접는다.

    Defender 이벤트 로그처럼 여러 줄짜리 증거가 7,842행 있으나, 청크 텍스트는
    한 줄 = 한 발화 형식이라 내부 줄바꿈을 남기면 화자 귀속이 흐트러진다.
    """
    return re.sub(r"\s+", " ", html.unescape(str(s))).strip()


def strip_domain(addr: str) -> str:
    """주소에서 닉네임만 남긴다.

    대부분은 '@' 앞부분이면 되지만 '@' 가 아예 없는 행이 10개 있다. 그냥
    split('@')[0] 하면 'twinq3mcco35auwcstmt.onion' 이 통째로 닉네임이 되어
    가짜 계정이 생긴다. 도메인 오타(oinon, 후행 공백, '>')는 '@' 뒤쪽에 있어
    자동으로 버려진다.
    """
    a = str(addr).strip().rstrip(">").strip()
    if "@" in a:
        return a.split("@", 1)[0].strip()
    return DOMAIN_SUFFIX.sub("", a).strip() or a


def preprocess(csv_path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path, index_col=0).rename_axis("src_row").reset_index()
    stat = {"원본": len(df)}

    # 행 순서는 원본 그대로 둔다. 원본은 149개 날짜 블록이 뒤섞인 순서지만,
    # 순서를 바꾸면 src_row 로 원본을 되짚기 어려워진다. 시간 순서가 필요한
    # 단계(브로드캐스트 탐지)는 그 안에서 자체적으로 정렬하고, 세션 분리는
    # chunking.py 가 ['dyad','ts'] 로 다시 정렬해서 처리한다.
    df["ts"] = pd.to_datetime(df["ts"])

    # 엔티티 디코딩 + 시스템 메시지 줄 제거 + 공백 정규화
    for src, dst in (("body", "body"), ("body_en", "body_en")):
        txt = df[src].astype(str).map(html.unescape)
        txt = txt.str.replace(SYSTEM_LINE, "", regex=True)
        df[dst] = txt.map(norm_text)

    # 닉네임 추출
    df["nick_from"] = df["from"].map(strip_domain)
    df["nick_to"] = df["to"].map(strip_domain)

    # conference(그룹채팅방) 제외 - 1:1 이 아니라 dyad 가 성립하지 않는다
    is_conf = (df["from"].astype(str).str.contains("conference", case=False, na=False)
               | df["to"].astype(str).str.contains("conference", case=False, na=False))
    df = df[~is_conf]
    stat["conference 제외"] = int(is_conf.sum())

    # 빈 본문 제거 (원래 빈 1행 + 시스템 메시지만 있던 행)
    empty = (df.body.str.len() == 0) | (df.body_en.str.len() == 0)
    df = df[~empty]
    stat["빈 본문 제거"] = int(empty.sum())

    df["dyad"] = np.where(df.nick_from < df.nick_to,
                          df.nick_from + " | " + df.nick_to,
                          df.nick_to + " | " + df.nick_from)

    # 화자 중복 오류: a->b 와 b->a 가 같은 본문·같은 시각으로 둘 다 기록된 로그
    # 재구성 아티팩트. 먼저 온 행만 남긴다.
    key = (df.dyad + "\x00" + df.body + "\x00"
           + df.ts.dt.floor(MIRROR_BUCKET).astype(str))
    mirror = key.duplicated(keep="first")
    df = df[~mirror]
    stat["화자 중복 오류 제거"] = int(mirror.sum())

    # 발화 중복 나열: 한 사람이 같은 공지를 수십~수백 명에게 개별 발송한 것.
    # 지우지 않고 대표 1행으로 접되 수신자 목록을 남긴다 - 197명 명부 자체가
    # 그 시점 조직 규모의 증거다.
    d = df.sort_values(["nick_from", "body", "ts"], kind="mergesort").copy()
    gap = d.groupby(d.nick_from + "\x00" + d.body)["ts"].diff().dt.total_seconds()
    d["_grp"] = (gap.isna() | (gap > BCAST_GAP)).cumsum()
    d["_bcast"] = d.groupby("_grp")["nick_to"].transform("nunique") >= BCAST_MIN_RECIPIENTS

    merged = d.groupby("_grp").agg(
        recipients=("nick_to", lambda s: ";".join(sorted(set(s)))),
        merged_rows=("src_row", lambda s: ";".join(str(int(x)) for x in sorted(s))))
    drop = d._bcast & d.duplicated("_grp", keep="first")
    stat["브로드캐스트 병합"] = int(drop.sum())

    kept = d[~drop].copy()
    kept["kind"] = np.where(kept._bcast, "broadcast", "conversation")
    kept["recipients"] = np.where(kept._bcast, kept._grp.map(merged.recipients), "")
    kept["merged_rows"] = np.where(kept._bcast, kept._grp.map(merged.merged_rows), "")

    cols = ["src_row", "ts", "nick_from", "nick_to", "dyad",
            "body", "body_en", "kind", "recipients", "merged_rows"]
    out = kept.sort_values("src_row", kind="mergesort")   # 원본 행 순서로 복원

    stat["정제 후 대화"] = int((out.kind == "conversation").sum())
    stat["공지"] = int((out.kind == "broadcast").sum())
    return out[cols], stat


def main() -> None:
    ap = argparse.ArgumentParser(description="jabberchat2020.csv 정제")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    out, stat = preprocess(args.csv)

    print("[정제]")
    for k, v in stat.items():
        print(f"  {k:<18}{v:>9,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    size = os.path.getsize(args.out) / 1e6
    print(f"\n저장: {args.out}  ({len(out):,}행 / {size:.1f}MB)")


if __name__ == "__main__":
    main()
