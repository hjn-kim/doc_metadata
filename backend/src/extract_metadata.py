#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0-1. 메타데이터 추출 — 질문 한 줄에서 조건을 뽑는다

    질문  ->  {sender, receiver, participants, since, until, keywords}
                                  |
                          filter_metadata.py

뽑는 것:

    sender       말한 사람으로 지목된 닉네임
    receiver     받는 사람으로 지목된 닉네임
    participants 방향 없이 대화에 낀 닉네임
    since/until  기간
    keywords     원문에 글자 그대로 나올 문자열 (IP·도메인·파일명·지갑주소)

    사람 칸을 셋으로 나눈 것은 질문이 그렇게 말하기 때문이다. 정작 필터는
    셋을 합쳐 "이 사람들이 다 낀 대화" 로만 건다 — 청크 메타데이터에 방향이
    없기 때문이다 (chunk_meta.py 참고). sender 하나면 그 사람이 낀 대화,
    sender + receiver 나 participants 두 명이면 그 둘의 대화쌍이 된다.
    나눠 두는 값어치는 화면에서 4B 가 무엇을 어떻게 읽었는지 보이는 것이다.

뽑는 일은 4B(Qwen3)가 한다:

    LLM         기본. 질문을 읽고 위 네 칸을 JSON 으로 채운다. 규칙으로는 못
                잡는 표현("지난 9월 말쯤", "스턴이랑 톰이 나눈 얘기")까지
                받으려면 결국 모델이 읽어야 한다.
    규칙(rule)  안전망. 두 자리에만 쓴다 — 4B 가 죽거나 JSON 이 깨졌을 때,
                그리고 4B 가 어떤 칸을 비웠는데 규칙에는 잡히는 것이 있을 때.
                모델 없이 파이프라인을 돌려 볼 수 있는 경로(--no-llm)이자
                4B 를 붙였을 때 넘어야 할 기준선이기도 하다.

    LLM 이 뽑았다고 그대로 쓰지는 않는다. 이름은 nicks.json 명부에 있는 것만
    남기고, 날짜는 파싱되는 것만, 키워드는 ASCII 만 쓴다. 4B 가 지어낸 이름
    하나에 후보가 0개가 되는 것을 여기서 한 번, filter_metadata 의 사다리에서
    또 한 번 막는다.

사람과 키워드를 가르는 기준 (여기서 제일 자주 틀린다):

    "ahtyng와 alarm 사이의 대화에서..."   -> participants  두 사람이 나눈 대화
    "stern이 말한 ..."                    -> sender        stern 이 화자
    "maze가 등장하는 기록에서..."          -> keywords      본문에 그 글자가 나온다

    'maze' 는 닉네임이 아니라 랜섬웨어 이름이다. 반대로 'target' 은 닉네임
    이기도 하다. 그래서 규칙 쪽은 '사이의 대화 / 가 보낸 / 와의 대화' 처럼
    관계를 말하는 표현에서만 사람을 뽑는다. 질문에 닉네임이 그냥 적혀 있다고
    사람 조건을 걸면, 그 사람이 낀 대화 밖에 있는 정답을 조용히 잘라낸다.

한국어 키워드를 넣지 않는 이유:

    원문은 러시아어와 그 영어 번역이다. '서버', '랜섬웨어' 같은 한국어 낱말은
    본문에 글자 그대로 나올 수 없다. 그런 말은 이미 질문 임베딩이 처리하고
    있으므로 키워드로는 ASCII 식별자만 넘긴다 (search.py 가 리터럴로 맞춘다).

단독 실행:
    python src/extract_metadata.py "ahtyng와 alarm 사이의 대화에서 ..."
    python src/extract_metadata.py --no-llm "2020-09-29 대화에서 ..."
    python src/extract_metadata.py --qa           # 규칙 기준선 (qa.json)
    python src/extract_metadata.py --qa --llm     # 4B 를 붙여 같은 표를 다시
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_meta import load_chunk_meta, load_nicks  # noqa: E402
from filter_metadata import MetaQuery, normalize  # noqa: E402

# 추출기를 어떻게 돌릴지. 파이프라인이 인자로 덮어쓸 수 있다.
#   llm   규칙 + 4B (기본)
#   rule  규칙만 (모델을 안 올린다. CPU 데모/테스트)
#   off   추출하지 않는다 (필터 없이 전체를 뒤진다)
DEFAULT_MODE = os.getenv("RAG_EXTRACT", "llm").strip().lower()

# 추출은 짧은 JSON 하나다. 길게 잡아 봐야 늘어지기만 한다.
MAX_NEW_TOKENS = 256

RESPONSE_SCHEMA = {
    "sender": ["말한 사람으로 지목된 닉네임. 없으면 []"],
    "receiver": ["받는 사람으로 지목된 닉네임. 없으면 []"],
    "participants": ["방향 없이 대화에 낀 닉네임. 없으면 []"],
    "since": "시작 날짜 YYYY-MM-DD. 없으면 null",
    "until": "끝 날짜 YYYY-MM-DD. 없으면 null",
    "keywords": ["원문에 그대로 나올 ASCII 문자열. 없으면 []"],
}


# --------------------------------------------------------------------------
# 규칙 추출
# --------------------------------------------------------------------------

# "A와 B 사이의 대화", "A와 B의 대화속에", "A와 B가 나눈"
DYAD_RE = re.compile(r"([A-Za-z0-9_.\-]{2,})\s*(?:와|과)\s*"
                     r"([A-Za-z0-9_.\-]{2,})\s*(?:의|가|이)?\s*"
                     r"(?:사이[의에]?\s*)?(?:대화|나눈|주고받)")
# "A가 B에게 보낸" — 방향이 다 드러난 경우
DIRECTED_RE = re.compile(r"([A-Za-z0-9_.\-]{2,})\s*[가이]\s*"
                         r"([A-Za-z0-9_.\-]{2,})\s*(?:에게|한테)\s*"
                         r"(?:보낸|말한|준|전한|시킨)")
# "A에게 보낸", "A한테 온" — 받는 사람만 드러난 경우
RECEIVER_RE = re.compile(r"([A-Za-z0-9_.\-]{2,})\s*(?:에게|한테)\s*"
                         r"(?:보낸|말한|온|전한)")
# "A가 보낸", "A와의 대화", "A의 대화", "A가 말한"
SOLO_RE = re.compile(r"([A-Za-z0-9_.\-]{2,})\s*"
                     r"(?:와의\s*대화|의\s*대화|[가이]\s*보낸|[가이]\s*말한)")

# 날짜. 일까지 없어도 받는다 — "2020-08 부터 2020-09 사이에" 처럼 달만 말하는
# 질문이 있고, 그건 filter_metadata.parse_span 이 그달 1일~말일로 편다.
DATE_RE = re.compile(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?")
KO_DATE_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?")
# 날짜 그 자체인 토큰. 키워드 후보에서 뺀다.
DATE_TOKEN_RE = re.compile(r"\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?")

# 본문에 그대로 나올 만한 토큰. 한글이 섞이면 버린다.
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@\-]{2,}")

# 질문에 흔히 섞이는 영문 잡음. 키워드로 걸면 축소가 아니라 방해가 된다.
TOKEN_STOP = {
    "http", "https", "www", "com", "org", "net", "the", "and", "for", "with",
    "jabber", "chat", "log", "logs", "conti", "id", "ip",
}


# 날짜 뒤에 붙는 방향 표지. "2020-06 이후" 는 시작만, "2020-10 이전" 은 끝만
# 말한다. 하나뿐인 날짜를 늘 양끝으로 쓰면 "이후" 질문이 그 달 하나로 좁혀져
# 정답이 통째로 걸러진다.
SINCE_MARK = ("이후", "부터", "이래", "다음")
UNTIL_MARK = ("이전", "까지", "전에", "앞서")
MARK_WINDOW = 8          # 날짜 바로 뒤 몇 글자까지 표지로 볼지


def _dates_of(q: str) -> tuple[list[str], list[str]]:
    """
    질문 -> (시작 후보, 끝 후보).

    날짜 하나에 표지가 없으면 시작이자 끝으로 본다 (그날 하루 / 그 달).
    "2020-08 부터 2020-09 사이에" 는 앞 날짜가 시작, 뒤 날짜가 양쪽이라
    시작 2020-08 · 끝 2020-09 로 잡힌다.
    """
    found: list[tuple[str, str]] = []       # (날짜, 표지)
    for pattern in (DATE_RE, KO_DATE_RE):
        for m in pattern.finditer(q):
            y, mo, d = m.group(1), m.group(2), m.group(3)
            iso = f"{y}-{int(mo):02d}-{int(d):02d}" if d else f"{y}-{int(mo):02d}"
            tail = q[m.end():m.end() + MARK_WINDOW]
            if any(mark in tail for mark in UNTIL_MARK):
                found.append((iso, "until"))
            elif any(mark in tail for mark in SINCE_MARK):
                found.append((iso, "since"))
            else:
                found.append((iso, "both"))

    since = [d for d, kind in found if kind in ("since", "both")]
    until = [d for d, kind in found if kind in ("until", "both")]
    return sorted(since), sorted(until)


def rule_extract(question: str) -> dict:
    """정규식으로 뽑는다. 모델 없이 도는 기준선이자 안전망."""
    q = question or ""

    # 방향이 드러난 표현부터 본다. 필터에서는 어차피 "그 사람이 낀 대화" 로
    # 같아지지만, 화면에는 질문이 말한 대로 칸을 나눠 보여 준다.
    sender: list[str] = []
    receiver: list[str] = []
    participants: list[str] = []

    m = DIRECTED_RE.search(q)
    if m:                                     # A 가 B 에게 보낸
        sender, receiver = [m.group(1)], [m.group(2)]
    elif DYAD_RE.search(q):                   # A 와 B 의 대화
        m = DYAD_RE.search(q)
        participants = [m.group(1), m.group(2)]
    else:
        m = SOLO_RE.search(q)
        if m and re.search(r"[가이]\s*(?:말한|보낸)", m.group(0)):
            sender = [m.group(1)]             # A 가 말한
        elif m:
            participants = [m.group(1)]       # A 와의 대화
        else:
            m = RECEIVER_RE.search(q)
            if m:
                receiver = [m.group(1)]       # A 에게 보낸

    people = [*sender, *receiver, *participants]

    since_list, until_list = _dates_of(q)

    # 날짜는 기간 조건으로 이미 갔다. 키워드로도 넘기면 본문에 '2020-09-29'
    # 라고 적혀 있는 청크만 남아 그날 대화를 통째로 놓친다 (로그 본문에는
    # 시각만 적혀 있고 날짜는 없다).
    taken = {p.lower() for p in people}
    keywords = [t for t in TOKEN_RE.findall(q)
                if t.lower() not in TOKEN_STOP and t.lower() not in taken
                and not t.isdigit() and not DATE_TOKEN_RE.fullmatch(t)]

    return {
        "sender": sender,
        "receiver": receiver,
        "participants": participants,
        "since": min(since_list) if since_list else None,
        "until": max(until_list) if until_list else None,
        "keywords": list(dict.fromkeys(keywords)),
    }


# --------------------------------------------------------------------------
# LLM 추출
# --------------------------------------------------------------------------

def build_system_prompt(nicks: list[str], max_nicks: int = 320) -> str:
    """
    명부를 통째로 실어 준다.

    289명이면 900토큰 남짓이다. 이걸 넣어 두면 4B 가 'Stern' 을 'stern' 으로
    맞추고, 명부에 없는 낱말(maze 같은 것)을 사람으로 착각하는 일이 크게 준다.
    명부가 더 커지면 앞에서 자른다 — 통째로 못 실을 바에는 규칙 쪽 결과를
    믿는 편이 낫다.
    """
    roster = ", ".join(nicks[:max_nicks])
    return f"""당신은 검색 질문에서 필터 조건만 뽑아내는 도구입니다.

자료는 2020-06-21 ~ 2020-11-16 사이의 Conti 조직 Jabber 대화 로그입니다.
원문은 러시아어이고 영어 번역본이 함께 있습니다. 질문은 한국어입니다.

뽑을 것:

  sender       보낸 사람 / 말한 사람.
               "A가 보낸", "A가 말한", "A가 지시한" -> sender: ["A"]
  receiver     받는 사람.
               "A에게 보낸", "A가 받은", "A한테 온" -> receiver: ["A"]
  participants 방향을 안 밝힌 대화. 낀 사람을 **모두** 넣습니다.
               "A와 B의 대화", "A와 B 사이의 대화", "A와 B가 나눈 이야기"
               -> participants: ["A", "B"]      (한 명만 넣으면 틀립니다)

           세 칸은 서로 배타적입니다. 한 사람은 한 칸에만 넣습니다.
             "A가 보낸"        -> sender ["A"],  나머지 두 칸은 []
             "A가 받은"        -> receiver ["A"], 나머지 두 칸은 []
             "A가 B에게 보낸"  -> sender ["A"], receiver ["B"], participants []
             "A와 B의 대화"    -> participants ["A","B"], 나머지 두 칸은 []
           같은 이름을 두 칸에 겹쳐 적으면 틀린 출력입니다.

           셋 다 아래 명부에 있는 닉네임만 넣습니다.

  since    기간 시작 (YYYY-MM-DD). 없으면 null
  until    기간 끝 (YYYY-MM-DD). 없으면 null

           "2020-09-29에"       -> since 2020-09-29, until 2020-09-29
           "2020-08 부터 2020-09 사이" -> since 2020-08-01, until 2020-09-30
           "2020-10 이전에"      -> since null,       until 2020-10-31
           "2020-06 이후에"      -> since 2020-06-01, until null

           '이전 / 까지' 는 끝만, '이후 / 부터' 는 시작만 말합니다. 한쪽만
           말한 것을 양쪽으로 채우면 그 달 하나로 좁혀져 답을 놓칩니다.
  keywords 대화 본문에 글자 그대로 나올 문자열. IP 주소, 도메인, 파일명,
           지갑 주소, 프로그램 이름 같은 것입니다.

지켜야 할 것:

  1. "X가 등장하는 기록에서" 의 X 는 사람 칸이 아니라 keywords 입니다.
     본문에 그 글자가 나온다는 뜻이지 그 사람이 대화했다는 뜻이 아닙니다.
  2. 명부에 없는 이름은 사람 칸에 넣지 마세요. 대신 keywords 로 넣으세요.
  3. 한국어 낱말은 keywords 에 넣지 마세요. 원문이 러시아어·영어라 그대로
     나오지 않습니다. ASCII 문자열만 넣습니다.
  4. 질문에 없는 조건은 지어내지 마세요. 없으면 [] 와 null 입니다.

보기:

  질문: ahtyng와 alarm 사이의 대화에서 서버 현황을 파악한 증거를 찾아주세요.
  출력: {{"sender": [], "receiver": [], "participants": ["ahtyng", "alarm"], "since": null, "until": null, "keywords": []}}

  질문: stern이 말한 locker에 대한 증거를 찾아주세요.
  출력: {{"sender": ["stern"], "receiver": [], "participants": [], "since": null, "until": null, "keywords": ["locker"]}}

  질문: bentley가 deploy에게 보낸 ransom 관련 지시를 찾아주세요.
  출력: {{"sender": ["bentley"], "receiver": ["deploy"], "participants": [], "since": null, "until": null, "keywords": ["ransom"]}}

  질문: 68.224.217.72가 등장하는 기록에서 내부 문서를 탈취한 정황을 찾아주세요.
  출력: {{"sender": [], "receiver": [], "participants": [], "since": null, "until": null, "keywords": ["68.224.217.72"]}}

  질문: 2020-09-29에 랜섬웨어 배포를 논의한 대화를 찾아주세요.
  출력: {{"sender": [], "receiver": [], "participants": [], "since": "2020-09-29", "until": "2020-09-29", "keywords": []}}

  질문: 2020-10 이전에 Wermgr에 대한 증거를 찾아주세요.
  출력: {{"sender": [], "receiver": [], "participants": [], "since": null, "until": "2020-10-31", "keywords": ["Wermgr"]}}

닉네임 명부 ({len(nicks)}명):
{roster}"""


def llm_extract(question: str, model_name: str | None = None,
                device: str | None = None) -> dict:
    """4B 에게 JSON 하나를 받는다. 실패하면 예외를 올린다 (부르는 쪽이 규칙으로 되돌린다)."""
    from local_llm import DEFAULT_MODEL, generate_json

    nicks = load_nicks().get("nicks", [])
    return generate_json(
        system=build_system_prompt(nicks),
        user=f"질문: {question}\n출력:",
        schema=RESPONSE_SCHEMA,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,          # 추출은 창의성이 필요한 일이 아니다
        model_name=model_name or DEFAULT_MODEL,
        device=device,
    )


# --------------------------------------------------------------------------
# 합치기
# --------------------------------------------------------------------------

@dataclass
class ExtractResult:
    """0단계 결과. 화면에는 raw 를, 필터에는 query 를 준다."""

    question: str
    query: MetaQuery                      # 정규화·명부 대조를 마친 조건
    raw: dict = field(default_factory=dict)      # 실제로 쓴 조건 (합친 것)
    rule: dict = field(default_factory=dict)     # 규칙이 뽑은 것
    llm: dict | None = None                      # 4B 가 뽑은 것 (안 썼으면 None)
    source: str = "rule"                  # rule / llm+rule / off
    elapsed: float = 0.0
    error: str = ""                       # 4B 가 실패했으면 그 이유

    def label(self) -> str:
        return self.query.label()


def _merge(rule: dict, llm: dict | None, meta) -> tuple[dict, bool]:
    """
    LLM 결과를 쓰고, LLM 이 비운 칸만 규칙으로 메운다. (합친 조건, 규칙을 썼나)

    뽑는 일은 4B 가 한다. 규칙은 두 자리에만 남는다.

        1. 4B 가 죽거나 JSON 이 깨졌을 때 (llm 이 None)
        2. 4B 가 어떤 칸을 비웠는데 규칙에는 잡히는 것이 있을 때

    2번을 두는 이유: 조건을 하나 더 거는 쪽이 아니라 놓치지 않는 쪽이 안전
    하기 때문이다. 규칙 패턴은 '사이의 대화 / 가 말한' 처럼 관계를 말하는
    표현에서만 사람을 뽑으므로 없는 조건을 지어내지 않고, 잘못 걸려도
    filter_metadata 의 사다리가 풀어 준다.

    검증은 그대로 남는다 — 사람은 명부(nicks.json)에 있는 것만, 날짜는
    파싱되는 것만, 키워드는 ASCII 만. 4B 가 지어낸 이름이 필터로 새는 것을
    막는 자리라 여기는 양보하지 않는다.
    """
    if not llm:
        return dict(rule), True

    out: dict = {}
    used_rule = False

    # 사람은 칸별로 옮긴다. 어느 칸이든 명부에 있는 이름만 남는다.
    roles = {}
    for key in ROLE_KEYS:
        names = [p for p in (meta.resolve(x) for x in _listify(llm.get(key)))
                 if p]
        roles[key] = list(dict.fromkeys(names))

    rule_names = {n for key in ROLE_KEYS for n in rule.get(key, [])}
    llm_names = {n for names in roles.values() for n in names}

    if not llm_names and rule_names:
        # 4B 가 사람을 한 명도 못 뽑았다. 규칙이 잡은 것을 그대로 쓴다.
        roles = {key: list(rule.get(key, [])) for key in ROLE_KEYS}
        used_rule = True
    elif rule_names and rule_names == llm_names:
        # 같은 사람을 두고 칸만 갈렸다. 규칙 쪽을 쓴다 — 규칙은 "A와 B의 대화",
        # "A가 B에게 보낸" 같은 표현을 보고 나눈 것이라 근거가 분명한 반면,
        # 4B 는 같은 이름을 세 칸에 다 적어 보내는 일이 잦다.
        if roles != {key: list(rule.get(key, [])) for key in ROLE_KEYS}:
            roles = {key: list(rule.get(key, [])) for key in ROLE_KEYS}
            used_rule = True

    out.update(_split_roles(roles))

    # 열린 구간("2020-10 이전")을 4B 가 닫힌 구간(그 달 하나)으로 읽는 일이
    # 잦다. 규칙은 '이전/이후' 표지를 보고 한쪽만 채우므로, 규칙이 한쪽만
    # 잡았는데 4B 가 양쪽을 채웠으면 규칙 쪽을 쓴다. 질문에 적힌 표지가
    # 모델의 짐작보다 확실하다.
    one_sided = bool(rule.get("since")) != bool(rule.get("until"))
    llm_both = all(llm.get(k) for k in ("since", "until"))

    for key in ("since", "until"):
        value = llm.get(key)
        if one_sided and llm_both:
            out[key] = rule.get(key)
            used_rule = True
        elif value and re.search(r"\d{4}[-/.]\d{1,2}", str(value)):
            out[key] = str(value).strip()
        else:
            out[key] = rule.get(key)
            used_rule = used_rule or bool(out[key])

    words = [w for w in _listify(llm.get("keywords"))
             if w.isascii() and len(w) >= 3 and w.lower() not in TOKEN_STOP]
    if not words and rule.get("keywords"):
        words, used_rule = rule["keywords"], True
    out["keywords"] = list(dict.fromkeys(words))

    return out, used_rule


# 사람 칸 세 개. 서로 배타적이다 (아래 _split_roles 참고).
ROLE_KEYS = ("sender", "receiver", "participants")


def _split_roles(roles: dict) -> dict:
    """
    세 칸을 배타적으로 정리한다.

    방향을 말한 칸(sender/receiver)이 이긴다. 같은 이름이 participants 에도
    적혀 있으면 그쪽에서 뺀다 — "A가 B에게 보낸" 을 4B 가 세 칸에 다 적어
    보내는 일이 잦은데, 그대로 두면 화면이 "이 사람이 보내기도 하고 그냥
    참여하기도 했다" 처럼 읽힌다.

    필터 결과는 어느 쪽이든 같다 (방향을 저장하지 않으므로 셋을 합쳐 쓴다).
    여기서 정리하는 것은 순전히 화면에 사실대로 적기 위해서다.
    """
    sender = list(dict.fromkeys(roles.get("sender") or []))
    receiver = [n for n in dict.fromkeys(roles.get("receiver") or [])
                if n not in sender]
    directed = set(sender) | set(receiver)
    participants = [n for n in dict.fromkeys(roles.get("participants") or [])
                    if n not in directed]
    return {"sender": sender, "receiver": receiver,
            "participants": participants}


def _listify(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,;|]", value) if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def extract(question: str, mode: str | None = None,
            model_name: str | None = None,
            device: str | None = None) -> ExtractResult:
    """
    질문 -> 조건. mode 는 llm / rule / off.

    4B 가 죽거나 JSON 이 깨져도 규칙 결과로 계속 간다. 추출은 검색을 돕는
    장치이지 검색을 막는 장치가 아니다.
    """
    started = time.time()
    mode = (mode or DEFAULT_MODE).lower()
    meta = load_chunk_meta()

    if mode == "off":
        return ExtractResult(question=question, query=MetaQuery(),
                             source="off", elapsed=time.time() - started)

    rule = rule_extract(question)
    llm_raw: dict | None = None
    error = ""

    if mode == "llm":
        try:
            llm_raw = llm_extract(question, model_name, device)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    raw, used_rule = _merge(rule, llm_raw, meta)
    if llm_raw is None:
        source = "rule"                      # 4B 를 못 썼다
    else:
        source = "llm+rule" if used_rule else "llm"
    return ExtractResult(
        question=question,
        query=normalize(raw, meta),
        raw=raw,
        rule=rule,
        llm=llm_raw,
        source=source,
        elapsed=time.time() - started,
        error=error,
    )


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def _chunk_texts() -> list[str]:
    """
    청크별 본문 (모든 언어를 이어 붙인 것). 키워드 대조에만 쓴다.

    임베딩 모델은 올리지 않는다 — .npz 의 texts 만 읽으면 되므로 GPU 없이도
    키워드 채널을 그대로 재볼 수 있다.
    """
    import numpy as np
    from corpus import documents

    joined: list[str] | None = None
    for doc in documents():
        with np.load(doc.emb_path, allow_pickle=False) as z:
            texts = [str(t).lower() for t in z["texts"]]
        if joined is None:
            joined = texts
        else:
            joined = [a + " " + b for a, b in zip(joined, texts)]
    return joined or []


def _qa_selfcheck(mode: str) -> None:
    """
    qa.json 문항을 추출 -> 필터 -> 키워드까지 통과시켜 축소율과 누락을 본다.

    filter_metadata.py --qa 와 같은 표에 키워드 열이 붙은 것이다. 거기서는
    정답만 아는 규칙으로 뽑고 여기서는 추출기로 뽑는다. 두 표의 차이가 곧
    추출기의 값어치다.

    '정답통과' 가 떨어지면 그 조건은 쓰면 안 된다. 후보를 아무리 줄여도 정답이
    그 안에 없으면 뒤 단계가 할 수 있는 일이 없기 때문이다.
    """
    import numpy as np

    from filter_metadata import build_mask
    from search import usable_keywords   # 같은 규칙으로 키워드를 고른다

    qa_path = Path(__file__).resolve().parent.parent / "data" / "qa" / "qa.json"
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))["qa_pairs"]
    meta = load_chunk_meta()
    row_of = {int(c): i for i, c in enumerate(meta.chunk_index)}
    texts = _chunk_texts()

    def keyword_mask(words: list[str]):
        """search.py 의 키워드 사다리를 청크 단위로 다시 만든 것."""
        words = usable_keywords(words)
        if not words or not texts:
            return None
        hits = np.zeros(len(texts), dtype=np.int32)
        for word in words:
            needle = word.lower()
            hits += np.fromiter((needle in t for t in texts), dtype=bool,
                                count=len(texts))
        # search.py 의 keyword_signal 과 같은 규칙: 하나라도 들어 있으면 후보.
        return (hits >= 1) if int(hits.max()) else None

    tally: dict[int, list] = {}
    started = time.time()
    for p in pairs:
        got = extract(p["question"], mode=mode)
        res = build_mask(got.query, meta)
        keep = res.mask
        kw = keyword_mask(got.query.keywords)
        if kw is not None:
            merged = kw if keep is None else keep & kw
            if merged.any():                      # 키워드 사다리 (search.py 와 같다)
                keep = merged

        gold = [row_of[c] for c in p["answer_chunk_indices"] if c in row_of]
        kept = int(keep[gold].sum()) if keep is not None else len(gold)
        n_kept = int(keep.sum()) if keep is not None else meta.size

        row = tally.setdefault(p["q_type"], [0, 0, 0, 0, [], 0, 0])
        row[0] += 1
        row[1] += 1 if kept else 0                # 하나라도 살아남았나
        row[2] += 1 if kept == len(gold) else 0   # 전부 살아남았나
        row[3] += n_kept
        if not kept:
            row[4].append(f"{p['id']} 0/{len(gold)}")
        row[5] += 1 if res.relaxed else 0
        row[6] += len(got.query.keywords)

    # 유형 이름은 qa.json 이 들고 있는 것을 그대로 쓴다. 문항 설계가 바뀌면
    # 이름도 같이 바뀌는데, 여기에 박아 두면 조용히 엉뚱한 이름이 붙는다.
    names = {p["q_type"]: p.get("q_type_name", f"유형 {p['q_type']}")
             for p in pairs}
    print(f"\n  추출 방식: {mode}   ({time.time() - started:.1f}초)")
    print(f"\n  {'유형':<14}{'문항':>5}{'정답1개+':>9}{'정답전부':>9}"
          f"{'평균후보':>10}{'축소율':>9}{'완화':>6}{'키워드':>7}")
    for qt in sorted(tally):
        n, any_ok, all_ok, pool, lost, relax, kw = tally[qt]
        avg = pool / n
        print(f"  {str(names.get(qt, qt)):<14}{n:>5}{any_ok:>9}{all_ok:>9}"
              f"{avg:>10,.0f}{100 * avg / meta.size:>8.1f}%{relax:>6}"
              f"{kw / n:>7.1f}")
        if lost:
            print(f"      [!] 정답이 통째로 걸러진 문항: {', '.join(lost)}")

    print("\n  정답1개+ : 정답 청크가 하나라도 후보에 남은 문항 수.")
    print("             qa.json 은 같은 증거가 512/128 overlap 으로 겹친 청크를")
    print("             모두 정답으로 인정하므로 답하는 데에는 하나만 남아도")
    print("             된다. 0개가 되는 것만 치명적이다.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="질문에서 메타데이터 조건을 뽑는다.")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--no-llm", action="store_true", help="규칙만 (모델 안 올림)")
    ap.add_argument("--llm", action="store_true", help="4B 를 쓴다 (기본)")
    ap.add_argument("--qa", action="store_true", help="qa.json 문항 표")
    args = ap.parse_args()

    mode = "rule" if args.no_llm else ("llm" if args.llm else DEFAULT_MODE)

    if args.qa:
        _qa_selfcheck(mode)
        return

    question = " ".join(args.question) or \
        "ahtyng와 alarm 사이의 대화에서 서버 현황을 파악한 증거를 찾아주세요."
    got = extract(question, mode=mode)

    print(f"\n질문 : {question}")
    print(f"방식 : {got.source}  ({got.elapsed:.1f}초)")
    if got.error:
        print(f"  [!] 4B 실패, 규칙으로 대체: {got.error}")
    print(f"규칙 : {json.dumps(got.rule, ensure_ascii=False)}")
    if got.llm is not None:
        print(f"4B   : {json.dumps(got.llm, ensure_ascii=False)}")
    print(f"최종 : {json.dumps(got.raw, ensure_ascii=False)}")
    print(f"조건 : {got.label()}")
    if got.query.unknown:
        print(f"  [!] 명부에 없어 버린 이름: {', '.join(got.query.unknown)}")
    if got.query.keywords:
        print(f"키워드: {', '.join(got.query.keywords)}")


if __name__ == "__main__":
    main()
