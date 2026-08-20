#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
코퍼스 레지스트리 — data/chunks 와 data/emb 를 짝지어 준다.

    data/chunks/{키}.jsonl                     청크 본문 + 세션 메타데이터
    data/chunks/sessions.jsonl                 본문 없는 메타데이터만 (언어 공통)
    data/emb/{모델}/{키}_embeddings.npz        청크 벡터

지금 코퍼스는 Conti 조직의 Jabber 1:1 대화 로그(2020-06-21~11-16) 하나이고,
같은 대화가 두 벌로 들어 있다.

    jabber_ru   원문 (러시아어 중심)
    jabber_en   같은 대화의 영어 번역

두 벌은 청크 경계를 영어 기준으로 한 번만 계산했기 때문에(chunking.py 의
--split-basis en) chunk_index 가 완전히 일치한다. 그래서 sessions.jsonl 의
메타데이터(참여자·시각) 한 벌이 두 언어에 그대로 들어맞고, QA 정답도 한 벌이면
된다. 메타데이터 필터가 언어와 무관하게 도는 것도 이 성질 덕분이다.

임베딩 모델은 문서가 아니라 '색인의 판본'이다. 같은 청크를 모델별로 따로
임베딩해 data/emb/{모델}/ 아래에 나눠 담는다.

    bgem3    BAAI/bge-m3        다국어 dense (+ sparse 헤드)
    kurev1   nlpai-lab/KURE-v1  한국어 특화 dense

어느 판본을 쓸지는 환경변수 RAG_EMB_MODEL 로 고른다 (기본 bgem3). 판본을 바꾸면
검색 점수가 통째로 달라지므로 프로세스를 새로 띄우는 것을 권한다.

코퍼스 코드는 언어 코드를 그대로 쓴다 (ru / en). 인용 표기가 "ru#1465" 처럼
짧아야 답변 안에서 읽히기 때문이다. 키 앞 두 글자를 쓰던 옛 규칙은 여기서는
둘 다 "ja" 가 되어 못 쓴다.

코퍼스를 추가하는 방법:
    1. data/chunks 에 {키}.jsonl 을 넣는다 (chunking.py 가 만든다)
    2. 임베딩을 만들어 data/emb/{모델}/{키}_embeddings.npz 로 저장한다
    3. 아래 DOC_TITLES 에 제목과 언어를 적는다
       적지 않아도 파일 이름에서 뽑아 쓰므로 앱은 그대로 돈다

단독 실행:
    python src/corpus.py
    RAG_EMB_MODEL=kurev1 python src/corpus.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHUNKS_ROOT = ROOT / "data" / "chunks"
EMB_ROOT = ROOT / "data" / "emb"
EMB_SUFFIX = "_embeddings.npz"

# 청크 메타데이터. **문서마다 한 벌씩** 둔다.
#
#     data/meta/{메타키}.npz          청크별 대화쌍(dyad)·시각
#     data/meta/{메타키}_nicks.json   닉네임 명부 (있는 문서만)
#
# '문서키' 가 아니라 '메타키' 인 이유는 jabber_ru 와 jabber_en 때문이다. 둘은
# 같은 대화를 번역만 한 것이라 청크 경계가 완전히 일치한다(chunking.py 의
# --split-basis en). 메타데이터 한 벌이 두 언어에 그대로 들어맞으므로 문서별로
# 쪼개면 똑같은 파일을 두 벌 들고 있게 된다. 그래서 아래 DOC_SPECS 에서
# 두 문서가 같은 meta_key("jabber") 를 가리킨다.
#
# 메타데이터가 아예 없는 문서도 된다 (meta_key=None). 그런 문서는 사람·기간으로
# 좁힐 수 없을 뿐, 검색·리랭킹·답변은 그대로 돈다.
META_ROOT = ROOT / "data" / "meta"
META_SUFFIX = ".npz"
NICKS_SUFFIX = "_nicks.json"

# build_chunk_meta.py 의 입력. jabber 전용이다 (다른 문서는 자기 청크 .jsonl 에
# 메타데이터를 이미 들고 있어서 중간 파일이 필요 없다).
SESSIONS_PATH = CHUNKS_ROOT / "sessions.jsonl"


def meta_path(meta_key: str) -> Path:
    """data/meta/{메타키}.npz"""
    return META_ROOT / f"{meta_key}{META_SUFFIX}"


def nicks_path(meta_key: str) -> Path:
    """data/meta/{메타키}_nicks.json"""
    return META_ROOT / f"{meta_key}{NICKS_SUFFIX}"


# 옛 이름. chunk_meta.py 의 기본값과 단독 실행 경로가 아직 쓴다.
CHUNK_META_PATH = meta_path("jabber")
NICKS_PATH = nicks_path("jabber")

# 정제된 원본 대화 CSV. chunking.py 의 입력이다. 청크 메타데이터는 이제
# sessions.jsonl 만 보고 만들므로 build_chunk_meta.py 는 이 파일을 읽지 않는다.
PROCESSED_CSV = ROOT / "data" / "jabberchat2020process.csv"

# 옛 법률 문서 시절의 경로. embed.py 가 아직 import 하므로 이름만 남겨 둔다.
TXT_ROOT = ROOT / "data" / "txt"

# 코퍼스를 하나로 좁히지 않고 전부 뒤질 때 쓰는 값.
ALL_DOCS = "all"

# 한글 음절·자모. 문서가 어느 글자를 쓰는지 판정하는 곳이 두 군데(키워드 채널,
# 이름 패턴)라 여기 한 벌만 둔다. 비ASCII 로 갈음하면 안 된다 — jabber 닉 289개
# 중 'стов' 하나가 키릴이라, 그것 때문에 jabber 를 한글 문서로 착각한다.
HANGUL_RE = re.compile("[" + chr(0xAC00) + "-" + chr(0xD7A3)
                       + chr(0x1100) + "-" + chr(0x11FF)
                       + chr(0x3130) + "-" + chr(0x318F) + "]")


def script_of(text: str) -> str:
    """글자 하나 묶음이 어느 문자 종류인가. ascii / hangul / other."""
    text = str(text)
    if text.isascii():
        return "ascii"
    if HANGUL_RE.search(text):
        return "hangul"
    return "other"

# 임베딩 판본. 디렉터리 이름 -> (모델 이름, 한 줄 설명)
EMB_MODELS: dict[str, tuple[str, str]] = {
    "bgem3": ("BAAI/bge-m3", "다국어 dense 1024차원 (+ sparse 헤드)"),
    "kurev1": ("nlpai-lab/KURE-v1", "한국어 특화 dense 1024차원"),
}

DEFAULT_EMB_MODEL = "bgem3"

@dataclass(frozen=True)
class DocSpec:
    """
    코퍼스 하나의 명세. **문서를 추가할 때 손대는 곳은 여기 한 줄뿐이다.**

    제목·언어는 화면용이고, 나머지 넷은 "이 문서에 무엇을 물어볼 수 있는가" 다.
    0단계 필터가 이 값을 보고 문서마다 다르게 군다.

        meta_key    어느 메타데이터 파일을 읽을지. None 이면 메타데이터 없음
        has_time    기간으로 좁힐 수 있는가 (원문에 시각이 있는가)
        has_people  사람으로 좁힐 수 있는가 (화자를 식별할 수 있는가)
        kw_scripts  본문에 글자 그대로 나올 수 있는 문자 종류

    kw_scripts 를 문서마다 두는 이유:

        키워드 채널은 질문에서 주워 온 문자열을 본문과 리터럴로 맞춰 본다.
        원문이 러시아어·영어인 jabber 에서는 '대포통장' 같은 한글이 본문에
        나올 수 없으므로 걸어 봐야 헛일이고, 반대로 한국어 원문인 ko_voice 에서는
        한글이야말로 제일 정확히 맞는 키워드다. 문서를 안 보고 하나로 정하면
        어느 한쪽이 반드시 손해를 본다.
    """

    title: str                              # 화면에 쓸 제목
    lang: str                               # 언어 코드 (ru / en / ko)
    lang_name: str                          # 언어 이름 (러시아어)
    note: str                               # 한 줄 설명
    meta_key: str | None = None             # data/meta/{meta_key}.npz
    has_time: bool = True
    has_people: bool = True
    kw_scripts: tuple[str, ...] = ("ascii",)


# 코퍼스 키 -> 명세.
#
# jabber_ru 와 jabber_en 이 meta_key 를 함께 쓴다 (같은 대화, 같은 청크 번호).
# ko_voice 는 자기 몫을 따로 갖되 시각이 없다 — 원문에 통화 시각이 적혀 있지
# 않아서 chunking_ko.py 가 ts 를 null 로 둔다. 기간으로는 영영 못 좁히는
# 문서이고, 그 사실을 여기 적어 두면 필터가 알아서 비켜 간다.
DOC_SPECS: dict[str, DocSpec] = {
    "jabber_ru": DocSpec(
        "ru_Contiransom", "ru", "러시아어",
        "2020-06-21~11-16 · dyad + 1시간 gap 세션",
        meta_key="jabber",
    ),
    "jabber_en": DocSpec(
        "en_Contiransom ", "en", "영어",
        "같은 대화의 영어 번역 · 청크 경계 동일",
        meta_key="jabber",
    ),
    "ko_voice": DocSpec(
        "ko_보이스피싱통화록", "ko", "한국어",
        "통화 70건 · 통화 1건 = 청크 1개 (긴 통화만 발화 경계 분할)",
        meta_key="ko_voice",
        has_time=False,                     # 원문에 시각이 없다
        kw_scripts=("ascii", "hangul"),
    ),
}


def active_model() -> str:
    """
    지금 쓸 임베딩 판본(디렉터리 이름).

    환경변수를 부를 때마다 읽는다. 모르는 이름이면 기본값으로 돌아가되 경고를
    남긴다 — 오타 하나에 검색이 조용히 빈 결과를 내는 것보다 낫다.
    """
    name = os.getenv("RAG_EMB_MODEL", DEFAULT_EMB_MODEL).strip()
    if name in EMB_MODELS:
        return name
    if name and (EMB_ROOT / name).is_dir():          # 등록만 안 된 새 판본
        return name
    print(f"[corpus] 모르는 임베딩 판본입니다: {name!r} -> {DEFAULT_EMB_MODEL}",
          file=sys.stderr)
    return DEFAULT_EMB_MODEL


@dataclass(frozen=True)
class Document:
    """코퍼스 하나. 청크와 색인이 둘 다 있는 것만 만든다."""

    key: str            # 파일 이름에서 온 코퍼스 키 (jabber_ru)
    code: str           # 짧은 코드 (ru). 인용 표기 "ru#1465" 의 앞부분
    title: str          # 화면에 쓸 제목
    lang: str           # 언어 코드 (ru / en)
    lang_name: str      # 언어 이름 (러시아어)
    note: str           # 한 줄 설명
    model_dir: str      # 임베딩 판본 디렉터리 (bgem3)
    model_name: str     # 임베딩 모델 이름 (BAAI/bge-m3)
    chunks_path: Path   # data/chunks/{키}.jsonl
    emb_path: Path      # data/emb/{모델}/{키}_embeddings.npz

    # --- 메타데이터 (DOC_SPECS 에서 온다) --------------------------------
    meta_key: str | None = None         # None 이면 메타데이터 파일이 없는 문서
    has_time: bool = True               # 기간으로 좁힐 수 있는가
    has_people: bool = True             # 사람으로 좁힐 수 있는가
    kw_scripts: tuple[str, ...] = ("ascii",)   # 리터럴로 맞춰 볼 문자 종류

    @property
    def label(self) -> str:
        """선택 상자에 넣을 한 줄. 제목만으로는 언어가 안 보인다."""
        return f"{self.title} · {self.lang_name}"

    # --- 메타데이터 경로 ------------------------------------------------
    @property
    def group(self) -> str:
        """
        청크를 세는 단위. 메타데이터를 함께 쓰는 문서는 같은 값을 갖는다.

        jabber_ru 와 jabber_en 은 같은 대화의 두 판본이라 "청크 15,522개" 로
        세어야 사람이 아는 수와 맞는다. 문서 키로 세면 두 배로 보인다.
        메타데이터가 없는 문서는 자기 혼자 한 묶음이다.
        """
        return self.meta_key or self.key

    @property
    def meta_path(self) -> Path | None:
        """data/meta/{메타키}.npz. 메타데이터가 없는 문서면 None."""
        return meta_path(self.meta_key) if self.meta_key else None

    @property
    def nicks_path(self) -> Path | None:
        """data/meta/{메타키}_nicks.json. 명부가 없는 문서면 None."""
        return nicks_path(self.meta_key) if self.meta_key else None

    @property
    def has_meta(self) -> bool:
        """메타데이터 파일이 실제로 있는가. 없으면 필터가 이 문서를 건너뛴다."""
        path = self.meta_path
        return path is not None and path.is_file()

    @property
    def has_nicks(self) -> bool:
        """
        명부 파일이 있는가. 0단계 추출기의 화이트리스트로 쓴다.

        ko_voice 처럼 화자가 '가해자1' 같은 역할 이름뿐인 문서는 명부를 두지
        않는다. 질문에 그런 이름이 적혀 올 일이 없어서 화이트리스트가 할 일이
        없기 때문이다 (.npz 안에는 dyad 를 적기 위한 이름표가 그대로 들어 있다).
        """
        path = self.nicks_path
        return path is not None and path.is_file()

    @property
    def has_chunks(self) -> bool:
        return self.chunks_path.is_file()

    # --- 옛 이름. server.py 가 아직 부른다 -------------------------------
    @property
    def txt_path(self) -> Path:
        return self.chunks_path

    @property
    def has_text(self) -> bool:
        return self.has_chunks


def _spec(key: str) -> DocSpec:
    """
    DOC_SPECS 에 없는 코퍼스도 굴러가게 파일 이름에서 뽑아 쓴다.

    등록하지 않은 문서는 '메타데이터 없음' 으로 본다. 있는지 없는지 모르는
    파일을 있다고 가정하는 것보다, 없다고 보고 검색만 돌리는 쪽이 안전하다.
    """
    got = DOC_SPECS.get(key)
    if got is not None:
        return got
    lang = key.rsplit("_", 1)[-1] if "_" in key else key[:2]
    return DocSpec(key, lang, lang, "", meta_key=None,
                   has_time=False, has_people=False)


@lru_cache(maxsize=8)
def load_documents(model_dir: str) -> tuple[Document, ...]:
    """
    data/emb/{model_dir}/ 의 .npz 를 기준으로 코퍼스 목록을 만든다.

    색인이 있어야 검색이 되므로 .npz 가 기준이다. 청크 .jsonl 은 없어도 되고
    (미리보기만 못 쓴다) 코드가 겹치면 뒤에 오는 코퍼스에 숫자를 붙여 구분한다.
    """
    model_name = EMB_MODELS.get(model_dir, ("", ""))[0]
    emb_dir = EMB_ROOT / model_dir

    docs: list[Document] = []
    seen: set[str] = set()

    for path in sorted(emb_dir.glob(f"*{EMB_SUFFIX}")):
        key = path.name[: -len(EMB_SUFFIX)]
        spec = _spec(key)

        code = spec.lang
        if code in seen:                       # 언어 코드가 겹치는 코퍼스
            code = f"{code}{len(seen)}"
        seen.add(code)

        docs.append(Document(
            key=key,
            code=code,
            title=spec.title,
            lang=spec.lang,
            lang_name=spec.lang_name,
            note=spec.note,
            model_dir=model_dir,
            model_name=model_name,
            chunks_path=CHUNKS_ROOT / f"{key}.jsonl",
            emb_path=path,
            meta_key=spec.meta_key,
            has_time=spec.has_time,
            has_people=spec.has_people,
            kw_scripts=spec.kw_scripts,
        ))
    return tuple(docs)


def documents(model_dir: str | None = None) -> list[Document]:
    """코퍼스 목록. 화면과 파이프라인이 모두 이 순서를 따른다."""
    return list(load_documents(model_dir or active_model()))


def find(name: str | None, model_dir: str | None = None) -> Document | None:
    """
    코퍼스 키 · 짧은 코드 · 언어 · 제목 중 무엇으로 찾아도 되게 한다.

    None 이나 "all" 은 '전체'라는 뜻이라 None 을 돌려준다.
    """
    if not name or name == ALL_DOCS:
        return None
    for doc in load_documents(model_dir or active_model()):
        if name in (doc.key, doc.code, doc.lang, doc.title, doc.label):
            return doc
    raise KeyError(f"모르는 코퍼스입니다: {name}")


def documents_for(name: str | None, model_dir: str | None = None) -> list[Document]:
    """
    검색 대상이 될 문서 목록. 이름을 주면 하나, 없거나 'all' 이면 전부.

    search.load_index(doc) 가 합치는 문서와 정확히 같은 목록이다. 0단계 필터가
    "어느 문서에 무엇을 물어볼 수 있는가" 를 따지려면 이 목록이 필요하다.
    """
    target = find(name, model_dir)
    return [target] if target else documents(model_dir)


def doc_label(name: str | None, model_dir: str | None = None) -> str:
    """화면 문구용. 전체 검색이면 코퍼스 수를 적어 준다."""
    doc = find(name, model_dir)
    if doc:
        return doc.label
    return f"전체 코퍼스 {len(load_documents(model_dir or active_model()))}종"


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    model_dir = active_model()
    model_name, model_note = EMB_MODELS.get(model_dir, ("(미등록)", ""))
    print(f"임베딩 판본 : {model_dir}  ({model_name}) {model_note}")
    print(f"청크        : {CHUNKS_ROOT}")
    print(f"색인        : {EMB_ROOT / model_dir}\n")

    docs = documents()
    if not docs:
        print(f"[!] 색인이 없습니다: {EMB_ROOT / model_dir}/*{EMB_SUFFIX}")
        if EMB_ROOT.is_dir():
            found = sorted(p.name for p in EMB_ROOT.iterdir() if p.is_dir())
            print(f"    쓸 수 있는 판본: {', '.join(found) or '(없음)'}")
        return

    print(f"{'코드':<6}{'언어':<12}{'청크':>10}  제목")
    for doc in docs:
        size = (f"{doc.chunks_path.stat().st_size // 1024:,}KB"
                if doc.has_chunks else "-")
        print(f"{doc.code:<6}{doc.lang_name:<12}{size:>10}  {doc.title}")
        if not doc.has_chunks:
            print(f"{'':<6}[!] 청크가 없습니다: {doc.chunks_path}")

    # 문서마다 메타데이터가 따로다. 어느 문서에 무엇을 물어볼 수 있는지를
    # 한눈에 보여 준다 — 필터가 "왜 이 문서는 안 좁혀졌지" 를 되짚는 자리다.
    print(f"\n메타데이터  : {META_ROOT}")
    print(f"{'코드':<6}{'파일':<24}{'기간':<6}{'사람':<6}{'명부':<6}키워드")
    for doc in docs:
        if doc.meta_key is None:
            name = "(없음)"
        else:
            name = doc.meta_path.name + ("" if doc.has_meta else "  [!]")
        print(f"{doc.code:<6}{name:<24}"
              f"{'O' if doc.has_time else '-':<6}"
              f"{'O' if doc.has_people and doc.has_meta else '-':<6}"
              f"{'O' if doc.has_nicks else '-':<6}"
              f"{'+'.join(doc.kw_scripts)}")

    missing = sorted({d.meta_key for d in docs if d.meta_key and not d.has_meta})
    if missing:
        print(f"\n[!] 메타데이터가 없습니다: {', '.join(missing)}")
        print("    python data/build_chunk_meta.py --doc all 를 돌리세요")



if __name__ == "__main__":
    main()
