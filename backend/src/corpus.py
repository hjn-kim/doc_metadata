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
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHUNKS_ROOT = ROOT / "data" / "chunks"
EMB_ROOT = ROOT / "data" / "emb"
EMB_SUFFIX = "_embeddings.npz"

# 언어와 무관한 청크 메타데이터. build_chunk_meta.py 의 입력이자,
# 메타데이터 필터가 읽는 원본이다.
SESSIONS_PATH = CHUNKS_ROOT / "sessions.jsonl"
CHUNK_META_PATH = CHUNKS_ROOT / "chunk_meta.npz"
NICKS_PATH = CHUNKS_ROOT / "nicks.json"

# 정제된 원본 대화 CSV. chunking.py 의 입력이다. 청크 메타데이터는 이제
# sessions.jsonl 만 보고 만들므로 build_chunk_meta.py 는 이 파일을 읽지 않는다.
PROCESSED_CSV = ROOT / "data" / "jabberchat2020process.csv"

# 옛 법률 문서 시절의 경로. embed.py 가 아직 import 하므로 이름만 남겨 둔다.
TXT_ROOT = ROOT / "data" / "txt"

# 코퍼스를 하나로 좁히지 않고 전부 뒤질 때 쓰는 값.
ALL_DOCS = "all"

# 임베딩 판본. 디렉터리 이름 -> (모델 이름, 한 줄 설명)
EMB_MODELS: dict[str, tuple[str, str]] = {
    "bgem3": ("BAAI/bge-m3", "다국어 dense 1024차원 (+ sparse 헤드)"),
    "kurev1": ("nlpai-lab/KURE-v1", "한국어 특화 dense 1024차원"),
}

DEFAULT_EMB_MODEL = "bgem3"

# 코퍼스 키 -> (화면에 쓸 제목, 언어 코드, 언어 이름, 한 줄 설명)
DOC_TITLES: dict[str, tuple[str, str, str, str]] = {
    "jabber_ru": (
        "Conti Jabber 로그 · 원문", "ru", "러시아어",
        "2020-06-21~11-16 · dyad + 1시간 gap 세션",
    ),
    "jabber_en": (
        "Conti Jabber 로그 · 영문", "en", "영어",
        "같은 대화의 영어 번역 · 청크 경계 동일",
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

    @property
    def label(self) -> str:
        """선택 상자에 넣을 한 줄. 제목만으로는 언어가 안 보인다."""
        return f"{self.title} · {self.lang_name}"

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


def _describe(key: str) -> tuple[str, str, str, str]:
    """DOC_TITLES 에 없는 코퍼스도 굴러가게 파일 이름에서 뽑아 쓴다."""
    if key in DOC_TITLES:
        return DOC_TITLES[key]
    lang = key.rsplit("_", 1)[-1] if "_" in key else key[:2]
    return (key, lang, lang, "")


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
        title, lang, lang_name, note = _describe(key)

        code = lang
        if code in seen:                       # 언어 코드가 겹치는 코퍼스
            code = f"{code}{len(seen)}"
        seen.add(code)

        docs.append(Document(
            key=key,
            code=code,
            title=title,
            lang=lang,
            lang_name=lang_name,
            note=note,
            model_dir=model_dir,
            model_name=model_name,
            chunks_path=CHUNKS_ROOT / f"{key}.jsonl",
            emb_path=path,
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

    meta_state = ("있음" if CHUNK_META_PATH.is_file()
                  else "없음 (data/build_chunk_meta.py 를 돌리세요)")
    print(f"\n메타데이터  : {SESSIONS_PATH.name} "
          f"{'있음' if SESSIONS_PATH.is_file() else '없음'}"
          f" / {CHUNK_META_PATH.name} {meta_state}")


if __name__ == "__main__":
    main()
