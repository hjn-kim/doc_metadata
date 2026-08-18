#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 서버 — 파이프라인을 HTTP 로 연다.

계산은 전부 여기서 한다 (임베딩·리랭킹·LLM). 화면은 CPU 서버의 app.py 가
그린다. 이 파일은 main.run_pipeline 을 감싸기만 하고 로직을 새로 만들지 않는다.

    POST /rag/search    질문 하나를 파이프라인에 태운다
                        기본      : 다 끝난 뒤 JSON 한 번  (is-web 등)
                        stream=true: 단계마다 한 줄씩 NDJSON (app.py)
    GET  /rag/meta      문서 목록·질문 세트·모델 이름  (화면이 그리는 값 전부)
    GET  /health        모델 로드 상태

data 도 여기에만 둔다. CPU 서버에는 app.py 하나뿐이라 문서 목록도 질문 세트도
들고 있지 않다. 그래서 /rag/meta 가 함께 필요하다. 원문 전체는 보내지 않는다
(화면의 원문 미리보기는 뺐다). 글자 수만 세어 /rag/meta 에 실어 보낸다.

왜 스트리밍을 따로 두나:
    전체가 1분을 훌쩍 넘는다. 다 끝나고 한 번에 주면 화면이 그 동안 멈춘 것처럼
    보인다. app.py 는 원래 단계가 끝날 때마다 카드를 그리므로 그 동작을 살리려면
    단계별로 흘려보내야 한다. 답만 필요한 호출자는 기본 모드를 쓰면 된다.

동시 실행:
    모델이 11GB 넘게 올라가 있어서 두 요청이 겹치면 OOM 이다. 락 하나로 줄을
    세운다. uvicorn 은 반드시 --workers 1 로 띄운다 (워커마다 모델을 새로
    올린다).

실행:
    uvicorn src.server:app --host 0.0.0.0 --port 58566 --workers 1

    RAG_API_KEY=...   설정하면 X-API-Key 헤더를 요구한다 (기본: 인증 없음)
    RAG_WARMUP=0      기동 시 모델 예열을 건너뛴다 (기본: 예열함)
    LOCAL_LLM_MODEL   답변 생성 모델 (기본: Qwen/Qwen3-4B)
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema  # noqa: E402
from corpus import documents  # noqa: E402
from grade import load_questions  # noqa: E402
from local_llm import DEFAULT_MODEL as LLM_MODEL  # noqa: E402
from main import run_pipeline  # noqa: E402
from rerank import FINAL_TOP_N  # noqa: E402
from rerank_gpu import DEFAULT_MODEL as RERANKER_MODEL  # noqa: E402
from search import DEFAULT_MODEL as EMBED_MODEL, DEFAULT_TOP_K  # noqa: E402
from embed import CHUNK_SIZE, OVERLAP  # noqa: E402

API_KEY = os.getenv("RAG_API_KEY", "").strip()
WARMUP = os.getenv("RAG_WARMUP", "1").lower() not in ("0", "false", "no")
WARMUP_QUESTION = os.getenv("RAG_WARMUP_QUESTION",
                            "대마재배자는 누구에게 허가를 받나요?")

# GPU 를 쓰는 구간 전체를 감싼다. 검색 -> 리랭킹 -> 생성이 한 요청 안에서
# 이어지므로 단계마다 잠그면 중간에 다른 요청이 끼어들어 VRAM 이 겹친다.
GPU_LOCK = threading.Lock()

# 예열 상태. /health 가 읽는다.
STATE = {"ready": False, "warmup_error": None, "started": time.time(),
         "requests": 0, "last_error": None}


# --------------------------------------------------------------------------
# 기동 / 예열
# --------------------------------------------------------------------------

def _warmup() -> None:
    """
    더미 질문을 한 번 통과시켜 모델 3개를 미리 올린다.

    lru_cache 라 첫 요청이 알아서 올리기는 하지만, 그 요청 하나가 3~4분 걸린다.
    데모 중에 첫 사람만 그 시간을 뒤집어쓰는 상황을 막는다.
    """
    try:
        with GPU_LOCK:
            started = time.time()
            run_pipeline(WARMUP_QUESTION, doc=None, top_k=3, final_n=2)
        STATE["ready"] = True
        print(f"[warmup] 모델 예열 완료 ({time.time() - started:.0f}초)",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        # 예열이 실패해도 서버는 살려 둔다. 실제 요청에서 다시 시도한다.
        STATE["warmup_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[warmup] 실패: {STATE['warmup_error']}", flush=True)
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs = documents()
    print(f"[start] 문서 {len(docs)}종, 질문 {len(load_questions())}개", flush=True)
    if not docs:
        print(f"[start] [!] 색인이 비어 있습니다: {ROOT / 'data' / 'emb'}",
              flush=True)
    if WARMUP:
        # 포트를 먼저 열고 뒤에서 예열한다. 예열이 끝날 때까지 기다리면
        # 헬스체크가 타임아웃으로 죽는다.
        threading.Thread(target=_warmup, daemon=True).start()
    else:
        STATE["ready"] = True
    yield


app = FastAPI(title="SNS RAG GPU 서버", version="1.0", lifespan=lifespan)

# is-web 처럼 브라우저에서 직접 부르는 경우를 위해 열어 둔다. 서버끼리
# 부르는 경우(app.py)에는 상관없다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("RAG_CORS", "*").split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_key(x_api_key: str | None) -> None:
    """RAG_API_KEY 가 설정돼 있을 때만 검사한다."""
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key 가 필요합니다.")


# --------------------------------------------------------------------------
# 요청 본문
# --------------------------------------------------------------------------

class SearchRequest(BaseModel):
    question: str = Field(..., description="질문 한 줄")
    doc: str | None = Field(None, description="문서 키나 짧은 코드. 없으면 전체")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)
    final_n: int = Field(FINAL_TOP_N, ge=1, le=20)
    gold: list[str] | None = Field(None, description="정답 후보. 주면 5단계까지")
    answer_language: str | None = Field(None, description="답변 언어 (기본 한국어)")
    meta: str | None = Field(None, pattern="^(llm|rule|off)$",
                             description="0단계 조건 추출 방식. "
                                         "llm(기본) / rule(규칙만) / off(안 씀)")
    stream: bool = Field(False, description="단계별 NDJSON 으로 받을지")


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def _run(req: SearchRequest, on_stage=None):
    """락을 잡고 파이프라인 한 번."""
    with GPU_LOCK:
        STATE["requests"] += 1
        return run_pipeline(
            req.question,
            doc=req.doc,
            top_k=req.top_k,
            final_n=req.final_n,
            gold=req.gold or None,
            answer_language=req.answer_language,
            on_stage=on_stage,
            meta_mode=req.meta,
        )


def _line(payload: dict) -> str:
    """NDJSON 한 줄. 한글을 그대로 보낸다 (Content-Type 이 utf-8 이다)."""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _stream(req: SearchRequest):
    """
    파이프라인을 다른 스레드에서 돌리고 단계 결과를 큐로 받아 흘려보낸다.

    run_pipeline 은 on_stage 콜백으로 진행을 알리는 동기 함수라, 생성기에서
    바로 yield 할 수가 없다. 스레드 + 큐가 그 사이를 잇는다.
    """
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            if GPU_LOCK.locked():
                # 앞 요청이 도는 중이다. 기다린다는 사실을 알려 준다.
                events.put(("wait", None))
            result = _run(req, on_stage=lambda s, p: events.put(("stage", (s, p))))
            events.put(("done", result))
        except Exception as exc:  # noqa: BLE001
            STATE["last_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            events.put(("error", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = events.get()
        if kind == "wait":
            yield _line({"stage": "queued",
                         "message": "앞선 요청이 끝나기를 기다리는 중입니다."})
        elif kind == "stage":
            stage, obj = payload
            to_dict = schema.STAGE_TO_DICT.get(stage)
            if to_dict is None:            # 모르는 단계는 조용히 건너뛴다
                continue
            yield _line({"stage": stage, "payload": to_dict(obj)})
        elif kind == "done":
            yield _line({"stage": "done",
                         "response": schema.response_envelope(payload)})
            return
        else:
            yield _line({"stage": "error", "message": payload})
            return


@app.post("/rag/search")
def rag_search(req: SearchRequest, request: Request,
               x_api_key: str | None = Header(None)):
    check_key(x_api_key)

    if not (req.question or "").strip():
        raise HTTPException(status_code=400, detail="question 이 비어 있습니다.")
    if not documents():
        raise HTTPException(status_code=503,
                            detail="색인(data/emb)이 비어 있습니다.")

    wants_stream = req.stream or "application/x-ndjson" in (
        request.headers.get("accept") or "")

    if wants_stream:
        return StreamingResponse(
            _stream(req),
            media_type="application/x-ndjson; charset=utf-8",
            # 중간 프록시가 버퍼링하면 스트리밍이 의미가 없다.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = _run(req)
    except Exception as exc:  # noqa: BLE001
        STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"{type(exc).__name__}: {exc}") from exc

    return JSONResponse(schema.response_envelope(result))


# --------------------------------------------------------------------------
# 메타 / 헬스
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _doc_rows() -> list[dict]:
    """
    코퍼스 한 줄씩 — 화면의 데이터셋 표에 그대로 들어간다.

    글자 수와 청크·토큰·차원은 파일을 읽어야 나온다. CPU 서버에는 파일이 없으므로
    여기서 세어 보낸다. 코퍼스가 2종뿐이고 바뀌지 않으므로 한 번만 센다.

    글자 수는 .npz 의 texts 에서 센다. 원문 .jsonl 을 읽으면 13MB 를 통째로
    파싱하는 데다 세션 메타데이터까지 글자 수에 섞인다.
    """
    import numpy as np

    rows = []
    for d in documents():
        chunks = tokens = dim = chars = 0
        try:
            with np.load(d.emb_path, allow_pickle=False) as z:
                n, dimension = z["embeddings"].shape
                chunks, dim = int(n), int(dimension)
                tokens = int(z["token_count"].sum())
                chars = int(np.char.str_len(z["texts"]).sum())
        except Exception:  # noqa: BLE001 - 깨진 파일은 0 으로 둔다
            pass

        rows.append({
            "key": d.key, "code": d.code, "title": d.title,
            "lang": d.lang, "lang_name": d.lang_name, "note": d.note,
            "model": d.model_name, "model_dir": d.model_dir,
            "has_text": d.has_chunks,
            "chars": chars, "chunks": chunks, "tokens": tokens, "dim": dim,
        })
    return rows


@app.get("/rag/meta")
def rag_meta(x_api_key: str | None = Header(None)):
    """
    화면이 그리는 값 전부. CPU 서버가 기동 때 한 번 읽어 캐시한다.

    문서 목록·질문 세트·모델 이름·청킹 설정이 여기서 나간다. 문서를 추가하면
    GPU 서버만 다시 띄우면 되고 CPU 쪽은 손댈 것이 없다.
    """
    check_key(x_api_key)
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "models": {
            "embed": EMBED_MODEL,
            "rerank": RERANKER_MODEL,
            "llm": LLM_MODEL,
        },
        "chunk": {"size": CHUNK_SIZE, "overlap": OVERLAP},
        "defaults": {"top_k": DEFAULT_TOP_K, "final_n": FINAL_TOP_N},
        "documents": _doc_rows(),
        "questions": [
            {"id": q.id, "doc": q.doc, "question": q.question,
             "answer": q.answer, "keywords": list(q.keywords)}
            for q in load_questions()
        ],
    }


@app.get("/health")
def health():
    """인증 없이 열어 둔다 (로드밸런서·감시 스크립트용)."""
    info = {
        "ok": True,
        "ready": STATE["ready"],
        "busy": GPU_LOCK.locked(),
        "uptime_sec": round(time.time() - STATE["started"], 1),
        "requests": STATE["requests"],
        "n_docs": len(documents()),
        "warmup_error": STATE["warmup_error"],
        "last_error": STATE["last_error"],
    }
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "free_gb": round(free / 1024 ** 3, 1),
                "total_gb": round(total / 1024 ** 3, 1),
            }
        else:
            info["gpu"] = None
    except Exception:  # noqa: BLE001
        info["gpu"] = None
    return info
