"""
serve.py
--------
HTTP server exposing the translator at the exact contract LearnVoca's
TranslationRepository already calls (see learn_voca/lib/features/translator/data/
translation_repository.dart, `_callService`):

    POST /translate
    body:     {"sentence": "...", "source_lang": <code>, "target_lang": <code>}
    response: {"translated_sentence": "..."}

Valid codes are whatever `languages.json` defines (GET /health lists them).
This is Tier 3 in the app's lookup chain (cache -> dictionary -> this service).

Two ways to run the model behind it:

  * LOCAL (default): loads base model + LoRA adapter with transformers, on the
    GPU when there is one, otherwise on the CPU (slow; ~6 GB RAM for 1.5B).
  * BACKEND: set TRANSLATOR_BACKEND_URL to an OpenAI-compatible chat endpoint
    -- Ollama (http://localhost:11434/v1) or llama.cpp's llama-server
    (http://localhost:8080/v1) running the GGUF that scripts/export_model.py
    helps you build. This process then needs only fastapi + uvicorn (no
    torch), so it fits a small, cheap CPU server.

Environment variables:
  BASE_MODEL, ADAPTER_DIR     local mode (defaults as in translate.py)
  TRANSLATOR_BACKEND_URL      backend mode, see above
  TRANSLATOR_BACKEND_MODEL    the model name the backend knows (default "translator")
  TRANSLATOR_API_KEY          if set, /translate requires the header
                              `X-API-Key: <key>` (or `Authorization: Bearer <key>`).
                              Set it before exposing the server through a
                              tunnel -- and make the app send the same header.
  MAX_INPUT_CHARS             longest sentence accepted (default 500)

Run locally:
    .\\.venv\\Scripts\\python.exe scripts/serve.py
    # then, for a real device during development, tunnel it:
    #   cloudflared tunnel --url http://localhost:8000
    # and put the https://*.trycloudflare.com URL into app_config.translator.base_url

The model loads ONCE at startup (not per-request) -- reloading a 1.5B model on
every call would make each translation take as long as this whole server's boot.
"""

import argparse
import hmac
import json
import os
import threading
import urllib.error
import urllib.request
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from common import build_messages, load_config

# Language codes come from languages.json, so adding a language does not require
# touching this file.
#
# HEADS UP: your Supabase `translations` table has its own CHECK constraint that
# allows only 'tl' and 'cyo'. Widen it before serving a third code, or
# record_translation() will reject every row it tries to cache.
_CONFIG = load_config()
_CODE_TO_LANG = dict(_CONFIG["code_to_name"])
_BACKEND_URL = os.environ.get("TRANSLATOR_BACKEND_URL", "").rstrip("/")
_BACKEND_MODEL = os.environ.get("TRANSLATOR_BACKEND_MODEL", "translator")
_API_KEY = os.environ.get("TRANSLATOR_API_KEY", "")
_MAX_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))

app = FastAPI(title=" <-> ".join(_CONFIG["language_names"]) + " Translator")
_state = {"model": None, "tokenizer": None, "translate": None}
# One generate() at a time. FastAPI runs this sync endpoint on a thread pool,
# and parallel calls on one model only compete for the same GPU memory.
_lock = threading.Lock()


class TranslateRequest(BaseModel):
    sentence: str
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translated_sentence: str


@app.on_event("startup")
def _load_model():
    if not _API_KEY:
        print("NOTE: TRANSLATOR_API_KEY is not set, so anyone who can reach this port "
              "can use it. Set it before exposing the server through a tunnel.")
    if _BACKEND_URL:
        print(f"Backend mode: forwarding to {_BACKEND_URL} (model '{_BACKEND_MODEL}').")
        return
    from translate import load, translate    # imports torch; skipped in backend mode
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = os.environ.get("ADAPTER_DIR", "outputs/translator-lora")
    print(f"Loading {base_model} + adapter from {adapter_dir} ...")
    _state["model"], _state["tokenizer"] = load(base_model, adapter_dir)
    _state["translate"] = translate
    print("Model ready.")


def _check_key(x_api_key, authorization):
    if not _API_KEY:
        return
    supplied = x_api_key or ""
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[len("bearer "):].strip()
    if not hmac.compare_digest(supplied.encode("utf-8"), _API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="missing or wrong API key")


def _translate_via_backend(sentence, src_lang, tgt_lang):
    """Ask an OpenAI-compatible server, with the exact prompt used in training."""
    body = json.dumps({
        "model": _BACKEND_MODEL,
        "messages": build_messages(sentence, src_lang, tgt_lang),
        "temperature": 0,
        # Same reason as in translate.py: a repetition penalty fights copying
        # names and shared words. Honoured by llama-server; ignored elsewhere.
        "repeat_penalty": 1.0,
        "max_tokens": min(512, len(sentence) + 64),
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(f"{_BACKEND_URL}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"translation backend failed: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": _BACKEND_URL or "local",
        "model_loaded": bool(_BACKEND_URL) or _state["model"] is not None,
        "languages": _CODE_TO_LANG,
        "auth_required": bool(_API_KEY),
        "max_input_chars": _MAX_CHARS,
    }


@app.post("/translate", response_model=TranslateResponse)
def do_translate(req: TranslateRequest,
                 x_api_key: Optional[str] = Header(default=None),
                 authorization: Optional[str] = Header(default=None)):
    _check_key(x_api_key, authorization)
    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="sentence must not be empty")
    if len(sentence) > _MAX_CHARS:
        raise HTTPException(status_code=413,
                            detail=f"sentence is longer than {_MAX_CHARS} characters")

    src_lang = _CODE_TO_LANG.get(req.source_lang)
    tgt_lang = _CODE_TO_LANG.get(req.target_lang)
    if src_lang is None or tgt_lang is None:
        valid = "', '".join(sorted(_CODE_TO_LANG))
        raise HTTPException(
            status_code=400,
            detail=f"source_lang/target_lang must be one of '{valid}'",
        )
    if src_lang == tgt_lang:
        raise HTTPException(status_code=400, detail="source_lang and target_lang must differ")

    if _BACKEND_URL:
        result = _translate_via_backend(sentence, src_lang, tgt_lang)
    else:
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="model is still loading")
        with _lock:
            result = _state["translate"](_state["model"], _state["tokenizer"],
                                         sentence, src_lang, tgt_lang)
    if not result:
        raise HTTPException(status_code=404, detail="No translation found for that phrase.")
    return TranslateResponse(translated_sentence=result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
