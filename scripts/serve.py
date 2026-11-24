"""
serve.py
--------
HTTP server exposing the translator at the exact contract the Learnut app
already calls (learnut repo: shared/src/commonMain/kotlin/com/app/tagacuyo/
core/data/OtherRepositories.kt, `TranslationRepository.translate`):

    POST /translate
    body:     {"sentence": "...", "source_lang": "Tagalog", "target_lang": "Cuyonon"}
    response: {"translated_sentence": "..."}

source_lang/target_lang may be a language's name (what Learnut sends) or its
code ("tl", "cyo"), in any case; both come from `languages.json` (GET /health
lists them). Extra body fields (Learnut also sends "translated": null) are
ignored. The app reads this server's URL from Supabase: table app_config, key
'app_translator', field 'get_api_translator'. It then logs every answer into
public.translations itself; this server never talks to Supabase.

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
                              Learnut does NOT send one yet: setting this makes
                              every request from the app fail with
                              "Unauthorized". Only set it once the app sends it.
  MAX_INPUT_CHARS             longest sentence accepted (default 500; Learnut
                              caps input at app_translator.translation_string_length)
  TRANSLATOR_DOCS             "1" turns on FastAPI's /docs, /redoc and /openapi.json
                              pages. They're off by default because this server is
                              public, and Learnut needs only /translate and /health.

Run locally:
    .\\.venv\\Scripts\\python.exe scripts/serve.py
    # then, for a real device during development, tunnel it:
    #   cloudflared tunnel --url http://localhost:8000
    # and put the https://*.trycloudflare.com URL into Supabase:
    #   app_config -> app_translator -> get_api_translator  (see the README)
    # That setting is global: it switches EVERY user's app to your machine.

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

from common import build_messages, load_config, resolve_language

# Languages come from languages.json, so adding a language does not require
# touching this file. Learnut sends the language NAME, so renaming one in
# languages.json breaks the app until the app is updated too.
_CONFIG = load_config()
_CODE_TO_LANG = dict(_CONFIG["code_to_name"])
_BACKEND_URL = os.environ.get("TRANSLATOR_BACKEND_URL", "").rstrip("/")
_BACKEND_MODEL = os.environ.get("TRANSLATOR_BACKEND_MODEL", "translator")
_API_KEY = os.environ.get("TRANSLATOR_API_KEY", "")
_MAX_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))

_DOCS = os.environ.get("TRANSLATOR_DOCS") == "1"
app = FastAPI(
    title=" <-> ".join(_CONFIG["language_names"]) + " Translator",
    docs_url="/docs" if _DOCS else None,
    redoc_url="/redoc" if _DOCS else None,
    openapi_url="/openapi.json" if _DOCS else None,
)
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
              "can use it. (Learnut sends no key yet, so leave it unset for the app.)")
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

    src_lang = resolve_language(_CONFIG, req.source_lang)
    tgt_lang = resolve_language(_CONFIG, req.target_lang)
    if src_lang is None or tgt_lang is None:
        valid = "', '".join(sorted(_CODE_TO_LANG) + list(_CONFIG["language_names"]))
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
