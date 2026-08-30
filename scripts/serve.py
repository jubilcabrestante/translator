"""
serve.py
--------
HTTP server exposing the fine-tuned model at the exact contract LearnVoca's
TranslationRepository already calls (see learn_voca/lib/features/translator/data/
translation_repository.dart, `_callService`):

    POST /translate
    body:     {"sentence": "...", "source_lang": <code>, "target_lang": <code>}
    response: {"translated_sentence": "..."}

Valid codes are whatever `languages.json` defines (GET /health lists them).

This is Tier 3 in the app's lookup chain (cache -> dictionary -> this service).
Nothing on the Flutter/Supabase side needs to change; point app_config.translator
.base_url at wherever this server ends up running.

Run locally:
    .\\.venv\\Scripts\\python.exe scripts/serve.py
    # then, for a real device during development, tunnel it:
    #   cloudflared tunnel --url http://localhost:8000
    # and put the https://*.trycloudflare.com URL into app_config.translator.base_url

The model loads ONCE at startup (not per-request) -- reloading a 1.5B model on
every call would make each translation take as long as this whole server's boot.
"""

import argparse
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import load_config
from translate import load, translate  # same load/translate used by the CLI + evaluate.py

# Language codes come from languages.json, so adding a language does not require
# touching this file.
#
# HEADS UP: your Supabase `translations` table has its own CHECK constraint that
# allows only 'tl' and 'cyo'. Widen it before serving a third code, or
# record_translation() will reject every row it tries to cache.
_CONFIG = load_config()
_CODE_TO_LANG = dict(_CONFIG["code_to_name"])

app = FastAPI(title=" <-> ".join(_CONFIG["language_names"]) + " Translator")
_state = {"model": None, "tokenizer": None}


class TranslateRequest(BaseModel):
    sentence: str
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translated_sentence: str


@app.on_event("startup")
def _load_model():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = os.environ.get("ADAPTER_DIR", "outputs/translator-lora")
    print(f"Loading {base_model} + adapter from {adapter_dir} ...")
    model, tokenizer = load(base_model, adapter_dir)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    print("Model ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "languages": _CODE_TO_LANG,
    }


@app.post("/translate", response_model=TranslateResponse)
def do_translate(req: TranslateRequest):
    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="sentence must not be empty")

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

    result = translate(_state["model"], _state["tokenizer"], sentence, src_lang, tgt_lang)
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
