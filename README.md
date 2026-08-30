# Tagalog ⇄ Cuyonon Translator (LLM fine-tune)

A **decoder-only LLM** (same architecture family as Claude/GPT) fine-tuned into a
specialized two-way translation engine using **QLoRA**, sized to train on a
**4 GB laptop GPU** (RTX 3050).

One model handles **both directions**:
- Tagalog → Cuyonon
- Cuyonon → Tagalog

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (open, ungated, multilingual, instruction-tuned).

---

## Why this instead of the old GRU seq2seq?

| | Old (GRU + attention, from scratch) | This (QLoRA fine-tune of an LLM) |
|---|---|---|
| Starting knowledge | none — learns language from your 2.7k pairs | already "knows" many languages; only learns *your mapping* |
| Architecture | RNN | Transformer (the modern standard) |
| Scales with more data | poorly (overfits) | yes — more data + bigger base model = better |
| Style | fixed input→output | instruction-following ("translate X to Y") |

The architecture does **not** change as your dataset grows. To get better results
later you (a) add more parallel data, and/or (b) swap `--model` for a bigger
checkpoint on a bigger GPU. Same code.

---

## Project layout

```
translator/
├── assets/
│   ├── taga-cuyo.txt        # Tagalog<sep>Cuyonon   (source data)
│   ├── cuyo-taga.txt        # reverse (not needed — we generate both directions)
│   └── entities/            # editable lists KEPT UNCHANGED by the model
│       ├── names.txt        #   people's names
│       ├── places.txt       #   cities / provinces / countries
│       └── events.txt       #   events/holidays with no Cuyonon term
├── scripts/
│   ├── common.py            # shared system prompt (used by prep AND translate)
│   ├── prepare_data.py      # raw .txt  ->  data/train.jsonl + data/val.jsonl
│   ├── train.py             # QLoRA fine-tune  ->  outputs/translator-lora/
│   ├── translate.py         # load model + adapter, translate text (CLI)
│   ├── evaluate.py          # chrF / BLEU on the validation set
│   ├── serve.py             # HTTP server the LearnVoca app calls (see below)
│   └── sync_from_supabase.py# pulls new real-world phrases back for retraining
├── train_all.ps1           # ONE command: prepare -> train -> evaluate
├── requirements.txt
└── README.md
```

---

## Setup — ALREADY DONE on this machine ✅

The environment is installed and verified:
- Python 3.11.9 (per-user) + `.venv`
- PyTorch 2.5.1 **+ CUDA 12.1** (sees the RTX 3050, bf16 supported)
- transformers / trl / peft / datasets / accelerate / bitsandbytes / sacrebleu

**One remaining manual step:** point VS Code at the venv so import errors clear —
`Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv\Scripts\python.exe`.

<details><summary>To reproduce the setup from scratch (e.g. on another PC)</summary>

```powershell
cd c:\projects\translator
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
</details>

---

## Run it

### The easy way — one command does everything
```powershell
.\train_all.ps1                              # prepare -> train -> evaluate
.\train_all.ps1 -Model "Qwen/Qwen2.5-0.5B-Instruct"   # smaller, if you hit OOM
```

### Or step by step
```powershell
$py = ".\.venv\Scripts\python.exe"

# 1) Build the dataset (both directions + entity passthrough, 90/10 split)
& $py scripts/prepare_data.py

# 2) Fine-tune (the long step)
& $py scripts/train.py

# 3) Translate
& $py scripts/translate.py --text "sino ang kasama mo?" --to Cuyonon
& $py scripts/translate.py                   # interactive mode
& $py scripts/translate.py --text "Pumunta si Maria sa Cuyo." --keep "Maria,Cuyo"

# 4) Measure quality on the held-out set
& $py scripts/evaluate.py
```

---

## Names, cities, events (terms with no Cuyonon word)

Proper nouns should be **copied through unchanged**, never invented. Three mechanisms
handle this:

1. **Instruction** — the system prompt (in `scripts/common.py`) tells the model to keep
   names, cities, brands, and event/holiday names unchanged.
2. **Teaching by example** — `prepare_data.py` reads `assets/entities/*.txt` and generates
   "term → itself" training examples (both directions). *Add your own* names/places to
   those files, then re-run training.
3. **Glossary override** — force specific terms verbatim at translation time:
   `translate.py --text "..." --keep "Maria,Cuyo,Palawan"`

⚠️ Only put a holiday/event in `events.txt` if it genuinely has **no** Cuyonon term.
If it *does* have one, add a normal translated line to `assets/taga-cuyo.txt` instead —
otherwise you teach the model the wrong thing. (I can't invent Cuyonon for you; wrong
translations would make the model worse.)

---

## ⚠️ 4 GB VRAM survival guide

Your GPU is small, and other apps were using ~1.2 GB. **Before training, close
browsers, games, and anything GPU-heavy.** Then, if you still get
`CUDA out of memory`, try these in order:

1. Use the smaller base model (fits easily in 4 GB):
   ```powershell
   python scripts/train.py    --model Qwen/Qwen2.5-0.5B-Instruct
   python scripts/translate.py --base-model Qwen/Qwen2.5-0.5B-Instruct
   ```
2. Shorten sequences: `--max-seq-len 128`
3. Smaller adapter: `--lora-r 8`
4. Bigger effective batch without more memory: `--grad-accum 32`

---

## When your dataset gets bigger (your other goal)

More data is the single biggest lever for accuracy. As it grows:

| Dataset size | Recommendation |
|---|---|
| ~3k (now) | 1.5B model on your laptop (or 0.5B if OOM) |
| ~20k–50k | Same code, run on free **Google Colab** (16 GB T4) with `--model Qwen/Qwen2.5-1.5B-Instruct`, larger batch |
| 100k+ | `--model Qwen/Qwen2.5-7B-Instruct` on Colab/A100 — genuinely strong quality |

Keep re-running `evaluate.py` after each training run and watch **chrF** climb —
that's your proof the model is getting more accurate.

Tips to grow data cheaply:
- Add more parallel sentences to `assets/taga-cuyo.txt` (same `<sep>` format).
- Include phrases/sentences, not just single words — right now ~62% of your data
  is single words, which limits sentence-level fluency.
- Keep spelling/orthography consistent across entries.

---

## Notes
- The LoRA adapter saved in `outputs/translator-lora/` is only a few MB. The base
  model is downloaded from Hugging Face on first run and cached.
- Training and inference both use the **same system prompt** (defined in the
  scripts) — don't change one without the other.

---

## Wiring this into the LearnVoca app (Flutter + Supabase)

**The Supabase side already exists and needs no changes.** `learn_voca`'s
`TranslationRepository` already does the right thing — cache → dictionary →
remote service, with cache write-back — via `public.translations`,
`lookup_translation()`, `record_translation()`, and `app_config.translator`
(migration `20260101000400_engagement.sql`). Edge Functions can't run this
model anyway (no GPU, tiny sandbox, second-scale timeouts) — that's not where
it belongs.

**The one missing piece is Tier 3**, the actual HTTP service the repository
calls: `POST {baseUrl}/translate` with `{sentence, source_lang, target_lang}`
(`"tl"`/`"cyo"`), returning `{translated_sentence}`. `scripts/serve.py`
implements exactly that contract:

```powershell
.\.venv\Scripts\python.exe scripts/serve.py           # loads the model once, listens on :8000
```

To test from a real device during development, tunnel it and point
`app_config.translator.base_url` (a row in Postgres, editable from the
Supabase dashboard — no app release needed) at the tunnel URL:
```powershell
cloudflared tunnel --url http://localhost:8000
```

**For a real, always-on `baseUrl`**, this laptop server isn't it — you need
somewhere the process stays up. Roughly cheapest → simplest, for a 1.5B model:
- A small VPS running the model via **Ollama/llama.cpp** (convert to GGUF,
  quantized 4-bit) — CPU inference is fine at this size, a few seconds/request.
- A serverless GPU host (**Modal**, **Replicate**, **HF Inference Endpoints**)
  — pay per call, nothing to run yourself.

That's a hosting decision worth making deliberately (cost, latency, who
maintains it) rather than something to wire up blind — happy to help set up
whichever you pick.

### The retrain loop — "add new data, will it update?"

Yes, and part of it is already running: every time Tier 3 answers a phrase the
cache didn't have, `record_translation()` saves it to `public.translations`.
That table is a live, growing log of real phrases people actually ask for.

```
app usage → new phrase → serve.py answers → record_translation() caches it
                                                      │
                                    scripts/sync_from_supabase.py  (pulls it back)
                                                      ▼
                                       data/review_new_pairs.txt  (YOU eyeball it)
                                                      ▼
                                    merge good lines into assets/taga-cuyo.txt
                                                      ▼
                                     .\train_all.ps1   (retrain from scratch)
                                                      ▼
                                    restart scripts/serve.py  (new adapter loads)
```

```powershell
$env:SUPABASE_URL = "https://xxxx.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "..."   # Project Settings -> API. NEVER put this in the Flutter app.
.\.venv\Scripts\python.exe scripts/sync_from_supabase.py
```

**Why review instead of auto-merging:** a cached row is the *model's own*
output. Feeding a model's possibly-wrong translation back in as new "ground
truth" teaches it to be more confidently wrong (self-training drift) — it
doesn't make the dataset better on its own. A quick human glance at
`data/review_new_pairs.txt` before moving lines into `assets/taga-cuyo.txt` is
what actually makes this loop safe.
