# Tagalog ⇄ Cuyonon Translator (LLM fine-tune)

A **decoder-only LLM** (same architecture family as Claude/GPT) fine-tuned into a
specialized translation engine using **QLoRA**, sized to train on a **4 GB laptop
GPU** (RTX 3050) or a free Colab T4.

One model handles **every direction** between the languages in `languages.json`
(today Tagalog, Cuyonon and English: 6 directions).

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (open, ungated, multilingual, instruction-tuned).

---

## What limits quality today (read this first)

The code is not the bottleneck. **The Cuyonon data is.**

- There are about **1,030 real Tagalog–Cuyonon sentence pairs**. The other
  ~1,500 rows are single words or rows where both sides are identical. A model
  cannot learn to write a language from that little. 5,000–10,000 sentence
  pairs is the realistic target.
- A **native Cuyonon speaker** is the one resource nothing else replaces: to
  translate new sentences, settle the review sheets, fill the test set, and
  judge the model's output.

See [Improving the data](#improving-the-data) for where more sentences can come
from, and [Measuring progress](#measuring-progress) for how to tell whether a
change actually helped.

---

## Why an LLM instead of the old GRU seq2seq?

| | Old (GRU + attention, from scratch) | This (QLoRA fine-tune of an LLM) |
|---|---|---|
| Starting knowledge | none — learns language from your 2.7k pairs | already "knows" many languages; only learns *your mapping* |
| Architecture | RNN | Transformer (the modern standard) |
| Scales with more data | poorly (overfits) | yes — more data + bigger base model = better |
| Style | fixed input→output | instruction-following ("translate X to Y") |

The architecture does **not** change as your dataset grows. To get better results
you (a) add more parallel data, and/or (b) swap `--model` for a newer or bigger
checkpoint. Same code.

---

## Project layout

```
translator/
├── assets/
│   ├── taga-cuyo.txt          # Tagalog<sep>Cuyonon  -- THE core data (one file, both directions)
│   ├── taga-eng.txt           # Tagalog<sep>English  (mined; see "Getting the English column")
│   ├── taga-eng-manual.txt    # hand/reviewed English that always wins over mined lines
│   ├── test-taga-cuyo.txt     # FROZEN test set -- fill once, never train on it
│   └── entities/              # names, places, events KEPT UNCHANGED by the model
├── scripts/
│   ├── common.py              # shared prompt + file reader (used by every script)
│   ├── lint_data.py           # checks the data; --fix applies safe fixes; --review for a speaker
│   ├── prepare_data.py        # corpora -> data/train.jsonl, val.jsonl, test.jsonl
│   ├── train.py               # QLoRA fine-tune -> outputs/translator-lora/ (needs an NVIDIA GPU)
│   ├── evaluate.py            # chrF/BLEU with confidence intervals + speaker review sheet
│   ├── translate.py           # translate text (GPU, or CPU when there is none)
│   ├── serve.py               # HTTP server the LearnVoca app calls
│   ├── export_model.py        # merge the adapter -> standalone model -> GGUF for CPU hosting
│   ├── back_translate.py      # monolingual Cuyonon -> synthetic training pairs
│   ├── sync_from_supabase.py  # pulls real app phrases back for review
│   ├── mine_english.py        # free English column from Wiktionary + OPUS
│   ├── bootstrap_english.py   # paid English column via the Anthropic API (optional)
│   └── import_csv_corpus.py   # Hugging Face style CSV -> Moses folder
├── notebooks/train_colab.ipynb
├── train_all.ps1              # ONE command: check -> prepare -> train -> evaluate
├── languages.json             # languages, corpora, limits, test set
└── requirements.txt
```

---

## Setup

**Hardware.** Training uses 4-bit QLoRA and needs an **NVIDIA GPU** (CUDA).
On a machine without one (e.g. Intel Iris graphics), train on **Google Colab**
with `notebooks/train_colab.ipynb` (free T4). Translating, evaluating and
serving also run on a plain CPU — slower, about 6 GB of RAM for the 1.5B model —
and fast on a CPU once exported to GGUF (see `export_model.py`).

**Software.** Python 3.11, then:

```powershell
cd c:\projects\translator\translator
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA machines
# pip install torch                                                    # CPU-only machines
pip install -r requirements.txt
```

Then point VS Code at the venv: `Ctrl+Shift+P` → *Python: Select Interpreter* →
`.venv\Scripts\python.exe`.

The data scripts (`lint_data.py`, `prepare_data.py`, `mine_english.py`,
`import_csv_corpus.py`) use only the standard library and run anywhere.

**Colab** clones this repository from GitHub, so commit and push your changes
before running the notebook.

---

## Run it

### The easy way — one command does everything
```powershell
.\train_all.ps1                              # check -> prepare -> train -> evaluate
.\train_all.ps1 -Model "Qwen/Qwen2.5-0.5B-Instruct"   # smaller, if you hit OOM
```

### Or step by step
```powershell
$py = ".\.venv\Scripts\python.exe"

& $py scripts/lint_data.py                   # 0) check the data (add --fix for safe fixes)
& $py scripts/prepare_data.py                # 1) build train / val / test
& $py scripts/train.py                       # 2) fine-tune (the long step; GPU)
& $py scripts/evaluate.py                    # 3) scores + a review sheet for a speaker

& $py scripts/translate.py --text "sino ang kasama mo?" --to Cuyonon
& $py scripts/translate.py                   # interactive mode
& $py scripts/translate.py --text "Pumunta si Maria sa Cuyo." --keep "Maria,Cuyo"
```

---

## Improving the data

### 1. Keep what you have clean

```powershell
& $py scripts/lint_data.py                                   # report
& $py scripts/lint_data.py --fix                             # spacing, doubled punctuation, curly quotes,
                                                             #   missing final mark, first-letter capital
& $py scripts/lint_data.py --review data/review/speaker-review.csv
```

`--fix` only does what a machine can decide. Anything that needs judgment —
a question translated as a statement, a capital in mid-sentence, a name spelled
differently, an English word inside Cuyonon, two spellings of one word — goes to
the review sheet (opens in Excel / Google Sheets) for a native speaker.

**The November 2026 cleanup** (already applied; `git log` has the diff):
`cuyo-taga.txt` was 97.5% a copy of `taga-cuyo.txt` and was merged into it;
~200 rows lost autocorrect damage ("pila" → *Libya*, `Ren`, `Ora's`), doubled
punctuation (`Ano Pa?.`), missing question marks and stray capitals; four rows
holding two answers were split; restored names (`Tomas`, `Tom`, `Mary`) and
one meaning fix (`sinigang` vs `tinola`). Sentence pairs flagged for formatting
problems went from 26% to 2%. The leftovers that need a speaker are in
`data/review/speaker-review.csv` and `data/review/spelling-questions.csv`
(regenerate with the lint command above).

**Spelling is the speaker's call.** `den`/`ren` is *not* an error — in the data
`ren` only follows vowels and `den` only follows consonants, like Tagalog
*rin*/*din*. But `ren`/`ron`, `dige`/`digi`, `indi`/`inde`, `indong`/`indung`,
`sadin`/`adin` look like one word spelled several ways. Standardize the ones that
are only spelling; with this little data, every split spelling costs accuracy.

### 2. Get more Cuyonon sentences (biggest gain)

- **Translate what users actually ask.** `sync_from_supabase.py` lists the most
  requested phrases. Have the speaker translate the top ones from scratch.
- **Add a "suggest a better translation" button** in LearnVoca. Corrections from
  speakers are verified data that keeps growing (an app change, in the
  `learn_voca` repo).
- **The New Testament.** A Cuyonon New Testament exists (1982; the Philippine
  Bible Society launched a revised one, *Matinlong Barita Biblia: Bagong Tipan*,
  in March 2023). It is not freely online, so ask the Bible Society for the text
  and permission. Aligned verse by verse with a Tagalog New Testament it is
  ~7,900 pairs. It is formal language: give it a `limit` so it stays under about
  a third of training.
- **Back-translation.** Collect text written only in Cuyonon (Facebook pages,
  local news, church bulletins, songs), then
  `python scripts/back_translate.py --input monolingual/cuyonon.txt`.
  It writes *machine Tagalog → human Cuyonon* pairs; list the file in
  `languages.json` with `"synthetic": true`. The model only ever learns to
  *write* the human side.
- **False friends.** Words that look the same in Filipino and Cuyonon but mean
  different things (*baba*: mouth vs chin) are where the model fails. SIL's
  *Contrastive Analysis of True and False Cognates in Filipino and Cuyonon* lists
  them; add sentences using each one.

---

## Measuring progress

- **Frozen test set.** Fill `assets/test-taga-cuyo.txt` once with 300–500
  sentences a speaker translated (ideally real app phrases), and never train on
  it. `prepare_data.py` keeps anything overlapping it out of training, and
  `evaluate.py` scores on it automatically once it has rows.
- **Stable validation split.** Until then, evaluation uses the validation split,
  which is now assigned by a hash of each concept cluster instead of a shuffle.
  Adding data no longer reshuffles it (a shuffle moved ~80% of it when one
  sentence was added), so runs before and after a data change are comparable.
- **Confidence intervals.** Each score row shows a 95% interval. With ~100
  sentences per direction, a 1–2 point chrF change is often noise.
- **A speaker's verdict.** Every evaluation writes
  `outputs/eval/<run>/review-sheet.csv`: 50 sentences to mark *correct /
  understandable / wrong*. Tally it with
  `python scripts/evaluate.py --tally outputs/eval/<run>/review-sheet.csv`.
  That number is what users feel.

Track the **`multi-word (sentences)`** row: single words are mostly answered by
the app's dictionary tier, and copy rows score near 100 for free.

---

## Names, cities, events (terms with no Cuyonon word)

Proper nouns should be **copied through unchanged**, never invented:

1. **Instruction** — the system prompt (in `scripts/common.py`) says so.
2. **Teaching by example** — `prepare_data.py` turns `assets/entities/*.txt`
   into "term → itself" examples, and also into **names in context**: every
   training sentence that keeps a place unchanged on both sides is reused with
   other places from `places.txt` swapped in, so the model practises keeping
   names *inside sentences*. All of this goes to training only.
3. **Glossary override** — `translate.py --keep "Maria,Cuyo"` adds a
   "Keep these terms unchanged" line. Training examples now carry the same line,
   so the model has seen the format; translate.py warns if a kept term still
   didn't come through verbatim.

⚠️ Only put a holiday/event in `events.txt` if it genuinely has **no** Cuyonon
term. If it does, add a translated line to `assets/taga-cuyo.txt` instead.

---

## Training

- **Loss on the answer only** (default). Every example repeats the same ~80-token
  system prompt; training on it wasted most of the gradient and made `eval_loss`
  measure prompt memory. `--full-sequence-loss` restores the old behaviour for an
  A/B comparison.
- **Best epoch, not last.** The checkpoint with the lowest validation loss is the
  one saved.
- **No NVIDIA GPU?** `train.py` says so and points to the Colab notebook.

### 4 GB VRAM survival guide

Close browsers, games, and anything GPU-heavy first. If you still get
`CUDA out of memory`, try in order:

1. The smaller base model: `--model Qwen/Qwen2.5-0.5B-Instruct`
   (and `--base-model` with the same value for translate/evaluate/serve)
2. Shorter sequences: `--max-seq-len 128`
3. Smaller adapter: `--lora-r 8`
4. Bigger effective batch without more memory: `--grad-accum 32`

### Trying a newer base model

Qwen2.5 is from 2024. Candidates worth comparing on the same test set: Qwen3
(1.7B/4B), Gemma 3 (1B/4B), Cohere's Tiny Aya (check its licence), or
NLLB-200-distilled-600M (a dedicated translation model, but it needs different
training code). Cuyonon is a Visayan language, so models with good Cebuano /
Hiligaynon coverage should transfer best. `train.py` reads the answer marker
from each model's own chat template, but the pinned `transformers==4.46.3` /
`trl==0.12.2` are too old for Qwen3 and Gemma 3: upgrading them comes first
(newer TRL renames some SFTConfig options and replaces
`DataCollatorForCompletionOnlyLM`). Try `--num-beams 4` in evaluate.py too.

---

## Adding a language

The language set lives in `languages.json`; no script hardcodes Tagalog or
Cuyonon. To add one:

1. Add it to `languages` (`code` is what your app sends over HTTP, `name` is what
   appears in the prompt the model is trained on).
2. Add a `<sep>`-delimited file to `corpora`, naming its two columns.
3. Re-run `prepare_data.py` and retrain.

**You do not need every pair.** A pair never stated directly is still emitted
when both sides share a specific common translation in a third language, so
`Tagalog<->English` data alone yields `English<->Cuyonon` examples in which the
Cuyonon is still your original human text. `prepare_data.py` counts `direct` and
`bridged` examples separately; pass `--no-bridged` to emit only attested pairs.
Bridging needs the shared sentence spelled identically in both files —
`lint_data.py` flags pivots that differ.

### Getting the English column

`assets/taga-eng.txt` (`<Tagalog><sep><English>`). Only the *Tagalog* side is
ever translated — the Cuyonon is never machine-touched. Routes, cheapest first:

| Route | Cost | Notes |
|---|---|---|
| **Ask Claude Code** | free on your plan | Have your Claude Code session translate the entries and write the file. Best quality; slowest per batch. |
| **`scripts/mine_english.py`** | free | Copies existing human translations from Wiktionary (single words) and OPUS (sentences). Output is grouped by source. |
| **Local MT model** | free | `Helsinki-NLP/opus-mt-tl-en` or NLLB-200, offline. Weaker on idioms. |
| **Hand-write** | free | Highest quality. Start with the sentences. |
| **`scripts/bootstrap_english.py`** | **spends money** | Calls the paid Anthropic API, billed per token to a console account — **separate from a Claude.ai / Claude Code subscription**. Needs `ANTHROPIC_API_KEY`; refuses to run without one. `--dry-run` estimates the cost first. |

Put corrections in `assets/taga-eng-manual.txt`, which always wins when
`mine_english.py` regenerates the file. In November 2026 all 320 mined English
*sentences* were reviewed: ~50 were subtitle noise or wrong ("Sa gabi." →
*Zumba at night.*, "Tumigil ka!" → *Shut up!*), 4 changed a name or place, ~30
needed polish; the corrections are in a marked section of the manual file.

⚠️ Your Supabase `translations` table has a CHECK constraint allowing only `tl`
and `cyo`. Widen it before serving a third code, or `record_translation()` will
reject every row it caches. `GET /health` lists the codes the server accepts.

---

## How the data is split and balanced

**Split.** Your own corpora are grouped into concept clusters — every spelling,
synonym and direction of a term — and each cluster goes to train, validation or
test as a whole:
- punctuation/case variants are welded together, so `Paalam po` and
  `Paalam po!` cannot land on opposite sides;
- the side is chosen by a hash of the cluster, so it stays put as data grows;
- clusters touching the test file are kept out of training and validation.

**Reference corpora are training-only.** Downloaded OPUS/HF folders never enter
validation or test, and any of their pairs that touch a held-out sentence is
dropped. Each is sampled to its `limit` in `languages.json` (a stable sample).
Uncapped, the full download made Cuyonon **0.57%** of the training data and one
epoch took ~10 days on a T4; `prepare_data.py` now prints each language's share
and warns when your own corpora fall below 30%.

**Rebalancing (training only):**

| | effect |
|---|---|
| copy rows (input == output) | capped at 10% (`--max-copy-frac`); ~25% of raw rows are copies |
| sentences from **your** corpora | repeated ×2 (`--multi-word-repeat`); reference rows never repeated |

Rows are written exactly as they appear in the files, so a capital that changes
the meaning survives: `Sa susunod na Linggo.` (next Sunday) and
`Sa susunod na linggo.` (next week) are two different examples.

---

## Notes
- The LoRA adapter saved in `outputs/translator-lora/` is only a few MB. The base
  model is downloaded from Hugging Face on first run and cached.
- Training and inference share the **same prompt** (`scripts/common.py`) — don't
  change one without the other.
- Generation turns off the base model's built-in repetition penalty: it pushes
  the model away from tokens already in the prompt, which is exactly what copying
  a name, or a word Tagalog and Cuyonon share, requires.

---

## Wiring this into the LearnVoca app (Flutter + Supabase)

**The Supabase side already exists.** `learn_voca`'s `TranslationRepository`
does cache → dictionary → remote service, with cache write-back, via
`public.translations`, `lookup_translation()`, `record_translation()`, and
`app_config.translator` (migration `20260101000400_engagement.sql`).

**Tier 3 is `scripts/serve.py`**: `POST {baseUrl}/translate` with
`{sentence, source_lang, target_lang}` (`"tl"`/`"cyo"`), returning
`{translated_sentence}`.

```powershell
.\.venv\Scripts\python.exe scripts/serve.py           # loads the model once, listens on :8000
```

Before exposing it (e.g. `cloudflared tunnel --url http://localhost:8000` for a
real device), set **`TRANSLATOR_API_KEY`**: the server then requires the header
`X-API-Key: <key>`, and the app must send it. Without it, anyone with the tunnel
URL can use your GPU. `MAX_INPUT_CHARS` (default 500) caps input length.

**For an always-on `baseUrl`**, the cheapest route is a small CPU server:

1. `python scripts/export_model.py` — merges the adapter into a standalone model
   and writes an Ollama `Modelfile`.
2. Convert to GGUF with llama.cpp (the script prints the exact commands; `q8_0`
   needs no compiling) and `ollama create translator -f Modelfile`.
3. Run `serve.py` in **backend mode** in front of it:
   `TRANSLATOR_BACKEND_URL=http://localhost:11434/v1 TRANSLATOR_BACKEND_MODEL=translator`.
   Same HTTP contract, no torch on the server.

A serverless GPU host (Modal, Replicate, HF Inference Endpoints) is the
pay-per-call alternative.

### The retrain loop — "add new data, will it update?"

Yes. Every time Tier 3 answers a phrase the cache didn't have,
`record_translation()` saves it to `public.translations`.

```
app usage → new phrase → serve.py answers → record_translation() caches it
                                                      │
                                    scripts/sync_from_supabase.py  (pulls it back)
                                                      ▼
                                       data/review_new_pairs.txt  (a speaker checks it)
                                                      ▼
                         merge good lines into the file named in each section header
                                                      ▼
                         lint_data.py → .\train_all.ps1 → restart serve.py
```

```powershell
$env:SUPABASE_URL = "https://xxxx.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "..."   # Project Settings -> API. NEVER put this in the Flutter app.
.\.venv\Scripts\python.exe scripts/sync_from_supabase.py
```

Rows are grouped by the corpus file they belong to (any language pair in
`languages.json`), and anything already in your corpora or the test set is
skipped.

**Why review instead of auto-merging:** a cached row is the *model's own*
output. Feeding it back as "ground truth" teaches the model to be more
confidently wrong (self-training drift). Better still: use the list to see which
phrases people ask most, and have a speaker translate those from scratch.

---

Sources for the data leads above: [Joshua Project – Cuyonon](https://joshuaproject.net/languages/cyo),
[Manila Times, March 2023](https://manilatimes.net/2023/03/11/news/regions/bible-to-be-released-in-palawans-cuyonon/1882225/amp),
[SIL – True and False Cognates in Filipino and Cuyonon](https://sil-philippines-languages.org/ical/papers/dangan-Contrastive%20Analysis%20of%20True%20and%20False%20Cognates%20in%20Filipino%20and%20Cuyonon.pdf),
[Tiny Aya (2026)](https://arxiv.org/html/2603.11510).
