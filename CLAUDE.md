# Translator: rules for coding agents

A LoRA fine-tune of Qwen2.5-1.5B-Instruct that translates between Tagalog,
Cuyonon and English. Setup, training, export and hosting are in `README.md`.

## Connected project: Learnut

This translator exists to serve the **Learnut** app. Treat the two repos as one
system: a change on one side can break the other.

- **Repo:** `jubilcabrestante/learnut` (Kotlin Multiplatform + Compose, Android
  first), checked out at `C:\projects\mobile app\learnut`. Its own rules are in
  its `AGENTS.md`.
- **Read the app side before changing** the HTTP contract, the language names in
  `languages.json`, or anything to do with Supabase:
  - the client is `TranslationRepository.translate` in
    `shared/src/commonMain/kotlin/com/app/tagacuyo/core/data/OtherRepositories.kt`
  - the Supabase schema, RLS and seed rows are in its `supabase/migrations/`.
    The Learnut repo owns the schema; this repo never migrates it.

### The contract `scripts/serve.py` must keep

- `POST {url}/translate` with
  `{"sentence": "...", "source_lang": "Tagalog", "target_lang": "Cuyonon", "translated": null}`.
  The answer is `{"translated_sentence": "..."}`.
- Languages arrive as display **names**. Renaming a language in `languages.json`
  breaks the app until the app changes too.
- The app sends **no API key** and gives up after **30 seconds**. Don't set
  `TRANSLATOR_API_KEY` until the app sends one.
- The app adds a `.` when the input has no end punctuation, and caps input at
  `translation_string_length` (30 characters).
- The app logs every answer in `public.translations`.
  `scripts/sync_from_supabase.py` reads that log back for retraining.

### Where the app finds this server

- Supabase table `app_config`, row `app_translator`, field
  `get_api_translator`. The app re-reads it on every launch and resume.
  `api_translator_enabled: false` makes the app show "service unavailable".
- Learnut has **one production Supabase project**, with no dev or staging copy,
  and its builds go to testers on Google Play. Changing that row switches every
  user at once, so **ask the owner first**.
- `start-translator.cmd` runs the translator on the owner's laptop. It changes
  the row only after the owner says yes, and switches it back on Ctrl+C.

### Changes that span both repos

A contract change needs a matching Learnut change. Every push to Learnut's
`main` ships to testers and follows its release rules (version bump, "What's
new" line), so don't push there unless the owner asks.

## This machine

- The laptop has no NVIDIA GPU. Training (`scripts/train.py`) runs on Kaggle or
  Colab (`notebooks/train_colab.ipynb`), which clone from GitHub, so push
  local changes before a training run.
- Serving runs on the laptop's CPU: Ollama serves the q8_0 GGUF, and `serve.py`
  runs in backend mode in front of it (`start-translator.cmd`).
