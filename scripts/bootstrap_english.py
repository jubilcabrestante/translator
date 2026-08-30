"""
bootstrap_english.py
--------------------
Generate the English column for the existing Tagalog<->Cuyonon corpus, so the
translator can serve English without anyone hand-translating Cuyonon twice.

The idea: you already have 2.6k human-verified Tagalog<->Cuyonon pairs. Tagalog
to English is a high-resource direction that an LLM does well, so translating
only the TAGALOG side produces Tagalog<->English pairs where the low-resource
side is never machine-generated. prepare_data.py then bridges the two corpora
through their shared Tagalog terms and emits English<->Cuyonon training data on
its own -- the Cuyonon in those pairs is still your original human text.

This means the ENGLISH column is machine-generated and worth spot-checking.
About 60% of the entries are isolated single words, where word sense is
genuinely ambiguous ('bata' -> child? young?), so that is where errors will be.
The file is plain text: fix any line and re-run prepare_data.py.

Setup:
    pip install anthropic
    # credentials: `ant auth login`, or export ANTHROPIC_API_KEY
    # (the SDK finds either on its own -- no key goes in this file)

Usage:
    python scripts/bootstrap_english.py --dry-run      # count + cost estimate
    python scripts/bootstrap_english.py                # run it
    python scripts/bootstrap_english.py --limit 50     # try a small slice first

Safe to interrupt: finished chunks are cached to data/english_cache.jsonl and
skipped on the next run.
"""

import argparse
import json
import os
import sys

from common import load_config

try:
    import anthropic
except ImportError:
    sys.exit(
        "This script needs the Anthropic SDK, which the training requirements "
        "deliberately leave out (Colab does not need it):\n    pip install anthropic"
    )

MODEL = "claude-opus-5"
# Rough per-1M-token rates for the estimate below. Real billing is authoritative.
INPUT_PER_MTOK, OUTPUT_PER_MTOK = 5.00, 25.00

INSTRUCTIONS = """You are translating entries from a Tagalog-Cuyonon dictionary and phrasebook into English.

These are corpus entries, not conversation. Follow these rules exactly:

1. Translate each Tagalog entry into natural English.
2. Many entries are a single word with no context. Give the most common,
   neutral English sense of that word. Do not add explanations, alternatives,
   parentheses, or slashes -- exactly one English rendering per entry.
3. Preserve the entry's surface style. If the Tagalog ends with a period,
   question mark or exclamation mark, end the English the same way. If it has
   no final punctuation, do not add any.
4. Proper nouns -- people's names, cities, provinces, countries, brands,
   organisations, holidays -- are copied through UNCHANGED, never translated.
5. If an entry is already English, or is a number or symbol, return it unchanged.
6. Return one result per input, with the same index. Never merge, skip, or
   reorder entries."""

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "english": {"type": "string"},
                },
                "required": ["index", "english"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def read_tagalog_terms(config):
    """Every distinct Tagalog surface form across the configured corpora."""
    terms = []
    seen = set()
    for corpus in config["corpora"]:
        if "Tagalog" not in corpus["columns"]:
            continue
        column = corpus["columns"].index("Tagalog")
        if not os.path.exists(corpus["path"]):
            continue
        with open(corpus["path"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "<sep>" not in line:
                    continue
                parts = line.split("<sep>")
                if len(parts) != 2:
                    continue
                term = parts[column].strip()
                if term and term.lower() not in seen:
                    seen.add(term.lower())
                    terms.append(term)
    return terms


def load_cache(path):
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done[row["tagalog"]] = row["english"]
                except (json.JSONDecodeError, KeyError):
                    continue          # a half-written line from an interrupt
    return done


def translate_chunk(client, chunk, effort):
    """Translate one chunk, returning a list of English strings the same length."""
    numbered = [{"index": i, "tagalog": t} for i, t in enumerate(chunk)]
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=INSTRUCTIONS,
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": "Translate these entries into English:\n\n"
                       + json.dumps(numbered, ensure_ascii=False, indent=1),
        }],
    )
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", "")
        raise RuntimeError(f"Request refused: {detail}")

    text = next(b.text for b in response.content if b.type == "text")
    by_index = {row["index"]: row["english"] for row in json.loads(text)["translations"]}
    missing = [i for i in range(len(chunk)) if i not in by_index]
    if missing:
        raise ValueError(f"model skipped {len(missing)} of {len(chunk)} entries")
    return [by_index[i].strip() for i in range(len(chunk))], response.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="assets/taga-eng.txt",
                    help="Written as <Tagalog><sep><English>, the format "
                         "languages.json already points at.")
    ap.add_argument("--cache", default="data/english_cache.jsonl")
    ap.add_argument("--chunk-size", type=int, default=40,
                    help="Entries per API request. Smaller is more robust, larger "
                         "is cheaper (the instructions are sent once per request).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only translate the first N new entries -- use this to "
                         "sanity-check quality before paying for the whole corpus.")
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="Thinking depth. Short dictionary entries do not need much; "
                         "raise it if you see word-sense mistakes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report how many entries and a rough cost, then exit.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = ap.parse_args()

    config = load_config(args.config)
    terms = read_tagalog_terms(config)
    cache = load_cache(args.cache)
    todo = [t for t in terms if t not in cache]
    if args.limit:
        todo = todo[:args.limit]

    chunks = [todo[i:i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    # Rough: ~4 chars/token, plus the instructions re-sent per request.
    in_tok = sum(len(t) for t in todo) / 4 + len(chunks) * len(INSTRUCTIONS) / 4
    out_tok = sum(len(t) for t in todo) / 4 * 1.3
    cost = in_tok / 1e6 * INPUT_PER_MTOK + out_tok / 1e6 * OUTPUT_PER_MTOK

    print(f"Tagalog entries found : {len(terms)}")
    print(f"Already translated    : {len(cache)}")
    print(f"To translate now      : {len(todo)}  in {len(chunks)} requests")
    print(f"Model                 : {MODEL} (effort={args.effort})")
    print(f"Rough cost estimate   : ${cost:.2f}  (thinking tokens not counted; "
          f"treat as a floor)")

    if args.dry_run:
        return
    if not todo:
        print("\nNothing new to translate.")
    elif not args.yes:
        if input("\nProceed and spend this? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    # The SDK resolves credentials lazily, so a missing key does not surface
    # until the first request, and then as a bare TypeError rather than
    # AuthenticationError. Check up front so the failure is one readable line
    # instead of a traceback.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.path.isdir(os.path.expanduser("~/.config/anthropic"))):
        sys.exit(
            "No Anthropic credentials found, so nothing was sent and nothing was "
            "billed.\n\n"
            "This script is the only part of this project that costs money: it "
            "calls the\npaid Anthropic API, which bills a console account per "
            "token. That is separate\nfrom a Claude.ai or Claude Code "
            "subscription -- those do not cover it.\n\n"
            "If you do want to run it, set a key from "
            "https://console.anthropic.com/settings/keys:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "and re-run. Finished chunks are cached, so it is safe to stop and "
            "resume.\n\n"
            "If you would rather not spend anything, skip this script entirely -- "
            "see the\n'Adding a language' section of README.md for the free routes."
        )

    client = anthropic.Anthropic()

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    total_in = total_out = 0
    with open(args.cache, "a", encoding="utf-8") as cache_file:
        for n, chunk in enumerate(chunks, 1):
            try:
                english, usage = translate_chunk(client, chunk, args.effort)
            except anthropic.AuthenticationError:
                sys.exit("Authentication failed. Run `ant auth login`, or export "
                         "ANTHROPIC_API_KEY, then re-run -- finished chunks are cached.")
            except anthropic.PermissionDeniedError:
                sys.exit("This credential lacks permission for the Messages API.")
            except anthropic.RateLimitError:
                sys.exit("Rate limited even after the SDK's retries. Re-run later; "
                         "finished chunks are cached.")
            except (anthropic.APIStatusError, anthropic.APIConnectionError,
                    ValueError, RuntimeError, TypeError, json.JSONDecodeError) as e:
                print(f"  chunk {n}/{len(chunks)} FAILED ({type(e).__name__}: {e}) "
                      f"-- skipping, re-run to retry it")
                continue

            for tagalog, translated in zip(chunk, english):
                cache_file.write(json.dumps(
                    {"tagalog": tagalog, "english": translated},
                    ensure_ascii=False) + "\n")
                cache[tagalog] = translated
            cache_file.flush()
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            print(f"  chunk {n}/{len(chunks)} ok  ({len(chunk)} entries)")

    if total_in or total_out:
        billed = total_in / 1e6 * INPUT_PER_MTOK + total_out / 1e6 * OUTPUT_PER_MTOK
        print(f"\nTokens: {total_in} in / {total_out} out  (~${billed:.2f})")

    # ---- Write the parallel file from everything cached so far --------------
    written = 0
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tagalog<sep>English -- MACHINE-GENERATED by "
                "scripts/bootstrap_english.py.\n")
        f.write("# The Tagalog side is your original data; the English side is not "
                "human-verified.\n")
        f.write("# Correct any line in place, then re-run scripts/prepare_data.py.\n")
        for tagalog in terms:
            if tagalog in cache:
                f.write(f"{tagalog}<sep>{cache[tagalog]}\n")
                written += 1
    print(f"Wrote {written} pairs -> {args.out}")
    print("\nNext: python scripts/prepare_data.py   (English pairs join automatically)")


if __name__ == "__main__":
    main()
