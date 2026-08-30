"""
mine_english.py
---------------
Fill in the English column for the existing Tagalog entries using FREE, openly
licensed data -- no API key, no spending, fully offline once downloaded.

This is the free alternative to bootstrap_english.py. It does not translate
anything; it looks up each of your Tagalog entries in reference data and copies
across an existing human translation. Anything it cannot find is simply left
out, and `prepare_data.py` carries on with whatever is present.

Two complementary sources, because your entries split in two:

  * OPUS parallel corpora (sentence pairs) -- good for phrases, poor for single
    words, because a lone word is rarely a whole sentence. Download the `moses`
    zip for the corpora you want from https://opus.nlpl.eu (pick the en-tl pair)
    and unzip each into its own folder under --opus-dir.

  * Wiktionary, via the machine-readable extracts at https://kaikki.org --
    this is a dictionary, so it is the one that actually covers single words.
    Grab the Tagalog .jsonl and pass it with --wiktionary.

Both are openly licensed (Wiktionary is CC-BY-SA; OPUS licensing varies per
corpus -- check the one you use before redistributing anything derived from it).
The output file records which source each line came from.

Usage:
    python scripts/mine_english.py --opus-dir path/to/opus --wiktionary tl.jsonl
    python scripts/mine_english.py --wiktionary tl.jsonl          # dictionary only
    python scripts/mine_english.py --opus-dir path/to/opus --report-only

Then:
    python scripts/prepare_data.py    # English joins automatically
"""

import argparse
import collections
import glob
import json
import os
import re

from common import load_config

# Wiktionary glosses are definitions, not translation equivalents. Anything long
# or parenthetical reads badly as a target sentence, so keep only short ones.
MAX_GLOSS_WORDS = 6


def normalize(text):
    """Loose key for matching: case, spacing and edge punctuation are ignored."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(' .!?,;:"“”‘’')


def read_target_terms(config, language="Tagalog"):
    """Every distinct surface form of `language` across the configured corpora."""
    terms, seen = [], set()
    for corpus in config["corpora"]:
        if language not in corpus["columns"] or not os.path.exists(corpus["path"]):
            continue
        column = corpus["columns"].index(language)
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


def mine_opus(opus_dir, wanted):
    """Scan extracted OPUS moses folders for sentences matching `wanted`.

    Expects one folder per corpus, each holding a matching pair of *.tl and
    *.en files with one aligned sentence per line -- exactly what the moses
    zips unpack to.
    """
    found = collections.defaultdict(list)
    if not opus_dir or not os.path.isdir(opus_dir):
        return found, collections.Counter()
    per_corpus = collections.Counter()
    for name in sorted(os.listdir(opus_dir)):
        folder = os.path.join(opus_dir, name)
        if not os.path.isdir(folder):
            continue
        tl_files = glob.glob(os.path.join(folder, "*.tl"))
        en_files = glob.glob(os.path.join(folder, "*.en"))
        if not tl_files or not en_files:
            continue
        with open(tl_files[0], encoding="utf-8", errors="replace") as ft, \
                open(en_files[0], encoding="utf-8", errors="replace") as fe:
            for tl_line, en_line in zip(ft, fe):
                key = normalize(tl_line)
                english = en_line.strip()
                if key and english and key in wanted:
                    found[key].append((english, name))
                    per_corpus[name] += 1
    return found, per_corpus


def mine_wiktionary(path, wanted):
    """Pull short English glosses for `wanted` out of a kaikki.org .jsonl extract.

    Each line is one dictionary entry; `senses[].glosses[]` holds the English
    definitions. Only the first few short glosses per word are kept -- a long
    definition is a description, not something the model should learn to emit.
    """
    found = collections.defaultdict(list)
    if not path or not os.path.exists(path):
        return found
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize(entry.get("word", ""))
            if not key or key not in wanted:
                continue
            for sense in entry.get("senses", []):
                for gloss in sense.get("glosses", []) or []:
                    gloss = gloss.strip()
                    # Drop parenthetical/usage notes and over-long definitions.
                    if not gloss or gloss.startswith("("):
                        continue
                    gloss = re.sub(r"\s*\([^)]*\)", "", gloss).strip()
                    if gloss and len(gloss.split()) <= MAX_GLOSS_WORDS:
                        if gloss not in [g for g, _ in found[key]]:
                            found[key].append((gloss, "wiktionary"))
                    if len(found[key]) >= 3:
                        break
                if len(found[key]) >= 3:
                    break
    return found


def match_style(source, english):
    """Echo the entry's final punctuation onto the gloss.

    The corpus is written as 'Aso.' / 'Magkano?' and the model is trained on
    exact strings, so a gloss that silently drops the period teaches it to drop
    punctuation in one language but not the other.
    """
    english = english.strip()
    if not english:
        return english
    # REPLACE the gloss's punctuation rather than only filling a gap. A corpus
    # sentence carries the punctuation of the context it was mined from, which
    # is not this entry's: 'Ako.' matched a line reading 'Me?', and training on
    # that teaches the model to answer a statement with a question.
    body = english.rstrip(".!?,;: ")
    tail = source.strip()[-1:] if source.strip() else ""
    return body + tail if tail in ".!?" else body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--opus-dir", default=None,
                    help="Folder holding unzipped OPUS moses corpora, one subfolder each.")
    ap.add_argument("--wiktionary", default=None,
                    help="kaikki.org .jsonl extract for the source language.")
    ap.add_argument("--out", default="assets/taga-eng.txt")
    ap.add_argument("--source-lang", default="Tagalog")
    ap.add_argument("--max-per-term", type=int, default=2,
                    help="Cap on how many English variants to record per entry. "
                         "prepare_data.py treats them as alternative references.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print coverage and write nothing.")
    args = ap.parse_args()

    if not args.opus_dir and not args.wiktionary:
        raise SystemExit("Give at least one of --opus-dir or --wiktionary. "
                         "See the docstring for where to download them.")

    config = load_config(args.config)
    terms = read_target_terms(config, args.source_lang)
    wanted = {normalize(t) for t in terms}
    print(f"{args.source_lang} entries to cover: {len(terms)}")

    opus_hits, per_corpus = mine_opus(args.opus_dir, wanted)
    if args.opus_dir:
        print(f"  OPUS matches   : {len(opus_hits)} entries "
              f"({sum(per_corpus.values())} sentence hits)")
        for name, count in per_corpus.most_common():
            print(f"      {name:<16}{count:6d}")

    wik_hits = mine_wiktionary(args.wiktionary, wanted)
    if args.wiktionary:
        print(f"  Wiktionary     : {len(wik_hits)} entries")

    # Dictionary glosses win for single words; corpus sentences win for phrases.
    results = {}
    for term in terms:
        key = normalize(term)
        single = len(term.split()) == 1
        primary = wik_hits.get(key, []) if single else opus_hits.get(key, [])
        backup = opus_hits.get(key, []) if single else wik_hits.get(key, [])
        merged, seen = [], set()
        for english, source in list(primary) + list(backup):
            styled = match_style(term, english)
            if styled and styled.lower() not in seen:
                seen.add(styled.lower())
                merged.append((styled, source))
            if len(merged) >= args.max_per_term:
                break
        if merged:
            results[term] = merged

    singles = [t for t in terms if len(t.split()) == 1]
    multis = [t for t in terms if len(t.split()) > 1]
    hit_single = sum(1 for t in singles if t in results)
    hit_multi = sum(1 for t in multis if t in results)
    print(f"\nCoverage: {len(results)}/{len(terms)} "
          f"({100 * len(results) / max(1, len(terms)):.1f}%)")
    print(f"  single-word : {hit_single}/{len(singles)} "
          f"({100 * hit_single / max(1, len(singles)):.1f}%)")
    print(f"  multi-word  : {hit_multi}/{len(multis)} "
          f"({100 * hit_multi / max(1, len(multis)):.1f}%)")

    if args.report_only:
        print("\n--report-only: nothing written.")
        return

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {args.source_lang}<sep>English -- assembled by "
                f"scripts/mine_english.py from open data.\n")
        f.write("# Sources: Wiktionary via kaikki.org (CC-BY-SA); OPUS parallel "
                "corpora (per-corpus licences).\n")
        f.write("# Not hand-checked. Single-word senses are the likeliest errors "
                "-- fix any line and\n# re-run scripts/prepare_data.py.\n")
        for term in terms:
            for english, source in results.get(term, []):
                f.write(f"{term}<sep>{english}\n")
                written += 1
    print(f"\nWrote {written} pairs -> {args.out}")
    print("Next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
