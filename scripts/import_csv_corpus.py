"""
import_csv_corpus.py
--------------------
Convert a two-column CSV of parallel text (the shape most Hugging Face
translation datasets ship in) into the OPUS Moses layout that
`prepare_data.py` already reads: one plain-text file per language, one aligned
sentence per line.

By default EVERY row is kept. Filtering is available but opt-in:

  --drop-untranslated  rows whose two columns are identical
  --drop-duplicates    repeated pairs
  --max-words N        rows longer than N words on either side

Those exist because train.py truncates at --max-seq-len (256 tokens), so rows
longer than that are cut mid-sentence during training. If you keep long rows,
raise --max-seq-len to match rather than letting them be silently cut.

Usage:
    python scripts/import_csv_corpus.py \\
        --csv path/to/train_data.csv path/to/test_data.csv \\
        --columns tagalog english \\
        --languages Tagalog English \\
        --out corpora/hf-tagalog-english

Then add it to languages.json:
    { "dir": "corpora/hf-tagalog-english",
      "columns": ["Tagalog", "English"], "limit": 8000 }
"""

import argparse
import collections
import csv
import os
import sys

from common import load_config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more CSV files. Splits are merged; "
                         "prepare_data.py makes its own leak-free split.")
    ap.add_argument("--columns", nargs=2, required=True,
                    metavar=("COL_A", "COL_B"),
                    help="Column names in the CSV, in order.")
    ap.add_argument("--languages", nargs=2, required=True,
                    metavar=("LANG_A", "LANG_B"),
                    help="Language names from languages.json, matching --columns.")
    ap.add_argument("--out", required=True,
                    help="Output folder; one file per language is written into it.")
    ap.add_argument("--max-words", type=int, default=0,
                    help="Drop rows where either side exceeds this many words. "
                         "0 (the default) keeps every row. Set it only if you want "
                         "filtering: train.py truncates at --max-seq-len, so rows "
                         "longer than that are cut mid-sentence during training.")
    ap.add_argument("--min-words", type=int, default=0)
    ap.add_argument("--drop-untranslated", action="store_true",
                    help="Drop rows whose two columns are identical.")
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="Drop repeated pairs.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    codes = config["name_to_code"]
    for lang in args.languages:
        if lang not in codes:
            raise SystemExit(f"{lang!r} is not in languages.json. Add it there first.")

    col_a, col_b = args.columns
    pairs, seen = [], set()
    stats = collections.Counter()

    for path in args.csv:
        if not os.path.exists(path):
            print(f"  {path}: MISSING - skipped")
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in (col_a, col_b) if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(
                    f"{path}: no column named {missing}. Found: {reader.fieldnames}")
            rows = 0
            for row in reader:
                rows += 1
                a = (row.get(col_a) or "").strip()
                b = (row.get(col_b) or "").strip()
                if not a or not b:
                    stats["empty"] += 1
                    continue
                if args.drop_untranslated and a.lower() == b.lower():
                    stats["untranslated"] += 1
                    continue
                na, nb = len(a.split()), len(b.split())
                if args.min_words and (na < args.min_words or nb < args.min_words):
                    stats["too short"] += 1
                    continue
                if args.max_words and (na > args.max_words or nb > args.max_words):
                    stats["too long"] += 1
                    continue
                if args.drop_duplicates:
                    key = (a.lower(), b.lower())
                    if key in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(key)
                pairs.append((a, b))
            print(f"  {os.path.basename(path)}: {rows} rows read")

    print(f"\nDropped:")
    for reason, n in stats.most_common():
        print(f"  {reason:<14}{n:8d}")
    print(f"Kept: {len(pairs)}")

    if not pairs:
        raise SystemExit("Nothing left after filtering -- check --columns and --max-words.")

    os.makedirs(args.out, exist_ok=True)
    base = os.path.basename(args.out.rstrip("/\\")) or "corpus"
    written = []
    for index, lang in enumerate(args.languages):
        out_path = os.path.join(args.out, f"{base}.{codes[lang]}")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for pair in pairs:
                # Newlines inside a cell would break the line alignment the
                # Moses format depends on.
                f.write(" ".join(pair[index].split()) + "\n")
        written.append(out_path)

    for path in written:
        print(f"Wrote {len(pairs)} lines -> {path}")
    print(f"\nNow add to languages.json:")
    print(f'    {{ "dir": "{args.out}", '
          f'"columns": ["{args.languages[0]}", "{args.languages[1]}"], "limit": 8000 }}')


if __name__ == "__main__":
    main()
