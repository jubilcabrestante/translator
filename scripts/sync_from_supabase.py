"""
sync_from_supabase.py
----------------------
Pulls newly-cached translations out of LearnVoca's `public.translations` table
(populated automatically by record_translation() every time Tier 3 answers a
query the cache didn't already have) and stages them for review as new training
data.

This is the "does it update the data" half of the retrain loop:

  1. People use the app  ->  new phrases hit the model server  ->
     record_translation() caches them in Postgres.        [already happens]
  2. THIS SCRIPT pulls those rows and writes them to a review file, grouped by
     the corpus file each one belongs to.
  3. YOU (ideally a Cuyonon speaker) read the review file and delete/fix
     anything wrong.
  4. Move the good lines into the corpus file named above their section.
  5. Run scripts/lint_data.py, then re-run .\\train_all.ps1.
  6. Restart scripts/serve.py so the new adapter is loaded.

Why review instead of auto-merging: a cached row is a machine translation the
model itself produced. Feeding a model's own possibly-wrong output back in as
new "ground truth" reinforces whatever mistakes it already makes (model
collapse / self-training drift). A human glance breaks that loop. Even better:
have the speaker translate the most-asked SOURCE phrases from scratch -- this
file lists them by hit count.

Every language pair in languages.json is handled. A row is filed under the
corpus whose two columns match its pair, in that file's column order. A pair
with no corpus file (say English<->Cuyonon) gets its own section instead of
being forced into the wrong file.

Requires env vars (never commit these, never put them in the Flutter app):
  SUPABASE_URL                e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   from Project Settings -> API (NOT the anon key --
                               the anon key can't read this table, by design;
                               the service role key bypasses RLS, so treat it
                               like a root password and only ever use it here,
                               offline, on your own machine)

Usage:
  $env:SUPABASE_URL = "https://xxxx.supabase.co"
  $env:SUPABASE_SERVICE_ROLE_KEY = "..."
  python scripts/sync_from_supabase.py
  python scripts/sync_from_supabase.py --min-hits 2   # only phrases asked >=2 times
"""

import argparse
import collections
import os
import sys

import requests

from common import load_config, loose_key, read_sep_file

REVIEW_FILE = "data/review_new_pairs.txt"
PAGE_SIZE = 1000


def fetch_translations(base_url, service_key, min_hits, max_rows):
    """Page through public.translations via PostgREST using the service role key."""
    url = f"{base_url.rstrip('/')}/rest/v1/translations"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }
    rows = []
    while len(rows) < max_rows:
        limit = min(PAGE_SIZE, max_rows - len(rows))
        params = {
            "select": "source_lang,target_lang,source_text,translated_text,hit_count",
            "hit_count": f"gte.{min_hits}",
            # A total order, so pages neither overlap nor skip rows.
            "order": "hit_count.desc,source_text.asc,target_lang.asc",
            "limit": str(limit),
            "offset": str(len(rows)),
        }
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < limit:
            break
    return rows


def pair_key(lang_a, text_a, lang_b, text_b):
    """Same pair regardless of direction, case or edge punctuation."""
    return tuple(sorted([(lang_a, loose_key(text_a)), (lang_b, loose_key(text_b))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Only pull phrases asked at least this many times "
                         "(repeated queries are more likely to be real, common phrases).")
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--out", default=REVIEW_FILE)
    args = ap.parse_args()

    base_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not base_url or not service_key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment "
              "variables first (service role key, not anon key). See the docstring.")
        sys.exit(1)

    config = load_config()
    code_to_name = config["code_to_name"]

    # Which corpus file each language pair belongs in (first match wins).
    corpus_for = {}
    for corpus in config["corpora"]:
        if corpus["format"] == "sep" and not corpus["synthetic"]:
            corpus_for.setdefault(frozenset(corpus["columns"]), corpus)

    # Everything already known -- training corpora AND the test set, so a test
    # sentence is never suggested for training.
    known = set()
    for corpus in config["corpora"] + config["test"]:
        if corpus["format"] != "sep":
            continue
        lang_a, lang_b = corpus["columns"]
        for a, b in read_sep_file(corpus["path"]):
            known.add(pair_key(lang_a, a, lang_b, b))

    print(f"Fetching cached translations with hit_count >= {args.min_hits} ...")
    rows = fetch_translations(base_url, service_key, args.min_hits, args.max_rows)
    print(f"Got {len(rows)} rows from public.translations")

    sections = collections.OrderedDict()     # (file label, left lang, right lang) -> {line: hits}
    unknown_codes = collections.Counter()
    already = 0
    for row in rows:
        src = code_to_name.get(row.get("source_lang"))
        tgt = code_to_name.get(row.get("target_lang"))
        if not src or not tgt or src == tgt:
            unknown_codes[(row.get("source_lang"), row.get("target_lang"))] += 1
            continue
        src_text = (row.get("source_text") or "").strip()
        tgt_text = (row.get("translated_text") or "").strip()
        if not src_text or not tgt_text or "<sep>" in src_text + tgt_text:
            continue
        if pair_key(src, src_text, tgt, tgt_text) in known:
            already += 1
            continue
        corpus = corpus_for.get(frozenset((src, tgt)))
        if corpus:
            left_lang, right_lang = corpus["columns"]
            label = os.path.relpath(corpus["path"])
        else:
            left_lang, right_lang, label = src, tgt, None
        left, right = (src_text, tgt_text) if left_lang == src else (tgt_text, src_text)
        lines = sections.setdefault((label, left_lang, right_lang), {})
        line = f"{left}<sep>{right}"
        lines[line] = max(row.get("hit_count") or 0, lines.get(line, 0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Review each line below. DELETE anything wrong, FIX anything close,\n")
        f.write("# then move the good lines into the file named in their section header,\n")
        f.write("# run scripts/lint_data.py, and re-run .\\train_all.ps1.\n")
        f.write("# These are the MODEL'S OWN outputs -- never merge them unread.\n")
        for (label, left_lang, right_lang), lines in sections.items():
            if label:
                f.write(f"\n# ---- for {label}   ({left_lang} | {right_lang}) ----\n")
            else:
                f.write(f"\n# ---- {left_lang} | {right_lang}: no corpus file for this pair yet."
                        f" Add one to languages.json before using these. ----\n")
            for line, hits in sorted(lines.items(), key=lambda kv: -kv[1]):
                f.write(f"# hits={hits}\n{line}\n")
                total += 1

    print(f"\nWrote {total} candidate new pairs -> {args.out} "
          f"({already} were already in your corpora)")
    for (src_code, tgt_code), n in unknown_codes.items():
        print(f"  skipped {n} rows with codes {src_code!r}->{tgt_code!r} "
              f"(not a configured language pair)")
    print("Review it, then merge the good lines into the files named in the headers.")


if __name__ == "__main__":
    main()
