"""
sync_from_supabase.py
----------------------
Pulls the phrases people translated in the Learnut app out of its
`public.translations` table and stages them for review as new training data.

Learnut logs every successful translation itself (TranslationRepository.translate
inserts sentence, source_lang, target_lang, translated, user_id), one row per
request. So the number of rows for a phrase is how often it was asked.

This is the "does it update the data" half of the retrain loop:

  1. People use the app  ->  phrases hit the model server  ->  the app logs
     each answer into public.translations.               [already happens]
  2. THIS SCRIPT pulls those rows, counts how often each phrase was asked, and
     writes them to a review file grouped by the corpus file each belongs to.
  3. YOU (ideally a Cuyonon speaker) read the review file and delete/fix
     anything wrong.
  4. Move the good lines into the corpus file named above their section.
  5. Run scripts/lint_data.py, then retrain (Colab notebook or .\\train_all.ps1).
  6. Redeploy the new model (see "Wiring this into the Learnut app" in the README).

Why review instead of auto-merging: a logged row is a machine translation the
model itself produced. Feeding a model's own possibly-wrong output back in as
new "ground truth" reinforces whatever mistakes it already makes (model
collapse / self-training drift). A human glance breaks that loop. Even better:
have the speaker translate the most-asked SOURCE phrases from scratch -- this
file lists them by how often they were asked.

Languages may be stored as names ("Tagalog", what Learnut sends) or codes
("tl"); both resolve through languages.json. Every language pair there is
handled. A row is filed under the corpus whose two columns match its pair, in
that file's column order. A pair with no corpus file (say English<->Cuyonon)
gets its own section instead of being forced into the wrong file.

Requires env vars (never commit these, never put them in the app):
  SUPABASE_URL                e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   from Project Settings -> API (NOT the anon key --
                               app users may only insert into this table, by
                               design; the service role key bypasses RLS, so
                               treat it like a root password and only ever use
                               it here, offline, on your own machine)

Usage:
  $env:SUPABASE_URL = "https://xxxx.supabase.co"
  $env:SUPABASE_SERVICE_ROLE_KEY = "..."
  python scripts/sync_from_supabase.py
  python scripts/sync_from_supabase.py --min-hits 1   # every phrase, even if asked once
"""

import argparse
import collections
import os
import sys

import requests

from common import load_config, loose_key, read_sep_file, resolve_language

REVIEW_FILE = "data/review_new_pairs.txt"
PAGE_SIZE = 1000


def fetch_translations(base_url, service_key, max_rows):
    """Page through public.translations via PostgREST, oldest first.

    Keyset pagination on the identity column (id > last id seen), so rows the
    app inserts while this runs can neither shift nor duplicate a page.
    """
    url = f"{base_url.rstrip('/')}/rest/v1/translations"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }
    rows = []
    last_id = None
    while len(rows) < max_rows:
        limit = min(PAGE_SIZE, max_rows - len(rows))
        params = {
            "select": "id,sentence,source_lang,target_lang,translated",
            "translated": "not.is.null",
            "order": "id.asc",
            "limit": str(limit),
        }
        if last_id is not None:
            params["id"] = f"gt.{last_id}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 400:
            print(f"ERROR from Supabase: {resp.text[:300]}\n"
                  "This script expects Learnut's schema: public.translations with "
                  "id, sentence, source_lang, target_lang, translated.")
            sys.exit(1)
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < limit:
            break
        last_id = page[-1]["id"]
    return rows


def count_phrases(rows, config):
    """One entry per (source, target, phrase): times asked + latest answer.

    Rows arrive oldest first, so the translation kept is the newest one -- the
    model currently deployed, not one it has since been retrained away from.
    """
    phrases = {}
    unknown = collections.Counter()
    for row in rows:
        src = resolve_language(config, row.get("source_lang"))
        tgt = resolve_language(config, row.get("target_lang"))
        if not src or not tgt or src == tgt:
            unknown[(row.get("source_lang"), row.get("target_lang"))] += 1
            continue
        src_text = (row.get("sentence") or "").strip()
        tgt_text = (row.get("translated") or "").strip()
        if not src_text or not tgt_text or "<sep>" in src_text + tgt_text:
            continue
        key = (src, tgt, loose_key(src_text))
        entry = phrases.setdefault(key, {"hits": 0})
        entry["hits"] += 1
        entry["src"], entry["tgt"] = src, tgt
        entry["src_text"], entry["tgt_text"] = src_text, tgt_text
    return list(phrases.values()), unknown


def pair_key(lang_a, text_a, lang_b, text_b):
    """Same pair regardless of direction, case or edge punctuation."""
    return tuple(sorted([(lang_a, loose_key(text_a)), (lang_b, loose_key(text_b))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Only keep phrases asked at least this many times "
                         "(repeated queries are more likely to be real, common phrases).")
    ap.add_argument("--max-rows", type=int, default=100000,
                    help="Stop after reading this many logged translations.")
    ap.add_argument("--out", default=REVIEW_FILE)
    args = ap.parse_args()

    base_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not base_url or not service_key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment "
              "variables first (service role key, not anon key). See the docstring.")
        sys.exit(1)

    config = load_config()

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

    print("Fetching logged translations from public.translations ...")
    rows = fetch_translations(base_url, service_key, args.max_rows)
    phrases, unknown_langs = count_phrases(rows, config)
    print(f"Got {len(rows)} rows = {len(phrases)} distinct phrases; "
          f"keeping those asked >= {args.min_hits} times")

    sections = collections.OrderedDict()     # (file label, left lang, right lang) -> {line: hits}
    already = 0
    for p in phrases:
        if p["hits"] < args.min_hits:
            continue
        src, tgt = p["src"], p["tgt"]
        if pair_key(src, p["src_text"], tgt, p["tgt_text"]) in known:
            already += 1
            continue
        corpus = corpus_for.get(frozenset((src, tgt)))
        if corpus:
            left_lang, right_lang = corpus["columns"]
            label = os.path.relpath(corpus["path"])
        else:
            left_lang, right_lang, label = src, tgt, None
        if left_lang == src:
            left, right = p["src_text"], p["tgt_text"]
        else:
            left, right = p["tgt_text"], p["src_text"]
        lines = sections.setdefault((label, left_lang, right_lang), {})
        line = f"{left}<sep>{right}"
        lines[line] = lines.get(line, 0) + p["hits"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Review each line below. DELETE anything wrong, FIX anything close,\n")
        f.write("# then move the good lines into the file named in their section header,\n")
        f.write("# run scripts/lint_data.py, and retrain.\n")
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
    for (src_lang, tgt_lang), n in unknown_langs.items():
        print(f"  skipped {n} rows with languages {src_lang!r}->{tgt_lang!r} "
              f"(not a configured language pair)")
    print("Review it, then merge the good lines into the files named in the headers.")


if __name__ == "__main__":
    main()
