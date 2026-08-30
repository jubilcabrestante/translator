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
  2. THIS SCRIPT pulls those rows and writes them to a review file.
  3. YOU read the review file and delete/fix anything wrong.
  4. Move the good lines into assets/taga-cuyo.txt.
  5. Re-run .\\train_all.ps1 to retrain on the combined data.
  6. Restart scripts/serve.py so the new adapter is loaded.

Why review instead of auto-merging: a cached row is a machine translation the
model itself produced. Feeding a model's own possibly-wrong output back in as
new "ground truth" reinforces whatever mistakes it already makes (model
collapse / self-training drift). A human glance breaks that loop.

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
import os
import sys

import requests

EXISTING_FILE = "assets/taga-cuyo.txt"
REVIEW_FILE = "data/review_new_pairs.txt"


def load_existing_pairs(path):
    """Read the current dataset so already-known pairs aren't re-suggested."""
    existing = set()
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "<sep>" in line:
                existing.add(line)
    return existing


def fetch_translations(base_url, service_key, min_hits):
    """Page through public.translations via PostgREST using the service role key."""
    url = f"{base_url.rstrip('/')}/rest/v1/translations"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }
    params = {
        "select": "source_lang,target_lang,source_text,translated_text,hit_count",
        "hit_count": f"gte.{min_hits}",
        "order": "hit_count.desc",
        "limit": "1000",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Only pull phrases asked at least this many times "
                         "(repeated queries are more likely to be real, common phrases).")
    ap.add_argument("--out", default=REVIEW_FILE)
    args = ap.parse_args()

    base_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not base_url or not service_key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment "
              "variables first (service role key, not anon key). See the docstring.")
        sys.exit(1)

    print(f"Fetching cached translations with hit_count >= {args.min_hits} ...")
    rows = fetch_translations(base_url, service_key, args.min_hits)
    print(f"Got {len(rows)} rows from public.translations")

    existing = load_existing_pairs(EXISTING_FILE)

    new_lines = []
    for row in rows:
        # Normalize to Tagalog<sep>Cuyonon regardless of which direction was asked.
        if row["source_lang"] == "tl":
            tagalog, cuyonon = row["source_text"], row["translated_text"]
        else:
            tagalog, cuyonon = row["translated_text"], row["source_text"]

        tagalog, cuyonon = tagalog.strip(), cuyonon.strip()
        if not tagalog or not cuyonon:
            continue

        line = f"{tagalog}<sep>{cuyonon}"
        if line in existing:
            continue
        new_lines.append((row["hit_count"], line))

    # Dedupe while keeping the highest hit_count for each line.
    seen = {}
    for hits, line in new_lines:
        if line not in seen or hits > seen[line]:
            seen[line] = hits

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Review each line below, DELETE anything wrong, then move the\n")
        f.write(f"# good ones into {EXISTING_FILE} and re-run .\\train_all.ps1\n")
        f.write("# format: hit_count | Tagalog<sep>Cuyonon\n\n")
        for line, hits in sorted(seen.items(), key=lambda kv: -kv[1]):
            f.write(f"# hits={hits}\n{line}\n")

    print(f"\nWrote {len(seen)} candidate new pairs -> {args.out}")
    print("Review it, then merge the good lines into assets/taga-cuyo.txt and retrain.")


if __name__ == "__main__":
    main()
