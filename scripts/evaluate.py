"""
evaluate.py
-----------
Score the trained model with chrF and BLEU -- on the frozen test set
(data/test.jsonl) once it has rows, otherwise on the validation set.

chrF is the primary metric: it works on character n-grams, so it is more
forgiving and more informative than BLEU for short text and low-resource
languages -- exactly this situation. Higher = better, range 0-100.

What this does that a plain corpus-level score does not:

1. MULTI-REFERENCE. ~300 sources have several valid answers
   ('Maganda.' -> Goapa. / Matinlo. / Mapostora.). Every valid target for a
   source is passed to sacrebleu as a parallel reference stream, so any of
   them counts as correct.

2. SEGMENTS. Single words and real sentences are different tasks, and the app
   answers most single words from its dictionary tier before the model is
   called. The number to track is 'multi-word (sentences)'.

3. ECHO RATE. The share of inputs needing a real change that the model just
   parroted back. Watch this stay low.

4. CONFIDENCE INTERVALS. ~100 sentences per direction is a small sample; chrF
   can move a couple of points between two runs by chance alone. Each row
   shows a 95% bootstrap interval. If two runs' intervals overlap heavily, the
   difference is probably noise -- collect more test sentences before
   concluding anything.

5. SAVED OUTPUT. Each run writes, under outputs/eval/<run>/:
     predictions.tsv   every source, prediction and reference
     review-sheet.csv  a sample of sentences for a native speaker to mark
                       correct / understandable / wrong
     summary.json      the scores
   chrF is a stand-in; the speaker's verdict is what users actually feel.
   Tally a filled-in sheet with --tally.

Usage:
  python scripts/evaluate.py                     # test set if filled in, else validation
  python scripts/evaluate.py --split val
  python scripts/evaluate.py --limit 200         # faster, evaluate a subset
  python scripts/evaluate.py --num-beams 4       # A/B beam search against greedy
  python scripts/evaluate.py --show 15           # print sample translations
  python scripts/evaluate.py --tally outputs/eval/<run>/review-sheet.csv
"""

import argparse
import collections
import csv
import datetime
import json
import os
import random

import sacrebleu

from common import parse_user_turn

RATINGS = ("correct", "understandable", "wrong")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_reference_streams(reference_lists):
    """Turn per-example reference lists into equal-length streams for sacrebleu.

    sacrebleu wants [[ref1_a, ref1_b, ...], [ref2_a, ref2_b, ...]] with every
    stream the same length, so examples with fewer references are padded by
    repeating one they already have. Padding with a duplicate adds no new way to
    match, which keeps the score honest.
    """
    if not reference_lists:
        return []
    width = max(len(r) for r in reference_lists)
    return [[refs[i] if i < len(refs) else refs[0] for refs in reference_lists]
            for i in range(width)]


def score(hyps, reference_lists):
    """chrF / BLEU / exact-match over one segment, counting any reference as correct."""
    if not hyps:
        return None
    streams = to_reference_streams(reference_lists)
    exact = 100.0 * sum(
        any(h.strip() == r.strip() for r in refs)
        for h, refs in zip(hyps, reference_lists)
    ) / len(hyps)
    return {
        "n": len(hyps),
        "chrf": sacrebleu.corpus_chrf(hyps, streams).score,
        "bleu": sacrebleu.corpus_bleu(hyps, streams).score,
        "exact": exact,
    }


def chrf_interval(hyps, reference_lists, rounds, seed=0):
    """95% bootstrap interval for corpus chrF: resample sentences with replacement."""
    if rounds <= 0 or len(hyps) < 10:
        return None
    rng = random.Random(seed)
    n = len(hyps)
    scores = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(sacrebleu.corpus_chrf(
            [hyps[i] for i in idx],
            to_reference_streams([reference_lists[i] for i in idx])).score)
    scores.sort()
    return scores[int(0.025 * rounds)], scores[max(0, int(0.975 * rounds) - 1)]


def tally(path):
    """Summarise a review sheet a speaker has filled in (Excel may re-save it)."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        column = next((c for c in reader.fieldnames or [] if c.lower().startswith("rating")), None)
        if column is None:
            raise SystemExit(f"{path}: no 'rating' column found.")
        overall = collections.Counter()
        by_direction = collections.defaultdict(collections.Counter)
        for row in reader:
            value = (row.get(column) or "").strip().lower()
            if not value:
                verdict = "(not rated)"
            else:
                verdict = next((r for r in RATINGS if value.startswith(r[0])), "(other)")
            overall[verdict] += 1
            by_direction[row.get("direction", "")][verdict] += 1

    def line(counts):
        rated = sum(counts[r] for r in RATINGS)
        if not rated:
            return "nothing rated yet"
        parts = [f"{r} {100 * counts[r] / rated:.0f}%" for r in RATINGS]
        return f"{' | '.join(parts)}   ({rated} rated)"

    print(f"{path}\n  overall: {line(overall)}")
    for direction, counts in sorted(by_direction.items()):
        print(f"  {direction:<24} {line(counts)}")
    if overall["(not rated)"]:
        print(f"  {overall['(not rated)']} rows not rated yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--split", choices=["auto", "test", "val"], default="auto",
                    help="auto = the test set when data/test.jsonl has rows, else validation.")
    ap.add_argument("--val-file", default="data/val.jsonl")
    ap.add_argument("--test-file", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N distinct sources.")
    ap.add_argument("--show", type=int, default=0,
                    help="Print this many sample translations for eyeballing.")
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="Bootstrap rounds for the chrF confidence interval (0 = off).")
    ap.add_argument("--review-size", type=int, default=50,
                    help="Sentences in the speaker review sheet.")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--tally", default=None, metavar="CSV",
                    help="Summarise a filled-in review sheet and exit (no model needed).")
    args = ap.parse_args()

    if args.tally:
        tally(args.tally)
        return

    def has_rows(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0

    if args.split == "test" or (args.split == "auto" and has_rows(args.test_file)):
        split, data_file = "test", args.test_file
    else:
        split, data_file = "val", args.val_file
    if not has_rows(data_file):
        raise SystemExit(f"{data_file} is empty or missing. Run scripts/prepare_data.py "
                         f"(and fill assets/test-taga-cuyo.txt for a test set).")
    print(f"Evaluating on the {split} set: {data_file}")
    if split == "val":
        print("  (no frozen test set yet -- fill assets/test-taga-cuyo.txt; until then "
              "validation is used, which is stable across runs but drawn from the "
              "same sources as training)")

    # ---- Collect every valid reference per (direction, source) ---------------
    references = collections.OrderedDict()
    for row in read_jsonl(data_file):
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        src_lang, tgt_lang, text = parse_user_turn(msgs["user"])
        refs = references.setdefault((src_lang, tgt_lang, text), [])
        if msgs["assistant"] not in refs:
            refs.append(msgs["assistant"])

    items = list(references.items())
    if args.limit:
        items = items[:args.limit]

    multi_ref = sum(1 for _, refs in items if len(refs) > 1)
    print(f"{len(items)} distinct sources to translate "
          f"({multi_ref} have more than one valid answer)")

    from translate import load, translate   # imports torch; not needed for --tally
    print("Loading model + adapter...")
    model, tokenizer = load(args.base_model, args.adapter_dir, device=args.device)

    rows = []
    for i, ((src_lang, tgt_lang, text), refs) in enumerate(items):
        prediction = translate(model, tokenizer, text, src_lang, tgt_lang,
                               num_beams=args.num_beams)
        rows.append((src_lang, tgt_lang, text, refs, prediction))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} done")

    # ---- Segment ------------------------------------------------------------
    def is_copy(text, refs):
        return all(text.strip().lower() == r.strip().lower() for r in refs)

    segments = collections.OrderedDict([
        ("ALL", rows),
        ("multi-word (sentences)", [r for r in rows if len(r[2].split()) > 1]),
        ("single-word (dictionary-ish)", [r for r in rows if len(r[2].split()) == 1]),
        ("non-copy only", [r for r in rows if not is_copy(r[2], r[3])]),
        ("copy rows (input == output)", [r for r in rows if is_copy(r[2], r[3])]),
    ])
    # One row per direction present in the data, so a 3-language model reports
    # all 6 without this file knowing which languages exist. Sentences only:
    # that is where directions differ.
    for src_lang, tgt_lang in sorted({(r[0], r[1]) for r in rows}):
        segments[f"{src_lang} -> {tgt_lang} (sentences)"] = [
            r for r in rows if r[0] == src_lang and r[1] == tgt_lang and len(r[2].split()) > 1]

    print("\n" + "=" * 92)
    print("RESULTS   (chrF is the primary metric; any valid synonym counts)")
    print("=" * 92)
    print(f"{'segment':<36}{'n':>6}{'chrF':>9}{'95% interval':>17}{'BLEU':>9}{'exact%':>9}")
    print("-" * 92)
    summary = collections.OrderedDict()
    for name, subset in segments.items():
        s = score([r[4] for r in subset], [r[3] for r in subset])
        if s is None:
            print(f"{name:<36}{0:>6}{'--':>9}{'':>17}{'--':>9}{'--':>9}")
            continue
        interval = chrf_interval([r[4] for r in subset], [r[3] for r in subset], args.bootstrap)
        shown = f"{interval[0]:.1f} - {interval[1]:.1f}" if interval else ""
        print(f"{name:<36}{s['n']:>6}{s['chrf']:>9.2f}{shown:>17}{s['bleu']:>9.2f}"
              f"{s['exact']:>9.2f}")
        s["chrf_95"] = list(interval) if interval else None
        summary[name] = s
    print("-" * 92)

    # ---- Echo rate: parroting the input back on rows that need a real change -
    non_copy = [r for r in rows if not is_copy(r[2], r[3])]
    if non_copy:
        echoed = sum(1 for r in non_copy if r[4].strip().lower() == r[2].strip().lower())
        echo = 100.0 * echoed / len(non_copy)
        summary["echo_rate"] = echo
        print(f"Echo rate (returned the input unchanged when it should not have): "
              f"{echo:.1f}%  ({echoed}/{len(non_copy)})")
    print("=" * 92)
    print("The number to track as the dataset grows is 'multi-word (sentences)'.")

    if args.show:
        print("\nSamples:")
        for src_lang, tgt_lang, text, refs, pred in rows[:args.show]:
            mark = "ok " if any(pred.strip() == r.strip() for r in refs) else "  X"
            print(f"  {mark} [{src_lang[:3]}->{tgt_lang[:3]}] {text!r}")
            print(f"        got {pred!r}")
            print(f"        ref {refs}")

    # ---- Save predictions, a speaker review sheet and the scores -------------
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    adapter = os.path.basename(os.path.normpath(args.adapter_dir))
    beams = f"-beams{args.num_beams}" if args.num_beams > 1 else ""
    run_dir = os.path.join(args.out_dir, f"{stamp}-{adapter}-{split}{beams}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "predictions.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["direction", "source", "prediction", "references", "exact", "sentence"])
        for src_lang, tgt_lang, text, refs, pred in rows:
            writer.writerow([f"{src_lang} -> {tgt_lang}", text, pred, " || ".join(refs),
                             int(any(pred.strip() == r.strip() for r in refs)),
                             int(len(text.split()) > 1)])

    candidates = [r for r in rows if len(r[2].split()) > 1 and not is_copy(r[2], r[3])]
    sample = random.Random(0).sample(candidates, min(args.review_size, len(candidates)))
    # utf-8-sig so Excel shows accented letters correctly.
    with open(os.path.join(run_dir, "review-sheet.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "direction", "source", "model output", "reference(s)",
                         "rating (correct / understandable / wrong)", "comment"])
        for i, (src_lang, tgt_lang, text, refs, pred) in enumerate(sample, 1):
            writer.writerow([i, f"{src_lang} -> {tgt_lang}", text, pred, " || ".join(refs), "", ""])

    summary["_run"] = {"split": split, "data_file": data_file, "adapter_dir": args.adapter_dir,
                       "base_model": args.base_model, "num_beams": args.num_beams,
                       "limit": args.limit, "sources": len(items)}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved predictions, scores and a {len(sample)}-sentence review sheet to {run_dir}")
    print(f"Have a Cuyonon speaker fill in the rating column, then:\n"
          f"  python scripts/evaluate.py --tally {os.path.join(run_dir, 'review-sheet.csv')}")


if __name__ == "__main__":
    main()
