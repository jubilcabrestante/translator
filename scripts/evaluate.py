"""
evaluate.py
-----------
Score the trained model on the held-out validation set using chrF and BLEU.

chrF is the primary metric: it works on character n-grams, so it is more
forgiving and more informative than BLEU for short text and low-resource
languages -- exactly this situation. Higher = better, range 0-100.

Three things this does that a plain corpus-level score does not:

1. MULTI-REFERENCE. The data has ~400 Tagalog terms with several valid Cuyonon
   answers ('Maganda.' -> Goapa. / Matinlo. / Mapostora.). Those are synonyms
   and dialect variants, not errors, but single-reference scoring marks all but
   one wrong. Every valid target for a source is collected and passed to
   sacrebleu as parallel reference streams, so any of them counts as correct.

2. SEGMENTS. Single words and real sentences are different tasks with very
   different difficulty, and the app answers most single words from its
   dictionary tier before the model is called. One blended number hides the
   score that actually matters, so they are reported separately.

3. ECHO RATE. The share of non-copy inputs the model just parroted back. Around
   a quarter of the raw data is legitimately identical across both languages, so
   a model can score respectably by learning to copy. Watch this stay low.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --limit 200         # faster, evaluate a subset
  python scripts/evaluate.py --show 15           # print sample translations
"""

import argparse
import collections
import json

import sacrebleu

from translate import load, translate   # reuse the loader + translate fn


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def parse_direction(user_content):
    """Recover 'Tagalog'/'Cuyonon' from a 'Translate from X to Y:' user message."""
    # user_content looks like: "Translate from Tagalog to Cuyonon:\n\n<text>"
    header = user_content.split(":", 1)[0]        # "Translate from Tagalog to Cuyonon"
    words = header.split()
    src_lang = words[words.index("from") + 1]
    tgt_lang = words[words.index("to") + 1]
    text = user_content.split("\n\n", 1)[1]
    return src_lang, tgt_lang, text


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--val-file", default="data/val.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N distinct sources.")
    ap.add_argument("--show", type=int, default=0,
                    help="Print this many sample translations for eyeballing.")
    args = ap.parse_args()

    # ---- Collect every valid reference per (direction, source) ---------------
    references = collections.OrderedDict()
    for row in read_jsonl(args.val_file):
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        src_lang, tgt_lang, text = parse_direction(msgs["user"])
        references.setdefault((src_lang, tgt_lang, text), [])
        if msgs["assistant"] not in references[(src_lang, tgt_lang, text)]:
            references[(src_lang, tgt_lang, text)].append(msgs["assistant"])

    items = list(references.items())
    if args.limit:
        items = items[:args.limit]

    multi_ref = sum(1 for _, refs in items if len(refs) > 1)
    print(f"{len(items)} distinct sources to translate "
          f"({multi_ref} have more than one valid answer)")

    print("Loading model + adapter...")
    model, tokenizer = load(args.base_model, args.adapter_dir)

    rows = []
    for i, ((src_lang, tgt_lang, text), refs) in enumerate(items):
        prediction = translate(model, tokenizer, text, src_lang, tgt_lang)
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
        ("Tagalog -> Cuyonon", [r for r in rows if r[0] == "Tagalog"]),
        ("Cuyonon -> Tagalog", [r for r in rows if r[0] == "Cuyonon"]),
    ])

    print("\n" + "=" * 74)
    print("RESULTS   (chrF is the primary metric; any valid synonym counts)")
    print("=" * 74)
    print(f"{'segment':<30}{'n':>6}{'chrF':>10}{'BLEU':>10}{'exact%':>10}")
    print("-" * 74)
    for name, subset in segments.items():
        s = score([r[4] for r in subset], [r[3] for r in subset])
        if s is None:
            print(f"{name:<30}{0:>6}{'--':>10}{'--':>10}{'--':>10}")
            continue
        print(f"{name:<30}{s['n']:>6}{s['chrf']:>10.2f}{s['bleu']:>10.2f}"
              f"{s['exact']:>10.2f}")
    print("-" * 74)

    # ---- Echo rate: parroting the input back on rows that need a real change -
    non_copy = [r for r in rows if not is_copy(r[2], r[3])]
    if non_copy:
        echoed = sum(1 for r in non_copy if r[4].strip().lower() == r[2].strip().lower())
        print(f"Echo rate (returned the input unchanged when it should not have): "
              f"{100.0 * echoed / len(non_copy):.1f}%  ({echoed}/{len(non_copy)})")
    print("=" * 74)
    print("The number to track as the dataset grows is 'multi-word (sentences)'.")

    if args.show:
        print("\nSamples:")
        for src_lang, tgt_lang, text, refs, pred in rows[:args.show]:
            mark = "ok " if any(pred.strip() == r.strip() for r in refs) else "  X"
            print(f"  {mark} [{src_lang[:3]}->{tgt_lang[:3]}] {text!r}")
            print(f"        got {pred!r}")
            print(f"        ref {refs}")


if __name__ == "__main__":
    main()
