"""
evaluate.py
-----------
Score the trained model on the held-out validation set using chrF and BLEU.

chrF is the better metric here: it works on character n-grams, so it is more
forgiving and more informative than BLEU for short text and low-resource
languages (exactly your situation). Higher = better; chrF ranges 0-100.

Track this number as your dataset grows -- it is how you'll *prove* the model
is getting more accurate.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --limit 200   # faster, evaluate on a subset
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--val-file", default="data/val.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    args = ap.parse_args()

    rows = read_jsonl(args.val_file)
    if args.limit:
        rows = rows[:args.limit]

    print("Loading model + adapter...")
    model, tokenizer = load(args.base_model, args.adapter_dir)

    hyps, refs = [], []
    for i, row in enumerate(rows):
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        src_lang, tgt_lang, text = parse_direction(msgs["user"])
        reference = msgs["assistant"]
        prediction = translate(model, tokenizer, text, src_lang, tgt_lang)
        hyps.append(prediction)
        refs.append(reference)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    exact = 100.0 * sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / len(hyps)

    print("\n================ RESULTS ================")
    print(f"Examples evaluated : {len(hyps)}")
    print(f"chrF               : {chrf:.2f}   (primary metric, 0-100)")
    print(f"BLEU               : {bleu:.2f}")
    print(f"Exact match        : {exact:.2f}%")
    print("=========================================")


if __name__ == "__main__":
    main()
