"""
prepare_data.py
---------------
Turn the raw `<sep>`-delimited parallel files in assets/ into an instruction-style
(chat) dataset for supervised fine-tuning (SFT) of a decoder-only LLM.

Two things this does:

1. BIDIRECTIONAL: from a single Tagalog<sep>Cuyonon file it generates TWO training
   examples per pair -- one for each translation direction -- so one model learns
   both ways (and it doubles the data).

2. ENTITY PASSTHROUGH: it reads editable lists in assets/entities/ (names, places,
   events) and emits "this term -> itself" examples in both directions. A person's
   name or a city name is identical across languages, so these examples are
   guaranteed-correct and teach the model to KEEP proper nouns unchanged instead of
   inventing a translation for them.

Output: data/train.jsonl and data/val.jsonl, one chat example per line.
"""

import argparse
import json
import os
import random

from common import build_messages  # shared system prompt lives here too


def read_pairs(path, src_lang, tgt_lang):
    """Read a `<sep>`-delimited file into (src_lang, src_text, tgt_lang, tgt_text) tuples."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "<sep>" not in line:
                continue
            parts = line.split("<sep>")
            if len(parts) != 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if src and tgt:
                pairs.append((src_lang, src, tgt_lang, tgt))
    return pairs


def load_entities(entity_dir):
    """Load every non-comment line from every .txt file under entity_dir."""
    entities = []
    if not os.path.isdir(entity_dir):
        return entities
    for fname in sorted(os.listdir(entity_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(entity_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term and not term.startswith("#"):
                    entities.append(term)
    return entities


def to_example(src_lang, src_text, tgt_lang, tgt_text):
    """Build one chat-format training example using the shared system prompt."""
    return {"messages": build_messages(src_text, src_lang, tgt_lang, assistant=tgt_text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="assets/taga-cuyo.txt",
                    help="Parallel file: <Tagalog><sep><Cuyonon> per line.")
    ap.add_argument("--src-lang", default="Tagalog")
    ap.add_argument("--tgt-lang", default="Cuyonon")
    ap.add_argument("--entity-dir", default="assets/entities",
                    help="Folder of .txt lists (names/places/events) kept unchanged.")
    ap.add_argument("--entity-repeat", type=int, default=2,
                    help="How many times to repeat each entity example (more = stronger signal).")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for validation.")
    ap.add_argument("--no-both-directions", action="store_true",
                    help="Only keep the forward direction (default is both).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    both = not args.no_both_directions

    # ---- 1. Real parallel data (both directions) -----------------------------
    pairs = read_pairs(args.data_file, args.src_lang, args.tgt_lang)
    print(f"Read {len(pairs)} parallel pairs from {args.data_file}")

    examples = []
    for src_lang, src, tgt_lang, tgt in pairs:
        examples.append(to_example(src_lang, src, tgt_lang, tgt))
        if both:
            examples.append(to_example(tgt_lang, tgt, src_lang, src))

    # ---- 2. Entity passthrough (term -> itself, both directions) -------------
    entities = load_entities(args.entity_dir)
    n_entity_examples = 0
    for term in entities:
        for _ in range(max(1, args.entity_repeat)):
            examples.append(to_example(args.src_lang, term, args.tgt_lang, term))
            if both:
                examples.append(to_example(args.tgt_lang, term, args.src_lang, term))
            n_entity_examples += 2 if both else 1
    print(f"Loaded {len(entities)} entities from {args.entity_dir} "
          f"-> {n_entity_examples} passthrough examples")

    # ---- Deduplicate identical examples --------------------------------------
    seen, unique = set(), []
    for ex in examples:
        key = json.dumps(ex["messages"], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    examples = unique

    # ---- Shuffle + split ------------------------------------------------------
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val, train = examples[:n_val], examples[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in [("train", train), ("val", val)]:
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split):5d} examples -> {out_path}")

    print("\nDone. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
