"""
prepare_data.py
---------------
Turn the raw `<sep>`-delimited parallel files in assets/ into an instruction-style
(chat) dataset for supervised fine-tuning (SFT) of a decoder-only LLM.

What this does, and why:

1. BIDIRECTIONAL: from a Tagalog<sep>Cuyonon pair it generates TWO training
   examples -- one per direction -- so one model learns both ways.

2. LEAK-FREE SPLIT: the two directions of a pair, and every synonym variant of
   the same term, are kept on the SAME side of the train/val split. Splitting
   individual examples instead (the old behaviour) put `kain->kaen` in train and
   `kaen->kain` in val, so ~90% of the validation set was already memorised and
   chrF measured almost nothing. Terms are grouped with a union-find over both
   languages, so "Maganda." and all of {Goapa., Matinlo., Mapostora.} stay
   together instead of straddling the split.

3. REBALANCING (train only): ~25% of the raw rows are pure copies (the Tagalog
   and Cuyonon are identical) and ~60% are single words. Left alone, "echo the
   input" becomes the most reinforced behaviour, and most of the model's
   capacity goes to single words that the app's dictionary tier answers before
   the model is ever called. `--max-copy-frac` caps copy rows and
   `--multi-word-repeat` oversamples real sentences, which are the only rows
   that teach grammar.

4. ENTITY PASSTHROUGH: `assets/entities/` lists names, places and events that
   are identical across the languages, emitted as "term -> itself" so the model
   keeps proper nouns unchanged. These are synthetic and guaranteed-correct, so
   they go to TRAIN ONLY -- scoring them in validation would inflate chrF.

Output: data/train.jsonl and data/val.jsonl, one chat example per line.
"""

import argparse
import collections
import json
import os
import random

from common import build_messages  # shared system prompt lives here too


def read_pairs(path, swap=False):
    """Read a `<sep>` file into (tagalog, cuyonon) tuples.

    `swap=True` for a cuyo-taga file, whose columns are the other way round, so
    every source normalises to the same (Tagalog, Cuyonon) orientation.
    """
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "<sep>" not in line:
                continue
            parts = line.split("<sep>")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b:
                continue
            pairs.append((b, a) if swap else (a, b))
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


class UnionFind:
    """Groups terms linked by at least one translation pair.

    Nodes are ('T', tagalog) and ('C', cuyonon). Uniting them per pair puts a
    whole synonym cluster in one component, and a component is what the
    train/val split moves around -- never a single example.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def to_example(src_lang, src_text, tgt_lang, tgt_text):
    """Build one chat-format example using the shared system prompt."""
    return {"messages": build_messages(src_text, src_lang, tgt_lang, assistant=tgt_text)}


def is_copy(src_text, tgt_text):
    return src_text.strip().lower() == tgt_text.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="assets/taga-cuyo.txt",
                    help="Parallel file: <Tagalog><sep><Cuyonon> per line.")
    ap.add_argument("--reverse-file", default="assets/cuyo-taga.txt",
                    help="Parallel file with the columns swapped: "
                         "<Cuyonon><sep><Tagalog>. Merged in and deduplicated; "
                         "it holds pairs the main file is missing.")
    ap.add_argument("--src-lang", default="Tagalog")
    ap.add_argument("--tgt-lang", default="Cuyonon")
    ap.add_argument("--entity-dir", default="assets/entities",
                    help="Folder of .txt lists (names/places/events) kept unchanged.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of term GROUPS held out for validation.")
    ap.add_argument("--multi-word-repeat", type=int, default=2,
                    help="How many times to repeat each multi-word (sentence) training "
                         "example. These are the only rows that teach grammar and they "
                         "are the minority, so oversampling them shifts capacity toward "
                         "the traffic the model actually serves. 1 = no oversampling.")
    ap.add_argument("--max-copy-frac", type=float, default=0.10,
                    help="Cap on the share of TRAINING rows that are pure copies "
                         "(input identical to output, including entity passthrough). "
                         "Raw data is ~25%% copies, which over-teaches echoing. "
                         "Use 1.0 to disable the cap.")
    ap.add_argument("--no-both-directions", action="store_true",
                    help="Only keep the forward direction (default is both).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    both = not args.no_both_directions

    # ---- 1. Load and merge every parallel source -----------------------------
    main_pairs = list(dict.fromkeys(read_pairs(args.data_file)))
    reverse_pairs = read_pairs(args.reverse_file, swap=True)
    pairs = list(dict.fromkeys(main_pairs + reverse_pairs))   # order-preserving
    print(f"Read {len(main_pairs)} unique pairs from {args.data_file}")
    print(f"Read {len(reverse_pairs)} pairs from {args.reverse_file} "
          f"-> {len(pairs) - len(main_pairs)} were new")
    print(f"Unique pairs after merge: {len(pairs)}")

    # ---- 2. Group linked terms so the split cannot leak ----------------------
    uf = UnionFind()
    for tl, cyo in pairs:
        uf.union(("T", tl.strip().lower()), ("C", cyo.strip().lower()))

    groups = collections.defaultdict(list)
    for tl, cyo in pairs:
        groups[uf.find(("T", tl.strip().lower()))].append((tl, cyo))

    group_keys = list(groups)
    random.shuffle(group_keys)
    biggest = max(len(v) for v in groups.values())
    print(f"Grouped into {len(group_keys)} term clusters "
          f"(largest holds {biggest} pairs)")

    # Fill validation until it holds ~val-frac of the PAIRS, whole groups only.
    target_val_pairs = int(len(pairs) * args.val_frac)
    val_groups, val_count = set(), 0
    for key in group_keys:
        if val_count >= target_val_pairs:
            break
        val_groups.add(key)
        val_count += len(groups[key])

    train_pairs = [p for k in group_keys if k not in val_groups for p in groups[k]]
    val_pairs = [p for k in group_keys if k in val_groups for p in groups[k]]

    # ---- 3. Emit both directions, staying inside each split ------------------
    def emit(pair_list):
        out = []
        for tl, cyo in pair_list:
            out.append(to_example(args.src_lang, tl, args.tgt_lang, cyo))
            if both:
                out.append(to_example(args.tgt_lang, cyo, args.src_lang, tl))
        return out

    train_examples = emit(train_pairs)
    val_examples = emit(val_pairs)

    # ---- 4. Entity passthrough -> TRAIN ONLY ---------------------------------
    entities = load_entities(args.entity_dir)
    for term in entities:
        train_examples.append(to_example(args.src_lang, term, args.tgt_lang, term))
        if both:
            train_examples.append(to_example(args.tgt_lang, term, args.src_lang, term))
    print(f"Loaded {len(entities)} entities from {args.entity_dir} -> "
          f"{len(entities) * (2 if both else 1)} passthrough examples (train only)")

    # ---- 5. Deduplicate exact examples within each split ---------------------
    def dedupe(examples):
        seen, unique = set(), []
        for ex in examples:
            key = json.dumps(ex["messages"], ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        return unique

    train_examples = dedupe(train_examples)
    val_examples = dedupe(val_examples)

    # ---- 6. Rebalance the TRAINING set ---------------------------------------
    # Validation is deliberately left alone: it has to keep the natural
    # distribution so scores stay comparable across runs. evaluate.py segments
    # it at scoring time instead.
    def source_and_target(ex):
        msgs = {m["role"]: m["content"] for m in ex["messages"]}
        return msgs["user"].split("\n\n", 1)[1], msgs["assistant"]

    copies, singles, multis = [], [], []
    for ex in train_examples:
        source, target = source_and_target(ex)
        if is_copy(source, target):
            copies.append(ex)
        elif len(source.split()) == 1:
            singles.append(ex)
        else:
            multis.append(ex)

    raw_total = len(train_examples)
    raw_copy_frac = len(copies) / max(1, raw_total)

    kept_multis = multis * max(1, args.multi_word_repeat)

    # Cap copies as a share of the FINAL training size:
    #   frac = copies / (copies + others)  ->  copies = frac/(1-frac) * others
    others = len(singles) + len(kept_multis)
    if args.max_copy_frac >= 1.0:
        kept_copies = copies
    else:
        allowed = int(args.max_copy_frac / (1 - args.max_copy_frac) * others)
        random.shuffle(copies)
        kept_copies = copies[:allowed]

    train_examples = kept_copies + singles + kept_multis
    random.shuffle(train_examples)
    final = max(1, len(train_examples))

    print("\nTraining mix (raw -> after rebalancing)")
    print(f"  copy rows       : {len(copies):5d} ({100 * raw_copy_frac:4.1f}%) -> "
          f"{len(kept_copies):5d} ({100 * len(kept_copies) / final:4.1f}%)")
    print(f"  single-word rows: {len(singles):5d} -> {len(singles):5d} "
          f"({100 * len(singles) / final:4.1f}%)")
    print(f"  multi-word rows : {len(multis):5d} -> {len(kept_multis):5d} "
          f"({100 * len(kept_multis) / final:4.1f}%)  "
          f"[x{max(1, args.multi_word_repeat)}]")

    # ---- 7. Write ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in [("train", train_examples), ("val", val_examples)]:
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(split):5d} examples -> {out_path}")

    print("\nDone. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
