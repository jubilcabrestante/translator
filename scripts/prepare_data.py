"""
prepare_data.py
---------------
Turn the `<sep>`-delimited parallel files listed in `languages.json` into an
instruction-style (chat) dataset for supervised fine-tuning (SFT) of a
decoder-only LLM.

What this does, and why:

1. EVERY DIRECTION. The language set comes from languages.json, and a training
   example is emitted for each ordered pair the data supports -- 2 directions
   for 2 languages, 6 for 3 -- all served by one model.

2. BRIDGING. A pair never stated directly is still emitted when both sides
   share a SPECIFIC common translation in a third language. Bootstrapping
   English off the Tagalog column therefore yields English<->Cuyonon training
   data without anyone translating Cuyonon to English by hand, and the Cuyonon
   in those pairs is still the original human text.

   The shared-term test is deliberately narrower than "same cluster". Clusters
   are connected components, and components chain through homographs -- Cuyonon
   'Baba.' is Tagalog 'Bibig.' (mouth) while Tagalog 'Baba.' means chin, which
   welds two unrelated concepts together. Component-based bridging produced 14
   wrong pairs out of 447 on a test corpus; requiring one real shared term
   produced 0 out of 433, losing none of the good ones. Bridged examples are
   counted separately in the summary since they are inferred, not attested.

3. LEAK-FREE SPLIT. A whole cluster goes to train or to val, never both. That
   keeps every direction of a term, and every synonym variant of it, on one
   side. Splitting individual examples instead (the original behaviour) put
   `kain->kaen` in train and `kaen->kain` in val, leaving ~90% of the validation
   set already memorised, so chrF measured recall rather than translation.

4. REBALANCING (train only): ~25% of the raw rows are pure copies (the two
   languages agree) and ~60% are single words. Left alone, "echo the input"
   becomes the most reinforced behaviour, and most of the model's capacity goes
   to single words that the app's dictionary tier answers before the model is
   called. `--max-copy-frac` caps copy rows, `--multi-word-repeat` oversamples
   the sentences that actually teach grammar.

5. ENTITY PASSTHROUGH: `assets/entities/` lists names, places and events that
   are identical in every language, emitted as "term -> itself" so the model
   keeps proper nouns unchanged. These are synthetic and guaranteed-correct, so
   they go to TRAIN ONLY -- scoring them in validation would inflate chrF.

Output: data/train.jsonl and data/val.jsonl, one chat example per line.
"""

import argparse
import collections
import glob
import itertools
import json
import os
import random

from common import build_messages, load_config


def read_corpus(path, columns):
    """Read a `<sep>`-delimited file into (lang_a, text_a, lang_b, text_b) links."""
    lang_a, lang_b = columns
    links = []
    if not os.path.exists(path):
        return links
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "<sep>" not in line:
                continue
            parts = line.split("<sep>")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                links.append((lang_a, a, lang_b, b))
    return links


def read_moses(path, columns, codes):
    """Read an OPUS Moses folder: one plain-text file per language, line-aligned.

    This is the shape the `moses` zips on https://opus.nlpl.eu unpack to, so a
    downloaded corpus can be pointed at directly with no conversion step.
    """
    lang_a, lang_b = columns
    links = []
    if not os.path.isdir(path):
        return links
    a_files = glob.glob(os.path.join(path, f"*.{codes[lang_a]}"))
    b_files = glob.glob(os.path.join(path, f"*.{codes[lang_b]}"))
    if not a_files or not b_files:
        return links
    with open(a_files[0], encoding="utf-8", errors="replace") as fa,             open(b_files[0], encoding="utf-8", errors="replace") as fb:
        for a, b in zip(fa, fb):
            a, b = a.strip(), b.strip()
            if a and b:
                links.append((lang_a, a, lang_b, b))
    return links


def load_entities(entity_dir):
    """Load every non-comment line from every .txt file under entity_dir."""
    entities = []
    if not entity_dir or not os.path.isdir(entity_dir):
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
    """Groups terms linked by at least one translation, across all languages.

    Nodes are (language, normalised text). Uniting the two sides of every link
    puts a whole concept -- all its synonyms, in every language -- into one
    component, and a component is what the train/val split moves around.
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
    ap.add_argument("--config", default=None,
                    help="Path to languages.json (default: repo root).")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of concept CLUSTERS held out for validation.")
    ap.add_argument("--max-per-direction", type=int, default=None,
                    help="Cap on training examples per ordered language pair. This is "
                         "the balance lever when a big reference corpus is in play: "
                         "the base model already knows English and Tagalog from "
                         "pretraining and needs far less data for that pair, whereas "
                         "it has never seen Cuyonon. Without a cap, ~1M subtitle "
                         "pairs make Cuyonon a rounding error and the model forgets "
                         "it. Validation is capped proportionally.")
    ap.add_argument("--max-variants", type=int, default=4,
                    help="Cap on how many synonym variants per language are crossed "
                         "when emitting a direction, so a large cluster cannot flood "
                         "the dataset with one concept.")
    ap.add_argument("--no-bridged", action="store_true",
                    help="Only emit language pairs actually attested in a corpus file. "
                         "Without this, pairs inferred through a shared third language "
                         "are emitted too (that is how English<->Cuyonon is obtained).")
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)

    # ---- 1. Load every corpus -----------------------------------------------
    links = []
    codes = config["name_to_code"]
    for corpus in config["corpora"]:
        if corpus["format"] == "moses":
            found = read_moses(corpus["path"], corpus["columns"], codes)
        else:
            found = read_corpus(corpus["path"], corpus["columns"])
        # A reference corpus can dwarf the hand-built data -- OpenSubtitles alone
        # is ~1.07M pairs against ~2.6k Cuyonon. Sample it down here, before
        # clustering, so it neither swamps the training mix nor blows up memory.
        limit = corpus.get("limit")
        note = ""
        if limit and len(found) > limit:
            random.shuffle(found)
            found = found[:limit]
            note = f" (sampled from {limit}+)"
        state = f"{len(found)} links{note}" if found else "MISSING - skipped"
        print(f"  {corpus['label']:<28} {' <-> '.join(corpus['columns']):<22} {state}")
        links.extend(found)
    if not links:
        raise SystemExit("No parallel data found. Check the `corpora` paths in "
                         "languages.json.")
    print(f"Loaded {len(links)} links across "
          f"{len(config['language_names'])} languages: "
          f"{', '.join(config['language_names'])}")

    # ---- 2. Build concept clusters ------------------------------------------
    def node(lang, text):
        return (lang, text.strip().lower())

    uf = UnionFind()
    adjacency = collections.defaultdict(set)   # node -> nodes it directly translates to
    surface = {}                               # node -> original casing/spelling
    for lang_a, a, lang_b, b in links:
        na, nb = node(lang_a, a), node(lang_b, b)
        uf.union(na, nb)
        adjacency[na].add(nb)
        adjacency[nb].add(na)
        surface.setdefault(na, a)
        surface.setdefault(nb, b)

    clusters = collections.defaultdict(list)   # cluster key -> member nodes
    for n in adjacency:
        clusters[uf.find(n)].append(n)

    cluster_keys = list(clusters)
    random.shuffle(cluster_keys)
    print(f"Grouped into {len(cluster_keys)} concept clusters "
          f"(largest holds {max(len(v) for v in clusters.values())} terms)")

    # ---- 3. Split whole clusters --------------------------------------------
    target_val = int(len(cluster_keys) * args.val_frac)
    val_keys = set(cluster_keys[:target_val])
    train_keys = [k for k in cluster_keys if k not in val_keys]

    # ---- 4. Emit every supported direction ----------------------------------
    counts = collections.Counter()

    def emit(keys, per_direction_cap=None):
        """Emit a pair when the two terms translate each other directly, or when
        they share a specific common translation in a third language.

        The shared-neighbour test is deliberately narrower than "same cluster".
        A cluster is a connected component, and components chain through
        homographs: Cuyonon 'Baba.' is Tagalog 'Bibig.' (mouth) while Tagalog
        'Baba.' means chin, which silently welds two unrelated concepts
        together. Requiring an actual shared term keeps the bridge to one hop
        and drops those.
        """
        out = []
        direction_totals = collections.Counter()
        for key in keys:
            members = clusters[key]
            emitted = collections.Counter()     # (src node, tgt lang) -> count
            for src_node in members:
                src_lang = src_node[0]
                # Walk outward from each node instead of crossing every member
                # against every other. A cluster is a connected component, and at
                # corpus scale one hub string ("Yes.") welds a component tens of
                # thousands of nodes wide -- permutations over that is quadratic
                # and never finishes. Direct neighbours are the attested pairs;
                # neighbours-of-neighbours are the bridged ones. Both are bounded
                # by degree, not by cluster size.
                candidates = []
                for neighbour in adjacency[src_node]:
                    if neighbour[0] != src_lang:
                        candidates.append((neighbour, "direct"))
                if not args.no_bridged:
                    direct = set(adjacency[src_node])
                    for neighbour in adjacency[src_node]:
                        for hop2 in adjacency[neighbour]:
                            if (hop2 != src_node and hop2[0] != src_lang
                                    and hop2 not in direct):
                                candidates.append((hop2, "bridged"))

                for tgt_node, kind in candidates:
                    tgt_lang = tgt_node[0]
                    if emitted[(src_node, tgt_lang)] >= args.max_variants:
                        continue
                    if (per_direction_cap is not None
                            and direction_totals[(src_lang, tgt_lang)] >= per_direction_cap):
                        continue
                    emitted[(src_node, tgt_lang)] += 1
                    direction_totals[(src_lang, tgt_lang)] += 1
                    counts[(src_lang, tgt_lang, kind)] += 1
                    out.append(to_example(src_lang, surface[src_node],
                                          tgt_lang, surface[tgt_node]))
        return out

    cap = args.max_per_direction
    train_examples = emit(train_keys, cap)
    # Keep validation proportional to the same val_frac, so a capped direction is
    # still scored on a held-out slice of comparable size.
    val_examples = emit(val_keys,
                        max(1, int(cap * args.val_frac)) if cap else None)

    print("\nExamples per direction (direct = stated in a file, "
          "bridged = inferred via a shared language)")
    for (src, tgt, kind), n in sorted(counts.items()):
        print(f"  {src:<10} -> {tgt:<10} {kind:<8} {n:6d}")

    # ---- 5. Entity passthrough -> TRAIN ONLY --------------------------------
    entities = load_entities(config.get("entity_path"))
    names = config["language_names"]
    for term in entities:
        for src_lang, tgt_lang in itertools.permutations(names, 2):
            train_examples.append(to_example(src_lang, term, tgt_lang, term))
    print(f"\nLoaded {len(entities)} entities -> "
          f"{len(entities) * len(names) * (len(names) - 1)} passthrough examples "
          f"(train only)")

    # ---- 6. Deduplicate exact examples within each split --------------------
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

    # ---- 7. Rebalance the TRAINING set --------------------------------------
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

    raw_total = max(1, len(train_examples))
    raw_copy_frac = len(copies) / raw_total
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

    # ---- 8. Write ------------------------------------------------------------
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
