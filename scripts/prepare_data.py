"""
prepare_data.py
---------------
Turn the parallel corpora listed in `languages.json` into an instruction-style
(chat) dataset for supervised fine-tuning (SFT) of a decoder-only LLM.

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
   chain through homographs -- Cuyonon 'Baba.' is Tagalog 'Bibig.' (mouth)
   while Tagalog 'Baba.' means chin -- so bridging only crosses one real shared
   sentence. Bridged examples are counted separately in the summary.

3. LEAK-FREE, STABLE SPLIT. Your own corpora are grouped into concept clusters
   (every spelling, synonym and direction of a term) and each cluster goes to
   train, validation or test as a whole. Two things keep that split honest
   across runs:
     * Punctuation and case variants of a sentence are welded into one cluster,
       so 'Paalam po' and 'Paalam po!' can never straddle the split.
     * A cluster's side is decided by a hash of its contents, not by shuffling.
       Adding data no longer reshuffles the validation set -- shuffling moved
       ~80% of it when a single sentence was added -- so scores from before and
       after a data change can be compared.

4. FROZEN TEST SET. Files listed under `test` in languages.json are written to
   data/test.jsonl exactly as they are, and any cluster touching a test
   sentence is kept out of training and validation. Fill that file once (a
   native speaker, real app sentences) and never train on it.

5. REFERENCE CORPORA ARE TRAINING-ONLY. Big downloaded corpora (Moses folders,
   or any corpus marked "train_only") never enter validation or test, and any
   of their pairs touching a held-out sentence is dropped. Each is sampled down
   to its `limit` with a stable per-corpus sample, so one corpus changing does
   not resample the others.

6. SYNTHETIC CORPORA ("synthetic": true -- e.g. back-translated text) are
   emitted one way only, machine-made column -> human column, and are never
   bridged: the model learns to WRITE only human text.

7. REBALANCING (train only). ~25% of the raw rows are copies (both languages
   agree), which over-teaches echoing, so `--max-copy-frac` caps them.
   `--multi-word-repeat` oversamples sentences from YOUR corpora only; repeating
   subtitle lines too would drown Cuyonon further.

8. NAMES. Entity passthrough ("Maria" -> "Maria") as before, plus names in
   context: sentences that keep a place unchanged on both sides are reused with
   other places swapped in, and a share of name-bearing examples carry the
   "Keep these terms unchanged" line, so translate.py --keep sends a prompt the
   model has actually seen. All of this is training-only.

Output: data/train.jsonl, data/val.jsonl and data/test.jsonl, one chat example
per line.
"""

import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import random

from common import (build_messages, contains_term, load_config, load_entities,
                    loose_key, parse_user_turn, read_sep_file, term_pattern)


def read_corpus(path, columns):
    """Read a `<sep>` file into (lang_a, text_a, lang_b, text_b) links."""
    lang_a, lang_b = columns
    return [(lang_a, a, lang_b, b) for a, b in read_sep_file(path)]


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
    with open(a_files[0], encoding="utf-8", errors="replace") as fa, \
            open(b_files[0], encoding="utf-8", errors="replace") as fb:
        for a, b in zip(fa, fb):
            a, b = a.strip(), b.strip()
            if a and b:
                links.append((lang_a, a, lang_b, b))
    return links


def read_any(corpus, codes):
    if corpus["format"] == "moses":
        return read_moses(corpus["path"], corpus["columns"], codes)
    return read_corpus(corpus["path"], corpus["columns"])


class UnionFind:
    """Groups sentences linked by at least one translation, across all languages."""

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


def stable_unit(*parts):
    """A deterministic pseudo-random number in [0, 1) for the given strings.

    Python's built-in hash() is salted per process, so it cannot drive anything
    that has to come out the same on every run.
    """
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def node(lang, text):
    """Graph node for a sentence. Case-insensitive, so 'Aso.' and 'aso.' meet."""
    return (lang, " ".join(text.lower().split()))


def loose_node(lang, text):
    """Welding node: one per sentence regardless of punctuation and case."""
    return ("~", lang, loose_key(text))


class Graph:
    """Sentences as nodes, translations as edges.

    `pairs` keeps every spelling pair actually written for an edge, so a direct
    example reproduces the file exactly -- including case that carries meaning
    ('Sa susunod na Linggo.' = next Sunday, '... linggo.' = next week). Bridged
    examples have no written pair and use the first spelling seen (`surface`).
    """

    def __init__(self):
        self.adj = collections.defaultdict(set)
        self.surface = {}
        self.pairs = collections.defaultdict(list)
        self.own = set()            # nodes that come from your own corpora

    def add(self, lang_a, a, lang_b, b, own):
        na, nb = node(lang_a, a), node(lang_b, b)
        self.adj[na].add(nb)
        self.adj[nb].add(na)
        self.surface.setdefault(na, a)
        self.surface.setdefault(nb, b)
        if (a, b) not in self.pairs[(na, nb)]:
            self.pairs[(na, nb)].append((a, b))
            self.pairs[(nb, na)].append((b, a))
        if own:
            self.own.update((na, nb))


def to_example(src_lang, src_text, tgt_lang, tgt_text, keep_terms=None):
    """Build one chat-format example using the shared system prompt."""
    return {"messages": build_messages(src_text, src_lang, tgt_lang,
                                       keep_terms=keep_terms, assistant=tgt_text)}


def is_copy(src_text, tgt_text):
    return src_text.strip().lower() == tgt_text.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Path to languages.json (default: repo root).")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Share of concept CLUSTERS from your own corpora held out for "
                         "validation. Assignment is by hash, so it is stable as data grows.")
    ap.add_argument("--max-per-direction", type=int, default=None,
                    help="Optional cap on training examples per ordered language pair. "
                         "Per-corpus `limit` in languages.json is the main balance lever "
                         "for reference corpora; this is a second, blunter one. "
                         "Validation is capped proportionally.")
    ap.add_argument("--max-variants", type=int, default=4,
                    help="Cap on how many targets per language one source sentence gets, "
                         "so a large cluster cannot flood the dataset with one concept.")
    ap.add_argument("--no-bridged", action="store_true",
                    help="Only emit language pairs actually attested in a corpus file. "
                         "Without this, pairs inferred through a shared third language "
                         "are emitted too (that is how English<->Cuyonon is obtained).")
    ap.add_argument("--multi-word-repeat", type=int, default=2,
                    help="How many times to repeat each multi-word (sentence) example from "
                         "YOUR corpora. These are the rows that teach grammar and they are "
                         "the minority. Reference and synthetic rows are never repeated. "
                         "1 = no oversampling.")
    ap.add_argument("--max-copy-frac", type=float, default=0.10,
                    help="Cap on the share of TRAINING rows that are pure copies "
                         "(input identical to output, including entity passthrough). "
                         "Raw data is ~25%% copies, which over-teaches echoing. "
                         "Use 1.0 to disable the cap.")
    ap.add_argument("--entity-swaps", type=int, default=3,
                    help="For each training sentence that keeps a place name unchanged on "
                         "both sides, add this many copies with other places from "
                         "assets/entities/places.txt swapped in. 0 disables.")
    ap.add_argument("--keep-frac", type=float, default=0.5,
                    help="Share of name-bearing synthetic examples that carry the "
                         "'Keep these terms unchanged' line used by translate.py --keep.")
    ap.add_argument("--min-own-share", type=float, default=0.30,
                    help="Warn when fewer than this share of training examples come from "
                         "your own corpora (the rest being reference/synthetic data).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Salts the split hash and the shuffles. Changing it gives a "
                         "different (but again stable) split.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seed = str(args.seed)
    config = load_config(args.config)
    codes = config["name_to_code"]

    # ---- 1. Load every corpus -----------------------------------------------
    own_links, ref_links, synthetic_links, test_links = [], [], [], []
    for corpus in config["corpora"]:
        found = read_any(corpus, codes)
        total = len(found)
        note = ""
        limit = corpus.get("limit")
        if limit and total > limit:
            # Rank by a hash of the pair rather than shuffling: this corpus's
            # sample does not change when another corpus does, and adding lines
            # to it only swaps a few rows in or out.
            found.sort(key=lambda link: stable_unit(seed, corpus["label"], link[1], link[3]))
            found = found[:limit]
            note = f" (sampled {limit} of {total})"
        if corpus["synthetic"]:
            kind, bucket = "synthetic", synthetic_links
        elif corpus["train_only"]:
            kind, bucket = "reference", ref_links
        else:
            kind, bucket = "own", own_links
        bucket.extend(found)
        state = f"{len(found)} links{note}" if found else "MISSING - skipped"
        print(f"  {corpus['label']:<28} {' <-> '.join(corpus['columns']):<22} "
              f"{kind:<9} {state}")
    for corpus in config["test"]:
        found = read_any(corpus, codes)
        test_links.extend(found)
        state = f"{len(found)} links" if found else "empty - evaluation falls back to validation"
        print(f"  {corpus['label']:<28} {' <-> '.join(corpus['columns']):<22} "
              f"{'test':<9} {state}")
    if not (own_links or ref_links or synthetic_links):
        raise SystemExit("No parallel data found. Check the `corpora` paths in "
                         "languages.json.")
    print(f"Loaded {len(own_links)} own, {len(ref_links)} reference, "
          f"{len(synthetic_links)} synthetic and {len(test_links)} test links across "
          f"{len(config['language_names'])} languages: "
          f"{', '.join(config['language_names'])}")

    # ---- 2. Cluster your own data together with the test set ----------------
    uf = UnionFind()
    for lang_a, a, lang_b, b in own_links + test_links:
        na, nb = node(lang_a, a), node(lang_b, b)
        uf.union(na, nb)
        # Weld punctuation/case variants of a sentence into one cluster so they
        # can never straddle the split.
        for lang, text, n in ((lang_a, a, na), (lang_b, b, nb)):
            if loose_key(text):
                uf.union(n, loose_node(lang, text))

    members = collections.defaultdict(set)         # cluster root -> sentence nodes
    for lang_a, a, lang_b, b in own_links + test_links:
        for n in (node(lang_a, a), node(lang_b, b)):
            members[uf.find(n)].add(n)

    test_nodes = set()
    for lang_a, a, lang_b, b in test_links:
        test_nodes.update((node(lang_a, a), node(lang_b, b)))

    def cluster_name(nodes):
        # Named after its alphabetically-first member in loose form, so edits
        # that only touch punctuation or case leave the cluster where it was.
        return min(f"{lang}\x1f{loose_key(text) or text}" for lang, text in nodes)

    split_of = {}                                  # sentence node -> split
    clusters_per_split = collections.Counter()
    for nodes in members.values():
        if nodes & test_nodes:
            split = "test"
        elif stable_unit("val", seed, cluster_name(nodes)) < args.val_frac:
            split = "val"
        else:
            split = "train"
        clusters_per_split[split] += 1
        for n in nodes:
            split_of[n] = split
    largest = max((len(v) for v in members.values()), default=0)
    print(f"Grouped your data into {len(members)} concept clusters "
          f"(largest holds {largest} sentences): {clusters_per_split['train']} train, "
          f"{clusters_per_split['val']} validation, "
          f"{clusters_per_split['test']} touching the test set")

    # ---- 3. One graph per split ----------------------------------------------
    graphs = {"train": Graph(), "val": Graph()}
    excluded_for_test = 0
    for lang_a, a, lang_b, b in own_links:
        split = split_of[node(lang_a, a)]
        if split == "test":
            excluded_for_test += 1
            continue
        graphs[split].add(lang_a, a, lang_b, b, own=True)

    held_out = {loose_node(lang, text) for (lang, text), split in split_of.items()
                if split != "train"}

    def touches_held_out(lang_a, a, lang_b, b):
        return loose_node(lang_a, a) in held_out or loose_node(lang_b, b) in held_out

    dropped_ref = 0
    for lang_a, a, lang_b, b in ref_links:
        if touches_held_out(lang_a, a, lang_b, b):
            dropped_ref += 1
            continue
        graphs["train"].add(lang_a, a, lang_b, b, own=False)
    if excluded_for_test or dropped_ref:
        print(f"Held out: {excluded_for_test} of your links overlap the test set; "
              f"{dropped_ref} reference links touched a validation/test sentence and "
              f"were dropped")

    # ---- 4. Emit every supported direction ----------------------------------
    def emit(graph, counts, per_direction_cap=None):
        """Direct pairs exactly as written, plus one-hop bridged pairs through a
        shared third-language sentence. Returns [(example, provenance)]."""
        out = []
        direction_totals = collections.Counter()
        adjacency = {n: sorted(neighbours) for n, neighbours in graph.adj.items()}
        # Stable pseudo-random visiting order: when a cap cuts the list short, the
        # kept examples are a fair sample, and the same sample on every run.
        order = sorted(adjacency, key=lambda n: stable_unit("order", seed, n[0], n[1]))
        for src in order:
            src_lang = src[0]
            candidates = []          # (src text, target node, target text, kind)
            for nb in adjacency[src]:
                if nb[0] != src_lang:
                    for src_text, tgt_text in graph.pairs[(src, nb)]:
                        candidates.append((src_text, nb, tgt_text, "direct"))
            if not args.no_bridged:
                direct = graph.adj[src]
                seen = set()
                for nb in adjacency[src]:
                    for hop2 in adjacency[nb]:
                        if (hop2 == src or hop2[0] == src_lang or hop2 in direct
                                or hop2 in seen):
                            continue
                        seen.add(hop2)
                        candidates.append((graph.surface[src], hop2,
                                           graph.surface[hop2], "bridged"))
            emitted = collections.Counter()  # target language -> examples so far
            for src_text, tgt, tgt_text, kind in candidates:
                tgt_lang = tgt[0]
                if emitted[tgt_lang] >= args.max_variants:
                    continue
                if (per_direction_cap is not None
                        and direction_totals[(src_lang, tgt_lang)] >= per_direction_cap):
                    continue
                emitted[tgt_lang] += 1
                direction_totals[(src_lang, tgt_lang)] += 1
                counts[(src_lang, tgt_lang, kind)] += 1
                provenance = "own" if (src in graph.own and tgt in graph.own) else "reference"
                out.append((to_example(src_lang, src_text, tgt_lang, tgt_text), provenance))
        return out

    cap = args.max_per_direction
    train_counts, val_counts = collections.Counter(), collections.Counter()
    train_items = emit(graphs["train"], train_counts, cap)
    val_items = emit(graphs["val"], val_counts,
                     max(1, int(cap * args.val_frac)) if cap else None)

    # Synthetic corpora: one way only, never bridged, never held out.
    dropped_synthetic = 0
    for lang_a, a, lang_b, b in synthetic_links:
        if touches_held_out(lang_a, a, lang_b, b):
            dropped_synthetic += 1
            continue
        train_items.append((to_example(lang_a, a, lang_b, b), "synthetic"))
        train_counts[(lang_a, lang_b, "synthetic")] += 1
    if dropped_synthetic:
        print(f"Dropped {dropped_synthetic} synthetic links that touched a held-out sentence")

    test_examples = []
    for lang_a, a, lang_b, b in test_links:
        test_examples.append(to_example(lang_a, a, lang_b, b))
        test_examples.append(to_example(lang_b, b, lang_a, a))

    def show(title, counts):
        print(f"\n{title} (direct = stated in a file, bridged = inferred via a "
              f"shared language)")
        for (src, tgt, kind), n in sorted(counts.items()):
            print(f"  {src:<10} -> {tgt:<10} {kind:<9} {n:6d}")

    show("Training examples per direction", train_counts)
    show("Validation examples per direction", val_counts)

    # ---- 5. Names: passthrough, names in context, the --keep line -----------
    entities = load_entities(config.get("entity_path"))
    all_terms = [term for terms in entities.values() for term in terms]
    patterns = {term: term_pattern(term) for term in all_terms}
    languages = config["language_names"]

    passthrough = keep_lines = 0
    for term in all_terms:
        for src_lang, tgt_lang in itertools.permutations(languages, 2):
            train_items.append((to_example(src_lang, term, tgt_lang, term), "synthetic"))
            passthrough += 1
            if stable_unit("keep", seed, term, src_lang, tgt_lang) < args.keep_frac:
                train_items.append((to_example(src_lang, term, tgt_lang, term,
                                               keep_terms=[term]), "synthetic"))
                keep_lines += 1

    places = entities.get("places", [])
    in_context = swapped = 0
    for lang_a, a, lang_b, b in own_links:
        if split_of[node(lang_a, a)] != "train" or len(a.split()) < 2:
            continue
        kept = [t for t in all_terms if patterns[t].search(a) and patterns[t].search(b)]
        # 'Puerto' inside 'Puerto Princesa' is not a separate name.
        kept = [t for t in kept if not any(t != u and contains_term(u, t) for u in kept)]
        if not kept:
            continue
        # The real sentence once more, with the keep line, in both directions.
        for sl, st, tl, tt in ((lang_a, a, lang_b, b), (lang_b, b, lang_a, a)):
            train_items.append((to_example(sl, st, tl, tt, keep_terms=kept), "synthetic"))
            in_context += 1
        # Swap each place for other places. The sentence frame stays human-written
        # and places carry no grammatical gender, so the result is still correct.
        for term in [t for t in kept if t in places]:
            options = sorted(
                (p for p in places
                 if p != term and not contains_term(a, p) and not contains_term(b, p)),
                key=lambda p: stable_unit("swap", seed, a, b, term, p))
            for new in options[:args.entity_swaps]:
                swap_a = patterns[term].sub(lambda match: new, a)
                swap_b = patterns[term].sub(lambda match: new, b)
                for sl, st, tl, tt in ((lang_a, swap_a, lang_b, swap_b),
                                       (lang_b, swap_b, lang_a, swap_a)):
                    keep = [new] if stable_unit("swapkeep", seed, st, sl) < args.keep_frac else None
                    train_items.append((to_example(sl, st, tl, tt, keep_terms=keep),
                                        "synthetic"))
                    swapped += 1
    print(f"\nNames (training only): {len(all_terms)} entities -> {passthrough} "
          f"passthrough + {keep_lines} with the keep line; {in_context} real sentences "
          f"with the keep line; {swapped} place-swapped sentences")

    # ---- 6. Deduplicate exact examples within each split --------------------
    def dedupe(items):
        best, order = {}, []
        for example, provenance in items:
            key = json.dumps(example["messages"], ensure_ascii=False)
            if key not in best:
                best[key] = [example, provenance]
                order.append(key)
            elif provenance == "own":
                best[key][1] = "own"
        return [tuple(best[key]) for key in order]

    train_items = dedupe(train_items)
    val_items = dedupe(val_items)
    test_examples = [ex for ex, _ in dedupe([(ex, "test") for ex in test_examples])]

    # ---- 7. Rebalance the TRAINING set --------------------------------------
    # Validation is deliberately left alone: it has to keep the natural
    # distribution so scores stay comparable across runs. evaluate.py segments
    # it at scoring time instead.
    def source_and_target(example):
        msgs = {m["role"]: m["content"] for m in example["messages"]}
        return parse_user_turn(msgs["user"])[2], msgs["assistant"]

    copies, singles, multis = [], [], []
    for example, provenance in train_items:
        source, target = source_and_target(example)
        if is_copy(source, target):
            copies.append((example, provenance))
        elif len(source.split()) == 1:
            singles.append((example, provenance))
        else:
            multis.append((example, provenance))

    repeat = max(1, args.multi_word_repeat)
    kept_multis = []
    for example, provenance in multis:
        kept_multis.extend([(example, provenance)] * (repeat if provenance == "own" else 1))

    # Cap copies as a share of the FINAL training size:
    #   frac = copies / (copies + others)  ->  copies = frac/(1-frac) * others
    others = len(singles) + len(kept_multis)
    if args.max_copy_frac >= 1.0:
        kept_copies = copies
    else:
        allowed = int(args.max_copy_frac / (1 - args.max_copy_frac) * others)
        rng.shuffle(copies)
        kept_copies = copies[:allowed]

    final = kept_copies + singles + kept_multis
    rng.shuffle(final)
    total = max(1, len(final))
    raw_total = max(1, len(train_items))

    print("\nTraining mix (raw -> after rebalancing)")
    print(f"  copy rows       : {len(copies):6d} ({100 * len(copies) / raw_total:4.1f}%) -> "
          f"{len(kept_copies):6d} ({100 * len(kept_copies) / total:4.1f}%)")
    print(f"  single-word rows: {len(singles):6d} -> {len(singles):6d} "
          f"({100 * len(singles) / total:4.1f}%)")
    print(f"  multi-word rows : {len(multis):6d} -> {len(kept_multis):6d} "
          f"({100 * len(kept_multis) / total:4.1f}%)  [own corpora x{repeat}]")

    involvement = collections.Counter()
    by_provenance = collections.Counter()
    for example, provenance in final:
        src_lang, tgt_lang, _ = parse_user_turn(example["messages"][1]["content"])
        involvement.update({src_lang, tgt_lang})
        by_provenance[provenance] += 1
    print("\nShare of training examples involving each language:")
    for lang in languages:
        print(f"  {lang:<10} {100 * involvement[lang] / total:5.1f}%")
    own_share = by_provenance["own"] / total
    print(f"From your own corpora: {100 * own_share:.1f}%   reference: "
          f"{100 * by_provenance['reference'] / total:.1f}%   synthetic: "
          f"{100 * by_provenance['synthetic'] / total:.1f}%")
    if own_share < args.min_own_share:
        print(f"\nWARNING: only {100 * own_share:.1f}% of training examples come from your "
              f"own corpora. The reference corpora will dominate what the model learns. "
              f"Lower their `limit` in languages.json, or pass --max-per-direction.")

    # ---- 8. Write ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    outputs = [("train", [ex for ex, _ in final]),
               ("val", [ex for ex, _ in val_items]),
               ("test", test_examples)]
    for name, split in outputs:
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for example in split:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split):6d} examples -> {out_path}")
    if not val_items:
        print("WARNING: the validation set is empty; train.py needs at least one row.")

    print("\nDone. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
