"""
lint_data.py
------------
Check the hand-built corpora for the problems that quietly hurt a small model:
inconsistent punctuation and capitals, autocorrect damage, rows that hold two
answers, names that changed between languages, and duplicate rows.

  python scripts/lint_data.py                    # report only
  python scripts/lint_data.py --fix              # also apply the safe fixes
  python scripts/lint_data.py --review data/review/speaker-review.csv

--fix only touches what a machine can decide: spacing, doubled punctuation,
curly quotes, a final mark missing on one side (copied from the other side), a
lowercase first letter, and exact duplicate rows. Everything that needs a
person -- a question translated as a statement, a capital in mid-sentence, a
name spelled differently, an English word inside Cuyonon -- is reported, and
written to the --review sheet (opens in Excel / Google Sheets) for a native
speaker to settle.

It reads every `<sep>` file listed under `corpora` and `test` in languages.json.
Run it after adding rows, before prepare_data.py.
"""

import argparse
import collections
import csv
import os
import re

from common import (contains_term, load_config, load_entities, loose_key,
                    term_pattern)

LETTER = r"[^\W\d_]"


# ---- Safe, mechanical fixes ---------------------------------------------------
def fix_text(s):
    """Spacing, quotes and doubled final punctuation. Never changes words."""
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[·•]", " ", s)
    s = " ".join(s.split())
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)          # "meting ?" -> "meting?"
    s = re.sub(r",(?=" + LETTER + ")", ", ", s)      # "ko,pero" -> "ko, pero"
    s = re.sub(r"\s*/\s*", "/", s)                   # "ma'am/ sir" -> "ma'am/sir"
    s = re.sub(r"([?!])\.+$", r"\1", s)              # "?." -> "?"
    s = re.sub(r"(^|[^.])\.([?!])$", r"\1\2", s)     # ".?" -> "?"  (not "...")
    s = re.sub(r"([?!])\1+$", r"\1", s)              # "??" -> "?"
    s = re.sub(r"(^|[^.])\.\.$", r"\1.", s)          # ".." -> "."  (keeps "...")
    return s.strip()


def end_mark(s):
    return s[-1] if s and s[-1] in ".?!" else ""


def cap_first(s):
    """Uppercase the first letter, allowing only leading punctuation ('yong)."""
    return re.sub(r"^([^\w\s]*)(" + LETTER + ")",
                  lambda m: m.group(1) + m.group(2).upper(), s, count=1)


def fix_pair(left, right):
    left, right = cap_first(fix_text(left)), cap_first(fix_text(right))
    ok = lambda s: not re.search(r"[,;:)\"']$", s)
    if end_mark(left) and not end_mark(right) and ok(right):
        right += end_mark(left)
    elif end_mark(right) and not end_mark(left) and ok(left):
        left += end_mark(right)
    return left, right


# ---- Checks that need a person --------------------------------------------------
def words(s):
    return s.split()


def bare(token):
    return re.sub(r"^\W+|\W+$", "", token)


def mid_caps(s):
    """Capitalised words that are not sentence-initial (and not after . ? ! :)."""
    toks = words(s)
    found = []
    for i in range(1, len(toks)):
        if toks[i - 1][-1:] in ".?!:" or toks[i][:1] in "\"'([":
            continue
        w = bare(toks[i])
        if re.fullmatch(r"[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+", w):
            found.append(w)
    return found


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fix", action="store_true",
                    help="Rewrite the files with the safe, mechanical fixes applied.")
    ap.add_argument("--review", default=None, metavar="CSV",
                    help="Write every item that needs a native speaker to this CSV.")
    ap.add_argument("--show", type=int, default=5,
                    help="Examples printed per issue type.")
    args = ap.parse_args()

    config = load_config(args.config)
    entities = load_entities(config.get("entity_path"))
    all_terms = [t for terms in entities.values() for t in terms]
    entity_tokens = {tok for t in all_terms for tok in t.split()}

    files = [c for c in config["corpora"] + config["test"] if c["format"] == "sep"]
    # Languages that only ever appear in your own files are the low-resource
    # ones -- spelling consistency matters most there.
    in_reference = {lang for c in config["corpora"] if c["train_only"] for lang in c["columns"]}
    own_only = [l for l in config["language_names"] if l not in in_reference]

    rows = []            # (corpus, line number, left, right)
    raw_lines = {}       # path -> list of lines (for --fix)
    for corpus in files:
        if not os.path.isfile(corpus["path"]):
            continue
        with open(corpus["path"], encoding="utf-8-sig") as f:
            lines = f.read().split("\n")
        raw_lines[corpus["path"]] = lines
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("<sep>")
            if len(parts) != 2:
                rows.append((corpus, i, stripped, None))
                continue
            rows.append((corpus, i, parts[0].strip(), parts[1].strip()))

    # Lowercase vocabulary per language: evidence that a capitalised word in
    # mid-sentence is an ordinary word, not a name.
    lower_vocab = collections.defaultdict(collections.Counter)
    for corpus, _, left, right in rows:
        if right is None:
            continue
        for lang, text in zip(corpus["columns"], (left, right)):
            for tok in set(words(text)):
                w = bare(tok)
                if w and w == w.lower() and re.search(LETTER, w):
                    lower_vocab[lang][w] += 1

    auto = collections.defaultdict(list)       # issue -> [description]
    review = []                                # (corpus, line, left, right, issue)
    seen_exact = {}
    by_loose = collections.defaultdict(list)

    for corpus, line, left, right in rows:
        label = f"{os.path.basename(corpus['path'])}:{line}"
        if right is None:
            review.append((corpus, line, left, "", "malformed: needs exactly one <sep>"))
            continue
        lang_l, lang_r = corpus["columns"]
        fixed = fix_pair(left, right)
        if fixed != (left, right):
            auto["mechanical fix"].append(f"{label}: {left}<sep>{right}  ->  {fixed[0]}<sep>{fixed[1]}")
        key = (corpus["path"], left, right)
        if key in seen_exact:
            auto["exact duplicate"].append(f"{label}: same as line {seen_exact[key]}")
        else:
            seen_exact[key] = line
        by_loose[(corpus["path"], loose_key(left), loose_key(right))].append((line, left, right))

        multi = len(words(left)) > 1
        el, er = end_mark(left), end_mark(right)
        if el and er and el != er:
            review.append((corpus, line, left, right,
                           f"sentence type differs: {lang_l} ends '{el}', {lang_r} ends '{er}'"))
        if multi and loose_key(left) == loose_key(right):
            review.append((corpus, line, left, right,
                           f"{lang_r} identical to {lang_l} -- untranslated, or really the same?"))
        for a_text, b_text, lang_b in ((left, right, lang_r), (right, left, lang_l)):
            if re.search(r"[.?!]\s+\S", b_text) and not re.search(r"[.?!]\s+\S", a_text):
                review.append((corpus, line, left, right,
                               f"{lang_b} holds two sentences -- two answers in one row?"))
            if re.search(r"[/()]", b_text) and not re.search(r"[/()]", a_text):
                review.append((corpus, line, left, right,
                               f"slash or brackets only in the {lang_b} -- alternatives in one row?"))
        # Capitals: a word capitalised mid-sentence on one side only.
        caps_l, caps_r = mid_caps(left), mid_caps(right)
        shared = set(caps_l) & set(caps_r)
        for text, caps, other_caps, lang in ((left, caps_l, caps_r, lang_l),
                                             (right, caps_r, caps_l, lang_r)):
            unexplained = [w for w in caps if w not in entity_tokens and w not in shared]
            other_unexplained = [w for w in other_caps if w not in entity_tokens and w not in shared]
            for w in unexplained:
                if not other_unexplained and lower_vocab[lang][w.lower()] >= 1:
                    review.append((corpus, line, left, right,
                                   f"stray capital '{w}' in the {lang}? (written lowercase elsewhere)"))
        # Names: a known entity missing on one side, or respelled.
        for term in all_terms:
            pattern = term_pattern(term)
            if pattern.search(left) and not re.search(pattern.pattern, right, re.IGNORECASE):
                review.append((corpus, line, left, right, f"name '{term}' is missing from the {lang_r}"))
        for w in caps_l:
            if contains_term(right.lower(), w.lower()) or lower_vocab[lang_l][w.lower()] >= 2:
                continue
            twins = [bare(t) for t in words(right)
                     if bare(t)[:1].isupper() and bare(t) != w
                     and levenshtein(bare(t).lower(), w.lower()) <= 2]
            if twins:
                review.append((corpus, line, left, right,
                               f"name '{w}' became '{twins[0]}' -- names stay unchanged; which spelling?"))

    # Near-duplicates whose punctuation disagrees.
    for (path, _, _), group in by_loose.items():
        variants = {(l, r) for _, l, r in group}
        marks = {(end_mark(l), end_mark(r)) for l, r in variants}
        if len(variants) > 1 and any(a != b for a, b in marks):
            corpus = next(c for c in files if c["path"] == path)
            first = group[0]
            review.append((corpus, first[0], first[1], first[2],
                           "versions of this row disagree on punctuation: "
                           + " | ".join(f"{l}<sep>{r}" for l, r in sorted(variants))))

    # Pivot alignment: bridging needs the shared language spelled identically
    # in every file. Same words with different inner punctuation or spacing
    # silently stops 'Tagalog in file A' meeting 'Tagalog in file B'.
    loose = collections.defaultdict(dict)
    for corpus, _, left, right in rows:
        if right is None:
            continue
        for lang, text in zip(corpus["columns"], (left, right)):
            loose[lang].setdefault(loose_key(text), set()).add(text)
    for lang, groups in loose.items():
        for spellings in groups.values():
            lowered = {s.lower() for s in spellings}
            if len(lowered) < 2:
                continue          # case-only variants are already the same sentence
            if len({s.rstrip(".?!").lower() for s in spellings}) == 1:
                continue          # 'Salamat.' vs 'Salamat!' are different sentences
            review.append((None, 0, lang, " | ".join(sorted(spellings)),
                           f"{lang} pivot spelled differently between rows -- bridging "
                           f"through it breaks; make the rows match"))

    # Spelling families in low-resource languages: one vowel swapped, both used.
    for lang in own_only:
        vocab = lower_vocab[lang]
        pairs = set()
        for w, n in vocab.items():
            if n < 2 or len(w) < 3:
                continue
            for x, y in (("e", "i"), ("o", "u"), ("e", "o")):
                for i, ch in enumerate(w):
                    if ch == x:
                        v = w[:i] + y + w[i + 1:]
                        if vocab.get(v, 0) >= 2:
                            pairs.add(tuple(sorted((w, v))))
        for w, v in sorted(pairs, key=lambda p: -(vocab[p[0]] + vocab[p[1]])):
            review.append((None, 0, w, v,
                           f"{lang} spelling: '{w}' ({vocab[w]} rows) vs '{v}' ({vocab[v]} rows) -- "
                           f"same word? pick one spelling, or keep both if both are real"))

    # ---- Report ---------------------------------------------------------------
    print(f"Checked {len(rows)} rows in {len(raw_lines)} files.\n")
    if not auto and not review:
        print("No issues found.")
    for issue, items in auto.items():
        print(f"[fixable] {issue}: {len(items)}")
        for item in items[:args.show]:
            print(f"    {item}")
    def kind(text):
        # "stray capital 'Ren' in ..." -> "stray capital"; "sentence type differs: ..." -> ...
        return re.split(r":| '| \d+ rows| --", text, maxsplit=1)[0]

    counts = collections.Counter(kind(item[4]) for item in review)
    for issue, n in counts.most_common():
        print(f"[needs a speaker] {issue}: {n}")
        for corpus, line, left, right, text in [r for r in review if kind(r[4]) == issue][:args.show]:
            where = f"{os.path.basename(corpus['path'])}:{line}" if corpus else ""
            print(f"    {where} {left}<sep>{right}  -- {text}")

    if args.review:
        os.makedirs(os.path.dirname(args.review) or ".", exist_ok=True)
        # utf-8-sig so Excel shows accented letters correctly.
        with open(args.review, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "line", "left", "right", "question",
                             "speaker answer (fix the row in the file, or write OK)"])
            for corpus, line, left, right, text in review:
                writer.writerow([os.path.relpath(corpus["path"]) if corpus else "",
                                 line or "", left, right, text, ""])
        print(f"\nWrote {len(review)} review items -> {args.review}")

    if args.fix:
        changed_files = 0
        for path, lines in raw_lines.items():
            out, seen, changed = [], set(), False
            for line in lines:
                stripped = line.strip()
                parts = stripped.split("<sep>")
                if not stripped or stripped.startswith("#") or len(parts) != 2:
                    out.append(line.rstrip("\r"))
                    continue
                left, right = fix_pair(parts[0].strip(), parts[1].strip())
                new = f"{left}<sep>{right}"
                if new in seen:
                    changed = True
                    continue
                seen.add(new)
                changed |= new != line.rstrip("\r")
                out.append(new)
            if changed:
                text = "\n".join(out).rstrip("\n") + "\n"
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                changed_files += 1
        print(f"\n--fix: rewrote {changed_files} file(s). Review the change with `git diff`.")


if __name__ == "__main__":
    main()
