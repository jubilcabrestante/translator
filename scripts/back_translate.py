"""
back_translate.py
-----------------
Turn text written only in Cuyonon (or any language) into extra training pairs:
back-translation, the standard trick when parallel text is scarce.

Collect sentences people wrote in Cuyonon -- Facebook pages, local news,
church bulletins, song lyrics, stories. This script translates each one into
Tagalog with your current model and writes

    <machine Tagalog><sep><human Cuyonon>

Training then teaches machine Tagalog -> human Cuyonon, so the side the model
has to learn to WRITE is always human text. prepare_data.py emits a corpus
marked "synthetic" in that one direction only, never bridges through it and
never scores on it. Text written only in Cuyonon is far easier to find than
parallel text, so this is how the sentence data can grow fastest.

Usage:
  python scripts/back_translate.py --input monolingual/cuyonon.txt
  python scripts/back_translate.py --input monolingual/cuyonon.txt --lang Cuyonon --via Tagalog --limit 500

Input: plain text, one sentence per line (blank lines and lines starting with
'#' are skipped). Safe to stop and re-run: sentences already in --out are
skipped.

Then add the printed snippet to `corpora` in languages.json, for example:
  { "file": "assets/synthetic/bt-cuyonon.txt", "columns": ["Tagalog", "Cuyonon"],
    "synthetic": true, "limit": 3000 }

Start with a `limit` no bigger than your human sentence data, and compare
evaluate.py before and after. If the score drops, lower it.
"""

import argparse
import os

from common import load_config, read_sep_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Plain-text file, one sentence per line, written in --lang.")
    ap.add_argument("--lang", default="Cuyonon",
                    help="Language the input is written in (the human side).")
    ap.add_argument("--via", default="Tagalog",
                    help="Language to translate it into (the machine side).")
    ap.add_argument("--out", default=None,
                    help="Default: assets/synthetic/bt-<lang>.txt")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-beams", type=int, default=4,
                    help="Beam search gives cleaner synthetic sources than greedy.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only back-translate the first N new sentences.")
    ap.add_argument("--max-words", type=int, default=40,
                    help="Skip longer lines; training truncates at --max-seq-len anyway.")
    args = ap.parse_args()

    config = load_config()
    for lang in (args.lang, args.via):
        if lang not in config["name_to_code"]:
            raise SystemExit(f"{lang!r} is not in languages.json.")
    if args.lang == args.via:
        raise SystemExit("--lang and --via must differ.")
    out = args.out or os.path.join("assets", "synthetic", f"bt-{args.lang.lower()}.txt")

    done = {human for _, human in read_sep_file(out)}
    todo, queued = [], set()
    with open(args.input, encoding="utf-8-sig") as f:
        for line in f:
            text = " ".join(line.split())
            if not text or text.startswith("#") or "<sep>" in text:
                continue
            if len(text.split()) > args.max_words or text in done or text in queued:
                continue
            todo.append(text)
            queued.add(text)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} new sentences to back-translate ({len(done)} already in {out})")
    if not todo:
        return

    from translate import load, translate   # imports torch
    model, tokenizer = load(args.base_model, args.adapter_dir, device=args.device)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    new_file = not os.path.exists(out)
    written = 0
    with open(out, "a", encoding="utf-8") as f:
        if new_file:
            f.write(f"# {args.via} | {args.lang} -- BACK-TRANSLATED by scripts/back_translate.py.\n"
                    f"# The {args.via} side is machine-made; the {args.lang} side is human text.\n"
                    f"# List this file in languages.json with \"synthetic\": true.\n")
        for i, human in enumerate(todo, 1):
            machine = " ".join(translate(model, tokenizer, human, args.lang, args.via,
                                         num_beams=args.num_beams).split())
            if machine and "<sep>" not in machine:
                f.write(f"{machine}<sep>{human}\n")
                f.flush()
                written += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

    print(f"\nWrote {written} pairs -> {out}")
    print("Add this to `corpora` in languages.json (if it is not there yet):")
    print(f'  {{ "file": "{out.replace(os.sep, "/")}", "columns": ["{args.via}", "{args.lang}"], '
          f'"synthetic": true, "limit": 3000 }}')


if __name__ == "__main__":
    main()
