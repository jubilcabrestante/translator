"""
translate.py
------------
Load the base model + your trained LoRA adapter and translate text.

The languages come from `languages.json`, so this handles every direction the
model was trained on -- one model, all pairs.

Runs on an NVIDIA GPU (4-bit, fast) when there is one, and otherwise on the
CPU: full-precision weights, about 6 GB of RAM for the 1.5B model and a few
seconds per sentence. For fast CPU serving, export to GGUF instead (see
scripts/export_model.py).

Usage:
  # Interactive REPL:
  python scripts/translate.py

  # One-shot (language names, or the short codes from languages.json):
  python scripts/translate.py --text "sino ang kasama mo?" --to Cuyonon
  python scripts/translate.py --text "Ano imong aran?" --from cyo --to en

  # Force specific terms to be kept EXACTLY as written (glossary override):
  python scripts/translate.py --text "Pumunta si Maria sa Cuyo." --keep "Maria,Cuyo"

  # Beam search (often a little better, slower):
  python scripts/translate.py --text "Saan ka pupunta?" --to cyo --num-beams 4

The prompt is built by common.py, so it's guaranteed identical to the one used
during training -- including the "Keep these terms unchanged" line, which
prepare_data.py now puts into training examples too.
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import build_messages, load_config  # identical prompt to training


def resolve_language(value, config):
    """Accept either a language name ('Cuyonon') or its code ('cyo')."""
    if value in config["name_to_code"]:
        return value
    if value in config["code_to_name"]:
        return config["code_to_name"][value]
    options = ", ".join(f"{lang['name']} ({lang['code']})"
                        for lang in config["languages"])
    raise SystemExit(f"Unknown language {value!r}. Configured: {options}")


def pick_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but no CUDA GPU is available.")
    return device


def load(base_model, adapter_dir, device="auto"):
    if not os.path.isdir(adapter_dir):
        raise SystemExit(
            f"Adapter dir '{adapter_dir}' not found locally. Train it first:\n"
            f"    python scripts/train.py --output-dir {adapter_dir}\n"
            f"or point --adapter-dir at an existing adapter. (A bare name is otherwise "
            f"treated as a Hugging Face repo id, which is what caused the 401 you saw.)"
        )

    device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=compute_dtype,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        # bitsandbytes 4-bit needs CUDA, so on CPU the full weights load
        # instead. float32 because most laptop CPUs have no fast bfloat16.
        # Merging the adapter in removes its per-token overhead.
        print("No CUDA GPU: running on the CPU (about 6 GB of RAM for the 1.5B model, "
              "a few seconds per sentence).", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def translate(model, tokenizer, text, src_lang, tgt_lang, keep_terms=None,
              max_new_tokens=None, num_beams=1):
    messages = build_messages(text, src_lang, tgt_lang, keep_terms)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if max_new_tokens is None:
        # A translation runs about as long as its source. The old fixed cap of
        # 64 tokens could cut a long input off mid-sentence; scale instead.
        source_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        max_new_tokens = min(512, 2 * source_tokens + 16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic: greedy, or beam search
            num_beams=num_beams,
            # The base model's generation_config ships sampling settings and a
            # repetition penalty. The penalty applies even to greedy decoding
            # and pushes the model away from tokens already in the prompt --
            # exactly what copying a name, or a word Tagalog and Cuyonon share,
            # requires. Turn it off and clear the unused sampling settings.
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]   # strip the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def missing_terms(output, keep_terms):
    """Kept terms that did not come through verbatim."""
    return [t for t in (keep_terms or []) if t not in output]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Must match the base model used in training.")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--text", default=None)
    ap.add_argument("--from", dest="src_lang", default=None,
                    help="Language name or code. Default: first in languages.json.")
    ap.add_argument("--to", dest="tgt_lang", default=None,
                    help="Language name or code. Default: second in languages.json.")
    ap.add_argument("--keep", default=None,
                    help="Comma-separated terms to keep EXACTLY as written (glossary).")
    ap.add_argument("--num-beams", type=int, default=1,
                    help="1 = greedy (default). 4 often scores a little higher; slower.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    config = load_config(args.config)
    names = config["language_names"]
    src_lang = resolve_language(args.src_lang, config) if args.src_lang else names[0]
    tgt_lang = resolve_language(args.tgt_lang, config) if args.tgt_lang else names[1]
    if src_lang == tgt_lang:
        raise SystemExit("--from and --to must differ.")

    keep_terms = [t.strip() for t in args.keep.split(",") if t.strip()] if args.keep else None

    def run(text):
        result = translate(model, tokenizer, text, src_lang, tgt_lang, keep_terms,
                           num_beams=args.num_beams)
        lost = missing_terms(result, keep_terms)
        if lost:
            print(f"  (warning: not kept verbatim: {', '.join(lost)})", file=sys.stderr)
        return result

    print("Loading model + adapter... (first run downloads the base model)")
    model, tokenizer = load(args.base_model, args.adapter_dir, device=args.device)

    if args.text is not None:
        print(run(args.text))
        return

    # Interactive REPL
    codes = config["name_to_code"]
    print("\nInteractive mode.")
    print("  Switch direction by typing a pair, e.g. "
          f"'{codes[names[0]]}>{codes[names[1]]}'. Available: "
          + ", ".join(f"{n} ({codes[n]})" for n in names))
    print("  'quit' to exit.\n")
    while True:
        try:
            text = input(f"[{src_lang}->{tgt_lang}] > ").strip()
        except EOFError:
            break
        low = text.lower()
        if low in ("quit", "exit"):
            break
        if not text:
            continue
        # A bare "<src>><tgt>" (or "<src>-<tgt>") switches direction.
        for sep in (">", "-"):
            if sep in text and len(text.split(sep)) == 2:
                left, right = (p.strip() for p in text.split(sep))
                if (left in codes or left in config["code_to_name"]) and \
                        (right in codes or right in config["code_to_name"]):
                    src_lang = resolve_language(left, config)
                    tgt_lang = resolve_language(right, config)
                    print(f"  -> now translating {src_lang} to {tgt_lang}")
                    text = None
                    break
        if text is None:
            continue
        print("  =>", run(text))


if __name__ == "__main__":
    main()
