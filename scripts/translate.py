"""
translate.py
------------
Load the base model + your trained LoRA adapter and translate text.

Usage:
  # Interactive REPL (asks for direction + text):
  python scripts/translate.py

  # One-shot:
  python scripts/translate.py --text "sino ang kasama mo?" --to Cuyonon
  python scripts/translate.py --text "Ano imong aran?" --from Cuyonon --to Tagalog

  # Force specific terms to be kept EXACTLY as written (glossary override):
  python scripts/translate.py --text "Pumunta si Maria sa Cuyo." --keep "Maria,Cuyo"

The system prompt is imported from common.py, so it's guaranteed identical to the
one used during training.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import SYSTEM_PROMPT, build_user_turn  # identical prompt to training


def load(base_model, adapter_dir):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def translate(model, tokenizer, text, src_lang, tgt_lang, keep_terms=None, max_new_tokens=64):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(text, src_lang, tgt_lang, keep_terms)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic / greedy = most literal
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]   # strip the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Must match the base model used in training.")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--text", default=None)
    ap.add_argument("--from", dest="src_lang", default="Tagalog")
    ap.add_argument("--to", dest="tgt_lang", default="Cuyonon")
    ap.add_argument("--keep", default=None,
                    help="Comma-separated terms to keep EXACTLY as written (glossary).")
    args = ap.parse_args()

    keep_terms = [t.strip() for t in args.keep.split(",")] if args.keep else None

    print("Loading model + adapter... (first run downloads the base model)")
    model, tokenizer = load(args.base_model, args.adapter_dir)

    if args.text is not None:
        print(translate(model, tokenizer, args.text, args.src_lang, args.tgt_lang, keep_terms))
        return

    # Interactive REPL
    print("\nInteractive mode. Commands: 'tc' Tagalog->Cuyonon, 'ct' Cuyonon->Tagalog, 'quit'.\n")
    src_lang, tgt_lang = "Tagalog", "Cuyonon"
    while True:
        text = input(f"[{src_lang}->{tgt_lang}] > ").strip()
        low = text.lower()
        if low in ("quit", "exit"):
            break
        if low == "tc":
            src_lang, tgt_lang = "Tagalog", "Cuyonon"; continue
        if low == "ct":
            src_lang, tgt_lang = "Cuyonon", "Tagalog"; continue
        if not text:
            continue
        print("  =>", translate(model, tokenizer, text, src_lang, tgt_lang, keep_terms))


if __name__ == "__main__":
    main()
