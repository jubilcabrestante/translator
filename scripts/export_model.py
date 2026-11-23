"""
export_model.py
---------------
Merge the LoRA adapter into its base model and save a standalone copy -- the
step that gets the translator off the GPU.

The merged folder loads with plain transformers anywhere, and converts to GGUF
for llama.cpp / Ollama, which run a 1.5B model on an ordinary CPU at about a
second or two per sentence. That is the cheap way to host the app's Tier 3:
a small CPU server running Ollama, with scripts/serve.py in backend mode in
front of it (same HTTP contract, no torch needed).

Merging needs the full-precision weights (~3 GB of RAM for the 1.5B model in
float16) but no GPU, so it runs on a laptop or a Colab CPU runtime.

Usage:
  python scripts/export_model.py
  python scripts/export_model.py --adapter-dir outputs/translator-lora --out-dir outputs/translator-merged

Then (also printed at the end):
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  python llama.cpp/convert_hf_to_gguf.py outputs/translator-merged \
      --outtype q8_0 --outfile outputs/translator-merged/translator-q8_0.gguf
  cd outputs/translator-merged && ollama create translator -f Modelfile
  # serve the app from it:
  TRANSLATOR_BACKEND_URL=http://localhost:11434/v1 TRANSLATOR_BACKEND_MODEL=translator \
      python scripts/serve.py

q8_0 is near-lossless and needs no compiling. Smaller quantisations (Q4_K_M)
need llama.cpp's llama-quantize binary; check evaluate-style spot checks after.
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import SYSTEM_PROMPT

MODELFILE = '''FROM ./translator-q8_0.gguf

# The system prompt the model was trained with (scripts/common.py). serve.py
# sends it with every request anyway; this is the default for `ollama run`.
SYSTEM """{system}"""

# Deterministic, and no repetition penalty: a penalty pushes the model away
# from tokens already in the prompt, which is exactly what copying a name, or a
# word Tagalog and Cuyonon share, requires. (Ollama's default is 1.1.)
PARAMETER temperature 0
PARAMETER repeat_penalty 1.0
PARAMETER num_ctx 1024

# Ollama reads the chat template from the GGUF. If `ollama run translator`
# answers in a strange format, the template was not detected: add a TEMPLATE
# block for your base model's chat format here.
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Must match the base model used in training.")
    ap.add_argument("--adapter-dir", default="outputs/translator-lora")
    ap.add_argument("--out-dir", default="outputs/translator-merged")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    if not os.path.isdir(args.adapter_dir):
        raise SystemExit(f"Adapter dir '{args.adapter_dir}' not found. Train first, or pass "
                         f"--adapter-dir.")

    print(f"Loading {args.base_model} in {args.dtype} (no quantization; merging needs "
          f"the real weights) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    print("Merging the adapter into the base weights ...")
    model = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter_dir).save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(MODELFILE.format(system=SYSTEM_PROMPT))

    out = args.out_dir
    print(f"\nMerged model saved to {out} (plus an Ollama Modelfile).")
    print("\nNext, to run it on a CPU with Ollama:")
    print("  git clone --depth 1 https://github.com/ggml-org/llama.cpp")
    print("  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {out} --outtype q8_0 "
          f"--outfile {out}/translator-q8_0.gguf")
    print(f"  cd {out} && ollama create translator -f Modelfile")
    print("Then point serve.py at it:")
    print("  TRANSLATOR_BACKEND_URL=http://localhost:11434/v1 "
          "TRANSLATOR_BACKEND_MODEL=translator python scripts/serve.py")


if __name__ == "__main__":
    main()
