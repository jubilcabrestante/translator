"""
train.py
--------
QLoRA fine-tune a small decoder-only instruction LLM into a Tagalog<->Cuyonon
translator, sized to run on a 4 GB laptop GPU.

Why this fits 4 GB:
  * 4-bit quantization (bitsandbytes NF4) loads a 1.5B model in ~0.9 GB.
  * LoRA only trains tiny adapter matrices, not the full model.
  * gradient_checkpointing trades compute for memory.
  * batch size 1 + gradient accumulation simulates a larger batch cheaply.
  * paged_adamw_8bit keeps optimizer state small and avoids OOM spikes.

If you still hit CUDA out-of-memory:
  1. Close every other program using the GPU (browsers, games, etc.).
  2. Switch to the 0.5B model:  --model Qwen/Qwen2.5-0.5B-Instruct
  3. Lower --lora-r to 8, or --max-seq-len to 128.

The SAME script scales up: on a bigger GPU / Colab, just pass a larger --model
(e.g. Qwen/Qwen2.5-7B-Instruct) and raise the batch size.
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Base LLM. Use Qwen/Qwen2.5-0.5B-Instruct if you run out of memory.")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--val-file", default="data/val.jsonl")
    ap.add_argument("--output-dir", default="outputs/translator-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="Path to a checkpoint dir (e.g. outputs/translator-lora/checkpoint-295) "
                         "to continue training from instead of starting over.")
    args = ap.parse_args()

    # ---- Detect bf16 support (Ampere GPUs like the RTX 3050 support it) -------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"CUDA available: {torch.cuda.is_available()} | using "
          f"{'bf16' if use_bf16 else 'fp16'} compute dtype")

    # ---- 4-bit quantization config -------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,   # extra ~0.4 bits/param saved
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},        # put everything on GPU 0
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- LoRA: which layers get trainable adapters ---------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- Load the chat-format dataset ----------------------------------------
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    # ---- Training configuration ----------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=args.max_seq_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=lora_config,   # SFTTrainer applies LoRA + k-bit prep for us
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the LoRA adapter (small, a few MB) + tokenizer.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDone. LoRA adapter saved to {args.output_dir}")
    print("Next: python scripts/translate.py")


if __name__ == "__main__":
    main()
