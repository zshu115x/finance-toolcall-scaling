"""
Step 2: LoRA fine-tune Qwen2.5-1.5B on the tool-call dataset.

Uses HuggingFace PEFT + trl SFTTrainer (no Unsloth).
Training data format: messages with tool_calls in the assistant turn.
The Qwen2.5 chat template natively handles tool_calls in the messages list.

CPU note: bf16/fp16 are both disabled by default. Training ~100-150 examples
for 3 epochs takes roughly 30-60 minutes on a modern laptop CPU.
On an A100/H100 this is under 5 minutes.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_training_summary(log_history: list[dict], ckpt_dir: Path):
    losses = [e for e in log_history if "loss" in e]
    if not losses:
        print("  No metrics logged.")
        return
    final = losses[-1]
    min_loss = min(e["loss"] for e in losses)
    print(f"  Final step   : {final.get('step', '?')}")
    print(f"  Final loss   : {final['loss']:.4f}")
    print(f"  Min loss     : {min_loss:.4f}")
    print(f"  Epochs done  : {final.get('epoch', '?')}")
    print(f"  Checkpoint   : {ckpt_dir.resolve()}")


def main():
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_path = ROOT / cfg["paths"]["train_data"]
    ckpt_dir = ROOT / train_cfg["output_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        print("Run step1_generate_data.py first.")
        sys.exit(1)

    with open(data_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    n_train = len(raw)
    logger.info(f"Training on {n_train} examples from {data_path}")

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "model": train_cfg["base_model"],
        "n_train_examples": n_train,
        **{k: v for k, v in train_cfg.items() if k != "base_model"},
    }
    with open(results_dir / "training_config.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print("\n── LoRA Fine-Tuning ──────────────────────────────────────────")
    print(f"  Model      : {train_cfg['base_model']}")
    print(f"  LoRA rank  : {train_cfg['lora_r']}  alpha: {train_cfg['lora_alpha']}")
    print(f"  Epochs     : {train_cfg['num_epochs']}  LR: {train_cfg['learning_rate']}")
    print(f"  Data       : {data_path}  ({n_train} examples)")
    print(f"  Output     : {ckpt_dir.resolve()}")
    print("─────────────────────────────────────────────────────────────\n")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(train_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16 if train_cfg["bf16"]
        else torch.float16 if train_cfg["fp16"]
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg["base_model"],
        torch_dtype=dtype,
        device_map=device,
    )

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(raw)

    training_args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        report_to="none",
        max_length=train_cfg["max_seq_len"],
        packing=train_cfg["sample_packing"],
        assistant_only_loss=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    print("\n── Training Complete ─────────────────────────────────────────")
    print_training_summary(trainer.state.log_history, ckpt_dir)
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
