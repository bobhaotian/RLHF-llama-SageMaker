"""
Proximal Policy Optimization (PPO) training on SageMaker for **Meta-Llama-3-8B**
with 4-bit quantisation + LoRA adapters, using the **TRL 0.16.4** stack:

* **transformers ≥ 4.38** (comes with SageMaker DLC 4.49.0 — fine)  
* **peft 0.10.0** – LoRA layers  
* **accelerate 0.28.0** – device map sharding  
* **trl 0.16.4** – PPOTrainer bug-fixed for tuple outputs and v_head  

All ad-hoc monkey-patches from earlier versions (tuple shims, device casts, etc.)
are **no longer required** and have been removed. The script is therefore much
simpler and easier to maintain.
"""

import os, tarfile, tempfile, pathlib, copy
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ───── SageMaker env vars ──────────────────────────────────────────────
BASE_ID       = os.environ["BASE_ID"]          # e.g. meta-llama/Meta-Llama-3-8B
PROMPTS       = os.environ["PROMPT_FILE"]      # s3://…/ppo_prompts.jsonl
HF_TOKEN      = os.environ["HF_TOKEN"]
SM_REWARD_DIR = os.environ["SM_CHANNEL_REWARD"]

login(HF_TOKEN)

# ───── Tokeniser ───────────────────────────────────────────────────────

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"  # PPO expects right-padding

# ───── 4-bit quant cfg ─────────────────────────────────────────────────

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ───── Policy (LM + Value Head) ────────────────────────────────────────
# TRL 0.16 automatically adds a scalar value head when you load the model
# with AutoModelForCausalLMWithValueHead.

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
)
policy.gradient_checkpointing_enable()

# Attach LoRA adapters
policy = get_peft_model(
    policy,
    LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]),
)
policy.gradient_checkpointing_enable()

# Make sure generation defaults exist
if not hasattr(policy, "generation_config"):
    policy.generation_config = GenerationConfig.from_pretrained(BASE_ID)

# Freeze most of the policy except LoRA — handled by PEFT automatically

# ───── Reference policy (frozen) ───────────────────────────────────────
ref_policy = copy.deepcopy(policy)
ref_policy.eval()
for p in ref_policy.parameters():
    p.requires_grad = False

# ───── Reward model ────────────────────────────────────────────────────
# Unpack the tarball SageMaker passes in SM_CHANNEL_REWARD

tar_path = pathlib.Path(SM_REWARD_DIR) / "model.tar.gz"
work_dir = pathlib.Path(tempfile.mkdtemp())
with tarfile.open(tar_path) as tar:
    tar.extractall(work_dir)

# Detect if artefact is full model or LoRA adapter
if (work_dir / "config.json").exists() or (work_dir / "model" / "config.json").exists():
    rm_root = work_dir if (work_dir / "config.json").exists() else work_dir / "model"
    reward = AutoModelForSequenceClassification.from_pretrained(
        rm_root,
        num_labels=1,
        problem_type="regression",
        quantization_config=bnb_cfg,
        device_map="auto",
    )
else:
    rm_root = work_dir if (work_dir / "adapter_config.json").exists() else work_dir / "model"
    base_rm = AutoModelForSequenceClassification.from_pretrained(
        BASE_ID,
        num_labels=1,
        problem_type="regression",
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    reward = PeftModel.from_pretrained(base_rm, rm_root)

# Freeze reward model
for p in reward.parameters():
    p.requires_grad = False

# ───── PPO dataset ─────────────────────────────────────────────────────
raw_ds = load_dataset("json", data_files={"train": PROMPTS})["train"]

def tok_map(batch):
    return tok(batch["prompt"], max_length=512, truncation=True, padding=False)

ppo_dataset = raw_ds.map(tok_map, batched=True, remove_columns=raw_ds.column_names)

# ───── PPO config & trainer ────────────────────────────────────────────
ppo_cfg = PPOConfig(
    batch_size=8,
    mini_batch_size=2,
    learning_rate=1e-6,
)

trainer = PPOTrainer(
    ppo_cfg,       # config
    policy,        # model
    ref_policy,    # reference
    tok,           # tokenizer
    train_dataset=ppo_dataset,   # dataset
    value_model=policy,
    reward_model = reward,
)


trainer.train()
trainer.save_pretrained("/opt/ml/model")
tok.save_pretrained("/opt/ml/model")