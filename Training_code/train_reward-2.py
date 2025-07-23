"""LoRA reward‑model fine‑tuning for Meta‑Llama‑3‑8B with TRL 0.8 on a single
A10 GPU (g5.12xlarge). The script expects a JSON‑Lines dataset containing
`chosen` and `rejected` responses (and optionally a `prompt`). It tokenises each
pair into the four tensors required by TRL’s `RewardTrainer`.
"""

import os, torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import RewardTrainer, RewardConfig

# ───── env vars ─────────────────────────────────────────────────────────
BASE_ID    = os.environ["BASE_ID"]
TRAIN_FILE = os.environ["TRAIN_FILE"]
HF_TOKEN   = os.environ["HF_TOKEN"]


prompt_key, chosen_key, rejected_key = "prompt", "chosen", "rejected"

# i put the old version in another file (_old)
login(HF_TOKEN)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(BASE_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token      # reuse </s> as PAD
    tok.pad_token_id = tok.eos_token_id

tok.padding_side = "right"

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_ID,
    num_labels=1,
    problem_type="regression",
    quantization_config=bnb_cfg,
    device_map="auto",
)

model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_CLS",
    ),
)
model.gradient_checkpointing_enable()

# ───── dataset → tensor features ────────────────────────────────────────
raw_ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

max_len = 512

def encode(example):
    prompt   = example.get(prompt_key, "")
    chosen   = example[chosen_key]
    rejected = example[rejected_key]

    enc_c = tok(prompt + chosen,   truncation=True, max_length=max_len)
    enc_r = tok(prompt + rejected, truncation=True, max_length=max_len)

    return {
        "input_ids_chosen":        enc_c["input_ids"],
        "attention_mask_chosen":  enc_c["attention_mask"],
        "input_ids_rejected":      enc_r["input_ids"],
        "attention_mask_rejected": enc_r["attention_mask"],
    }

train_ds = raw_ds.map(encode, remove_columns=raw_ds.column_names)
train_ds.set_format(type="torch")

# ───── training config ─────────────────────────────────────────────────
cfg = RewardConfig(
    output_dir="/opt/ml/model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_length=max_len,
    remove_unused_columns=False,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=50,
    fp16=False,
)

trainer = RewardTrainer(
    model=model,
    processing_class=tok,   # 0.9+ expects this name
    args=cfg,
    train_dataset=train_ds,
)
trainer.train()

model.save_pretrained("/opt/ml/model")
tok.save_pretrained("/opt/ml/model")
