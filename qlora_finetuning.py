"""
qlora ファインチューニング
"""

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ==== モデルとデータのパス ====
MODEL_PATH = "./sarashina2-vision-8b"
DATA_PATH = "dataset/train.jsonl"

# ==== 量子化（4bit）設定 ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # または "fp4"
    bnb_4bit_compute_dtype=torch.float16,
)

# ==== LoRA 設定 ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 必要に応じて変更
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ==== データ読み込み ====
def load_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ==== カスタムデータセット ====
class SarashinaVisionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        prompt = self.processor.apply_chat_template(item["messages"], add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}

# ==== モデルとプロセッサのロード ====
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# ==== LoRAの適用 ====
model = get_peft_model(model, lora_config)

# ==== データセットの構築 ====
train_data = load_data(DATA_PATH)
train_dataset = SarashinaVisionDataset(train_data, processor)

# ==== トレーニング設定 ====
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_steps=10,
    logging_steps=5,
    fp16=True,
    remove_unused_columns=False,
)

# ==== Trainer 実行 ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,  # v5.0以降は processing_class もOK
)

trainer.train()
