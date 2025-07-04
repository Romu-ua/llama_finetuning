# raw_sarashina_style.py

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ==== パス設定 ====
MODEL_PATH = "./merged_model"             # LoRA統合済み or 通常ファインチューニング済みモデル
IMAGE_PATH = "dataset/test2.jpg"          # 推論対象画像

# ==== モデルとプロセッサの読み込み ====
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

# ==== プロンプト（Chat形式）====
messages = [
    {"role": "user", "content": "この画像が示す日本の行事は何ですか？"}
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# ==== 画像読み込み ====
image = Image.open(IMAGE_PATH).convert("RGB")

# ==== モデルへの入力変換 ====
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
    padding=True,
).to(model.device)

# ==== 応答生成 ====
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,                  # raw_sarashina風（再現性重視）
        temperature=0.0,                 # deterministic出力
        top_p=1.0,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

# ==== 応答デコード ====
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)[0]

# ==== 出力 ====
print("=== モデルの応答 ===")
print(output_text)
