from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

# モデルパス
base_model_path = "./sarashina2-vision-8b"
lora_model_path = "./checkpoints/checkpoint-3"  # Trainerのoutput_dirと一致させる

# ベースモデルを読み込み
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype="auto",
)

# LoRAを適用
model = PeftModel.from_pretrained(model, lora_model_path)

# LoRA層だけを保存（再利用用）
model.save_pretrained("./lora_weights_only")

# LoRAをベースにマージして保存（推論用モデル）
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# ✅ 追加：プロセッサも一緒に保存（←これが重要！）
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained("./merged_model")
