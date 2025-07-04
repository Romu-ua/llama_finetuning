# llama_finetuning


環境構築
```bash
nvidia-smi
> NVIDIA-SMI 570.153.02             Driver Version: 570.153.02     CUDA Version: 12.8  
```

仮想環境
```bash
conda create -n llama python=3.10
conda activate llama
```

ライブラリ
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.47.0
pip install accelerate
pip install protobuf
pip install sentencepiece
pip install bitsandbytes
pip install peft
```

使い方  
sarashina2-vision
```bash
python raw_sarashina.py
```

フルファインチューニング
```bash
python full_finetuning_sarashina.py
```
qlora ファインチューニング
```bash
python qlora_finetuning.py
```
ファインチューニングした層を本体にマージ
```bash
python save_lora.py
```

finetuning済みsarashinaを使用
```bash
python finetuning_sarashina.py
```