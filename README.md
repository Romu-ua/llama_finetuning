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

