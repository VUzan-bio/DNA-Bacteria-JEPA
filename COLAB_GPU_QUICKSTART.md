# Google Colab GPU Quickstart

This project now supports mixed precision (`bf16`/`fp16`) and gradient accumulation for faster Colab training.

## 1. Start Colab with GPU

In Colab: `Runtime -> Change runtime type -> T4 GPU` (or any GPU).

## 2. Setup project

Run these cells in order.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone https://github.com/<your-user>/<your-repo>.git
%cd /content/<your-repo>
!pip install -r requirements-colab.txt
```

If you already uploaded this repo to Drive instead of GitHub:

```bash
%cd /content/drive/MyDrive
!cp -r DNA-Bacteria-JEPA /content/
%cd /content/DNA-Bacteria-JEPA
!pip install -r requirements-colab.txt
```

## 3. Put data where Colab can read it

Recommended: keep data/checkpoints on Drive for persistence.

- Pretraining CSV:
  `/content/drive/MyDrive/DNA-Bacteria-JEPA/data/processed/pretrain_sequences.csv`
- Fine-tuning CSV:
  `/content/drive/MyDrive/DNA-Bacteria-JEPA/data/processed/cas12a_efficiency.csv`

## 4. Pretrain on GPU (proof-of-concept run)

```bash
!python scripts/01_pretrain_jepa.py \
  --data-path /content/drive/MyDrive/DNA-Bacteria-JEPA/data/processed/pretrain_sequences.csv \
  --output-dir /content/drive/MyDrive/DNA-Bacteria-JEPA/checkpoints/pretrain \
  --epochs 5 \
  --batch-size 128 \
  --grad-accum-steps 2 \
  --num-workers 2 \
  --precision auto \
  --save-every 1 \
  --max-samples 20000
```

Notes:
- Effective batch size = `batch_size * grad_accum_steps` (here 256).
- If GPU runs out of memory, lower `--batch-size` first.
- Remove `--max-samples` for full training.

## 5. Fine-tune on GPU

```bash
!python scripts/02_finetune_cas12a.py \
  --data-path /content/drive/MyDrive/DNA-Bacteria-JEPA/data/processed/cas12a_efficiency.csv \
  --pretrained-path /content/drive/MyDrive/DNA-Bacteria-JEPA/checkpoints/pretrain/checkpoint_epoch5.pt \
  --output-dir /content/drive/MyDrive/DNA-Bacteria-JEPA/checkpoints/finetune \
  --epochs 20 \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --num-workers 2 \
  --precision auto
```

## 6. Check GPU availability

```python
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

