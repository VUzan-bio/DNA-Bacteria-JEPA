# Data Format Guide

This file explains what data you need to provide and in what format.

## 1. Bacterial Genome Pretraining

**File:** `data/processed/pretrain_sequences.csv`

**Format:**
```csv
sequence,genome,position
ATCGATCGATCGATCG...,E_coli_K12,0
GCTAGCTAGCTAGCTA...,B_subtilis_168,256
...
```

**Requirements:**
- `sequence`: 512 bp DNA string (ACGT only, no N or ambiguous bases)
- `genome`: Identifier for source genome
- `position`: Starting position in genome

**How to create:**
1. Download bacterial genomes from NCBI (your choice which ones)
2. Extract 512 bp windows with 256 bp stride
3. Filter out sequences with ambiguous bases

**Expected size:** 50,000-500,000 sequences (more is better)

---

## 2. Cas12a Efficiency Fine-Tuning

**File:** `data/processed/cas12a_efficiency.csv`

**Format:**
```csv
crRNA_seq,target_seq,PAM,efficiency_normalized,source
ATCGATCGATCGATCGATCG,GGCATCGATCGATCG...,TTTA,0.75,EasyDesign
GCTAGCTAGCTAGCTAGCTA,CCGATCGATCGATCG...,TTTC,-0.42,OBrien2023
...
```

**Requirements:**
- `crRNA_seq`: Guide RNA sequence (20-24 nt)
- `target_seq`: Target DNA sequence (50-100 nt recommended)
- `PAM`: PAM sequence (must be TTTV format: TTTA, TTTC, or TTTG)
- `efficiency_normalized`: Normalized efficiency score (use robust scaling)
- `source`: Dataset identifier

**Where to get data:**
1. EasyDesign - Email authors for supplementary data
2. Published papers - Extract from supplementary tables
3. Your own experiments - If you have wet lab access

**Expected size:** 1,000-15,000 guides

---

## 3. RPA Primer Design (Optional)

**File:** `data/processed/rpa_primers.csv`

**Format:**
```csv
forward_primer,reverse_primer,target_seq,amplifies,time_to_positive
ATCGATCG...,GCTAGCTA...,GGCATCGATC...,1,18.5
CGATCGAT...,TAGCTAG...,CCGATCGAT...,0,
...
```

**Requirements:**
- `forward_primer`: Forward primer sequence
- `reverse_primer`: Reverse primer sequence
- `target_seq`: Expected amplicon sequence
- `amplifies`: Binary (0 or 1)
- `time_to_positive`: Time in minutes (optional, can be empty)

**Expected size:** 100-500 primer pairs

---

## Normalization

Use the provided `Cas12aDataManager.robust_normalize()` function:

```python
from src.cas12a.data_manager import Cas12aDataManager

# Robust scaling (median + IQR)
df["efficiency_normalized"] = Cas12aDataManager.robust_normalize(df["efficiency_raw"])
```

---

## Data Split

The code automatically handles leakage-safe splitting:

```python
from src.cas12a.data_manager import Cas12aDataManager

splits = Cas12aDataManager.split_data(df, test_size=0.15, val_size=0.15)
# Returns: SplitResult(train=..., val=..., test=...)
```

This ensures guides with the same sequence do not appear in multiple splits.

---

## Usage

```bash
# 1. Install dependencies
pip install torch pandas numpy scipy scikit-learn tqdm

# 2. Option A: build pretraining CSV automatically
python scripts/download_bacterial_genomes.py
python scripts/extract_pretraining_sequences.py

# 2. Option B: add data manually
#    - data/processed/pretrain_sequences.csv
#    - data/processed/cas12a_efficiency.csv

# 3. Pretrain JEPA
python scripts/01_pretrain_jepa.py \
    --data-path data/processed/pretrain_sequences.csv \
    --epochs 100 \
    --batch-size 128

# 4. Fine-tune on Cas12a
python scripts/02_finetune_cas12a.py \
    --data-path data/processed/cas12a_efficiency.csv \
    --pretrained-path checkpoints/pretrain/checkpoint_epoch100.pt \
    --epochs 50

# 5. Results are saved in checkpoints/finetune/best_model.pt
```

## Google Colab

For GPU training on free Colab, use:

- `COLAB_GPU_QUICKSTART.md`
- `requirements-colab.txt`

This avoids reinstalling Colab's preloaded CUDA PyTorch and uses GPU-friendly training flags.
