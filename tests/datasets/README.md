# tests/datasets/README.md

# VisionEdit — Dataset Setup Guide

This directory contains dataset loaders for the VisionEdit validation test suite.
No data files are stored here — the datasets live on your local machine and
are pointed to via `config.yaml`.

---

## Kinetics-400

**What it is**: Large-scale action recognition benchmark by Google DeepMind.
- 400 human action categories (~300K training clips, ~20K validation clips)
- Each clip ≈ 10 seconds, 25fps, from YouTube

**Download** (validation split only, ~100GB):
```bash
# Option 1: Using the cvdfoundation downloader (recommended)
git clone https://github.com/cvdfoundation/kinetics-dataset
cd kinetics-dataset
python download.py --val

# Option 2: Google Cloud Storage (requires gcloud CLI)
gsutil -m cp -r gs://deepmind-kinetics/kinetics400/val /your/path/
```

**Expected structure**:
```
/your/path/kinetics400/val/
  abseiling/
    abcd1234.mp4
    efgh5678.mp4
    ...
  air_drumming/
    ...
  running/
    ...
```

**Configure in `config.yaml`**:
```yaml
datasets:
  kinetics400_root: "D:/datasets/kinetics400/val"
```

**Used in tests**: `tests/test_datasets.py::TestKinetics400Pipeline`

---

## HMDB51

**What it is**: Human Motion Database — 51 action categories.
- 6,849 clips total (~130 per category)
- Clips are 2–5 seconds, 320×240, ~30fps
- 3 official train/test splits available

**Download**:
```bash
# Step 1: Download main archive (~2GB)
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

# Step 2: Extract outer .rar
unrar x hmdb51_org.rar

# Step 3: Extract all inner .rar files (one per category)
for f in *.rar; do unrar x "$f" -op/your/path/hmdb51/; done

# Step 4 (optional): Download split files for train/test splits
wget https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
unrar x test_train_splits.rar -op/your/path/hmdb51_splits/
```

**Expected structure**:
```
/your/path/hmdb51/
  brush_hair/
    April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    ...
  cartwheel/
    ...
  laugh/
    ...
  (51 category dirs total)

/your/path/hmdb51_splits/   (optional)
  brush_hair_test_split1.txt
  cartwheel_test_split1.txt
  ...
```

**Configure in `config.yaml`**:
```yaml
datasets:
  hmdb51_root: "D:/datasets/hmdb51"
  hmdb51_splits_dir: "D:/datasets/hmdb51_splits"   # optional
```

**Used in tests**: `tests/test_datasets.py::TestHMDB51Pipeline`

---

## Running the Dataset Tests

```bash
# Skip all dataset tests (no datasets needed, default CI behaviour)
python -m pytest tests/test_dummy.py -v

# Run with HMDB51 (once configured in config.yaml)
python -m pytest tests/test_datasets.py::TestHMDB51Pipeline -v

# Run with Kinetics-400 (once configured in config.yaml)
python -m pytest tests/test_datasets.py::TestKinetics400Pipeline -v

# Run everything
python -m pytest tests/ -v
```

Tests that require a dataset are automatically **skipped** if the corresponding
`root` path in `config.yaml` is `null` — you will never get a failure just
because a dataset isn't downloaded.
