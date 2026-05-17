# Automated Detection of Dead-Ball Events in Football Broadcasts
## Introduction to Data Science — Group Project

**Group Members:** Ahmed Umar Mirza (01-134232-028), Bisma Rauf (01-134232-047)
**Dataset:** SoccerNet (https://www.soccer-net.org/data)

---

## Scope

Detects the following dead-ball restart events from a broadcast match video:

- **Penalty**
- **Free-kick** (Direct + Indirect combined)
- **Kick-off**
- **Corner**
- **Throw-in**

A **Background** class is trained alongside these to suppress false positives during sliding-window prediction.

## Project Structure

```
IDS_Project/
├── 01_download_data.py        # SoccerNet labels + per-game video downloads
├── build_index_only.py        # Build data/event_index.csv from Labels-v2.json
├── 02_extract_features.py     # YOLO + hand-crafted features + background sampling
├── 03_train_model.py          # EDA, SMOTE, Random Forest, evaluation
├── 04_predict.py              # Sliding-window prediction on a new video
├── utils.py                   # Shared feature extraction (used by 02 and 04)
├── requirements.txt
└── README.md
```

## Setup

This project uses **Python 3.14**.

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
# CPU-only torch first to avoid pulling multi-GB CUDA wheels via ultralytics
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## Run Order

### Step 1 — Download labels and videos

Labels first (small, ~5 min):

```bash
python 01_download_data.py --labels-only
```

Then show the planned 15-game training corpus:

```bash
python 01_download_data.py --list
```

Download games **one at a time** (each game = ~400MB at 224p, both halves):

```bash
python 01_download_data.py --game 1
python 01_download_data.py --game 2
# ... etc, up to --game 15
```

Check progress anytime:

```bash
python 01_download_data.py --status
```

If you prefer a single long-running command:

```bash
python 01_download_data.py --all
```

### Step 2 — Build the event index

```bash
python build_index_only.py
```

Writes `data/event_index.csv` with one row per target event. Only rows where
`video_exists == True` are processed in step 3.

### Step 3 — Extract features (slow)

```bash
python 02_extract_features.py
```

- Pass 1: features around each labelled event (~10 s window).
- Pass 2: 40 random background windows per video.
- Output: `data/features.csv`. Resume support — safe to interrupt and re-run.

### Step 4 — Train

```bash
python 03_train_model.py
```

Saves `models/rf_model.joblib`, `models/label_encoder.joblib`, and EDA / evaluation plots into `plots/`.

### Step 5 — Predict on a new video

```bash
python 04_predict.py --video path/to/match.mkv --half 1 --out results.csv
# Limit to the first N seconds (useful for quick tests):
python 04_predict.py --video path/to/match.mkv --max-seconds 300
```

Outputs:
- `results.csv` — timestamped events (event class, confidence, match minute)
- `plots/06_event_timeline.png` — annotated timeline plot
