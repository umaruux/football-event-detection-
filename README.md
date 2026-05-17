# Automated Detection of Dead-Ball Events in Football Broadcasts
## Introduction to Data Science — Group Project

**Group Members:** Ahmed Umar Mirza (01-134232-028), Bisma Rauf (01-134232-047)
**Dataset:** SoccerNet England EPL subset, 720p broadcast video + `Labels-v2.json`
(https://www.soccer-net.org/data). Training uses **half 1 only** of the games
already present under `england_epl/`.

---

## Scope

Detects the following dead-ball restart events from a broadcast match video:

- **Free-kick** (Direct + Indirect combined)
- **Kick-off**
- **Corner**
- **Throw-in**
- **Penalty** *(in scope but only 1 labelled sample in the EPL half-1 corpus, so it
  is dropped at training — `03_train_model.py` removes any class with < 6 samples
  before SMOTE)*

A **Background** class is trained alongside these to suppress false positives during
sliding-window prediction. After the Penalty drop, the model has **5 trained
classes**: Background, Corner, Free-kick, Kick-off, Throw-in.

## Project Structure

```
IDS_Project/
├── 01_download_data.py        # SoccerNet labels + per-game video downloads
├── build_index_only.py        # Build data/event_index.csv from Labels-v2.json
├── 02_extract_features.py     # YOLO + hand-crafted features + background sampling
├── 03_train_model.py          # EDA, SMOTE, Random Forest, evaluation
├── 04_predict.py              # Sliding-window prediction on a new video
├── utils.py                   # Shared feature extraction (used by 02 and 04)
├── england_epl/               # Local SoccerNet EPL games (Labels-v2.json + *_720p.mkv)
├── testing_dataset/           # Held-out match used for prediction (testing_video.mkv)
├── requirements.txt
└── README.md
```

## Setup

This project uses **Python 3.14**.

Developed and run on **Windows (PowerShell)**:

```powershell
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
# CPU-only torch first to avoid pulling multi-GB CUDA wheels via ultralytics
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
```

On Linux/macOS substitute `.venv/bin/pip` for `.venv\Scripts\pip`.

## Run Order

### Step 1 — Data

The EPL games are already present locally under
`england_epl/<season>/<game>/`, each containing `Labels-v2.json` plus the 720p
broadcast halves (`1_720p.mkv`, `2_720p.mkv`). The held-out match used for
prediction lives in `testing_dataset/` as `testing_video.mkv` (+ its
`Labels-v2.json`). **No download step is required to reproduce the current
results.**

`01_download_data.py` remains available if you need to (re-)fetch games from
SoccerNet (`--labels-only`, `--list`, `--game N`, `--status`, `--all`); it is
not part of the standard run order here.

### Step 2 — Build the event index

```bash
python build_index_only.py
```

Writes `data/event_index.csv` with one row per target event found under
`england_epl/<season>/<game>/Labels-v2.json`. The index expects `1_720p.mkv` /
`2_720p.mkv` next to each labels file; only rows where `video_exists == True`
are processed in step 3, so missing halves are skipped automatically.

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

The current `results.csv` was produced by scanning the held-out match:

```bash
python 04_predict.py --video testing_dataset/testing_video.mkv --half 1 --out results.csv
# Limit to the first N seconds (useful for quick tests):
python 04_predict.py --video testing_dataset/testing_video.mkv --max-seconds 300
```

Outputs:
- `results.csv` — timestamped events (event class, confidence, match minute)
- `plots/06_event_timeline.png` — annotated timeline plot

## Results

Feature matrix (`data/features.csv`): **1,622 windows**. Class distribution
(after background sampling, before SMOTE):

| Class      | Windows |
|------------|---------|
| Background | 762     |
| Throw-in   | 466     |
| Free-kick  | 220     |
| Corner     | 119     |
| Kick-off   | 54      |
| Penalty    | 1 *(dropped at training)* |

Random Forest (200 trees, `class_weight="balanced"`, SMOTE on the training
split only):

- **5-fold CV F1 (weighted): ≈ 0.78**
- **Held-out test F1 (weighted): ≈ 0.57**

The CV/test gap is expected: the train/test split is row-level, so windows from
the same game leak across splits and inflate CV. A game-level split is the
highest-value next improvement — see `HANDOVER.md` for the full limitations and
priority list.

Running prediction on `testing_video.mkv` (full ~45 min half) yields **57
detected events** in `results.csv`, dominated by Throw-in and Free-kick with a
few Corner and Kick-off detections. `index.html` is a self-contained dashboard
that visualises this `results.csv` (auto-loads the embedded run; you can also
upload a fresh `results.csv`).
