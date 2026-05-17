"""
02_extract_features.py
-----------------------
For every event in data/event_index.csv where the video is downloaded:
  1. Extract a ±WINDOW_SEC frame window around the event timestamp.
  2. Run YOLOv8 on each sampled frame (players + ball).
  3. Compute hand-crafted spatial / temporal features.
  4. Append to data/features.csv.

Then, for each unique video that has at least one labelled event:
  5. Sample N_BACKGROUND random timestamps at least MIN_GAP seconds from
     ANY annotation in that game's Labels-v2.json, extract features, label
     them "Background".

Resume support: rows already in features.csv (identified by source_video +
position_sec + label) are skipped on re-run.
"""

import json
import random
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_model,
    detect_objects,
    compute_features,
    FEATURE_COLS,
    WINDOW_SEC,
    FRAME_SKIP,
)

EVENT_INDEX  = Path("data/event_index.csv")
OUTPUT_CSV   = Path("data/features.csv")
N_BACKGROUND = 40        # background samples per video
MIN_GAP      = 20        # seconds away from any annotation
RANDOM_SEED  = 42

CSV_COLUMNS = FEATURE_COLS + ["label", "source_video", "position_sec"]


def extract_frames(video_path, position_sec, window=WINDOW_SEC):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int((position_sec - window) * fps))
    end_frame   = int((position_sec + window) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    idx = start_frame
    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % FRAME_SKIP == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def append_row(features, label, video_path, position_sec):
    row = dict(features)
    row["label"] = label
    row["source_video"] = video_path
    row["position_sec"] = position_sec
    write_header = not OUTPUT_CSV.exists()
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
        OUTPUT_CSV, mode="a", header=write_header, index=False
    )


def load_already_done():
    """Return a set of (source_video, position_sec, label) keys already in features.csv."""
    if not OUTPUT_CSV.exists():
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    if set(FEATURE_COLS) - set(df.columns):
        raise SystemExit(
            f"features.csv has stale schema. Delete it and re-run:\n"
            f"  rm {OUTPUT_CSV}"
        )
    return {
        (r["source_video"], int(r["position_sec"]), r["label"])
        for _, r in df.iterrows()
    }


def get_all_annotation_seconds(game_dir, half):
    """Return all annotation times (seconds) for a given (game_dir, half)."""
    label_file = Path(game_dir) / "Labels-v2.json"
    if not label_file.exists():
        return []
    with open(label_file) as f:
        data = json.load(f)
    times = []
    for ann in data.get("annotations", []):
        gt = ann.get("gameTime", "")
        try:
            h_str, t_str = gt.split(" - ")
            if int(h_str) != half:
                continue
            mm, ss = t_str.split(":")
            times.append(int(mm) * 60 + int(ss))
        except (ValueError, IndexError):
            continue
    return sorted(times)


def video_duration_sec(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps


def sample_background_times(annotation_times, total_sec, n, min_gap):
    """Pick n timestamps that are >= min_gap seconds from every annotation."""
    rng = random.Random(RANDOM_SEED + int(total_sec))
    picked = []
    attempts = 0
    while len(picked) < n and attempts < n * 50:
        attempts += 1
        t = rng.uniform(WINDOW_SEC, max(WINDOW_SEC + 1, total_sec - WINDOW_SEC))
        if all(abs(t - a) >= min_gap for a in annotation_times):
            if all(abs(t - p) >= min_gap for p in picked):
                picked.append(int(t))
    return picked


def main():
    if not EVENT_INDEX.exists():
        raise SystemExit("event_index.csv missing. Run `python build_index_only.py` first.")

    index = pd.read_csv(EVENT_INDEX)
    index = index[index["video_exists"] == True].reset_index(drop=True)
    if len(index) == 0:
        raise SystemExit("No videos downloaded yet. Run `python 01_download_data.py --game N` first.")

    done = load_already_done()
    print(f"Loaded {len(done)} already-processed rows from features.csv.")

    model = load_model()

    # ── Pass 1: labelled events ───────────────────────────────────────────────
    print(f"\nPass 1: extracting features for {len(index)} labelled events...")
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Events"):
        video_path   = row["video_path"]
        position_sec = int(row["position_sec"])
        label        = row["label"]

        if (video_path, position_sec, label) in done:
            continue

        frames = extract_frames(video_path, position_sec)
        if len(frames) < 2:
            continue
        dets = detect_objects(model, frames)
        feats = compute_features(frames, dets)
        append_row(feats, label, video_path, position_sec)
        done.add((video_path, position_sec, label))

    # ── Pass 2: background sampling per video ────────────────────────────────
    print(f"\nPass 2: sampling background windows...")
    videos = index.groupby(["game_dir", "half", "video_path"]).size().reset_index()
    for _, vrow in tqdm(videos.iterrows(), total=len(videos), desc="Videos"):
        video_path = vrow["video_path"]
        existing_bg = sum(
            1 for (vp, _, lbl) in done if vp == video_path and lbl == "Background"
        )
        if existing_bg >= N_BACKGROUND:
            continue

        ann_times = get_all_annotation_seconds(vrow["game_dir"], int(vrow["half"]))
        total_sec = video_duration_sec(video_path)
        if total_sec < 60:
            continue

        needed = N_BACKGROUND - existing_bg
        bg_times = sample_background_times(ann_times, total_sec, needed, MIN_GAP)

        for t in bg_times:
            if (video_path, t, "Background") in done:
                continue
            frames = extract_frames(video_path, t)
            if len(frames) < 2:
                continue
            dets = detect_objects(model, frames)
            feats = compute_features(frames, dets)
            append_row(feats, "Background", video_path, t)
            done.add((video_path, t, "Background"))

    print(f"\nDone. Features saved to: {OUTPUT_CSV}")
    final = pd.read_csv(OUTPUT_CSV)
    print(final["label"].value_counts())


if __name__ == "__main__":
    main()
