"""
04_predict.py
--------------
Runs the trained model on a new match video and outputs a timeline of
detected events (free kicks, penalties, fouls, cards) with timestamps.

Usage:
    python 04_predict.py --video path/to/match.mkv [--half 1] [--out results.csv]

Output:
    - Prints detected events to console.
    - Saves a CSV of events with timestamps.
    - Saves an annotated timeline plot.
"""

import argparse
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Re-use feature functions from 02_extract_features.py
from extract_features_utils import (   # see note below *
    detect_objects,
    compute_features,
    load_model as load_yolo,
    FEATURE_COLS,
)

# * If running standalone without import, copy the functions from
#   02_extract_features.py directly into this file.

# ── CONFIG ─────────────────────────────────────────────────────────────────────
RF_MODEL_PATH  = Path("models/rf_model.joblib")
ENCODER_PATH   = Path("models/label_encoder.joblib")
YOLO_MODEL     = "yolov8n.pt"

SCAN_STEP_SEC  = 3        # slide the detection window every N seconds
WINDOW_SEC     = 5        # half-width of each detection window (seconds)
FRAME_SKIP     = 3        # process every Nth frame
CONFIDENCE_THR = 0.55     # minimum class probability to register an event
COOLDOWN_SEC   = 30       # ignore further detections of the same class for N sec
SMOOTHING_WIN  = 5        # majority-vote over this many consecutive windows

CLASS_COLORS = {
    "Penalty":     "#534AB7",
    "Foul":        "#993C1D",
    "Yellow card": "#BA7517",
    "Red card":    "#A32D2D",
    "Free kick":   "#0F6E56",
}
# ───────────────────────────────────────────────────────────────────────────────


def load_models():
    rf  = joblib.load(RF_MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    yolo = load_yolo()
    print(f"Loaded RF model from: {RF_MODEL_PATH}")
    return rf, le, yolo


def sliding_window_positions(total_sec, step=SCAN_STEP_SEC):
    """Generate centre timestamps for each sliding window."""
    t = WINDOW_SEC
    while t < total_sec - WINDOW_SEC:
        yield t
        t += step


def extract_window_frames(cap, fps, position_sec):
    """Extract frames around `position_sec` from an open VideoCapture."""
    start_frame = max(0, int((position_sec - WINDOW_SEC) * fps))
    end_frame   = int((position_sec + WINDOW_SEC) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_SKIP == 0:
            frames.append(frame)
    return frames


def rule_based_override(event_class, features):
    """
    Post-model rule layer.
    Returns the final class (may override the model's prediction).
    """
    # If model says Penalty but ball is not in penalty area → demote to Foul
    if event_class == "Penalty" and features.get("mean_ball_in_box", 0) < 0.3:
        return "Foul"

    # If model says Free kick but wall index is very low → demote to Foul
    if event_class == "Free kick" and features.get("mean_wall_index", 0) < 1:
        return "Foul"

    return event_class


def apply_cooldown(detections, cooldown=COOLDOWN_SEC):
    """
    Remove duplicate detections of the same class within `cooldown` seconds.
    """
    last_seen = {}
    filtered = []
    for det in sorted(detections, key=lambda x: x["time_sec"]):
        cls = det["event"]
        t   = det["time_sec"]
        if cls not in last_seen or (t - last_seen[cls]) > cooldown:
            filtered.append(det)
            last_seen[cls] = t
    return filtered


def format_time(sec, half=1):
    """Convert seconds-in-half to match minute string, e.g. '23:14'."""
    offset = (half - 1) * 45
    total_min = int(sec // 60) + offset
    s = int(sec % 60)
    return f"{total_min}'{s:02d}\""


def predict_video(video_path, half=1, out_csv="results.csv"):
    rf, le, yolo = load_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec    = total_frames / fps
    print(f"Video: {video_path}  |  FPS={fps:.1f}  |  Duration={total_sec/60:.1f} min")

    raw_detections = []   # before cooldown
    window_preds   = []   # (time, class_proba_vector) for smoothing

    positions = list(sliding_window_positions(total_sec))
    for pos in tqdm(positions, desc="Scanning video"):
        frames = extract_window_frames(cap, fps, pos)
        if len(frames) < 2:
            continue

        dets     = detect_objects(yolo, frames)
        feat_dict = compute_features(frames, dets)
        X        = np.array([[feat_dict[c] for c in FEATURE_COLS]])
        proba    = rf.predict_proba(X)[0]
        window_preds.append((pos, proba))

    cap.release()

    # ── Temporal smoothing: majority vote over SMOOTHING_WIN windows ──────────
    confirmed = []
    for i in range(len(window_preds)):
        start = max(0, i - SMOOTHING_WIN // 2)
        end   = min(len(window_preds), i + SMOOTHING_WIN // 2 + 1)
        window_slice = [window_preds[j][1] for j in range(start, end)]
        mean_proba = np.mean(window_slice, axis=0)
        best_idx   = int(np.argmax(mean_proba))
        best_prob  = float(mean_proba[best_idx])
        best_class = le.classes_[best_idx]

        if best_prob >= CONFIDENCE_THR and best_class != "background":
            pos = window_preds[i][0]
            # Re-compute features for rule layer (use stored window)
            feat_dict_i = {}   # in a full implementation, cache these above
            final_class = rule_based_override(best_class, feat_dict_i)
            confirmed.append({
                "time_sec": pos,
                "event":    final_class,
                "confidence": round(best_prob, 3),
                "match_time": format_time(pos, half),
            })

    # ── Cooldown deduplication ────────────────────────────────────────────────
    results = apply_cooldown(confirmed)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n── Detected Events ({len(results)}) ───────────────────────────────")
    for r in results:
        print(f"  {r['match_time']:<10} {r['event']:<15} conf={r['confidence']:.2f}")
    print("──────────────────────────────────────────────────────────────────")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")

    # ── Timeline plot ─────────────────────────────────────────────────────────
    plot_timeline(results, total_sec, half, video_path)

    return results


def plot_timeline(results, total_sec, half, video_path):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, total_sec / 60)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#888", lw=1.5, zorder=1)

    for r in results:
        color = CLASS_COLORS.get(r["event"], "#888")
        ax.scatter(r["time_sec"] / 60, 0, color=color, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.text(r["time_sec"] / 60, 0.15, r["match_time"],
                ha="center", va="bottom", fontsize=7.5,
                color=color, rotation=45)

    patches = [mpatches.Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.5)
    ax.set_xlabel("Match minute")
    ax.set_title(f"Event timeline — {Path(video_path).name} (Half {half})")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = Path("plots") / "06_event_timeline.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Timeline plot saved: {plot_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict football events in a video")
    parser.add_argument("--video", required=True, help="Path to .mkv match video")
    parser.add_argument("--half",  type=int, default=1, help="Match half (1 or 2)")
    parser.add_argument("--out",   default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    predict_video(args.video, half=args.half, out_csv=args.out)
