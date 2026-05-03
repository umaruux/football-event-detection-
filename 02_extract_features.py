"""
02_extract_features.py
-----------------------
For every event in the index (data/event_index.csv):
  1. Extract a ±WINDOW_SEC frame window around the event timestamp.
  2. Run YOLOv8 on each frame to detect players and the ball.
  3. Compute 14 hand-crafted spatial / temporal features per event window.
  4. Save the resulting feature matrix to data/features.csv.

Requirements:
    pip install ultralytics opencv-python numpy pandas tqdm

Usage:
    python 02_extract_features.py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EVENT_INDEX   = Path("data/event_index.csv")
OUTPUT_CSV    = Path("data/features.csv")
YOLO_MODEL    = "yolov8n.pt"      # nano; use yolov8s.pt for better accuracy
WINDOW_SEC    = 5                  # seconds before/after event to analyse
FRAME_SKIP    = 3                  # process every Nth frame (speed vs quality)
IMG_SIZE      = 640                # YOLO input resolution

# Approximate penalty spot in homography-mapped coordinates (0-1 normalised)
PENALTY_SPOT_NORM = np.array([0.5, 0.88])   # centre-x, near-end-y
# ───────────────────────────────────────────────────────────────────────────────


def load_model():
    model = YOLO(YOLO_MODEL)
    print(f"Loaded YOLO model: {YOLO_MODEL}")
    return model


# ── FRAME EXTRACTION ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, position_sec: float, window: int = WINDOW_SEC):
    """
    Return a list of BGR frames in the [position_sec - window, position_sec + window]
    range, sampled every FRAME_SKIP frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int((position_sec - window) * fps))
    end_frame   = int((position_sec + window) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % FRAME_SKIP == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


# ── YOLO DETECTION ─────────────────────────────────────────────────────────────

def detect_objects(model, frames):
    """
    Run YOLO on each frame. Returns a list of dicts, one per frame:
        {
          'players': np.ndarray of shape (N, 4) in xyxy format (normalised 0-1),
          'ball':    np.ndarray of shape (4,)  or None
        }

    COCO class IDs: 0 = person, 32 = sports ball
    """
    all_detections = []

    for frame in frames:
        h, w = frame.shape[:2]
        results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]

        boxes   = results.boxes.xyxy.cpu().numpy()    # (N, 4) absolute xyxy
        classes = results.boxes.cls.cpu().numpy()     # (N,)

        # Normalise to [0, 1]
        boxes_norm = boxes / np.array([w, h, w, h])

        players = boxes_norm[classes == 0]            # person
        ball_candidates = boxes_norm[classes == 32]   # sports ball
        ball = ball_candidates[0] if len(ball_candidates) > 0 else None

        all_detections.append({"players": players, "ball": ball})

    return all_detections


# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────

def centre(box):
    """Return (cx, cy) from normalised xyxy box."""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def player_clustering_score(players):
    """
    Mean nearest-neighbour distance between player centres.
    Low score → players tightly bunched (e.g. wall formation).
    Returns 1.0 if fewer than 2 players detected.
    """
    if len(players) < 2:
        return 1.0
    centres = np.array([centre(p) for p in players])
    dists = []
    for i, c in enumerate(centres):
        others = np.delete(centres, i, axis=0)
        dists.append(np.min(np.linalg.norm(others - c, axis=1)))
    return float(np.mean(dists))


def ball_distance_to_penalty_spot(ball):
    """
    Euclidean distance from ball centre to the (normalised) penalty spot.
    Returns 1.0 if ball not detected.
    """
    if ball is None:
        return 1.0
    bc = centre(ball)
    return float(np.linalg.norm(bc - PENALTY_SPOT_NORM))


def wall_formation_index(players, ball, tolerance=0.05):
    """
    Count of players whose centres are approximately collinear (within
    `tolerance` of the line perpendicular to the ball) and within
    ~9.15m (≈0.10 normalised width) of the ball.
    Heuristic indicator of a free-kick wall.
    Returns 0 if ball not detected.
    """
    if ball is None or len(players) < 3:
        return 0

    bc = centre(ball)
    nearby = [p for p in players if np.linalg.norm(centre(p) - bc) < 0.15]

    if len(nearby) < 3:
        return 0

    centres = np.array([centre(p) for p in nearby])
    # Count players whose y-coordinates (rows) are within `tolerance` of the same horizontal band
    y_vals = centres[:, 1]
    median_y = np.median(y_vals)
    collinear = int(np.sum(np.abs(y_vals - median_y) < tolerance))
    return collinear


def optical_flow_magnitude(frames):
    """
    Mean Dense optical flow magnitude across all consecutive frame pairs.
    Low magnitude → little motion → likely set-piece setup.
    Returns 0.0 if fewer than 2 frames.
    """
    if len(frames) < 2:
        return 0.0

    mags = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(np.mean(magnitude)))
        prev_gray = gray
    return float(np.mean(mags))


def penalty_area_player_count(players):
    """
    Count players whose centres fall inside the penalty area region
    (approximately the bottom sixth of the pitch, full width).
    """
    count = sum(1 for p in players if centre(p)[1] > 0.75)
    return count


def ball_in_penalty_area(ball):
    """1 if ball centre is in the penalty area, 0 otherwise."""
    if ball is None:
        return 0
    bc = centre(ball)
    return int(bc[1] > 0.75 and 0.18 < bc[0] < 0.82)


def temporal_delta(detections):
    """
    Mean absolute change in the number of players detected frame-over-frame.
    High delta → players repositioning (action / foul); low delta → static setup.
    """
    counts = [len(d["players"]) for d in detections]
    if len(counts) < 2:
        return 0.0
    deltas = [abs(counts[i] - counts[i - 1]) for i in range(1, len(counts))]
    return float(np.mean(deltas))


def ball_detected_ratio(detections):
    """Fraction of frames in which the ball was detected."""
    if not detections:
        return 0.0
    detected = sum(1 for d in detections if d["ball"] is not None)
    return detected / len(detections)


def avg_player_count(detections):
    """Mean number of players detected per frame."""
    if not detections:
        return 0.0
    return float(np.mean([len(d["players"]) for d in detections]))


def compute_features(frames, detections):
    """
    Aggregate all per-frame detections into a single 14-dimensional
    feature vector for the event window.

    Returns a dict of feature name → value.
    """
    # Use the middle frame's detections as the "snapshot" features
    mid = len(detections) // 2
    snap = detections[mid]
    players_snap = snap["players"]
    ball_snap    = snap["ball"]

    # --- Snapshot features (from middle frame) ---
    f_clustering    = player_clustering_score(players_snap)
    f_ball_dist     = ball_distance_to_penalty_spot(ball_snap)
    f_wall          = wall_formation_index(players_snap, ball_snap)
    f_in_box_count  = penalty_area_player_count(players_snap)
    f_ball_in_box   = ball_in_penalty_area(ball_snap)

    # --- Window-aggregate features ---
    f_flow          = optical_flow_magnitude(frames)
    f_temporal_d    = temporal_delta(detections)
    f_ball_ratio    = ball_detected_ratio(detections)
    f_avg_players   = avg_player_count(detections)

    # --- Mean versions of snapshot features across all frames ---
    f_mean_wall     = float(np.mean([
        wall_formation_index(d["players"], d["ball"]) for d in detections
    ]))
    f_mean_dist     = float(np.mean([
        ball_distance_to_penalty_spot(d["ball"]) for d in detections
    ]))
    f_mean_cluster  = float(np.mean([
        player_clustering_score(d["players"]) for d in detections
    ]))
    f_mean_in_box   = float(np.mean([
        penalty_area_player_count(d["players"]) for d in detections
    ]))
    f_mean_ball_box = float(np.mean([
        ball_in_penalty_area(d["ball"]) for d in detections
    ]))

    return {
        "clustering_mid":      f_clustering,
        "ball_dist_spot_mid":  f_ball_dist,
        "wall_index_mid":      f_wall,
        "players_in_box_mid":  f_in_box_count,
        "ball_in_box_mid":     f_ball_in_box,
        "optical_flow":        f_flow,
        "temporal_delta":      f_temporal_d,
        "ball_detected_ratio": f_ball_ratio,
        "avg_player_count":    f_avg_players,
        "mean_wall_index":     f_mean_wall,
        "mean_ball_dist_spot": f_mean_dist,
        "mean_clustering":     f_mean_cluster,
        "mean_players_in_box": f_mean_in_box,
        "mean_ball_in_box":    f_mean_ball_box,
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    index = pd.read_csv(EVENT_INDEX)
    # Keep only rows where the video file exists
    index = index[index["video_exists"] == True].reset_index(drop=True)
    print(f"Processing {len(index)} events with valid video paths...")

    model = load_model()
    records = []

    for _, row in tqdm(index.iterrows(), total=len(index), desc="Extracting features"):
        video_path   = row["video_path"]
        position_sec = float(row["position_sec"])
        label        = row["label"]

        frames = extract_frames(video_path, position_sec)
        if len(frames) < 2:
            print(f"  Skipping (not enough frames): {video_path} @ {position_sec}s")
            continue

        detections = detect_objects(model, frames)
        features   = compute_features(frames, detections)
        features["label"] = label
        records.append(features)

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeature matrix saved: {OUTPUT_CSV}  shape={df.shape}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
