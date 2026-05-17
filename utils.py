"""
utils.py
---------
Shared feature extraction functions used by both
02_extract_features.py and 04_predict.py.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
YOLO_MODEL         = "yolov8n.pt"
WINDOW_SEC         = 5
FRAME_SKIP         = 10
IMG_SIZE           = 640
PENALTY_SPOTS_NORM = np.array([[0.5, 0.12], [0.5, 0.88]])
FIELD_CORNERS_NORM = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
FIELD_CENTER_NORM  = np.array([0.5, 0.5])

FEATURE_COLS = [
    "clustering_mid",
    "ball_dist_spot_mid",
    "wall_index_mid",
    "players_in_box_mid",
    "ball_in_box_mid",
    "ball_dist_center_mid",
    "ball_dist_corner_mid",
    "ball_dist_sideline_mid",
    "optical_flow",
    "temporal_delta",
    "ball_detected_ratio",
    "avg_player_count",
    "mean_wall_index",
    "mean_ball_dist_spot",
    "mean_clustering",
    "mean_players_in_box",
    "mean_ball_in_box",
    "mean_ball_dist_center",
    "mean_ball_dist_corner",
    "mean_ball_dist_sideline",
]
# ───────────────────────────────────────────────────────────────────────────────


def load_model():
    model = YOLO(YOLO_MODEL)
    print(f"Loaded YOLO model: {YOLO_MODEL}")
    return model


def centre(box):
    """Return (cx, cy) from normalised xyxy box."""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def detect_objects(model, frames):
    all_detections = []
    for frame in frames:
        h, w = frame.shape[:2]
        results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        boxes_norm = boxes / np.array([w, h, w, h])
        players = boxes_norm[classes == 0]
        ball_candidates = boxes_norm[classes == 32]
        ball = ball_candidates[0] if len(ball_candidates) > 0 else None
        all_detections.append({"players": players, "ball": ball})
    return all_detections


def player_clustering_score(players):
    if len(players) < 2:
        return 1.0
    centres = np.array([centre(p) for p in players])
    dists = []
    for i, c in enumerate(centres):
        others = np.delete(centres, i, axis=0)
        dists.append(np.min(np.linalg.norm(others - c, axis=1)))
    return float(np.mean(dists))


def ball_distance_to_penalty_spot(ball):
    if ball is None:
        return 1.0
    bc = centre(ball)
    return float(np.min(np.linalg.norm(PENALTY_SPOTS_NORM - bc, axis=1)))


def ball_distance_to_center(ball):
    if ball is None:
        return 1.0
    return float(np.linalg.norm(centre(ball) - FIELD_CENTER_NORM))


def ball_distance_to_corner(ball):
    if ball is None:
        return 1.0
    bc = centre(ball)
    return float(np.min(np.linalg.norm(FIELD_CORNERS_NORM - bc, axis=1)))


def ball_distance_to_sideline(ball):
    """Throw-ins occur at the touchlines which appear as horizontal edges in broadcast."""
    if ball is None:
        return 0.5
    bc = centre(ball)
    return float(min(bc[1], 1.0 - bc[1]))


def wall_formation_index(players, ball, tolerance=0.05):
    if ball is None or len(players) < 3:
        return 0
    bc = centre(ball)
    nearby = [p for p in players if np.linalg.norm(centre(p) - bc) < 0.15]
    if len(nearby) < 3:
        return 0
    centres = np.array([centre(p) for p in nearby])
    median_y = np.median(centres[:, 1])
    return int(np.sum(np.abs(centres[:, 1] - median_y) < tolerance))


def optical_flow_magnitude(frames):
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
    return sum(1 for p in players if centre(p)[1] > 0.75)


def ball_in_penalty_area(ball):
    if ball is None:
        return 0
    bc = centre(ball)
    return int(bc[1] > 0.75 and 0.18 < bc[0] < 0.82)


def temporal_delta(detections):
    counts = [len(d["players"]) for d in detections]
    if len(counts) < 2:
        return 0.0
    return float(np.mean([abs(counts[i] - counts[i-1]) for i in range(1, len(counts))]))


def ball_detected_ratio(detections):
    if not detections:
        return 0.0
    return sum(1 for d in detections if d["ball"] is not None) / len(detections)


def avg_player_count(detections):
    if not detections:
        return 0.0
    return float(np.mean([len(d["players"]) for d in detections]))


def compute_features(frames, detections):
    mid  = len(detections) // 2
    snap = detections[mid]
    players_snap = snap["players"]
    ball_snap    = snap["ball"]

    return {
        "clustering_mid":         player_clustering_score(players_snap),
        "ball_dist_spot_mid":     ball_distance_to_penalty_spot(ball_snap),
        "wall_index_mid":         wall_formation_index(players_snap, ball_snap),
        "players_in_box_mid":     penalty_area_player_count(players_snap),
        "ball_in_box_mid":        ball_in_penalty_area(ball_snap),
        "ball_dist_center_mid":   ball_distance_to_center(ball_snap),
        "ball_dist_corner_mid":   ball_distance_to_corner(ball_snap),
        "ball_dist_sideline_mid": ball_distance_to_sideline(ball_snap),
        "optical_flow":           optical_flow_magnitude(frames),
        "temporal_delta":         temporal_delta(detections),
        "ball_detected_ratio":    ball_detected_ratio(detections),
        "avg_player_count":       avg_player_count(detections),
        "mean_wall_index":        float(np.mean([wall_formation_index(d["players"], d["ball"]) for d in detections])),
        "mean_ball_dist_spot":    float(np.mean([ball_distance_to_penalty_spot(d["ball"]) for d in detections])),
        "mean_clustering":        float(np.mean([player_clustering_score(d["players"]) for d in detections])),
        "mean_players_in_box":    float(np.mean([penalty_area_player_count(d["players"]) for d in detections])),
        "mean_ball_in_box":       float(np.mean([ball_in_penalty_area(d["ball"]) for d in detections])),
        "mean_ball_dist_center":  float(np.mean([ball_distance_to_center(d["ball"]) for d in detections])),
        "mean_ball_dist_corner":  float(np.mean([ball_distance_to_corner(d["ball"]) for d in detections])),
        "mean_ball_dist_sideline":float(np.mean([ball_distance_to_sideline(d["ball"]) for d in detections])),
    }