"""
build_index_only.py
--------------------
Run this instead of 01_download_data.py if you already have
annotations and videos downloaded locally.
"""

import json
import csv
from pathlib import Path

DATA_DIR = Path("C:/Users/Bisma/AppData/Roaming/Python/Python314/site-packages/SoccerNet")   # <-- update this to your actual folder path
TARGET_CLASSES = {"Penalty", "Foul", "Yellow card", "Red card"}

def filter_and_summarise():
    counts = {cls: 0 for cls in TARGET_CLASSES}
    total_games = 0
    for label_file in DATA_DIR.rglob("Labels-v2.json"):
        total_games += 1
        with open(label_file, "r") as f:
            data = json.load(f)
        for annotation in data.get("annotations", []):
            if annotation.get("label", "") in TARGET_CLASSES:
                counts[annotation["label"]] += 1
    print(f"Total games found: {total_games}")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<15}: {count} events")

def build_event_index():
    rows = []
    for label_file in DATA_DIR.rglob("Labels-v2.json"):
        game_dir = label_file.parent
        with open(label_file, "r") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            if ann.get("label", "") not in TARGET_CLASSES:
                continue
            half = int(ann.get("gameTime", "1 - 00:00").split(" - ")[0])
            time_str = ann.get("gameTime", "1 - 00:00").split(" - ")[1]
            mm, ss = time_str.split(":")
            position_sec = int(mm) * 60 + int(ss)
            video_path = game_dir / f"{half}_720p.mkv"
            rows.append({
                "game_dir":     str(game_dir),
                "half":         half,
                "position_sec": position_sec,
                "label":        ann["label"],
                "video_path":   str(video_path),
                "video_exists": video_path.exists(),
            })
    out_path = Path("data/event_index.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Event index saved: {out_path}  ({len(rows)} events)")

if __name__ == "__main__":
    filter_and_summarise()
    build_event_index()