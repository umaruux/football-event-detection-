"""
build_index_only.py
--------------------
Scans Labels-v2.json files under data/soccernet/ and builds data/event_index.csv
with one row per target event.

Direct + Indirect free-kicks are merged into a single "Free-kick" class.
"""

import csv
from pathlib import Path
import json


DATA_DIR = Path("data/soccernet")
INDEX_OUT = Path("data/event_index.csv")

TARGET_CLASSES = {
    "Penalty",
    "Direct free-kick",
    "Indirect free-kick",
    "Kick-off",
    "Corner",
    "Throw-in",
}
LABEL_RENAMES = {
    "Direct free-kick":   "Free-kick",
    "Indirect free-kick": "Free-kick",
}


def filter_and_summarise(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print("── Event Summary ─────────────────────")
    for label, count in c.most_common():
        print(f"  {label:<15} {count}")
    print("───────────────────────────────────────")


def build_event_index():
    rows = []
    for label_file in DATA_DIR.rglob("Labels-v2.json"):
        game_dir = label_file.parent
        with open(label_file) as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            label = ann.get("label", "")
            if label not in TARGET_CLASSES:
                continue
            game_time = ann.get("gameTime", "1 - 00:00")
            try:
                half_str, time_str = game_time.split(" - ")
                half = int(half_str)
                mm, ss = time_str.split(":")
                position_sec = int(mm) * 60 + int(ss)
            except (ValueError, IndexError):
                continue

            video_path = game_dir / f"{half}_224p.mkv"
            rows.append({
                "game_dir":     game_dir.as_posix(),
                "half":         half,
                "position_sec": position_sec,
                "label":        LABEL_RENAMES.get(label, label),
                "video_path":   video_path.as_posix(),
                "video_exists": video_path.exists(),
            })

    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEvent index saved: {INDEX_OUT}  ({len(rows)} events)")
    return rows


if __name__ == "__main__":
    rows = build_event_index()
    filter_and_summarise(rows)
    have_video = sum(1 for r in rows if r["video_exists"])
    print(f"\nEvents with downloaded video: {have_video}/{len(rows)}")
