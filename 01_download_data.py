"""
01_download_data.py
--------------------
Downloads SoccerNet annotation labels and video clips for the target event
classes: Penalty, Foul, Yellow card, Red card.

Requirements:
    pip install SoccerNet

Usage:
    python 01_download_data.py

Outputs:
    data/soccernet/          <- annotation JSON files per match
"""

import os
import json
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("C:/Users/Bisma/AppData/Roaming/Python/Python314/site-packages/SoccerNet")
TARGET_CLASSES = {"Penalty", "Foul", "Yellow card", "Red card"}
SPLITS = ["train", "valid", "test"]

# SoccerNet requires a password — register at https://www.soccer-net.org/data
# to get one and paste it below.
SOCCERNET_PASSWORD = "s0cc3rn3t"
# ───────────────────────────────────────────────────────────────────────────────


def download_annotations():
    """Download Labels-v2.json annotation files for all splits."""
    from SoccerNet.Downloader import SoccerNetDownloader

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(DATA_DIR))
    downloader.password = SOCCERNET_PASSWORD

    print("Downloading annotation labels...")
    downloader.downloadGames(
        files=["Labels-v2.json"],
        split=SPLITS,
        overwrite=False
    )
    print(f"Annotations saved to: {DATA_DIR}")


def download_videos():
    """
    Download 720p MKV video clips.
    WARNING: Full dataset is large (~1TB). Download only what you need.
    """
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=str(DATA_DIR))
    downloader.password = SOCCERNET_PASSWORD

    print("Downloading 720p videos (this may take a while)...")
    downloader.downloadGames(
        files=["1_720p.mkv", "2_720p.mkv"],  # first and second halves
        split=SPLITS,
        overwrite=False
    )
    print("Videos downloaded.")


def filter_and_summarise():
    """
    Walk all downloaded annotation files, filter for target classes,
    and print a summary of event counts per class.
    """
    counts = {cls: 0 for cls in TARGET_CLASSES}
    total_games = 0

    for label_file in DATA_DIR.rglob("Labels-v2.json"):
        total_games += 1
        with open(label_file, "r") as f:
            data = json.load(f)

        for annotation in data.get("annotations", []):
            label = annotation.get("label", "")
            if label in TARGET_CLASSES:
                counts[label] += 1

    print("\n── Dataset Summary ──────────────────")
    print(f"  Total games found : {total_games}")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<15}: {count} events")
    print("─────────────────────────────────────")
    return counts


def build_event_index():
    """
    Build a flat CSV index of all target events with their:
    - game path
    - half (1 or 2)
    - position (seconds)
    - label
    - video path

    Saves to data/event_index.csv
    """
    import csv

    rows = []
    for label_file in DATA_DIR.rglob("Labels-v2.json"):
        game_dir = label_file.parent
        with open(label_file, "r") as f:
            data = json.load(f)

        for ann in data.get("annotations", []):
            label = ann.get("label", "")
            if label not in TARGET_CLASSES:
                continue

            half = int(ann.get("gameTime", "1 - 00:00").split(" - ")[0])
            time_str = ann.get("gameTime", "1 - 00:00").split(" - ")[1]
            mm, ss = time_str.split(":")
            position_sec = int(mm) * 60 + int(ss)

            video_name = f"{half}_720p.mkv"
            video_path = game_dir / video_name

            rows.append({
                "game_dir": str(game_dir),
                "half": half,
                "position_sec": position_sec,
                "label": label,
                "video_path": str(video_path),
                "video_exists": video_path.exists(),
            })

    out_path = Path("data/event_index.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvent index saved to: {out_path}  ({len(rows)} events)")
    return out_path


if __name__ == "__main__":
    download_annotations()
    # Uncomment below to also download videos (large download):
    # download_videos()

    filter_and_summarise()
    build_event_index()
