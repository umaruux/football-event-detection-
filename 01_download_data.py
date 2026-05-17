"""
01_download_data.py
--------------------
Downloads SoccerNet annotation labels for the test split, and lets you
download 224p videos for individual games one at a time.

Step A (fast, ~5 min): all Labels-v2.json for the test split.
Step B (slow, ~25 min per half): videos, downloaded ONE GAME AT A TIME so
failures only affect that one game.

Usage:
    # Step A — get all labels for the test split
    python 01_download_data.py --labels-only

    # Show the planned 15-game list (1-indexed)
    python 01_download_data.py --list

    # Download one game (by 1-indexed slot in the planned list)
    python 01_download_data.py --game 1
    python 01_download_data.py --game 2
    ...

    # Show which games already have both halves downloaded
    python 01_download_data.py --status

    # (Optional) download every game in the planned list in one go
    python 01_download_data.py --all
"""

import argparse
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader


LOCAL_DIR = Path("data/soccernet")
PASSWORD  = "s0cc3rn3t"
SPLIT     = "test"
NUM_GAMES = 15
VIDEO_FILES = ["1_224p.mkv", "2_224p.mkv"]


def make_downloader():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    d = SoccerNetDownloader(LocalDirectory=str(LOCAL_DIR))
    d.password = PASSWORD
    return d


def all_games_with_labels():
    """Return a sorted list of game paths (relative to LOCAL_DIR) that have Labels-v2.json.
    Forward slashes only — SoccerNet's URL construction needs them on Windows too."""
    return sorted(
        p.parent.relative_to(LOCAL_DIR).as_posix()
        for p in LOCAL_DIR.rglob("Labels-v2.json")
    )


def planned_games():
    """The fixed list of games to use for training (first NUM_GAMES alphabetically)."""
    games = all_games_with_labels()
    if not games:
        raise SystemExit(
            "No labels found. Run `python 01_download_data.py --labels-only` first."
        )
    return games[:NUM_GAMES]


def game_status(game):
    """Return dict of which files exist locally for this game."""
    game_dir = LOCAL_DIR / game
    return {fname: (game_dir / fname).exists() for fname in VIDEO_FILES}


def cmd_labels_only():
    d = make_downloader()
    print(f"Downloading Labels-v2.json for split={SPLIT}...")
    d.downloadGames(files=["Labels-v2.json"], split=[SPLIT])
    print(f"Labels downloaded into {LOCAL_DIR}.")


def cmd_list():
    games = planned_games()
    print(f"Planned training corpus ({len(games)} games):\n")
    for i, g in enumerate(games, 1):
        st = game_status(g)
        marks = "".join("✓" if st[f] else "·" for f in VIDEO_FILES)
        print(f"  [{i:>2}] {marks}  {g}")
    print("\nLegend: [✓ ·] = [half1 half2]   ✓=downloaded  ·=missing")
    print("\nRun `python 01_download_data.py --game N` to download game N (1-indexed).")


def cmd_status():
    games = planned_games()
    done = 0
    for i, g in enumerate(games, 1):
        st = game_status(g)
        if all(st.values()):
            done += 1
        marks = "".join("✓" if st[f] else "·" for f in VIDEO_FILES)
        print(f"  [{i:>2}] {marks}  {g}")
    print(f"\n{done}/{len(games)} games fully downloaded.")


def download_one(game):
    d = make_downloader()
    print(f"\nDownloading: {game}")
    try:
        d.downloadGame(game=game, files=VIDEO_FILES, spl=SPLIT)
        print(f"  Done: {game}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def cmd_game(n):
    games = planned_games()
    if not (1 <= n <= len(games)):
        raise SystemExit(f"--game must be between 1 and {len(games)}")
    download_one(games[n - 1])


def cmd_all():
    games = planned_games()
    print(f"Downloading {len(games)} games sequentially...")
    failed = []
    for i, g in enumerate(games, 1):
        print(f"\n[{i}/{len(games)}]")
        if not download_one(g):
            failed.append(g)
    if failed:
        print(f"\n{len(failed)} games failed:")
        for g in failed:
            print(f"  {g}")
        print("\nRetry individually with --game N (see --list for indices).")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labels-only", action="store_true",
                       help="Download Labels-v2.json for the whole test split.")
    group.add_argument("--list", action="store_true",
                       help="Show the planned 15-game list with download status.")
    group.add_argument("--status", action="store_true",
                       help="Show download status of the planned 15 games.")
    group.add_argument("--game", type=int, metavar="N",
                       help="Download one game by 1-indexed slot in the planned list.")
    group.add_argument("--all", action="store_true",
                       help="Download every planned game sequentially (long).")
    args = parser.parse_args()

    if args.labels_only:
        cmd_labels_only()
    elif args.list:
        cmd_list()
    elif args.status:
        cmd_status()
    elif args.game is not None:
        cmd_game(args.game)
    elif args.all:
        cmd_all()


if __name__ == "__main__":
    main()
