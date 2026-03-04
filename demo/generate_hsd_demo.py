#!/usr/bin/env python3
"""
Generate HSD demo assets (collages, GIFs, scenarios_hsd.json).

Selects 20 diverse evaluation intervals from HSD, creates 1x8 collages
and 8-frame GIFs at 500ms intervals, then writes the demo JSON.

Usage:
    cd /home/hiwi/VENUSS_supplementary
    python demo/generate_hsd_demo.py
"""

import json
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Add VENUSS project root for imports
VENUSS_ROOT = Path("/home/hiwi/VENUSS")
sys.path.insert(0, str(VENUSS_ROOT))

from config.datasets import get_dataset_config
from utils.collage_manager import CollageManager
from utils.gif_manager import GifManager

# Paths
EVAL_INTERVALS_FILE = VENUSS_ROOT / "datasets" / "hsd" / "evaluation_intervals_100.json"
SCENARIO_ANNOTATIONS_FILE = VENUSS_ROOT / "datasets" / "hsd" / "scenario_annotations.json"

DEMO_DIR = Path(__file__).parent
COLLAGES_DIR = DEMO_DIR / "assets" / "collages_hsd"
GIFS_DIR = DEMO_DIR / "assets" / "gifs_hsd"
OUTPUT_JSON = DEMO_DIR / "data" / "scenarios_hsd.json"

NUM_SCENARIOS = 20
GRID_CONFIG = (1, 8)  # 1x8 horizontal collage
NUM_FRAMES = 8
INTERVAL = "500ms"
RANDOM_SEED = 42


def load_data():
    with open(EVAL_INTERVALS_FILE) as f:
        intervals = json.load(f)
    with open(SCENARIO_ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    ann_by_id = {a["id"]: a for a in annotations}
    return intervals, ann_by_id


def select_diverse_intervals(intervals, num=NUM_SCENARIOS):
    """Select intervals with maximum scenario diversity (~1-2 per unique scenario_id)."""
    random.seed(RANDOM_SEED)

    # Group by scenario_id
    by_sid = defaultdict(list)
    for iv in intervals:
        by_sid[iv["scenario_id"]].append(iv)

    unique_sids = sorted(by_sid.keys())
    print(f"Unique scenario types: {len(unique_sids)}")

    selected = []

    # Round 1: one per scenario_id
    for sid in unique_sids:
        candidates = by_sid[sid]
        # Prefer intervals with reasonable frame range (not too short)
        candidates_sorted = sorted(candidates, key=lambda x: x["end_frame"] - x["start_frame"], reverse=True)
        selected.append(candidates_sorted[0])

    # Round 2: fill remaining slots with second picks from different scenario_ids
    remaining = num - len(selected)
    if remaining > 0:
        selected_keys = {(s["video_name"], s["start_frame"]) for s in selected}
        pool = []
        for sid in unique_sids:
            for iv in by_sid[sid]:
                key = (iv["video_name"], iv["start_frame"])
                if key not in selected_keys:
                    pool.append(iv)
        random.shuffle(pool)
        # Prefer scenario_ids we've only seen once, spread across types
        seen_counts = defaultdict(int)
        for s in selected:
            seen_counts[s["scenario_id"]] += 1
        pool.sort(key=lambda x: seen_counts[x["scenario_id"]])
        for iv in pool:
            if remaining <= 0:
                break
            selected.append(iv)
            seen_counts[iv["scenario_id"]] += 1
            remaining -= 1

    random.shuffle(selected)
    return selected[:num]


def generate_assets(selected_intervals, ann_by_id):
    """Generate collages and GIFs for selected intervals."""
    hsd_config = get_dataset_config("hsd")

    collage_mgr = CollageManager(storage_mode="temp", dataset_config=hsd_config)
    gif_mgr = GifManager(storage_mode="temp", dataset_config=hsd_config)

    COLLAGES_DIR.mkdir(parents=True, exist_ok=True)
    GIFS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_out = []
    success_count = 0

    for idx, iv in enumerate(selected_intervals):
        video_name = iv["video_name"]
        start_frame = iv["start_frame"]
        scenario_id = iv["scenario_id"]
        annotation = ann_by_id[scenario_id]

        print(f"\n[{idx+1}/{len(selected_intervals)}] {iv['scenario']}")
        print(f"  Video: {video_name}, frames {start_frame}-{iv['end_frame']}, fps={iv['fps']}")

        # Create collage
        collage_path = collage_mgr.create_collage(
            video_name=video_name,
            start_frame=start_frame,
            grid_config=GRID_CONFIG,
            interval=INTERVAL,
        )

        if collage_path is None:
            print(f"  FAILED to create collage, skipping")
            continue

        # Copy collage to demo assets
        dest_collage = COLLAGES_DIR / f"scenario_{idx:03d}.jpg"
        shutil.copy2(collage_path, dest_collage)
        print(f"  Collage: {dest_collage.name}")

        # Create GIF
        gif_path = gif_mgr.create_gif(
            video_name=video_name,
            start_frame=start_frame,
            num_frames=NUM_FRAMES,
            interval=INTERVAL,
        )

        if gif_path is None:
            print(f"  FAILED to create GIF, skipping")
            dest_collage.unlink(missing_ok=True)
            continue

        # Copy GIF to demo assets
        dest_gif = GIFS_DIR / f"scenario_{idx:03d}.gif"
        shutil.copy2(gif_path, dest_gif)
        print(f"  GIF: {dest_gif.name}")

        # Build scenario entry
        scenarios_out.append({
            "index": success_count,
            "original_id": scenario_id,
            "scenario": iv["scenario"],
            "collage": f"assets/collages_hsd/scenario_{idx:03d}.jpg",
            "gif": f"assets/gifs_hsd/scenario_{idx:03d}.gif",
            "questions": annotation["questions"],
            "answer_key": annotation["answer_key"],
        })
        success_count += 1

    return scenarios_out


def main():
    print("=== HSD Demo Asset Generation ===\n")

    intervals, ann_by_id = load_data()
    print(f"Loaded {len(intervals)} evaluation intervals, {len(ann_by_id)} annotations")

    selected = select_diverse_intervals(intervals)
    print(f"\nSelected {len(selected)} intervals:")
    sid_counts = defaultdict(int)
    for iv in selected:
        sid_counts[iv["scenario_id"]] += 1
    for sid, count in sorted(sid_counts.items()):
        ann = ann_by_id[sid]
        print(f"  {ann['scenario'][:60]:60s} x{count}")

    scenarios = generate_assets(selected, ann_by_id)

    # Re-index after potential skips
    for i, s in enumerate(scenarios):
        s["index"] = i

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Collages:  {len(list(COLLAGES_DIR.glob('*.jpg')))} files in {COLLAGES_DIR}")
    print(f"GIFs:      {len(list(GIFS_DIR.glob('*.gif')))} files in {GIFS_DIR}")
    print(f"JSON:      {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
