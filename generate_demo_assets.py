#!/usr/bin/env python3
"""
Generate demo assets for VENUSS Human Evaluation Demo.

Selects 20 diverse scenarios from CoVLA, generates collages and GIFs,
and outputs scenarios.json for the static demo app.

Usage:
    python generate_demo_assets.py
"""

import json
import csv
import cv2
import math
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

SCENARIO_ANNOTATIONS = Path("/home/hiwi/VENUSS/datasets/covla/scenario_annotations.json")
FRAME_METADATA_CSV = Path("/home/hiwi/VENUSS/datasets/covla/frame_extraction_metadata.csv")
VIDEO_DIR = Path("/home/siyanli/CoVLA-Dataset/videos")

OUTPUT_DIR = Path(__file__).parent / "demo"
COLLAGE_DIR = OUTPUT_DIR / "assets" / "collages"
GIF_DIR = OUTPUT_DIR / "assets" / "gifs"
DATA_DIR = OUTPUT_DIR / "data"

NUM_SCENARIOS = 20
NUM_FRAMES = 8
INTERVAL_NAME = "500ms"
FRAME_SKIP = 9  # 500ms at 20fps => skip 9 frames

# Collage: 1x4 grid, 960x540 per frame
COLLAGE_FRAME_W = 960
COLLAGE_FRAME_H = 540

# GIF: 640x360 per frame
GIF_FRAME_W = 640
GIF_FRAME_H = 360
GIF_DURATION_MS = 500

# ============================================================================
# Scenario Selection — maximize diversity across all category values
# ============================================================================

CATEGORY_NAMES = ["Motion", "Direction", "Speed", "Following", "Acceleration", "TrafficLight", "Curve"]


def select_diverse_scenarios(scenarios, n=20):
    """Select n scenarios that cover all answer-key values as evenly as possible."""
    # Build coverage targets: for each position, which values exist?
    all_values = {}
    for i in range(7):
        all_values[i] = sorted(set(s["answer_key"][i] for s in scenarios))

    selected = []
    selected_ids = set()

    # Phase 1: Greedy coverage — pick scenarios that cover the rarest value
    remaining = list(scenarios)
    coverage = Counter()  # (position, value) -> count in selected

    while len(selected) < n and remaining:
        best_score = -1
        best_scenario = None

        for s in remaining:
            if s["id"] in selected_ids:
                continue
            # Score: how many (position, value) pairs have zero coverage
            score = 0
            for i in range(7):
                key = (i, s["answer_key"][i])
                if coverage[key] == 0:
                    score += 10  # High weight for uncovered values
                else:
                    # Prefer less-covered values
                    score += 1.0 / (coverage[key] + 1)
            if score > best_score:
                best_score = score
                best_scenario = s

        if best_scenario is None:
            break

        selected.append(best_scenario)
        selected_ids.add(best_scenario["id"])
        remaining = [s for s in remaining if s["id"] != best_scenario["id"]]
        for i in range(7):
            coverage[(i, best_scenario["answer_key"][i])] += 1

    # Report coverage
    print(f"\nSelected {len(selected)} scenarios. Coverage:")
    for i, name in enumerate(CATEGORY_NAMES):
        vals = Counter(s["answer_key"][i] for s in selected)
        print(f"  {name}: {dict(vals)}")

    return selected


# ============================================================================
# Frame Extraction
# ============================================================================

def load_frame_metadata():
    """Load and group frame metadata by scenario_id."""
    metadata_by_scenario = {}
    with open(FRAME_METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["valid"] != "True":
                continue
            sid = int(row["scenario_id"])
            if sid not in metadata_by_scenario:
                metadata_by_scenario[sid] = []
            metadata_by_scenario[sid].append(row)
    return metadata_by_scenario


def extract_frames(video_path, start_frame, num_frames, frame_skip):
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i * (frame_skip + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frames.append(frame)

    cap.release()
    return frames


def find_video_and_frames(metadata_rows):
    """Find a valid video and extract frames for a scenario."""
    for row in metadata_rows:
        video_name = row["video_name"]
        video_path = VIDEO_DIR / f"{video_name}.mp4"
        if not video_path.exists():
            continue

        start_frame = int(row["validated_start_frame"])
        total_frames = int(row["validated_frame_count"])

        # Check if we have enough frames for 4 images at 500ms intervals
        frames_needed = NUM_FRAMES * (FRAME_SKIP + 1)
        if total_frames < frames_needed:
            continue

        frames = extract_frames(video_path, start_frame, NUM_FRAMES, FRAME_SKIP)
        if frames is not None and len(frames) == NUM_FRAMES:
            return frames, video_name

    return None, None


# ============================================================================
# Asset Generation
# ============================================================================

def create_collage(frames, output_path):
    """Create a 1x8 horizontal collage at 960x540 per frame."""
    resized = []
    for frame in frames:
        r = cv2.resize(frame, (COLLAGE_FRAME_W, COLLAGE_FRAME_H), interpolation=cv2.INTER_AREA)
        resized.append(r)

    collage = np.hstack(resized)
    cv2.imwrite(str(output_path), collage, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


def create_gif(frames, output_path):
    """Create animated GIF at 640x360 with gray ending frames."""
    pil_frames = []
    for frame in frames:
        resized = cv2.resize(frame, (GIF_FRAME_W, GIF_FRAME_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.convert("P", palette=Image.ADAPTIVE, colors=256)
        pil_frames.append(pil_img)

    # Add gray ending frames — at least 500ms worth
    num_gray = max(1, math.ceil(500 / GIF_DURATION_MS))
    for i in range(num_gray):
        gray_rgb = Image.new("RGB", (GIF_FRAME_W, GIF_FRAME_H), (125, 125, 125))
        pixels = gray_rgb.load()
        pixels[i % GIF_FRAME_W, i % GIF_FRAME_H] = (124, 124, 124)
        gray_p = gray_rgb.convert("P", palette=Image.ADAPTIVE, colors=256)
        pil_frames.append(gray_p)

    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=False,
    )
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== VENUSS Demo Asset Generator ===\n")

    # Load data
    print("Loading scenario annotations...")
    with open(SCENARIO_ANNOTATIONS) as f:
        all_scenarios = json.load(f)
    print(f"  {len(all_scenarios)} scenarios loaded")

    print("Loading frame metadata...")
    metadata_by_scenario = load_frame_metadata()
    print(f"  {len(metadata_by_scenario)} scenarios with valid metadata")

    # Select diverse scenarios
    print("\nSelecting diverse scenarios...")
    selected = select_diverse_scenarios(all_scenarios, NUM_SCENARIOS)

    # Create output directories
    COLLAGE_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate assets — keep trying until we have NUM_SCENARIOS successes
    demo_scenarios = []
    used_ids = set()
    # Build a fallback pool of remaining scenarios sorted by id
    fallback_pool = [s for s in all_scenarios if s["id"] not in {sc["id"] for sc in selected}]

    candidates = list(selected) + fallback_pool

    for scenario in candidates:
        if len(demo_scenarios) >= NUM_SCENARIOS:
            break

        sid = scenario["id"]
        if sid in used_ids:
            continue

        idx = len(demo_scenarios)
        print(f"\n[{idx+1}/{NUM_SCENARIOS}] Scenario {sid}: {scenario['scenario'][:60]}...")

        metadata_rows = metadata_by_scenario.get(sid, [])
        if not metadata_rows:
            print(f"  SKIP: No metadata for scenario {sid}")
            continue

        frames, video_name = find_video_and_frames(metadata_rows)
        if frames is None:
            print(f"  SKIP: Could not extract frames for scenario {sid}")
            continue

        used_ids.add(sid)

        # Generate collage
        collage_filename = f"scenario_{idx:03d}.jpg"
        collage_path = COLLAGE_DIR / collage_filename
        create_collage(frames, collage_path)
        print(f"  Collage: {collage_path.name} ({collage_path.stat().st_size / 1024:.0f} KB)")

        # Generate GIF
        gif_filename = f"scenario_{idx:03d}.gif"
        gif_path = GIF_DIR / gif_filename
        create_gif(frames, gif_path)
        print(f"  GIF: {gif_path.name} ({gif_path.stat().st_size / 1024:.0f} KB)")

        # Build demo scenario entry
        demo_entry = {
            "index": idx,
            "original_id": sid,
            "scenario": scenario["scenario"],
            "collage": f"assets/collages/{collage_filename}",
            "gif": f"assets/gifs/{gif_filename}",
            "questions": scenario["questions"],
            "answer_key": scenario["answer_key"],
        }
        demo_scenarios.append(demo_entry)

    # Write scenarios.json
    scenarios_json_path = DATA_DIR / "scenarios.json"
    with open(scenarios_json_path, "w") as f:
        json.dump(demo_scenarios, f, indent=2)
    print(f"\n\nWrote {scenarios_json_path} ({len(demo_scenarios)} scenarios)")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Scenarios generated: {len(demo_scenarios)}/{NUM_SCENARIOS}")
    print(f"  Collages: {COLLAGE_DIR}")
    print(f"  GIFs: {GIF_DIR}")
    print(f"  Data: {scenarios_json_path}")

    # Check total size
    collage_size = sum(f.stat().st_size for f in COLLAGE_DIR.glob("*.jpg"))
    gif_size = sum(f.stat().st_size for f in GIF_DIR.glob("*.gif"))
    print(f"  Total collage size: {collage_size / (1024*1024):.1f} MB")
    print(f"  Total GIF size: {gif_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
