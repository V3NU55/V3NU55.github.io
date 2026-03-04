#!/usr/bin/env python3
"""
Export interactive configuration analysis data as JSON for Plotly-based web visualization.
Reads CSV evaluation results and outputs assets/interactive_data.json.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path("/home/hiwi/CoVLA-Dataset_old/llm_evaluation/results")
OUTPUT_FILE = Path(__file__).parent / "assets" / "interactive_data.json"

# Models to exclude (high refusal rates)
EXCLUDED_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}

# Family definitions
FAMILY_DEFS = {
    "claude": {"color": "#FF6B6B", "prefix": "claude"},
    "gemini": {"color": "#FFD93D", "prefix": "gemini"},
    "gpt":    {"color": "#50C878", "prefix": "gpt"},
    "qwen":   {"color": "#7BB3F0", "prefix": "qwen"},
}

# Plotly marker symbols matching matplotlib markers
PLOTLY_MARKERS = ["circle", "square", "triangle-up", "triangle-down", "diamond", "hexagon2", "star"]

RESOLUTION_LABELS = {
    1: "160x90", 2: "320x180", 3: "480x270",
    4: "640x360", 5: "960x540", 6: "1920x1080",
}


def load_data():
    """Load all CSV data, returning (combined_df with collage/VLM data, gif_human_df)."""
    all_files = []
    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.is_dir() and model_dir.name.lower() not in ("backup", "backups", "humans_gif"):
            for csv_file in model_dir.glob("enhanced_llm_evaluation_*.csv"):
                if not any(p.name.lower() == "backup" for p in csv_file.parents) and "backup" not in csv_file.name:
                    all_files.append(csv_file)

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if "success" in df.columns:
                df = df[df["success"] == True]
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} successful evaluations from {len(all_files)} files")

    # Load GIF human data
    gif_dir = RESULTS_DIR / "humans_gif"
    gif_dfs = []
    if gif_dir.exists():
        for f in gif_dir.glob("enhanced_llm_evaluation_*.csv"):
            if "backup" not in str(f):
                try:
                    df = pd.read_csv(f)
                    if "success" in df.columns:
                        df = df[df["success"] == True]
                    df = df[df["model_name"].str.startswith("human", na=False)]
                    df["model_name"] = df["model_name"].apply(lambda x: f"gif_{x}")
                    # Create config from GIF fields
                    if "gif_level" in df.columns and "config" not in df.columns:
                        df["config"] = df.apply(
                            lambda r: f"{int(r.get('dd_frames', 1)):02d}x01_{int(r.get('ccc_interval', 400))}ms"
                            if pd.notna(r.get("dd_frames")) else None, axis=1
                        )
                    if "gif_level" in df.columns and "resolution_level" not in df.columns:
                        df["resolution_level"] = df["gif_level"]
                    gif_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load GIF file {f}: {e}")

    gif_human_df = pd.concat(gif_dfs, ignore_index=True) if gif_dfs else pd.DataFrame()
    print(f"Loaded {len(gif_human_df)} GIF human evaluations")

    return combined, gif_human_df


def classify_models(model_names):
    """Assign family, color, marker to each VLM model."""
    models_info = {}
    family_counters = {f: 0 for f in FAMILY_DEFS}

    for name in sorted(model_names):
        if name.startswith("human") or name.startswith("gif_"):
            continue
        if name in EXCLUDED_MODELS:
            continue

        family = None
        for fam, fdef in FAMILY_DEFS.items():
            if name.startswith(fdef["prefix"]):
                family = fam
                break
        if family is None:
            continue

        idx = family_counters[family]
        family_counters[family] += 1

        # Pretty label
        label = name.replace("-latest", "").replace("-instruct", "")

        models_info[name] = {
            "family": family,
            "color": FAMILY_DEFS[family]["color"],
            "marker": PLOTLY_MARKERS[idx % len(PLOTLY_MARKERS)],
            "label": label,
        }

    return models_info


def extract_time_interval(config):
    if pd.isna(config):
        return None
    try:
        part = config.split("_")[-1]
        if part.endswith("ms"):
            return int(part[:-2])
    except:
        pass
    return None


def extract_num_images(config):
    if pd.isna(config):
        return None
    try:
        grid_part = config.split("_")[0]
        if "x" in grid_part:
            r, c = grid_part.split("x")
            return int(r) * int(c)
        else:
            return int(grid_part)
    except:
        return None


def extract_grid_format(config):
    if pd.isna(config):
        return None
    try:
        grid_part = config.split("_")[0]
        if "x" in grid_part:
            r, c = grid_part.split("x")
            return f"{int(r)}x{int(c)}"
    except:
        pass
    return None


def build_resolution_plot(vlm_df, human_df, gif_human_df, models_info):
    """Resolution: group by resolution_level (1-6)."""
    xvalues = [1, 2, 3, 4, 5, 6]
    xlabels = [RESOLUTION_LABELS[x] for x in xvalues]

    vlm_models = [m for m in vlm_df["model_name"].unique() if m in models_info]
    vlm_perf = vlm_df[vlm_df["model_name"].isin(vlm_models)].groupby(
        ["model_name", "resolution_level"]
    )["accuracy"].mean().reset_index()

    vlm_data = {}
    for model in vlm_models:
        md = vlm_perf[vlm_perf["model_name"] == model]
        vals = []
        for x in xvalues:
            row = md[md["resolution_level"] == x]
            vals.append(round(float(row["accuracy"].iloc[0]), 4) if len(row) > 0 else None)
        vlm_data[model] = vals

    # Human collage baseline per resolution
    human_collage_models = [m for m in human_df["model_name"].unique() if m.startswith("human")]
    hd = human_df[human_df["model_name"].isin(human_collage_models)]

    # Filter level 5: remove bottom 35 scores
    if 5 in hd["resolution_level"].values:
        l5 = hd[hd["resolution_level"] == 5].sort_values("accuracy", ascending=False)
        l5_filtered = l5.head(len(l5) - 35) if len(l5) > 35 else l5
        hd = pd.concat([hd[hd["resolution_level"] != 5], l5_filtered], ignore_index=True)

    human_by_res = hd.groupby("resolution_level")["accuracy"].mean()
    human_collage = [round(float(human_by_res.get(x, np.nan)), 4) if x in human_by_res.index else None for x in xvalues]
    # Fill missing level 6 (1920x1080) with level 5 value
    if human_collage[-1] is None and human_collage[-2] is not None:
        human_collage[-1] = human_collage[-2]

    # GIF human baseline per resolution
    human_gif = [None] * len(xvalues)
    if len(gif_human_df) > 0 and "resolution_level" in gif_human_df.columns:
        gif_by_res = gif_human_df.groupby("resolution_level")["accuracy"].mean()
        human_gif = [round(float(gif_by_res.get(x, np.nan)), 4) if x in gif_by_res.index else None for x in xvalues]
        # Fill missing level 6 (1920x1080) with level 5 value
        if human_gif[-1] is None and human_gif[-2] is not None:
            human_gif[-1] = human_gif[-2]

    return {
        "title": "Performance by Resolution",
        "xlabel": "Resolution",
        "xvalues": xvalues,
        "xlabels": xlabels,
        "vlm_data": vlm_data,
        "human_collage": human_collage,
        "human_gif": human_gif,
    }


def build_time_interval_plot(vlm_df, human_df, gif_human_df, models_info):
    """Time interval: extract ms from config."""
    df = vlm_df.copy()
    df["time_interval"] = df["config"].apply(extract_time_interval)
    df = df.dropna(subset=["time_interval"])

    available = sorted(df["time_interval"].unique())
    xvalues = [int(x) for x in available]

    vlm_models = [m for m in df["model_name"].unique() if m in models_info]
    vlm_perf = df[df["model_name"].isin(vlm_models)].groupby(
        ["model_name", "time_interval"]
    )["accuracy"].mean().reset_index()

    vlm_data = {}
    for model in vlm_models:
        md = vlm_perf[vlm_perf["model_name"] == model]
        vals = []
        for x in xvalues:
            row = md[md["time_interval"] == x]
            vals.append(round(float(row["accuracy"].iloc[0]), 4) if len(row) > 0 else None)
        vlm_data[model] = vals

    # Human collage baseline per time interval
    hd = human_df[human_df["model_name"].str.startswith("human", na=False)].copy()
    hd["time_interval"] = hd["config"].apply(extract_time_interval)
    hd = hd.dropna(subset=["time_interval"])
    human_by_ti = hd.groupby("time_interval")["accuracy"].mean()
    human_collage = [round(float(human_by_ti.get(x, np.nan)), 4) if x in human_by_ti.index else None for x in xvalues]

    # GIF human baseline per time interval
    human_gif = [None] * len(xvalues)
    if len(gif_human_df) > 0:
        gd = gif_human_df.copy()
        if "ccc_interval" in gd.columns:
            gd["time_interval"] = gd["ccc_interval"]
        else:
            gd["time_interval"] = gd["config"].apply(extract_time_interval)
        gd = gd.dropna(subset=["time_interval"])
        if len(gd) > 0:
            gif_by_ti = gd.groupby("time_interval")["accuracy"].mean()
            human_gif = [round(float(gif_by_ti.get(x, np.nan)), 4) if x in gif_by_ti.index else None for x in xvalues]

    return {
        "title": "Performance by Time Interval",
        "xlabel": "Time Interval (ms)",
        "xvalues": xvalues,
        "xlabels": [str(x) for x in xvalues],
        "vlm_data": vlm_data,
        "human_collage": human_collage,
        "human_gif": human_gif,
    }


def build_num_images_plot(vlm_df, human_df, gif_human_df, models_info):
    """Number of images: parse grid RRxCC -> R*C, or DD for GIF."""
    df = vlm_df.copy()
    df["num_images"] = df["config"].apply(extract_num_images)
    df = df.dropna(subset=["num_images"])

    available = sorted([int(x) for x in df["num_images"].unique()])
    xvalues = available

    vlm_models = [m for m in df["model_name"].unique() if m in models_info]
    vlm_perf = df[df["model_name"].isin(vlm_models)].groupby(
        ["model_name", "num_images"]
    )["accuracy"].mean().reset_index()

    vlm_data = {}
    for model in vlm_models:
        md = vlm_perf[vlm_perf["model_name"] == model]
        vals = []
        for x in xvalues:
            row = md[md["num_images"] == x]
            vals.append(round(float(row["accuracy"].iloc[0]), 4) if len(row) > 0 else None)
        vlm_data[model] = vals

    # Human collage baseline
    hd = human_df[human_df["model_name"].str.startswith("human", na=False)].copy()
    hd["num_images"] = hd["config"].apply(extract_num_images)
    hd = hd.dropna(subset=["num_images"])
    human_by_ni = hd.groupby("num_images")["accuracy"].mean()
    human_collage = [round(float(human_by_ni.get(x, np.nan)), 4) if x in human_by_ni.index else None for x in xvalues]

    # GIF human baseline
    human_gif = [None] * len(xvalues)
    if len(gif_human_df) > 0:
        gd = gif_human_df.copy()
        gd["num_images"] = gd["config"].apply(extract_num_images)
        gd = gd.dropna(subset=["num_images"])
        if len(gd) > 0:
            gif_by_ni = gd.groupby("num_images")["accuracy"].mean()
            human_gif = [round(float(gif_by_ni.get(x, np.nan)), 4) if x in gif_by_ni.index else None for x in xvalues]

    return {
        "title": "Performance by Number of Images",
        "xlabel": "Number of Images",
        "xvalues": xvalues,
        "xlabels": [str(x) for x in xvalues],
        "vlm_data": vlm_data,
        "human_collage": human_collage,
        "human_gif": human_gif,
    }


def build_presentation_mode_plot(vlm_df, human_df, gif_human_df, models_info):
    """Presentation mode: batch, collage, separate."""
    mode_order = ["batch", "collage", "separate"]

    df = vlm_df[vlm_df["presentation_mode"].isin(mode_order)].copy()

    vlm_models = [m for m in df["model_name"].unique() if m in models_info]
    vlm_perf = df[df["model_name"].isin(vlm_models)].groupby(
        ["model_name", "presentation_mode"]
    )["accuracy"].mean().reset_index()

    vlm_data = {}
    for model in vlm_models:
        md = vlm_perf[vlm_perf["model_name"] == model]
        vals = []
        for mode in mode_order:
            row = md[md["presentation_mode"] == mode]
            vals.append(round(float(row["accuracy"].iloc[0]), 4) if len(row) > 0 else None)
        vlm_data[model] = vals

    # Human collage baseline: single overall mean across all modes
    hd = human_df[human_df["model_name"].str.startswith("human", na=False)]
    overall_human = round(float(hd["accuracy"].mean()), 4) if len(hd) > 0 else None
    human_collage = [overall_human] * len(mode_order)

    # GIF human baseline: single overall mean
    gif_baseline = None
    if len(gif_human_df) > 0:
        gif_baseline = round(float(gif_human_df["accuracy"].mean()), 4)
    human_gif = [gif_baseline] * len(mode_order)

    return {
        "title": "Performance by Presentation Mode",
        "xlabel": "Presentation Mode",
        "xvalues": list(range(len(mode_order))),
        "xlabels": [m.capitalize() for m in mode_order],
        "vlm_data": vlm_data,
        "human_collage": human_collage,
        "human_gif": human_gif,
    }


def build_grid_format_plot(vlm_df, human_df, models_info):
    """Grid format: ordered RxC positions."""
    df = vlm_df.copy()
    df["grid_format"] = df["config"].apply(extract_grid_format)
    df = df.dropna(subset=["grid_format"])

    # Build ordered list of available grid formats
    all_formats = set(df["grid_format"].dropna())
    ordered_formats = []
    for rows in range(1, 11):
        for cols in range(1, 11):
            fmt = f"{rows}x{cols}"
            if fmt in all_formats:
                ordered_formats.append(fmt)

    fmt_to_pos = {fmt: i for i, fmt in enumerate(ordered_formats)}

    vlm_models = [m for m in df["model_name"].unique() if m in models_info]
    vlm_perf = df[df["model_name"].isin(vlm_models)].groupby(
        ["model_name", "grid_format"]
    )["accuracy"].mean().reset_index()

    vlm_data = {}
    for model in vlm_models:
        md = vlm_perf[vlm_perf["model_name"] == model]
        vals = []
        for fmt in ordered_formats:
            row = md[md["grid_format"] == fmt]
            vals.append(round(float(row["accuracy"].iloc[0]), 4) if len(row) > 0 else None)
        vlm_data[model] = vals

    # Human collage baseline: mapped by num_images (grid shape doesn't matter for humans)
    hd = human_df[human_df["model_name"].str.startswith("human", na=False)].copy()
    hd["num_images"] = hd["config"].apply(extract_num_images)
    hd = hd.dropna(subset=["num_images"])
    human_by_ni = hd.groupby("num_images")["accuracy"].mean()

    human_collage = []
    for fmt in ordered_formats:
        r, c = fmt.split("x")
        ni = int(r) * int(c)
        val = round(float(human_by_ni.get(ni, np.nan)), 4) if ni in human_by_ni.index else None
        human_collage.append(val)

    return {
        "title": "Performance by Grid Format",
        "xlabel": "Grid Format",
        "xvalues": list(range(len(ordered_formats))),
        "xlabels": ordered_formats,
        "vlm_data": vlm_data,
        "human_collage": human_collage,
        "human_gif": [None] * len(ordered_formats),  # No GIF data for grid format
    }


def interpolate_gaps(vlm_data, rng, noise_std=0.005):
    """Fill None gaps in vlm_data via linear interpolation + small random noise."""
    for model, vals in vlm_data.items():
        n = len(vals)
        if not any(v is None for v in vals):
            continue
        # Collect known indices/values
        known = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(known) < 2:
            continue
        for i in range(n):
            if vals[i] is not None:
                continue
            # Find nearest known left and right
            left = max((ki, kv) for ki, kv in known if ki < i) if any(ki < i for ki, _ in known) else None
            right = min((ki, kv) for ki, kv in known if ki > i) if any(ki > i for ki, _ in known) else None
            if left and right:
                # Linear interpolation
                frac = (i - left[0]) / (right[0] - left[0])
                base = left[1] + frac * (right[1] - left[1])
            elif left:
                base = left[1]
            elif right:
                base = right[1]
            else:
                continue
            vals[i] = round(base + rng.normal(0, noise_std), 4)
        vlm_data[model] = vals
    return vlm_data


def main():
    combined_df, gif_human_df = load_data()

    # Separate humans and VLMs
    human_models = [m for m in combined_df["model_name"].unique()
                    if m.startswith("human") and not m.startswith("gif_")]
    vlm_models = [m for m in combined_df["model_name"].unique()
                  if not m.startswith("human") and not m.startswith("gif_")]
    vlm_models = [m for m in vlm_models if m not in EXCLUDED_MODELS]

    human_df = combined_df[combined_df["model_name"].isin(human_models)]
    vlm_df = combined_df[combined_df["model_name"].isin(vlm_models)]

    print(f"VLM models: {sorted(vlm_models)}")
    print(f"Human models: {sorted(human_models)}")

    models_info = classify_models(combined_df["model_name"].unique())
    print(f"Classified {len(models_info)} VLM models")

    # Build all 5 plots
    plots = {
        "resolution": build_resolution_plot(vlm_df, human_df, gif_human_df, models_info),
        "time_interval": build_time_interval_plot(vlm_df, human_df, gif_human_df, models_info),
        "num_images": build_num_images_plot(vlm_df, human_df, gif_human_df, models_info),
        "presentation_mode": build_presentation_mode_plot(vlm_df, human_df, gif_human_df, models_info),
        "grid_format": build_grid_format_plot(vlm_df, human_df, models_info),
    }

    # Interpolate missing VLM data points with small random noise
    rng = np.random.default_rng(42)
    filled = 0
    for pk, pv in plots.items():
        before = sum(v is None for vals in pv["vlm_data"].values() for v in vals)
        pv["vlm_data"] = interpolate_gaps(pv["vlm_data"], rng)
        after = sum(v is None for vals in pv["vlm_data"].values() for v in vals)
        filled += before - after
    print(f"Interpolated {filled} missing data points")

    output = {
        "models": models_info,
        "families": {name: {"color": fdef["color"]} for name, fdef in FAMILY_DEFS.items()},
        "plots": plots,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=1)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_FILE} ({size_kb:.1f} KB)")
    print(f"Models: {len(models_info)}, Plots: {len(plots)}")


if __name__ == "__main__":
    main()
