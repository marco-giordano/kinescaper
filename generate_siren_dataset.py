#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_siren_dataset.py
=========================
Dataset generator for KineScaper using config_siren.yaml.

Balancing logic:
  - F = count of the reference class (biggest or smallest, per 'balance_wrt').
  - target_count = F * class_percentage  (subset size, same for every class).
  - For each class a subset of 'target_count' files is built:
      * sample without replacement if the class has enough unique files;
      * oversample (with replacement for the extra items) if not.
  - Loop: for each class -> for each sample in subset ->
          for each iteration -> for each (bg_spl, fg_spl) combo.
  - Total per class = target_count * iterations * |bg_grid| * |fg_grid|.
  - Velocity is drawn randomly from a quantized grid inside velocity_range.
  - Onset is drawn uniformly at random within onset_range.
  - Pre-flight shows count + estimated disk usage before starting.
"""

import os
import sys
import time
import math
import random
import datetime
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from src.core import KineScaper
from src.utils import (
    load_config,
    load_random_background,
    load_foreground_metadata,
    get_event_path,
    save_dcase_annotations,
    save_extended_metadata,
)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class TeeLogger:
    """Mirrors all writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._log = open(log_path, 'w', buffering=1)

    def write(self, msg: str):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()

    def fileno(self):
        return self._stdout.fileno()


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_spl_grid(spl_range: List[float], quantization: float) -> List[float]:
    """
    Build an inclusive grid from spl_range[0] to spl_range[1] with the
    given step size (quantization).  Both boundaries are included.

    Example:
        build_spl_grid([50.0, 71.0], 3)  →  [50, 53, 56, 59, 62, 65, 68, 71]
    """
    lo, hi = spl_range
    steps = math.floor((hi - lo) / quantization)
    return [lo + i * quantization for i in range(steps + 1) if lo + i * quantization <= hi + 1e-9]


def quantized_random_velocity(velocity_range: List[float], quantization: float) -> float:
    """
    Pick a random velocity uniformly from the quantized grid inside velocity_range.
    """
    grid = build_spl_grid(velocity_range, quantization)   # reuse the same helper
    return float(random.choice(grid))


# ---------------------------------------------------------------------------
# Class analysis / balancing
# ---------------------------------------------------------------------------

def analyze_class_frequencies(fg_metadata: pd.DataFrame) -> Dict[str, int]:
    """Return {mode_label: count} sorted descending by count."""
    counts = fg_metadata['mode'].value_counts().to_dict()
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def compute_target_count(class_frequencies: Dict[str, int],
                         balance_wrt: str,
                         class_percentage: float) -> int:
    """
    Return the target subset size shared by every class.

    target_count = reference_class_size * class_percentage

    where reference_class_size is the size of the biggest (or smallest)
    class, depending on balance_wrt.
    """
    if not class_frequencies:
        raise ValueError("Foreground metadata has no 'mode' classes.")
    counts = list(class_frequencies.values())
    if balance_wrt == "bigger":
        reference = max(counts)
    elif balance_wrt == "smaller":
        reference = min(counts)
    else:
        raise ValueError(f"balance_wrt must be 'bigger' or 'smaller', got '{balance_wrt}'")
    return max(1, int(reference * class_percentage))


def build_class_subset(fg_metadata: pd.DataFrame,
                       cls: str,
                       target_count: int) -> List[Dict]:
    """
    Build a list of 'target_count' rows for *cls*.

    Strategy:
      1. Take all unique rows for the class (sample without replacement).
      2. If the class has fewer unique rows than target_count, append
         random samples WITH replacement until target_count is reached.

    Returns a list of dicts (one per row) with at least 'filename' and 'mode'.
    """
    class_rows = fg_metadata[fg_metadata['mode'] == cls]
    if class_rows.empty:
        return []

    unique_count = len(class_rows)

    if unique_count >= target_count:
        # Enough unique files — sample without replacement
        sampled = class_rows.sample(n=target_count, replace=False)
    else:
        # Not enough unique files — take all, then oversample the remainder
        extra_needed = target_count - unique_count
        extra = class_rows.sample(n=extra_needed, replace=True)
        sampled = pd.concat([class_rows, extra], ignore_index=True)

    return sampled.to_dict('records')


# ---------------------------------------------------------------------------
# Scene geometry helper
# ---------------------------------------------------------------------------

def random_trajectory(config: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Returns (trajectory_array [2,3], z) using circle-randomized positions.
    """
    start_center = config.get('start_circle_center', [-200.0, 5.0])
    end_center   = config.get('end_circle_center',   [200.0, 5.0])
    pos_radius   = config.get('position_radius', 0.0)
    z_range      = config.get('z_range', [1.8, 1.8])
    if isinstance(z_range, (int, float)):
        z_range = [z_range, z_range]

    z       = random.uniform(z_range[0], z_range[1])
    theta_s = random.uniform(0, 2 * np.pi)
    theta_e = random.uniform(0, 2 * np.pi)
    start_pos = np.array([
        start_center[0] + pos_radius * np.cos(theta_s),
        start_center[1] + pos_radius * np.sin(theta_s),
        z,
    ])
    end_pos = np.array([
        end_center[0] + pos_radius * np.cos(theta_e),
        end_center[1] + pos_radius * np.sin(theta_e),
        z,
    ])
    return np.array([start_pos, end_pos]), z


# ---------------------------------------------------------------------------
# Pre-flight estimate
# ---------------------------------------------------------------------------

def estimate_dataset(
    all_classes: List[str],
    target_count: int,
    iterations: int,
    bg_grid: List[float],
    fg_grid: List[float],
    output_duration: float,
    fs: int,
    n_mics: int,
) -> Tuple[int, float]:
    """
    Returns (total_samples, estimated_disk_GB).

    Loop: class -> sample in subset -> iteration -> (bg, fg) combo.
    total = n_classes * target_count * iterations * |bg_grid| * |fg_grid|
    """
    combos = len(bg_grid) * len(fg_grid)
    total_samples = len(all_classes) * target_count * iterations * combos

    # float32 WAV: 4 bytes per sample
    bytes_per_wav = output_duration * fs * 4 * n_mics
    # +20% overhead for metadata files
    total_bytes = total_samples * bytes_per_wav * 1.20
    total_gb = total_bytes / (1024 ** 3)

    return total_samples, total_gb


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_siren_dataset():
    # 1. Load Config
    config_path = 'config_siren.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)

    # 2. Setup Output Directories
    output_folder = config.get('output_folder', 'output/siren_dataset')
    audio_dir = os.path.join(output_folder, 'audio')
    tsv_dir   = os.path.join(output_folder, 'csv')
    json_dir  = os.path.join(output_folder, 'json')

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(tsv_dir,   exist_ok=True)
    os.makedirs(json_dir,  exist_ok=True)

    # 3. Start Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(output_folder, f"generation_log_{timestamp}.txt")
    logger    = TeeLogger(log_path)
    sys.stdout = logger

    print("KineScaper — Siren Dataset Generation Log")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:  {os.path.abspath(config_path)}")
    print(f"Output:  {os.path.abspath(output_folder)}")
    print()

    # 4. Load Foreground Metadata
    fg_folder    = config.get('foreground_dir')
    fg_meta_path = config.get('metadata_path')
    bg_folder    = config.get('background_dir')

    for label, path in [("metadata_path", fg_meta_path), ("foreground_dir", fg_folder), ("background_dir", bg_folder)]:
        if not path:
            print(f"[ERROR] '{label}' not set in {config_path}")
            logger.close()
            return
        if not os.path.exists(path):
            print(f"[ERROR] Path not found: {path}")
            logger.close()
            return

    fg_metadata = load_foreground_metadata(fg_meta_path)

    if 'mode' not in fg_metadata.columns:
        print("[ERROR] Foreground metadata CSV has no 'mode' column.")
        logger.close()
        return

    # 5. Class Analysis
    class_freqs = analyze_class_frequencies(fg_metadata)
    print("=== Foreground Dataset Class Frequencies ===")
    for cls, cnt in class_freqs.items():
        print(f"  {cls:<20s} : {cnt:>5d} samples")
    print()

    balance_wrt      = config.get('balance_wrt', 'bigger')
    class_percentage = float(config.get('class_percentage', 1.0))
    iterations       = int(config.get('iterations', 1))

    # All classes are always used; class_percentage only scales the subset size.
    all_classes  = list(class_freqs.keys())
    target_count = compute_target_count(class_freqs, balance_wrt, class_percentage)

    counts       = list(class_freqs.values())
    ref_count    = max(counts) if balance_wrt == 'bigger' else min(counts)
    ref_label    = [k for k, v in class_freqs.items() if v == ref_count][0]

    print(f"Balance w.r.t.: '{balance_wrt}' class  ('{ref_label}' with {ref_count} samples)")
    print(f"class_percentage = {class_percentage*100:.0f}%  =>  target subset size per class = {target_count}")
    print(f"All {len(all_classes)} classes will be processed: {all_classes}")
    print(f"Iterations per (class x sample): {iterations}")
    print()

    # 5b. Build per-class subsets (done once, before generation)
    print("Building per-class subsets...")
    class_subsets: Dict[str, List[Dict]] = {}
    for cls in all_classes:
        subset = build_class_subset(fg_metadata, cls, target_count)
        n_unique = len(fg_metadata[fg_metadata['mode'] == cls])
        oversampled = target_count > n_unique
        print(
            f"  {cls:<20s}: {n_unique:>4d} unique  ->  subset={len(subset)}"
            + (f"  [oversampled x{target_count/n_unique:.1f}]" if oversampled else "")
        )
        class_subsets[cls] = subset
    print()

    # 6. Build SPL Grids
    bg_spl_range   = config.get('required_background_SPL', [50.0, 70.0])
    bg_quantiz     = float(config.get('background_SPL_quantization', 3.0))
    fg_spl_range   = config.get('required_foreground_SPL', [80.0, 110.0])
    fg_quantiz     = float(config.get('foreground_SPL_quantization', 3.0))

    if isinstance(bg_spl_range, (int, float)):
        bg_spl_range = [bg_spl_range, bg_spl_range]
    if isinstance(fg_spl_range, (int, float)):
        fg_spl_range = [fg_spl_range, fg_spl_range]

    bg_grid = build_spl_grid(bg_spl_range, bg_quantiz)
    fg_grid = build_spl_grid(fg_spl_range, fg_quantiz)

    print(f"Background SPL grid ({len(bg_grid)} points): {[f'{v:.1f}' for v in bg_grid]}")
    print(f"Foreground SPL grid ({len(fg_grid)} points): {[f'{v:.1f}' for v in fg_grid]}")
    print()

    # 7. Other config
    velocity_range  = config.get('velocity', [40.0, 90.0])
    velocity_quant  = float(config.get('velocity_quantization', 1.0))
    if isinstance(velocity_range, (int, float)):
        velocity_range = [velocity_range, velocity_range]

    onset_range = config.get('onset_range', [1.0, 5.0])
    if isinstance(onset_range, (int, float)):
        onset_range = [onset_range, onset_range]

    output_duration = config.get('output_duration', 40.0)
    fs              = config.get('sampling_frequency', 32000)
    n_mics          = len(config.get('mic_positions', [[0, 0, 1.8]]))

    # 8. Pre-flight Estimate
    total_samples, est_gb = estimate_dataset(
        all_classes, target_count, iterations, bg_grid, fg_grid,
        output_duration, fs, n_mics
    )

    combos = len(bg_grid) * len(fg_grid)
    print("=" * 60)
    print("  PRE-FLIGHT ESTIMATE")
    print("=" * 60)
    print(f"  Classes                : {len(all_classes)}")
    print(f"  Subset size per class  : {target_count}")
    print(f"  Iterations per sample  : {iterations}")
    print(f"  BG SPL grid points     : {len(bg_grid)}")
    print(f"  FG SPL grid points     : {len(fg_grid)}")
    print(f"  SPL combos             : {len(bg_grid)} x {len(fg_grid)} = {combos}")
    print(f"  Per-class total        : {target_count} x {iterations} x {combos} = {target_count*iterations*combos}")
    print(f"  " + "-" * 38)
    print(f"  TOTAL SAMPLES          : {total_samples}")
    print(f"  Estimated disk usage   : {est_gb:.2f} GB")
    print("=" * 60)
    print()

    # --- Pause for user confirmation ---
    # Temporarily restore stdout so input() works properly in a terminal
    sys.stdout = logger._stdout
    answer = input("Do you want to proceed? [y/N] ").strip().lower()
    sys.stdout = logger
    logger.write(f"User confirmation: '{answer}'\n")

    if answer not in ('y', 'yes'):
        print("Aborted by user.")
        logger.close()
        return

    print()
    print("Starting generation …")
    print()

    # 9. Main Generation Loop
    # Loop order: class -> sample in subset -> iteration -> (bg_spl, fg_spl)
    dataset_annotations: List[Dict] = []
    global_sample_idx = 0
    t_start = time.time()

    # Pre-build the flat job list for the progress bar total
    all_jobs = [
        (cls, sample_row, sample_idx, it, bg_spl, fg_spl)
        for cls in all_classes
        for sample_idx, sample_row in enumerate(class_subsets[cls])
        for it in range(iterations)
        for bg_spl in bg_grid
        for fg_spl in fg_grid
    ]

    pbar = tqdm(
        total=len(all_jobs),
        desc="Generating",
        unit="sample",
        file=logger._stdout,
        dynamic_ncols=True,
        colour="green",
    )

    for cls, sample_row, sample_idx, iteration_idx, bg_spl, fg_spl in all_jobs:
        t_sample_start = time.time()

        label    = sample_row['mode']
        fname    = sample_row['filename']
        src_path = get_event_path(fg_folder, fname)

        if not os.path.exists(src_path):
            print(f"[WARN] Source file not found: {src_path}, skipping.")
            pbar.update(1)
            continue

        # Override SPL values in config for this iteration
        iter_config = dict(config)
        iter_config['required_background_SPL'] = bg_spl
        iter_config['required_foreground_SPL']  = fg_spl

        ks = KineScaper(iter_config)

        # Background
        try:
            bg_chunk = load_random_background(bg_folder, output_duration, fs)
            ks.add_background(bg_chunk)
        except Exception as e:
            print(f"[ERROR] Loading background: {e}, skipping.")
            pbar.update(1)
            continue

        # Random velocity (quantized grid pick)
        velocity = quantized_random_velocity(velocity_range, velocity_quant)
        onset    = random.uniform(onset_range[0], onset_range[1])

        # Geometry
        trajectory, z = random_trajectory(config)

        print(
            f"  [cls={cls} | s={sample_idx}/{target_count-1} | iter={iteration_idx} "
            f"| bg={bg_spl:.1f}dB | fg={fg_spl:.1f}dB] "
            f"vel={velocity:.1f}km/h  onset={onset:.2f}s  z={z:.2f}m"
        )

        ks.add_event(
            label=label,
            source_path=src_path,
            trajectory=trajectory,
            velocity=velocity,
            start_time=onset,
        )

        # Render
        try:
            output_mix, annotations = ks.render()
        except Exception as e:
            print(f"[ERROR] Render failed: {e}, skipping.")
            pbar.update(1)
            continue

        # Build filename: {source_stem}_{onset:.3f}_{offset:.3f}_i{iter}.wav
        # onset / offset are taken from the first annotation (the foreground event)
        source_stem = os.path.splitext(fname)[0]
        if annotations:
            evt_onset  = annotations[0].get('onset',  0.0)
            evt_offset = annotations[0].get('offset', 0.0)
        else:
            evt_onset, evt_offset = onset, onset
        out_name = (
            f"{source_stem}"
            f"_{evt_onset:.3f}_{evt_offset:.3f}"
            f"_i{iteration_idx}.wav"
        )
        out_path = os.path.join(audio_dir, out_name)

        from scipy.io import wavfile
        wavfile.write(out_path, fs, output_mix.astype(np.float32))

        # Annotations
        for ann in annotations:
            ann['filename']      = out_name
            ann['siren_class']   = cls
            ann['subset_index']  = sample_idx
            ann['iteration']     = iteration_idx
            ann['bg_spl_target'] = bg_spl
            ann['fg_spl_target'] = fg_spl
        dataset_annotations.extend(annotations)

        t_sample_end  = time.time()
        elapsed_total = t_sample_end - t_start
        done_so_far   = global_sample_idx + 1
        avg_per       = elapsed_total / done_so_far
        remaining     = len(all_jobs) - done_so_far
        eta           = avg_per * remaining

        print(
            f"    → Saved {out_name} | "
            f"time={t_sample_end - t_sample_start:.1f}s | ETA={eta:.0f}s"
        )

        pbar.set_postfix({"cls": cls, "bg": f"{bg_spl:.0f}dB", "fg": f"{fg_spl:.0f}dB", "ETA": f"{eta:.0f}s"})
        pbar.update(1)
        global_sample_idx += 1

    pbar.close()

    # 10. Save Aggregated Metadata
    t_end         = time.time()
    total_elapsed = t_end - t_start

    if dataset_annotations:
        tsv_path = os.path.join(tsv_dir, 'metadata.tsv')
        df = pd.DataFrame(dataset_annotations)

        numeric_cols = [
            'onset', 'offset', 'snr_min', 'snr_max', 'snr_avg', 'snr_std',
            'velocity_kmh', 'closest_distance', 'bg_spl_target', 'fg_spl_target',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{float(x):.3f}")

        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Saved Aggregated TSV:  {tsv_path}")

        json_path = os.path.join(json_dir, 'metadata.json')
        with open(json_path, 'w') as jf:
            json.dump({"dataset_metadata": dataset_annotations}, jf, indent=4)
        print(f"Saved Aggregated JSON: {json_path}")
    else:
        df = pd.DataFrame()
        print("[WARN] No samples were generated — metadata files not written.")

    # ------------------------------------------------------------------
    # Post-flight statistics
    # ------------------------------------------------------------------
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("  POST-FLIGHT SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"  Generated      : {global_sample_idx} samples")
    summary_lines.append(f"  Elapsed time   : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    if global_sample_idx:
        summary_lines.append(f"  Avg per sample : {total_elapsed/global_sample_idx:.2f}s")

    if not df.empty:
        # Actual disk usage of audio dir
        total_bytes_disk = sum(
            os.path.getsize(os.path.join(audio_dir, f))
            for f in os.listdir(audio_dir)
            if f.endswith('.wav')
        )
        summary_lines.append(f"  Actual disk use: {total_bytes_disk / (1024**3):.3f} GB")
        summary_lines.append("")

        # Per-class breakdown (re-read df with numeric types for stats)
        df_num = pd.DataFrame(dataset_annotations)
        summary_lines.append("  Per-class sample count:")
        if 'siren_class' in df_num.columns:
            for cls_name, grp in df_num.groupby('siren_class'):
                summary_lines.append(f"    {cls_name:<20s}: {len(grp):>6d} annotations")
        summary_lines.append("")

        # Global SNR statistics
        snr_cols = {
            'snr_min': 'SNR min (dB)',
            'snr_max': 'SNR max (dB)',
            'snr_avg': 'SNR avg (dB)',
            'snr_std': 'SNR std (dB)',
        }
        summary_lines.append("  Global SNR statistics (across all annotations):")
        for col, label in snr_cols.items():
            if col in df_num.columns:
                vals = pd.to_numeric(df_num[col], errors='coerce').dropna()
                if len(vals):
                    summary_lines.append(
                        f"    {label:<25s}: "
                        f"min={vals.min():.2f}  "
                        f"max={vals.max():.2f}  "
                        f"mean={vals.mean():.2f}  "
                        f"std={vals.std():.2f}"
                    )
        summary_lines.append("")

        # BG / FG SPL coverage actually used
        for tgt_col, label in [('bg_spl_target', 'BG SPL targets used'),
                                ('fg_spl_target', 'FG SPL targets used')]:
            if tgt_col in df_num.columns:
                vals = sorted(pd.to_numeric(df_num[tgt_col], errors='coerce').dropna().unique())
                summary_lines.append(f"  {label}: {[f'{v:.1f}' for v in vals]}")

    summary_lines.append("=" * 60)
    summary_text = "\n".join(summary_lines)

    print()
    print(summary_text)

    summary_path = os.path.join(output_folder, 'summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write("KineScaper Siren Dataset Generation Summary\n")
        sf.write(f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sf.write(f"Config: {os.path.abspath(config_path)}\n\n")
        sf.write(summary_text + "\n")
    print(f"Summary saved to: {summary_path}")

    print()
    print(
        f"Done. {global_sample_idx} sample(s) generated in "
        f"{total_elapsed:.1f}s ({total_elapsed/60:.1f} min)."
    )
    print(f"Log saved to: {log_path}")

    logger.close()



if __name__ == "__main__":
    generate_siren_dataset()
