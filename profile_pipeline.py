"""
profile_pipeline.py
--------------------
Profiling harness for the KineScaper sample-generation pipeline.

Usage:
    python profile_pipeline.py

Outputs:
    - profile_pipeline.prof   (raw cProfile data, inspectable with snakeviz / pstats)
    - profile_report.txt      (sorted text summary)
    - profile_stages.txt      (manual wall-clock breakdown per pipeline stage)

Uses n_samples=1, output_duration taken from config.yaml so the numbers
reflect one full real-world generation pass.
"""

import cProfile
import pstats
import io
import os
import sys
import time
import random
import datetime
import json

import numpy as np
import scipy.io.wavfile as wavfile

# ── path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.utils import (
    load_config, load_random_background, load_foreground_metadata, get_event_path
)
from src.core import KineScaper

# ── helpers ───────────────────────────────────────────────────────────────────
PROF_OUT   = os.path.join(ROOT_DIR, "profile_pipeline.prof")
REPORT_OUT = os.path.join(ROOT_DIR, "profile_report.txt")
STAGES_OUT = os.path.join(ROOT_DIR, "profile_stages.txt")

CLOCK = time.perf_counter


def _short_path(full_path: str) -> str:
    """Return path relative to project root for readability."""
    try:
        return os.path.relpath(full_path, ROOT_DIR)
    except ValueError:
        return full_path


# ──────────────────────────────────────────────────────────────────────────────
# Stage-timed pipeline — mirrors generate_dataset() for a single sample
# ──────────────────────────────────────────────────────────────────────────────
def run_once_staged(config: dict) -> dict:
    """Run one full generate-sample pass and record wall-clock per stage."""
    stages = {}

    # Pull config params (same logic as generate_dataset.py)
    bg_folder    = config.get('background_dir')
    fg_folder    = config.get('foreground_dir')
    fg_meta_path = config.get('metadata_path')
    duration     = config.get('output_duration', 10.0)
    fs           = config.get('sampling_frequency', 44100)
    velocity_range = config.get('velocity', [30.0, 50.0])
    if isinstance(velocity_range, (int, float)):
        velocity_range = [velocity_range, velocity_range]
    onset_range = config.get('onset_range', [1.0, 5.0])
    if isinstance(onset_range, (int, float)):
        onset_range = [onset_range, onset_range]
    start_center = config.get('start_circle_center', [-200.0, 5.0])
    end_center   = config.get('end_circle_center',   [ 200.0, 5.0])
    pos_radius   = config.get('position_radius', 0.0)
    z_range      = config.get('z_range', [1.8, 1.8])
    if isinstance(z_range, (int, float)):
        z_range = [z_range, z_range]

    # ── Stage 0: KineScaper init ───────────────────────────────────────────────
    t0 = CLOCK()
    ks = KineScaper(config)
    stages['0_ks_init'] = CLOCK() - t0

    # ── Stage 1: load_random_background ───────────────────────────────────────
    t0 = CLOCK()
    bg_chunk = load_random_background(bg_folder, duration, fs)
    stages['1_load_background'] = CLOCK() - t0

    # ── Stage 2: add_background (RMS norm + tile)  ──────────────────────────
    t0 = CLOCK()
    ks.add_background(bg_chunk)
    stages['2_add_background'] = CLOCK() - t0

    # ── Stage 3: load foreground metadata + pick row ────────────────────────
    t0 = CLOCK()
    fg_metadata  = load_foreground_metadata(fg_meta_path)
    row          = fg_metadata.sample(n=1).iloc[0]
    label        = row['mode']
    fname        = row['filename']
    source_path  = get_event_path(fg_folder, fname)
    stages['3_load_fg_meta'] = CLOCK() - t0

    # ── Stage 4: trajectory generation ─────────────────────────────────────
    t0 = CLOCK()
    velocity = random.uniform(velocity_range[0], velocity_range[1])
    onset    = random.uniform(onset_range[0], onset_range[1])
    z        = random.uniform(z_range[0], z_range[1])
    theta_s  = random.uniform(0, 2 * np.pi)
    theta_e  = random.uniform(0, 2 * np.pi)
    start_pos = np.array([
        start_center[0] + pos_radius * np.cos(theta_s),
        start_center[1] + pos_radius * np.sin(theta_s),
        z
    ])
    end_pos = np.array([
        end_center[0] + pos_radius * np.cos(theta_e),
        end_center[1] + pos_radius * np.sin(theta_e),
        z
    ])
    trajectory = np.array([start_pos, end_pos])
    ks.add_event(label=label, source_path=source_path,
                 trajectory=trajectory, velocity=velocity, start_time=onset)
    stages['4_trajectory_setup'] = CLOCK() - t0

    # ── Stage 5: _load_and_normalize_foreground (isolated) ─────────────────
    t0 = CLOCK()
    norm_sig = ks._load_and_normalize_foreground(source_path)
    stages['5_load_norm_fg_audio'] = CLOCK() - t0

    # ── Stage 6: full render() (includes simulation + SNR + mixing) ─────────
    t0 = CLOCK()
    output_mix, annotations = ks.render()
    stages['6_render_total'] = CLOCK() - t0

    # ── Stage 7: write WAV ──────────────────────────────────────────────────
    import tempfile, scipy.io.wavfile as _wf
    t0 = CLOCK()
    tmp = tempfile.mktemp(suffix='.wav')
    _wf.write(tmp, fs, output_mix.astype(np.float32))
    os.remove(tmp)
    stages['7_write_wav'] = CLOCK() - t0

    return stages


# ──────────────────────────────────────────────────────────────────────────────
# cProfile wrapper (runs the same pipeline internally)
# ──────────────────────────────────────────────────────────────────────────────
def run_once_profiled(config: dict):
    """Target function for cProfile — identical to generate_dataset one-shot."""
    bg_folder    = config.get('background_dir')
    fg_folder    = config.get('foreground_dir')
    fg_meta_path = config.get('metadata_path')
    duration     = config.get('output_duration', 10.0)
    fs           = config.get('sampling_frequency', 44100)
    velocity_range = config.get('velocity', [30.0, 50.0])
    if isinstance(velocity_range, (int, float)):
        velocity_range = [velocity_range, velocity_range]
    onset_range = config.get('onset_range', [1.0, 5.0])
    if isinstance(onset_range, (int, float)):
        onset_range = [onset_range, onset_range]
    start_center = config.get('start_circle_center', [-200.0, 5.0])
    end_center   = config.get('end_circle_center',   [ 200.0, 5.0])
    pos_radius   = config.get('position_radius', 0.0)
    z_range      = config.get('z_range', [1.8, 1.8])
    if isinstance(z_range, (int, float)):
        z_range = [z_range, z_range]

    ks = KineScaper(config)

    fg_metadata  = load_foreground_metadata(fg_meta_path)
    bg_chunk     = load_random_background(bg_folder, duration, fs)
    ks.add_background(bg_chunk)

    row          = fg_metadata.sample(n=1).iloc[0]
    label        = row['mode']
    fname        = row['filename']
    source_path  = get_event_path(fg_folder, fname)
    velocity     = random.uniform(velocity_range[0], velocity_range[1])
    onset        = random.uniform(onset_range[0], onset_range[1])
    z            = random.uniform(z_range[0], z_range[1])
    theta_s      = random.uniform(0, 2 * np.pi)
    theta_e      = random.uniform(0, 2 * np.pi)
    start_pos    = np.array([
        start_center[0] + pos_radius * np.cos(theta_s),
        start_center[1] + pos_radius * np.sin(theta_s),
        z
    ])
    end_pos = np.array([
        end_center[0] + pos_radius * np.cos(theta_e),
        end_center[1] + pos_radius * np.sin(theta_e),
        z
    ])
    trajectory = np.array([start_pos, end_pos])
    ks.add_event(label=label, source_path=source_path,
                 trajectory=trajectory, velocity=velocity, start_time=onset)
    ks.render()


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    config = load_config('config.yaml')

    fs       = config.get('sampling_frequency', 44100)
    duration = config.get('output_duration', 10.0)
    n_total  = int(fs * duration)

    print("=" * 70)
    print(f"KineScaper — Pipeline Profiling")
    print(f"  fs={fs} Hz  |  duration={duration}s  |  total_samples={n_total:,}")
    print(f"  acceleration='{config.get('acceleration','cpu')}'")
    print("=" * 70)
    print()

    # ── 1. Stage-timer run ────────────────────────────────────────────────────
    print("Running staged wall-clock breakdown ...")
    stages = run_once_staged(config)
    total_wall = sum(stages.values())

    lines_stage = []
    lines_stage.append(f"KineScaper — Stage Wall-Clock Breakdown")
    lines_stage.append(f"Generated: {datetime.datetime.now()}")
    lines_stage.append(f"fs={fs}  duration={duration}s  total_audio_samples={n_total:,}")
    lines_stage.append("")
    lines_stage.append(f"{'Stage':<35}  {'Time (s)':>10}  {'% total':>8}")
    lines_stage.append("-" * 58)
    for name, t in sorted(stages.items()):
        pct = 100.0 * t / total_wall if total_wall > 0 else 0
        lines_stage.append(f"  {name:<33}  {t:>10.3f}  {pct:>7.1f}%")
    lines_stage.append("-" * 58)
    lines_stage.append(f"  {'TOTAL':<33}  {total_wall:>10.3f}  {'100.0%':>8}")
    lines_stage.append("")

    stage_text = "\n".join(lines_stage)
    print(stage_text)

    with open(STAGES_OUT, 'w') as f:
        f.write(stage_text)
    print(f"→ Saved to {STAGES_OUT}")
    print()

    # ── 2. cProfile run ───────────────────────────────────────────────────────
    print("Running cProfile (this will take a while on CPU) ...")
    pr = cProfile.Profile()
    pr.enable()
    run_once_profiled(config)
    pr.disable()
    pr.dump_stats(PROF_OUT)
    print(f"→ Raw profile saved to {PROF_OUT}")

    # Text report
    sio = io.StringIO()
    ps  = pstats.Stats(pr, stream=sio)
    ps.sort_stats('cumulative')
    ps.print_stats(60)  # top 60 lines
    report_text = sio.getvalue()

    with open(REPORT_OUT, 'w') as f:
        f.write(f"KineScaper — cProfile Report (sorted by cumulative time)\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"fs={fs}  duration={duration}s  total_audio_samples={n_total:,}\n\n")
        f.write(report_text)
    print(f"→ Text report saved to {REPORT_OUT}")
    print()

    # ── 3. Quick summary printed to console ───────────────────────────────────
    print("=" * 70)
    print("TOP FUNCTIONS (by cumulative time)")
    print("=" * 70)
    sio2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=sio2)
    ps2.sort_stats('cumulative')
    ps2.print_stats(20)
    print(sio2.getvalue())

    print("=" * 70)
    print("TOP FUNCTIONS (by per-call time — real CPU hogs)")
    print("=" * 70)
    sio3 = io.StringIO()
    ps3 = pstats.Stats(pr, stream=sio3)
    ps3.sort_stats('tottime')
    ps3.print_stats(20)
    print(sio3.getvalue())

    print(f"\nAll outputs written to project root:\n"
          f"  {STAGES_OUT}\n  {REPORT_OUT}\n  {PROF_OUT}\n")

    # ── 4. Hint for optional snakeviz visualisation ────────────────────────
    print("To explore the profile interactively, run:")
    print(f"  snakeviz {PROF_OUT}")
    print()


if __name__ == "__main__":
    main()
