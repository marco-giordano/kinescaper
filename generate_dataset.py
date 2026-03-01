import os
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd
import json
from typing import List
from tqdm import tqdm

from src.core import KineScaper
from src.utils import load_config, load_random_background, load_foreground_metadata, get_event_path, save_dcase_annotations, save_extended_metadata


class TeeLogger:
    """Mirrors all writes to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._log = open(log_path, 'w', buffering=1)  # line-buffered

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


def generate_dataset():
    # 1. Load Config
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)

    # 2. Setup Output
    output_folder = config.get('output_folder', 'output/dataset')
    audio_dir = os.path.join(output_folder, 'audio')
    tsv_dir   = os.path.join(output_folder, 'csv')
    json_dir  = os.path.join(output_folder, 'json')

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(tsv_dir,   exist_ok=True)
    os.makedirs(json_dir,  exist_ok=True)

    # 3. Start logging — mirrors all print() output to a .txt file in output root
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_folder, f"generation_log_{timestamp}.txt")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"KineScaper — Dataset Generation Log")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:  {os.path.abspath(config_path)}")
    print(f"Output:  {os.path.abspath(output_folder)}")
    print()

    print("Loaded Config.")

    # 4. Load Metadata
    bg_folder    = config.get('background_dir')
    fg_folder    = config.get('foreground_dir')
    fg_meta_path = config.get('metadata_path')

    if not fg_meta_path:
        print("'metadata_path' not set in config.yaml")
        logger.close()
        return
    if not os.path.exists(fg_meta_path):
        print(f"Foreground metadata not found: {fg_meta_path}")
        logger.close()
        return

    fg_metadata = load_foreground_metadata(fg_meta_path)

    # 5. Batch Loop Setup
    n_samples     = config.get('n_samples', 1)
    velocity_range = config.get('velocity', [30.0, 50.0])
    if isinstance(velocity_range, (int, float)):
        velocity_range = [velocity_range, velocity_range]

    onset_range = config.get('onset_range', [1.0, 5.0])
    if isinstance(onset_range, (int, float)):
        onset_range = [onset_range, onset_range]

    # Circle-based geometry
    start_center = config.get('start_circle_center', [-200.0, 5.0])
    end_center   = config.get('end_circle_center',   [ 200.0, 5.0])
    pos_radius   = config.get('position_radius', 0.0)
    z_range      = config.get('z_range', [1.8, 1.8])
    if isinstance(z_range, (int, float)):
        z_range = [z_range, z_range]

    print(f"Starting generation of {n_samples} samples...")
    print(f"Velocity Range:  {velocity_range} km/h")
    print(f"Onset Range:     {onset_range} s")
    print(f"Position Radius: {pos_radius} m")
    print()

    dataset_annotations = []
    t_start = time.time()

    # Progress bar — writes to the real stdout so it renders correctly in the terminal
    pbar = tqdm(
        total=n_samples,
        desc="Generating",
        unit="sample",
        file=logger._stdout,    # always the real stdout, not the tee
        dynamic_ncols=True,
        colour="green",
    )

    for i in range(n_samples):
        t_sample_start = time.time()
        print(f"--- Sample {i+1}/{n_samples} ---")

        # Foreground Event
        row   = fg_metadata.sample(n=1).iloc[0]
        label = row['mode']
        fname = row['filename']
        source_path = get_event_path(fg_folder, fname)

        velocity = random.uniform(velocity_range[0], velocity_range[1])
        onset    = random.uniform(onset_range[0], onset_range[1])

        duration = config.get('output_duration', 10.0)
        bg_fs    = config.get('sampling_frequency', 44100)

        _MAX_TRAJ_RETRIES = 20

        for _traj_attempt in range(_MAX_TRAJ_RETRIES):
            # Re-instantiate KineScaper so event list is fresh on each retry
            ks = KineScaper(config)
            try:
                bg_chunk = load_random_background(bg_folder, duration, bg_fs)
                ks.add_background(bg_chunk)
            except Exception as e:
                print(f"Error loading background: {e}")
                break   # background errors are not trajectory-related; skip sample

            # Circle-randomized trajectory — z fixed per simulation
            z       = random.uniform(z_range[0], z_range[1])
            theta_s = random.uniform(0, 2 * np.pi)
            theta_e = random.uniform(0, 2 * np.pi)
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

            if _traj_attempt > 0:
                print(f"  [retry {_traj_attempt}/{_MAX_TRAJ_RETRIES}] new trajectory: start={start_pos.round(2)}, end={end_pos.round(2)}")
            else:
                print(f"  label={label}, velocity={velocity:.1f} km/h, onset={onset:.2f}s")
                print(f"  start={start_pos.round(2)}, end={end_pos.round(2)}, z={z:.2f}m")

            ks.add_event(
                label=label,
                source_path=source_path,
                trajectory=trajectory,
                velocity=velocity,
                start_time=onset,
            )

            # C. Render
            output_mix, annotations = ks.render()

            if not annotations:
                # Minimum Distance Constraint was violated — try a new trajectory
                print(f"  Minimum Distance Constraint violated — drawing new trajectory…")
                continue

            # Valid simulation — exit retry loop
            break
        else:
            print(f"  [WARN] Could not find a valid trajectory after {_MAX_TRAJ_RETRIES} attempts. Skipping sample.")
            pbar.update(1)
            continue


        # D. Save audio
        out_name = f"dataset_{i:04d}.wav"
        out_path = os.path.join(audio_dir, out_name)

        from scipy.io import wavfile
        wavfile.write(out_path, bg_fs, output_mix.astype(np.float32))

        # E. Annotations
        for ann in annotations:
            ann['filename'] = out_name
        dataset_annotations.extend(annotations)

        # ETA + progress bar update
        t_sample_end  = time.time()
        elapsed_total = t_sample_end - t_start
        avg_per_sample = elapsed_total / (i + 1)
        eta_seconds   = avg_per_sample * (n_samples - (i + 1))

        print(f"  Saved {out_name}  |  Sample time: {t_sample_end - t_sample_start:.1f}s  |  ETA: {eta_seconds:.0f}s remaining")
        print()

        pbar.set_postfix({"last": label, "ETA": f"{eta_seconds:.0f}s"})
        pbar.update(1)

    pbar.close()

    # 6. Save Aggregated Metadata
    t_end         = time.time()
    total_elapsed = t_end - t_start

    tsv_path = os.path.join(tsv_dir, 'metadata.tsv')
    df = pd.DataFrame(dataset_annotations)

    for col in ['onset', 'offset', 'snr_min', 'snr_max', 'snr_avg', 'snr_std', 'velocity_kmh', 'closest_distance']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{float(x):.3f}")

    df.to_csv(tsv_path, sep='\t', index=False)
    print(f"Saved Aggregated TSV:  {tsv_path}")

    json_path = os.path.join(json_dir, 'metadata.json')
    json_data = {"dataset_metadata": dataset_annotations}
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved Aggregated JSON: {json_path}")

    print()
    print(f"✓ Done. Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) for {n_samples} sample(s).")
    print(f"Log saved to: {log_path}")

    logger.close()


if __name__ == "__main__":
    generate_dataset()
