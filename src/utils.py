
import os
import yaml
import random
import pandas as pd
import numpy as np
import json
import csv
from scipy.io import wavfile
from typing import Tuple, List, Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_random_background(bg_dir: str, duration: float, fs: int) -> np.ndarray:
    """
    Loads a random chunk of 'duration' seconds from a random wav file in bg_dir.
    """
    files = [f for f in os.listdir(bg_dir) if f.lower().endswith('.wav')]
    if not files:
        raise FileNotFoundError(f"No wav files found in {bg_dir}")
    
    choice = random.choice(files)
    path = os.path.join(bg_dir, choice)
    
    file_fs, data = wavfile.read(path)
    
    # Handle stereo/mono
    if data.ndim > 1:
        data = np.mean(data, axis=1) # Downmix to mono for now
        
    import scipy.signal
    
    # Resampling
    if file_fs != fs:
        # Calculate number of samples
        new_len = int(len(data) * fs / file_fs)
        data = scipy.signal.resample(data, new_len)
        print(f"Resampled {choice} from {file_fs} to {fs} Hz")
    
    # Normalize to float
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    data = data.astype(float)

    n_samples = int(duration * fs)
    if len(data) <= n_samples:
        # Loop it
        reps = int(np.ceil(n_samples / len(data)))
        chunk = np.tile(data, reps)[:n_samples]
    else:
        # Random crop
        max_start = len(data) - n_samples
        start = random.randint(0, max_start)
        chunk = data[start:start+n_samples]
        
    return chunk

def load_foreground_metadata(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def get_event_path(foreground_dir: str, filename: str) -> str:
    return os.path.join(foreground_dir, filename)

def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)

def linear_to_db(linear: float) -> float:
    if linear == 0:
        return -np.inf
    return 20 * np.log10(np.abs(linear))

def save_dcase_annotations(annotations: List[Dict], output_folder: str, audio_filename: str):
    """
    Saves DCASE-style CSV: filename, onset, offset, event_label, snr_min, snr_max, snr_avg
    """
    csv_filename = os.path.splitext(audio_filename)[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_filename)
    
    headers = ["filename", "onset", "offset", "event_label", "snr_min", "snr_max", "snr_avg"]
    
    with open(csv_path, 'w', newline='') as f:
        # DCASE format is often tab-separated, but checking requirement...
        # Project spec says "tab-separated" (TSV) effectively but calls it CSV.
        # "widely adopted tab-separated format used in DCASE challenges"
        writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        
        for ann in annotations:
            row = {
                "filename": audio_filename,
                "onset": f"{ann['onset']:.3f}",
                "offset": f"{ann['offset']:.3f}",
                "event_label": ann['event_label'],
                "snr_min": f"{ann.get('snr_min', 0):.1f}", 
                "snr_max": f"{ann.get('snr_max', 0):.1f}",
                "snr_avg": f"{ann.get('snr_avg', 0):.1f}"
            }
            writer.writerow(row)

def save_extended_metadata(annotations: List[Dict], output_folder: str, audio_filename: str):
    """
    Saves JSON sidecar with full physics details.
    """
    json_filename = os.path.splitext(audio_filename)[0] + ".json"
    json_path = os.path.join(output_folder, json_filename)
    
    data = {
        "audio_filename": audio_filename,
        "events": annotations
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
