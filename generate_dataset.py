import os
import random
import numpy as np
import pandas as pd
from typing import List

from src.core import KineScaper
from src.utils import load_config, load_random_background, load_foreground_metadata, get_event_path, save_dcase_annotations, save_extended_metadata

def generate_dataset():
    # 1. Load Config
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    print("Loaded Config.")
    
    # 2. Setup Output
    output_folder = config.get('output_folder', 'output/dataset')
    audio_dir = os.path.join(output_folder, 'audio')
    csv_dir = os.path.join(output_folder, 'csv')
    json_dir = os.path.join(output_folder, 'json')
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # 3. Load Metadata
    bg_folder = config.get('background_dir')
    fg_folder = config.get('foreground_dir')
    fg_meta_path = os.path.join(fg_folder, 'metadata.csv')
    
    if not os.path.exists(fg_meta_path):
        print(f"Foreground metadata not found: {fg_meta_path}")
        return

    fg_metadata = load_foreground_metadata(fg_meta_path)
    
    # 4. Batch Loop
    n_samples = config.get('n_samples', 1)
    velocity_range = config.get('velocity', [30.0, 50.0]) # Default fallback
    if isinstance(velocity_range, (int, float)):
        velocity_range = [velocity_range, velocity_range]
        
    print(f"Starting generation of {n_samples} samples...")
    print(f"Velocity Range: {velocity_range} km/h")
    
    for i in range(n_samples):
        print(f"\n--- Sample {i+1}/{n_samples} ---")
        
        # Initialize Engine per sample to keep state clean (or reuse?)
        # Reuse is better for efficiency if expensive resources (Metal backend?), 
        # but KineScaper holds state (self.events). We should re-init or clear events.
        # Let's re-init for safety for now.
        ks = KineScaper(config)
        
        # A. Background
        # Random duration? Use config duration.
        duration = config.get('output_duration', 10.0)
        bg_fs = config.get('sampling_frequency', 44100)
        
        try:
            bg_chunk = load_random_background(bg_folder, duration, bg_fs)
            ks.add_background(bg_chunk)
        except Exception as e:
            print(f"Error loading background: {e}")
            continue

        # B. Foreground Event
        # Pick random row
        row = fg_metadata.sample(n=1).iloc[0]
        label = row.get('event_label', row.get('siren_mode', 'unknown')) # Fallback
        fname = row['filename']
        source_path = get_event_path(fg_folder, fname)
        
        # Random Velocity
        velocity = random.uniform(velocity_range[0], velocity_range[1])
        
        # Trajectory (Simple Linear for now, from config)
        start_pos = np.array(config['start_position'])
        end_pos = np.array(config['end_position'])
        
        # Calculate Checkpoints
        # Distance = norm(end - start)
        dist_total = np.linalg.norm(end_pos - start_pos)
        
        # Time = Distance / Speed (m/s)
        speed_ms = velocity / 3.6
        travel_time = dist_total / speed_ms
        
        # We need trajectory points for the duration of the event?
        # Or just the full path? KineScaper takes full path.
        # But we need time points? pyroadacoustics generates time steps based on speed.
        # We just need geometric points.
        # Let's pass the 2 points [start, end].
        trajectory = np.array([start_pos, end_pos])
        
        # Start Time: Randomize onset?
        # Let's say we want the CPA (Closest Point of Approach) to be within the file?
        # Or just start at random time t0.
        # Config currently implies full trajectory simulation.
        # Simpler: Start event at t=2.0s?
        # Let's randomize onset between 1.0s and (Duration - EventLen - 1.0s)
        # We don't know event len without loading?
        # Just use a sensible default or config.
        # Default onset = 2.0s
        onset = 2.0 
        
        ks.add_event(label=label, 
                     source_path=source_path, 
                     trajectory=trajectory, 
                     velocity=velocity, 
                     start_time=onset)
        
        # C. Render
        output_mix, annotations = ks.render()
        
        # D. Save
        # Unique filename: dataset_001.wav
        out_name = f"dataset_{i:04d}.wav"
        out_path = os.path.join(audio_dir, out_name)
        
        # Normalize Output?
        # KineScaper output might be unbounded float.
        # Clip or Norm?
        # Usually DCASE datasets are normalized to -0.1dB or similar, but
        # we have calibrated SPL. We should keep the relative levels?
        # Writing float32 wav is safer.
        # scipy.io.wavfile writes based on dtype.
        
        # Convert to int16 for compatibility? Or float32?
        # Let's stick to float32 if possible, or int16 PCM.
        # For standard audio, int16 is common.
        # Let's normalize peak to -1dB to avoid clipping if it's too hot,
        # BUT that destroys our SPL calibration!
        # If we want to preserve SPL calibration, we must not normalize arbitrary peaks.
        # But we must ensure it fits in integer range if saving as int.
        # Let's save as float32.
        
        from scipy.io import wavfile
        wavfile.write(out_path, bg_fs, output_mix.astype(np.float32))
        
        # E. Annotations
        # Update filename in annotations
        save_dcase_annotations(annotations, csv_dir, out_name)
        save_extended_metadata(annotations, json_dir, out_name)
        
        print(f"Saved {out_name}")

if __name__ == "__main__":
    generate_dataset()
