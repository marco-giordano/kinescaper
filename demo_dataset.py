
import os
import numpy as np
import scipy.io.wavfile as wavfile
from src.core import KineScaper
from src.utils import load_config, load_random_background, load_foreground_metadata, get_event_path, save_dcase_annotations, save_extended_metadata

def run_demo():
    print("Initializing KineScaper Demo with Config...")
    
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    # Resolve relative paths
    root_dir = os.path.dirname(__file__)
    bg_dir = os.path.join(root_dir, config['background_dir'])
    fg_dir = os.path.join(root_dir, config['foreground_dir'])
    metadata_path = os.path.join(root_dir, config['metadata_path'])
    
    fs = config['sampling_frequency']
    duration = config['output_duration']
    
    ks = KineScaper(config)
    
    # 1. Add Background
    print("Loading Background...")
    try:
        bg_chunk = load_random_background(bg_dir, duration, fs)
        ks.add_background(bg_chunk) # internal logic handles validation against config
    except Exception as e:
        print(f"Background loading failed: {e}")
        
    # 2. Add Foreground Event
    print("Loading Foreground Metadata...")
    try:
        df = load_foreground_metadata(metadata_path)
        # Pick a random "siren"
        siren_row = df.sample(1).iloc[0]
        filename = siren_row['filename']
        label = siren_row['siren_mode']
        
        path = get_event_path(fg_dir, filename)
        
        # Define Trajectory from Config
        start_pos = np.array(config['start_position'])
        end_pos = np.array(config['end_position'])
        velocity = config['velocity']
        
        traj = np.vstack([start_pos, end_pos])
        
        print(f"Adding Event: {label} ({filename})")
        ks.add_event(
            label=label,
            source_path=path,
            trajectory=traj,
            velocity=velocity,
            start_time=2.0
        )
        
        # 3. Render
        print("Rendering...")
        mix, annotations = ks.render()
        
        # 4. Save
        out_folder_name = config.get('output_folder', 'output')
        out_dir = os.path.join(root_dir, out_folder_name)
        
        # Create subfolders
        audio_dir = os.path.join(out_dir, 'audio')
        csv_dir = os.path.join(out_dir, 'csv')
        json_dir = os.path.join(out_dir, 'json')
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        filename_base = 'demo_config_mix.wav'
        out_path = os.path.join(audio_dir, filename_base)
        
        # Normalize final mix just to be safe for playback (avoid hard clipping if calibration was aggressive)
        max_val = np.max(np.abs(mix))
        if max_val > 1.0:
            print(f"Warning: Mix clipped (Peak {max_val:.2f}). Normalizing to -1dB.")
            mix = mix / max_val * 0.89
        
        # Handle Multichannel
        if mix.ndim > 1:
            mix = mix.T # Scipy expects (N, M)
            
        wavfile.write(out_path, fs, mix.astype(np.float32))
        print(f"Saved audio to {out_path}")
        
        # Save Annotations
        save_dcase_annotations(annotations, csv_dir, filename_base)
        save_extended_metadata(annotations, json_dir, filename_base)
        print("Annotations Saved.")
        
    except Exception as e:
        print(f"Foreground workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
