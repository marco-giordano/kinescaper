
import sys
import os
import numpy as np
from scipy.io import wavfile

# Add vendored library to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'libs', 'pyroadacoustics'))

try:
    from pyroadacoustics.environment import Environment
    from pyroadacoustics.soundSource import SoundSource
    from pyroadacoustics.microphoneArray import MicrophoneArray
except ImportError as e:
    print(f"Error importing pyroadacoustics: {e}")
    sys.exit(1)

def run_proof_of_concept():
    print("Initializing Environment...")
    # 1. Setup Environment
    fs = 44100
    env = Environment(fs=fs)

    # 2. Setup Sound Source (Moving)
    # Start: (-50m, 10m, 1m) -> End: (50m, 10m, 1m)
    # Velocity: 15 m/s (~54 km/h)
    start_pos = np.array([-50.0, 10.0, 1.0])
    end_pos = np.array([50.0, 10.0, 1.0])
    
    # Generate a simple source signal (White Noise burst or tone)
    # Let's make it a 500Hz tone for clear Doppler audibility
    duration_approx = 100.0 / 15.0 # ~6.6 seconds
    t = np.linspace(0, duration_approx, int(fs * duration_approx))
    # Using a sawtooth-like or complex tone to hear phase effects better? Simple sine is fine for Doppler.
    # The library has a default signal if we pass None, let's try custom to be sure.
    source_signal = 0.5 * np.sin(2 * np.pi * 440 * t) 
    
    # Trajectory points
    trajectory_points = np.vstack([start_pos, end_pos])
    speed = 15.0
    
    # Add source
    print("Adding Source...")
    env.add_source(
        position=start_pos,
        signal=source_signal,
        trajectory_points=trajectory_points,
        source_velocity=[speed]
    )

    # 3. Setup Microphone (Static observer)
    # Position: (0, 0, 1.8) typical ear height
    mic_loc = np.array([[0.0, 0.0, 1.8]])
    print(f"Adding Microphone at {mic_loc}...")
    env.add_microphone_array(mic_locs=mic_loc)

    # 4. Simulate
    print("Running Simulation... (this might take a moment)")
    # This runs the sample-by-sample loop
    signals = env.simulate()
    
    # signals shape should be (N_mics, N_samples)
    output_signal = signals[0]
    
    # Normalize to avoid clipping
    output_signal = output_signal / np.max(np.abs(output_signal)) * 0.9

    # 5. Save output
    output_path = os.path.join("output", "proof_of_concept.wav")
    print(f"Saving output to {output_path}...")
    wavfile.write(output_path, fs, (output_signal * 32767).astype(np.int16))
    
    print("Done!")

if __name__ == "__main__":
    run_proof_of_concept()
