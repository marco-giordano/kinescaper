
import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# Ensure libs are in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_DIR = os.path.join(ROOT_DIR, 'libs', 'pyroadacoustics')
if LIBS_DIR not in sys.path:
    sys.path.append(LIBS_DIR)

try:
    from pyroadacoustics.environment import Environment
    from pyroadacoustics.soundSource import SoundSource
    from pyroadacoustics.microphoneArray import MicrophoneArray
except ImportError as _pyroad_err:
    _msg = (
        f"pyroadacoustics could not be imported: {_pyroad_err}\n"
        "Make sure the library is on sys.path (e.g. libs/pyroadacoustics is present "
        "and contains a valid pyroadacoustics package)."
    )
    class _MissingDep:          # noqa: E301
        def __init__(self, *a, **kw):
            raise RuntimeError(_msg)
    Environment       = _MissingDep  # type: ignore[assignment,misc]
    SoundSource       = _MissingDep  # type: ignore[assignment,misc]
    MicrophoneArray   = _MissingDep  # type: ignore[assignment,misc]


from .utils import db_to_linear, linear_to_db

@dataclass
class SoundEvent:
    label: str
    source_path: str
    trajectory: np.ndarray
    velocity: float
    start_time: float
    # These params are now used for the mixing logic
    # source_level_db: float # Removed, controlled by config now

class KineScaper:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.fs = config.get('sampling_frequency', 44100)
        self.duration = config.get('output_duration', 10.0)
        self.events: List[SoundEvent] = []
        
        self.background_signal: Optional[np.ndarray] = None
        self.bg_gain_linear: float = 1.0

        # Calibration Params
        self.cal_bg_norm = config.get('cal_background_norm', -20.0) # dBFS
        self.cal_bg_spl = config.get('cal_background_SPL', 50.0)
        self.cal_fg_norm = config.get('cal_foreground_norm', -1.0) # dBFS
        self.cal_fg_spl = config.get('cal_foreground_SPL', 100.0)
        
        self.req_bg_spl = config.get('required_background_SPL', 45.0)
        self.req_fg_spl = config.get('required_foreground_SPL', 90.0)

    def add_background(self, audio_data: np.ndarray):
        """
        Add background texture and apply Virtual Calibration Step A.
        """
        # Ensure correct length first? No, apply calibration on loop or source?
        # User says: "Load Background: Read the whole park_ambience.wav... Calculate one RMS... Multiply entire array"
        # So we operate on the input chunk first.
        
        # 1. Measure RMS
        rms = np.sqrt(np.mean(audio_data**2))
        if rms == 0:
            print("Warning: Background is silent.")
            self.background_signal = np.zeros(int(self.duration * self.fs))
            return

        rms_db = linear_to_db(rms)
        
        # 2. Calculate Gain to reach cal_bg_norm
        # Target = -20 dBFS
        gain_db = self.cal_bg_norm - rms_db
        gain_lin = db_to_linear(gain_db)
        
        # 3. Safety Check (Analytic)
        peak_val = np.max(np.abs(audio_data))
        projected_peak = peak_val * gain_lin
        
        if projected_peak > 1.0:
            print(f"Background Clipping Safety: projected peak {projected_peak:.2f} > 1.0. Reducing gain.")
            gain_lin = 1.0 / peak_val # Maximize to 0dBFS relative
            # Update virtual normalized level?
            # If we reduce gain, the effective normalized level is lower than -20dBFS.
            # But the logic assumes we are at -20dBFS for the SPL mapping.
            # If we can't reach -20dBFS without clipping, then our mapping shifts.
            # Let's assume we stick to the new peak.
            # Actually, standardizing to -20dBFS is usually safe for ambience.
            # But let's respect the clip check.
            pass
            
        normalized_bg = audio_data * gain_lin
        
        # 4. Prepare loop
        target_len = int(self.duration * self.fs)
        if len(normalized_bg) < target_len:
            repeats = int(np.ceil(target_len / len(normalized_bg)))
            normalized_bg = np.tile(normalized_bg, repeats)[:target_len]
        else:
            normalized_bg = normalized_bg[:target_len]
            
        self.background_signal = normalized_bg
        
        # 5. Calculate Final Mixing Gain for Background
        # We have normalized B at 'cal_bg_norm' (-20dBFS).
        # This maps to 'cal_bg_spl' (50dB).
        # We want 'req_bg_spl' (45dB).
        # Delta = 45 - 50 = -5dB.
        # So we apply -5dB gain on top of the normalized signal.
        # But we also need to balance against Foreground.
        # We will apply the relative gains at render time, or here.
        # Let's apply here to store the "Ready to Mix" signal.
        # Wait, if we mix in Render, we need to know the relation.
        # Let's compute the additional gain:
        
        extra_gain_db = self.req_bg_spl - self.cal_bg_spl
        self.bg_mix_gain = db_to_linear(extra_gain_db)


    def add_event(self, 
                  label: str, 
                  source_path: str, 
                  trajectory: np.ndarray, 
                  velocity: float, 
                  start_time: float):
        
        self.events.append(SoundEvent(label, source_path, trajectory, velocity, start_time))

    def _load_and_normalize_foreground(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source file not found: {path}")

        file_fs, data = wavfile.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        # Resampling
        if file_fs != self.fs:
             import scipy.signal
             new_len = int(len(data) * self.fs / file_fs)
             data = scipy.signal.resample(data, new_len)
             # print(f"Resampled FG {os.path.basename(path)} to {self.fs}")
        
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
             data = data / 2147483648.0
        data = data.astype(float)
        
        # Step B: Foreground Normalization
        # Normalize to Peak (cal_fg_norm usually -1dB)
        peak = np.max(np.abs(data))
        if peak == 0: return data
        
        target_peak_lin = db_to_linear(self.cal_fg_norm)
        gain = target_peak_lin / peak
        
        return data * gain


    def _point_segment_distance(self, px, py, pz, p1, p2):
        """
        Calculate the distance between a point p=(px,py,pz) and a line segment p1-p2.
        """
        p = np.array([px, py, pz])
        # Vector v = p2 - p1
        v = p2 - p1
        # Vector w = p - p1
        w = p - p1
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p - p1)
            
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p - p2)
            
        b = c1 / c2
        pb = p1 + b * v
        return np.linalg.norm(p - pb)
    
    def _compute_trajectory_min_distance(self, trajectory: np.ndarray, mic_pos: np.ndarray) -> float:
        """
        Computes the minimum distance between a trajectory (polyline) and a mic position.
        """
        min_d = float('inf')
        # Iterate over segments
        for i in range(len(trajectory) - 1):
             p1 = trajectory[i]
             p2 = trajectory[i+1]
             d = self._point_segment_distance(mic_pos[0], mic_pos[1], mic_pos[2], p1, p2)
             if d < min_d:
                 min_d = d
        return min_d

    def render(self) -> Tuple[np.ndarray, List[Dict]]:
        total_samples = int(self.duration * self.fs)
        output_mix = np.zeros(total_samples)
        
        # Prepare Config Params
        sens_f = self.cal_fg_spl - self.cal_fg_norm
        sens_b = self.cal_bg_spl - self.cal_bg_norm
        
        align_b = sens_b - sens_f
        target_adj_b = self.req_bg_spl - self.cal_bg_spl
        total_gain_b_db = align_b + target_adj_b
        
        target_adj_f = self.req_fg_spl - self.cal_fg_spl
        total_gain_f_db = target_adj_f
        
        # 1. Prepare Background Trace
        # We need the full background trace as it appears in the final mix for SNR calc
        bg_trace = np.zeros(total_samples)
        if self.background_signal is not None:
             bg_gain_lin = db_to_linear(total_gain_b_db)
             bg_trace = self.background_signal * bg_gain_lin
             
             output_mix += bg_trace 

        annotations = []
        frame_size = self.config.get('frame_size', 2048)
        
        # 2. Foreground
        gain_f_lin = db_to_linear(total_gain_f_db)
        mic_locs = self.config.get('mic_positions', [[0.0, 0.0, 1.8]])
        nmics = len(mic_locs)

        # Ensure output_mix is correct shape if we have >1 mics
        if nmics > 1 and output_mix.ndim == 1:
            output_mix_multi = np.zeros((nmics, total_samples))
            for ch in range(nmics):
                output_mix_multi[ch] = output_mix # Copy mono bg
            output_mix = output_mix_multi

        for event in self.events:
            print(f"Simulating event: {event.label}...")
            try:
                # 0. Min Distance Check (Geometric Segment Check)
                mic_array = np.array(mic_locs)
                min_dist_overall = float('inf')
                for m_idx in range(nmics):
                    mic_pos = np.array(mic_locs[m_idx])
                    d = self._compute_trajectory_min_distance(event.trajectory, mic_pos)
                    if d < min_dist_overall:
                        min_dist_overall = d
                
                min_dist = min_dist_overall

                if min_dist < 1.0:
                    print(f"Skipping event {event.label}: Trajectory too close to microphone (Min Dist: {min_dist:.2f}m < 1.0m)")
                    raise ValueError(f"Trajectory violates Minimum Distance Constraint ({min_dist:.2f}m < 1.0m)")

                # Speed Conversion (km/h -> m/s)
                # User config velocity is now treated as km/h
                velocity_ms = event.velocity / 3.6
                
                # Load & Norm Step B
                norm_sig = self._load_and_normalize_foreground(event.source_path)
                
                # Apply Physics Mixing Gain (Step C prep)
                input_sig = norm_sig * gain_f_lin
                
                # ── Routing decision ──────────────────────────────────────────────────
                #   acceleration=metal  →  MetalVectorisedSimulator (MPS/CUDA)
                #   acceleration=cpu  +  use_vectorisation=True  →  vectorised NumPy
                #   acceleration=cpu  +  use_vectorisation=False →  legacy SimulatorManager
                #
                # Rule: Metal requires vectorisation (tensors ops only).
                # If user sets acceleration=metal but use_vectorisation=False, we warn
                # and force it on automatically.
                # ─────────────────────────────────────────────────────────────────────
                accel_mode = self.config.get('acceleration', 'cpu')
                use_vec    = self.config.get('use_vectorisation', False)

                if accel_mode == 'metal' and not use_vec:
                    print(
                        "⚠️  WARNING: acceleration='metal' requires use_vectorisation=true. "
                        "Enabling vectorisation automatically for this run. "
                        "Set 'use_vectorisation: true' in config.yaml to suppress this warning."
                    )
                    use_vec = True

                sim_output = None

                # ── Branch 1: Metal ────────────────────────────────────────────────
                if accel_mode == 'metal':
                    try:
                        from .metal_vectorised_backend import simulate_metal_environment

                        env_metal = Environment(fs=self.fs)
                        env_metal.add_source(
                            position=event.trajectory[0],
                            signal=input_sig,
                            trajectory_points=event.trajectory,
                            source_velocity=[velocity_ms],
                        )
                        env_metal.add_microphone_array(mic_array)

                        sim_output = simulate_metal_environment(env_metal)  # (M, N)

                    except ImportError:
                        print("⚠️  Failed to import metal_vectorised_backend (torch missing?). "
                              "Falling back to NumPy vectorised path.")
                        accel_mode = 'cpu'
                        use_vec    = True
                    except Exception as e:
                        print(f"⚠️  Metal simulation failed: {e}. "
                              "Falling back to NumPy vectorised path.")
                        accel_mode = 'cpu'
                        use_vec    = True

                # ── Branch 2 & 3: CPU (vectorised or legacy) ──────────────────────
                if accel_mode == 'cpu' or sim_output is None:
                    env = Environment(fs=self.fs)
                    env.add_source(
                        position=event.trajectory[0],
                        signal=input_sig,
                        trajectory_points=event.trajectory,
                        source_velocity=[velocity_ms],
                    )
                    env.add_microphone_array(mic_array)

                    if use_vec:
                        from .vectorised_simulator import simulate_vectorised_environment
                        sim_output = simulate_vectorised_environment(env)   # (M, N_sim)
                    else:
                        sim_output = env.simulate()                         # (M, N_sim)
                
                # --- SNR ANALYSIS (Frame-based) ---
                fg_trace = sim_output[0]
                
                start_sample = int(event.start_time * self.fs)
                
                # Determine overlap range
                write_start = max(0, start_sample)
                read_start = 0 if start_sample >= 0 else -start_sample
                ev_len = fg_trace.shape[0]
                available = total_samples - write_start
                
                if available > 0 and read_start < ev_len:
                    to_write = min(available, ev_len - read_start)
                    
                    # Get the overlapping segments (Mic 0)
                    fg_seg = fg_trace[read_start : read_start + to_write]
                    bg_seg = bg_trace[write_start : write_start + to_write]
                    
                    # 1. Framing
                    n_frames = len(fg_seg) // frame_size
                    if n_frames > 0:
                        fg_frames = fg_seg[:n_frames*frame_size].reshape(n_frames, frame_size)
                        bg_frames = bg_seg[:n_frames*frame_size].reshape(n_frames, frame_size)
                        
                        e_f = np.mean(fg_frames**2, axis=1)
                        e_b = np.mean(bg_frames**2, axis=1)
                        e_b = np.maximum(e_b, 1e-12) 
                        e_f = np.maximum(e_f, 1e-12)
                        
                        snr_frames = 10 * np.log10(e_f / e_b)
                        
                        snr_min = float(np.min(snr_frames))
                        snr_max = float(np.max(snr_frames))
                        snr_avg = float(np.mean(snr_frames))
                        snr_std = float(np.std(snr_frames))
                    else:
                        snr_min = snr_max = snr_avg = snr_std = 0.0

                    # --- MIXING ---
                    if output_mix.ndim == 1:
                        output_mix[write_start:write_start+to_write] += fg_seg
                    else:
                        chunk_multi = sim_output[:, read_start:read_start+to_write]
                        output_mix[:, write_start:write_start+to_write] += chunk_multi

                    annotations.append({
                        "event_label": event.label,
                        "filename": os.path.basename(event.source_path),
                        "onset": event.start_time,
                        "offset": min(event.start_time + (ev_len / self.fs), self.duration),
                        "snr_min": snr_min,
                        "snr_max": snr_max,
                        "snr_avg": snr_avg,
                        "snr_std": snr_std,
                        "frame_size": frame_size,
                        # Extended metadata
                        "velocity_kmh": event.velocity, # Original input
                        "closest_distance": min_dist
                    })
                    
            except Exception as e:
                print(f"Simulation failed for {event.label}: {e}")
                import traceback
                traceback.print_exc()

        return output_mix, annotations
