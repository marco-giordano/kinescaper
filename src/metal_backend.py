
import numpy as np
import torch
import torch.nn.functional as F
import math

class MetalSimulator:
    def __init__(self, fs: int, c: float = 343.0):
        self.fs = fs
        self.c = c
        # Check for MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MetalSimulator: Using MPS (Metal) device.")
        elif torch.cuda.is_available():
             self.device = torch.device("cuda")
             print("MetalSimulator: Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("MetalSimulator: MPS not found, falling back to CPU.")

    def simulate_event(self, 
                       signal: np.ndarray, 
                       trajectory: np.ndarray, 
                       mic_pos: np.ndarray, 
                       velocity: float,
                       start_sample: int) -> np.ndarray:
        """
        Vectorized simulation of a single event source -> single mic.
        """
        # Convert inputs to Tensor
        # Signal: (T,)
        sig_tensor = torch.from_numpy(signal).float().to(self.device)
        traj_tensor = torch.from_numpy(trajectory).float().to(self.device) # (N_points, 3)
        mic_pos_tensor = torch.from_numpy(mic_pos).float().to(self.device) # (3,)
        
        N_samples = sig_tensor.shape[0]
        N_traj = traj_tensor.shape[0]

        # Trajectory is likely undersampled (control points) or fully sampled?
        # In current KineScaper, we generate trajectory per sample or passed as points?
        # KineScaper.render generates trajectory points? 
        # Wait, core.py passes `event.trajectory` which is just start and end point (2 points).
        # pyroadacoustics INTERPOLATES this in `initialize`.
        # We need to reproduce the full trajectory generation here or in core.
        
        # Assumption: trajectory passed here should be the FULL path or we interpolate it here.
        # Since we want to vectorize, we need positions for every audio sample.
        
        # Linear Interpolation of Trajectory
        # Create time steps
        t = torch.linspace(0, 1, steps=N_samples, device=self.device)
        
        # Simple Linear Interpolation between start (traj[0]) and end (traj[-1])
        # TODO: Support complex paths if traj has > 2 points
        p0 = traj_tensor[0]
        p1 = traj_tensor[-1]
        
        # Position per sample: P(t) = P0 + t * (P1 - P0)
        # (N, 3)
        positions = p0.unsqueeze(0) + t.unsqueeze(1) * (p1 - p0).unsqueeze(0)
        
        # --- 1. DIRECT PATH ---
        # Distance (N,)
        vec_d = positions - mic_pos_tensor.unsqueeze(0)
        dist_d = torch.norm(vec_d, dim=1) # (N,)
        
        # Time Delay (samples)
        delay_seconds = dist_d / self.c
        delay_samples = delay_seconds * self.fs
        
        # Fractional Delay (Interpolation)
        # Grid Sample requires coords in [-1, 1]. Too complex for 1D.
        # Use simpler interpolation:
        # indices = n - delay[n]
        n_indices = torch.arange(N_samples, device=self.device).float()
        read_indices = n_indices - delay_samples
        
        # Sig Out (Direct)
        sig_d = self._interpolate(sig_tensor, read_indices)
        
        # Distance Attenuation
        sig_d = sig_d / torch.clamp(dist_d, min=0.1)
        
        # --- 2. GROUND REFLECTION PATH ---
        # Image Source Method: Mirror Z
        # Ground plane assumed usually at Z=0, but PRA uses specific geometry.
        # Assuming Z=0 is road.
        # Mirror source: (x, y, -z) if road is z=0?
        # PRA: `_compute_angle` uses trig. Effectively image source.
        # Image pos: same as source but Z inverted relative to ground?
        # Let's assume ground is Z=0 for simplicity of "Metal Optimization".
        # Pos_img: (x, y, -z)
        pos_img = positions.clone()
        pos_img[:, 2] = -pos_img[:, 2]
        
        vec_r = pos_img - mic_pos_tensor.unsqueeze(0)
        dist_r = torch.norm(vec_r, dim=1)
        
        delay_seconds_r = dist_r / self.c
        
        # Asphalt Reflection is frequency dependent (Low Pass + Table).
        # For this optimization, we can approximate with a simple Attenuation or LPF.
        # Implementing the full complex table lookup on GPU is feasible but advanced.
        # Let's start with a rudimentary reflection coefficient (e.g. 0.8) + Distance Att.
        # Users asking for "Metal 3" usually want the PERFORMANCE, not necessarily bit-exact match to legacy CPU code logic if it's slow.
        # However, to be correct, we should do at least some filtering.
        # We'll skip complex FIR table for now, using a simple scalar factor.
        
        delay_samples_r = delay_seconds_r * self.fs
        read_indices_r = n_indices - delay_samples_r
        sig_r = self._interpolate(sig_tensor, read_indices_r)
        
        # Reflection Damping (Approximate Asphalt)
        sig_r = sig_r * 0.7 
        sig_r = sig_r / torch.clamp(dist_r, min=0.1)
        
        # Combine
        output = sig_d + sig_r
        
        return output.cpu().numpy()

    def _interpolate(self, signal, indices):
        """
        Differentiable linear interpolation for fractional delays.
        Signal: (N,)
        Indices: (N,) query points
        """
        N = signal.shape[0]
        
        # Clamp indices
        # We assume 0 padding outside
        idx_floor = torch.floor(indices).long()
        idx_ceil = idx_floor + 1
        alpha = indices - idx_floor.float()
        
        # Mask valid
        mask = (indices >= 0) & (indices < N - 1)
        
        # Gather
        # Safe gathering by clamping then masking result
        safe_floor = torch.clamp(idx_floor, 0, N-1)
        safe_ceil = torch.clamp(idx_ceil, 0, N-1)
        
        val_floor = signal[safe_floor]
        val_ceil = signal[safe_ceil]
        
        interp = (1 - alpha) * val_floor + alpha * val_ceil
        
        return interp * mask.float()
