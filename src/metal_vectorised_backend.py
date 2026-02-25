"""
src/metal_vectorised_backend.py
--------------------------------
PyTorch port of vectorised_simulator.py for Metal (MPS) / CUDA acceleration.

This module provides the same full physics as the NumPy vectorised engine, but
all tensor operations run on the device selected at init time:
  - MPS   (Apple Silicon Metal)
  - CUDA  (NVIDIA GPU)
  - CPU   (fallback — same as vectorised_simulator.py but via torch)

Device selection is automatic:  MPS  →  CUDA  →  CPU.

Public API
----------
MetalVectorisedSimulator(fs, c, Z0, road_material, air_abs_coeffs, simulation_params)
    .simulate(source_trajectory, source_signal, mic_pos) -> np.ndarray (N,)

simulate_metal_environment(env) -> np.ndarray (M, N)
    Drop-in replacement for Environment.simulate() using this backend.
"""

import math
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from typing import Dict

# ---------------------------------------------------------------------------

def _build_sinc_table_torch(device: torch.device):
    """Build the windowed Sinc look-up table identical to DelayLine.__init__."""
    SINC_SMP  = 11
    _delta    = 1.0 / 50.0
    delays    = torch.arange(0, 1 + _delta, _delta, device=device)   # (51,)
    table_size = len(delays)

    window = torch.hann_window(SINC_SMP, device=device)               # (11,)
    n_taps = torch.arange(SINC_SMP, device=device).float()            # (11,)

    table = torch.zeros(table_size, SINC_SMP, device=device)
    for idx in range(table_size):
        row = torch.sinc(n_taps - (SINC_SMP - 1) / 2.0 - delays[idx])
        row = row * window
        row = row / row.sum()
        table[idx] = row

    return table, delays, _delta, SINC_SMP


def _sinc_gather_torch(signal: torch.Tensor, read_indices: torch.Tensor,
                        sinc_table: torch.Tensor, table_delays: torch.Tensor,
                        delta: float, sinc_smp: int) -> torch.Tensor:
    """
    Vectorised fractional-delay read (same logic as vectorised_simulator._sinc_gather).
    All tensors on the same device.
    """
    N_sig = signal.shape[0]
    half  = sinc_smp // 2

    floor_idx = torch.floor(read_indices).long()          # (N,)
    frac      = read_indices - floor_idx.float()          # (N,) in [0, 1)

    alpha    = frac % delta
    pos      = torch.searchsorted(table_delays.contiguous(), (frac - alpha).contiguous())
    pos      = pos.clamp(0, len(table_delays) - 2)

    h_lo  = sinc_table[pos]                               # (N, sinc_smp)
    h_hi  = sinc_table[pos + 1]                           # (N, sinc_smp)
    needs_hi = (alpha > 1e-9).unsqueeze(1)
    h_sinc = torch.where(needs_hi,
                         alpha.unsqueeze(1) * (h_lo - h_hi) + h_hi,
                         h_lo)                            # (N, sinc_smp)

    tap_offsets = torch.arange(sinc_smp, device=signal.device) - half   # (sinc_smp,)
    gather_idx  = floor_idx.unsqueeze(1) + tap_offsets.unsqueeze(0)      # (N, sinc_smp)

    valid    = (gather_idx >= 0) & (gather_idx < N_sig)
    safe_idx = gather_idx.clamp(0, N_sig - 1)
    gathered = torch.where(valid, signal[safe_idx], torch.zeros_like(signal[safe_idx]))

    in_range = (read_indices >= 0) & (read_indices < N_sig)
    out = (h_sinc * gathered).sum(dim=1) * in_range.float()
    return out


def _apply_fir_causal_torch(signal: torch.Tensor, fir_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Causal per-sample varying FIR: out[n] = sum_k fir[n,k] * signal[n-k].

    Matches SimulatorManager._read1Buf deque semantics.
    Uses torch.unfold for a zero-copy strided view.
    """
    N, numtaps = fir_coeffs.shape
    # Pad start only (causal)
    sig_padded = F.pad(signal.unsqueeze(0).unsqueeze(0),
                       (numtaps - 1, 0), mode='constant', value=0.0)   # (1,1,N+L-1)
    # Sliding windows: shape (N, numtaps), windows[n, j] = signal[n - (numtaps-1) + j]
    windows = sig_padded.squeeze(0).squeeze(0).unfold(0, numtaps, 1)   # (N, numtaps)
    # fir[n, 0] * signal[n], fir[n, 1] * signal[n-1] → flip fir
    fir_flipped = fir_coeffs.flip(1)
    return (windows * fir_flipped).sum(dim=1)


def _build_air_absorption_filters_torch(
    air_abs_coeffs: torch.Tensor,   # (nbands,) on device
    pseud_A: torch.Tensor,          # (half_L, nbands) on device
    distances: torch.Tensor,         # (N,) on device
    numtaps: int = 11,
    n_bins: int = 200,
) -> torch.Tensor:
    """Return (N, numtaps) causal FIR coefficients, one row per sample."""
    d_min = distances.min().item()
    d_max = distances.max().item()

    if abs(d_max - d_min) < 1e-6:
        alpha = torch.pow(10.0, -air_abs_coeffs * d_min / 20.0)
        mid   = (numtaps + 1) // 2 - 1
        h_half = pseud_A @ alpha                               # (half_L,)
        h = torch.zeros(numtaps, device=distances.device)
        h[mid:]  = h_half
        h[:mid]  = h_half[1:].flip(0)
        return h.unsqueeze(0).expand(len(distances), -1)

    edges    = torch.linspace(d_min, d_max, n_bins + 1, device=distances.device)
    bin_ids  = torch.searchsorted(edges[1:].contiguous(), distances.contiguous())
    bin_ids  = bin_ids.clamp(0, n_bins - 1)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])              # (n_bins,)
    mid = (numtaps + 1) // 2 - 1
    filters = torch.zeros(n_bins, numtaps, device=distances.device)
    for i in range(n_bins):
        alpha        = torch.pow(10.0, -air_abs_coeffs * bin_centers[i] / 20.0)
        h_half       = pseud_A @ alpha
        h            = torch.zeros(numtaps, device=distances.device)
        h[mid:]      = h_half
        h[:mid]      = h_half[1:].flip(0)
        filters[i]   = h

    return filters[bin_ids]                                    # (N, numtaps)


# ---------------------------------------------------------------------------

class MetalVectorisedSimulator:
    """
    Full-physics vectorised simulator on Metal (MPS) / CUDA / CPU (torch).

    Physics identical to vectorised_simulator.py — the same equations, the same
    filter tables — but all operations are torch tensor ops that run on the
    selected accelerated device.
    """

    def __init__(self, fs: int, c: float, Z0: float, road_material,
                 air_abs_coeffs: np.ndarray, simulation_params: Dict):
        # Pick device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MetalVectorisedSimulator: using MPS (Metal).")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("MetalVectorisedSimulator: using CUDA.")
        else:
            self.device = torch.device("cpu")
            print("MetalVectorisedSimulator: MPS/CUDA unavailable, falling back to CPU torch.")

        self.fs    = fs
        self.c     = c
        self.Z0    = Z0
        self.road_material  = road_material
        self.sim_params     = simulation_params

        # ── Pre-compute on CPU, move to device ────────────────────────────
        # Sinc table
        self._sinc_table, self._sinc_delays, self._sinc_delta, self._SINC_SMP = \
            _build_sinc_table_torch(self.device)

        # Air absorption pseudo-inverse
        air_coeff_np = air_abs_coeffs
        nbands = len(air_coeff_np)
        freqs  = np.linspace(0, fs / 2, nbands)
        w      = 2 * np.pi * freqs / (fs / 2)
        L_air  = 10
        half_L = int(L_air / 2 + 1)
        A = np.zeros((nbands, half_L))
        A[:, 0] = 1.0
        for j in range(1, half_L):
            A[:, j] = 2 * np.cos(w * j)
        self._pseud_A = torch.from_numpy(
            np.linalg.inv(A.T @ A) @ A.T
        ).float().to(self.device)                              # (half_L, nbands)
        self._air_abs_coeffs = torch.from_numpy(
            air_coeff_np.astype(np.float32)
        ).to(self.device)                                      # (nbands,)

        # Asphalt reflection FIR table (Material branch only)
        self._refl_table = None
        self._n_taps_refl = 11
        if hasattr(road_material, 'absorption'):
            theta_vec = np.arange(-89, 90, 1)
            refl_np   = np.sqrt(1 - np.array(road_material.absorption["coeffs"]))
            Z_surf    = -Z0 * (refl_np + 1) / (refl_np - 1)
            cfreqs    = np.array(road_material.absorption["center_freqs"])
            t = np.zeros((len(theta_vec), self._n_taps_refl))
            for i, th in enumerate(theta_vec):
                R = np.abs((Z_surf * np.cos(math.radians(th)) - Z0) /
                           (Z_surf * np.cos(math.radians(th)) + Z0))
                t[i] = scipy.signal.firwin2(self._n_taps_refl, cfreqs / (fs / 2), R)
            self._refl_table = torch.from_numpy(t.astype(np.float32)).to(self.device)

    # ------------------------------------------------------------------

    def simulate(self,
                 source_trajectory: np.ndarray,    # (N, 3)
                 source_signal: np.ndarray,         # (N,)
                 mic_pos: np.ndarray,               # (3,)
                 ) -> np.ndarray:
        """Simulate one source → one mic. Returns (N,) numpy array."""
        dev = self.device

        src = torch.from_numpy(source_trajectory.astype(np.float32)).to(dev)  # (N,3)
        sig = torch.from_numpy(source_signal.astype(np.float32)).to(dev)      # (N,)
        mic = torch.from_numpy(mic_pos.astype(np.float32)).to(dev)            # (3,)
        N   = sig.shape[0]

        include_refl   = self.sim_params.get("include_reflected_path", True)
        include_airabs = self.sim_params.get("include_air_absorption", True)

        # ── 1. Geometry ────────────────────────────────────────────────
        diff_d   = src - mic.unsqueeze(0)                                      # (N,3)
        dist_d   = torch.norm(diff_d, dim=1)                                   # (N,)
        delay_d  = dist_d / self.c * self.fs                                   # (N,)

        image_src        = src.clone()
        image_src[:, 2]  = -src[:, 2]
        diff_im          = image_src - mic.unsqueeze(0)
        dist_im_mic      = torch.norm(diff_im, dim=1).clamp(min=1e-9)
        sin_theta        = ((src[:, 2] + mic[2]) / dist_im_mic).clamp(-1.0, 1.0)
        theta            = torch.arcsin(sin_theta)                             # (N,)

        safe_sin = torch.abs(torch.sin(theta)).clamp(min=1e-9) * torch.sign(torch.sin(theta) + 1e-30)
        a        = src[:, 2] / safe_sin                                        # (N,)
        b        = mic[2]    / safe_sin                                        # (N,)
        delay_a  = a / self.c * self.fs
        delay_b  = b / self.c * self.fs

        n_idx = torch.arange(N, device=dev, dtype=torch.float32)

        # ── 2. Direct path delay gather ────────────────────────────────
        read_d  = n_idx - delay_d
        sig_dir = _sinc_gather_torch(sig, read_d,
                                     self._sinc_table, self._sinc_delays,
                                     self._sinc_delta, self._SINC_SMP)

        if include_airabs:
            fir_d   = _build_air_absorption_filters_torch(
                self._air_abs_coeffs, self._pseud_A, dist_d)
            sig_dir = _apply_fir_causal_torch(sig_dir, fir_d)

        sig_dir = sig_dir / dist_d.clamp(min=1e-9)

        out = sig_dir  # mic directivity = 1.0 (omnidirectional)

        # ── 3. Reflected path ──────────────────────────────────────────
        if include_refl:
            read_a  = n_idx - delay_a
            sig_a   = _sinc_gather_torch(sig, read_a,
                                         self._sinc_table, self._sinc_delays,
                                         self._sinc_delta, self._SINC_SMP)
            if include_airabs:
                fir_a  = _build_air_absorption_filters_torch(
                    self._air_abs_coeffs, self._pseud_A, a)
                sig_a  = _apply_fir_causal_torch(sig_a, fir_a)

            sig_a = sig_a / a.clamp(min=1e-9)

            # Asphalt FIR (Material branch)
            if self._refl_table is not None:
                theta_deg_refl = 90.0 - torch.rad2deg(theta)
                theta_deg_refl = theta_deg_refl.clamp(-89.0, 89.0)
                refl_idx       = torch.round(theta_deg_refl).long() + 89
                refl_idx       = refl_idx.clamp(0, self._refl_table.shape[0] - 1)
                fir_refl       = self._refl_table[refl_idx]                    # (N, taps)
                sig_a          = _apply_fir_causal_torch(sig_a, fir_refl)

            # Second delay leg (reflection point → mic)
            read_b  = n_idx - delay_b
            sig_b   = _sinc_gather_torch(sig_a, read_b,
                                         self._sinc_table, self._sinc_delays,
                                         self._sinc_delta, self._SINC_SMP)
            if include_airabs:
                fir_b  = _build_air_absorption_filters_torch(
                    self._air_abs_coeffs, self._pseud_A, b)
                sig_b  = _apply_fir_causal_torch(sig_b, fir_b)

            sig_b = sig_b / b.clamp(min=1e-9)
            out   = sig_dir + sig_b

        return out.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------

def simulate_metal_environment(env) -> np.ndarray:
    """
    Drop-in replacement for Environment.simulate() using Metal/CUDA acceleration.

    Parameters
    ----------
    env : pyroadacoustics.Environment  (fully configured)

    Returns
    -------
    np.ndarray  shape (M, N)
    """
    if env.source is None or env.mic_array is None:
        raise RuntimeError('Before running simulation, define a source and mic_array')

    N = len(env.source.trajectory)
    M = env.mic_array.nmics

    # Instantiate the simulator once (builds tables) and reuse across mics
    sim = MetalVectorisedSimulator(
        fs                  = env.fs,
        c                   = env.c,
        Z0                  = env.Z0,
        road_material       = env.road_material,
        air_abs_coeffs      = env.air_absorption_coefficients,
        simulation_params   = env.simulation_params,
    )

    signals = np.zeros((M, N))
    for m in range(M):
        signals[m] = sim.simulate(
            source_trajectory = env.source.trajectory,
            source_signal     = env.source.signal,
            mic_pos           = env.mic_array.mic_positions[m],
        )

    return signals
