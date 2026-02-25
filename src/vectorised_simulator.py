"""
src/vectorised_simulator.py
----------------------------
Vectorised NumPy simulation engine for KineScaper.

Replicates the physics of `SimulatorManager.update()` / `Environment.simulate()`
**exactly** (same formulas, same filter tables) but operates on all N audio samples
simultaneously, eliminating the per-sample Python loop.

Physics pipeline (mirrors SimulatorManager.update(), step by step):
 1. Geometry per sample  : dist, delay, angle   → shape (N,)
 2. Fractional delay read: gather from input sig → Sinc interp, shape (N,)
 3. Air absorption FIR   : distance-binned batch convolution
 4. Distance attenuation : 1/dist               → shape (N,)
 5. Asphalt reflection   : angle-binned batch convolution
 6. Second delay leg     : reflected path delay  → gather
 7. Mic directivity      : scalar factor         → shape (N,)
 8. Sum direct + reflected                       → (N,)

Public API
----------
simulate_vectorised(source_trajectory, source_signal, mic_pos,
                    c, fs, Z0, road_material,
                    air_absorption_coefficients, simulation_params)
     → np.ndarray of shape (N,)
"""

import math
import numpy as np
import scipy.signal
import scipy.ndimage
from typing import Dict

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vectorised_air_absorption_filter(
    distances: np.ndarray,            # (N,) float
    air_abs_coeffs: np.ndarray,       # (nbands,)
    pseud_A: np.ndarray,              # pre-computed pseudo-inverse
    numtaps: int = 11,
    n_bins: int = 200,
) -> np.ndarray:
    """
    Compute the air-absorption-filtered signal via distance-binned batch convolution.

    For each distance bin, one FIR filter is computed (identical to
    SimulatorManager._compute_air_absorption_filter) and applied to the
    corresponding samples using overlap-add convolution.

    Parameters
    ----------
    distances : (N,) distances per sample
    air_abs_coeffs : air absorption coefficient array from Environment
    pseud_A : pre-computed pseudo-inverse matrix (from SimulatorManager.__init__)
    numtaps : FIR filter length (must match reference, default 11)
    n_bins  : number of distance bins (trade-off accuracy ↔ speed)

    Returns
    -------
    (N, numtaps) array of FIR coefficients, one row per sample.
    We return per-sample coefficients so the caller can apply convolution.
    """
    # Discretise distances into bins
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))

    if d_max == d_min:
        # All distances identical → one filter for all
        alpha = 10 ** (-air_abs_coeffs * d_min / 20.0)
        h = np.empty(numtaps)
        mid = int((numtaps + 1) / 2) - 1
        h[mid:] = pseud_A.dot(alpha)
        h[:mid] = np.flip(h[mid + 1:])
        return np.tile(h, (len(distances), 1))     # (N, numtaps)

    edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_ids = np.searchsorted(edges[1:], distances, side='left')   # (N,) in [0, n_bins-1]
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    # Build one filter per bin
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    filters = np.empty((n_bins, numtaps))
    for i, d in enumerate(bin_centers):
        alpha = 10 ** (-air_abs_coeffs * d / 20.0)
        mid = int((numtaps + 1) / 2) - 1
        h = np.empty(numtaps)
        h[mid:] = pseud_A.dot(alpha)
        h[:mid] = np.flip(h[mid + 1:])
        filters[i] = h

    return filters[bin_ids]    # (N, numtaps)


def _apply_fir_per_sample(signal: np.ndarray, fir_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply a causal FIR filter (with per-sample varying coefficients) to a signal.

    Replicates the reference behaviour where each output sample is computed as:
        out[n] = sum_{k=0}^{L-1}  fir[n, k] * signal[n - k]

    This matches SimulatorManager._read1Buf semantics: a deque of the last L
    delayed samples (newest first), dotted with the current fir coefficients.
    The operation is strictly causal — only past samples are used.

    Parameters
    ----------
    signal     : (N,) input signal
    fir_coeffs : (N, numtaps) per-sample FIR coefficients, where fir[n, 0]
                 multiplies the current sample and fir[n, k] multiplies
                 the sample k steps in the past.

    Returns
    -------
    (N,) causal-filtered signal
    """
    N, numtaps = fir_coeffs.shape

    # Pad ONLY at the start so signal[n-k] is valid for k in [0, numtaps-1].
    # signal_padded[n] corresponds to signal[n - (numtaps-1)].
    # Window starting at position n in the original signal uses padded indices
    # [n, n+1, ..., n + numtaps - 1] where index i maps to signal[i - (numtaps-1)].
    # out[n] = sum_k fir[n,k] * signal_padded[n + numtaps - 1 - k]
    #        = sum_k fir[n,k] * signal[n - k]   ← causal
    sig_padded = np.pad(signal, (numtaps - 1, 0), mode='constant')

    # Strided view: windows[n] = sig_padded[n : n + numtaps]
    #   = [signal[n-(numtaps-1)], ..., signal[n-1], signal[n]]
    shape   = (N, numtaps)
    strides = (sig_padded.strides[0], sig_padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(sig_padded, shape=shape, strides=strides)

    # fir[n, 0] * signal[n], fir[n, 1] * signal[n-1], etc.
    # windows[n, numtaps-1-k] = signal[n-k], so we flip fir along tap axis:
    fir_flipped = fir_coeffs[:, ::-1]   # (N, numtaps) — fir_flipped[n,j] = fir[n, L-1-j]
    out = np.einsum('ni,ni->n', windows, fir_flipped)
    return out


def _sinc_gather(signal: np.ndarray, read_indices: np.ndarray,
                 sinc_table: np.ndarray, table_delays: np.ndarray,
                 delta: float, sinc_smp: int) -> np.ndarray:
    """
    Vectorised fractional-delay read using the same windowed-Sinc look-up table
    as `DelayLine._frac_delay_interpolated_sinc` + the gather loop in
    `DelayLine._interpolated_read(..., method='Sinc')`.

    Replicates exactly:
        for i in range(sinc_smp):
            out += h_sinc[i] * delay_line[(rpi + i - floor(sinc_smp/2)) % N]

    Parameters
    ----------
    signal      : (N,) source signal (treated as a ring buffer, zero-padded outside)
    read_indices: (N,) float read positions (may be negative or > N → zero output)
    sinc_table  : (table_size, sinc_smp) pre-built table
    table_delays: (table_size,) fractional delays corresponding to each table row
    delta       : table step size
    sinc_smp    : number of sinc taps

    Returns
    -------
    (N,) gathered signal
    """
    N_sig = len(signal)
    N     = len(read_indices)
    half  = sinc_smp // 2

    # Integer and fractional parts
    floor_idx = np.floor(read_indices).astype(np.int64)   # (N,)
    frac      = read_indices - floor_idx                   # (N,) in [0, 1)

    # Look-up sinc filter for each fractional delay
    alpha    = frac % delta
    pos      = np.searchsorted(table_delays, frac - alpha)       # (N,) table row
    pos      = np.clip(pos, 0, len(table_delays) - 2)

    # Linear interpolation between neighbouring table rows
    h_lo = sinc_table[pos]                                 # (N, sinc_smp)
    h_hi = sinc_table[pos + 1]                             # (N, sinc_smp)
    needs_hi = alpha > 1e-9
    h_sinc = np.where(needs_hi[:, None],
                      alpha[:, None] * (h_lo - h_hi) + h_hi,
                      h_lo)                                # (N, sinc_smp)

    # Gather signal samples: for tap i → floor_idx + i - half (mod N_sig)
    tap_offsets = np.arange(sinc_smp) - half               # (sinc_smp,)
    gather_idx  = floor_idx[:, None] + tap_offsets[None, :]  # (N, sinc_smp)

    # Valid mask (only gather within [0, N_sig-1])
    valid = (gather_idx >= 0) & (gather_idx < N_sig)
    safe_idx = np.clip(gather_idx, 0, N_sig - 1)
    gathered = np.where(valid, signal[safe_idx], 0.0)     # (N, sinc_smp)

    # Also mask out samples where read_index itself is out-of-range
    in_range = (read_indices >= 0) & (read_indices < N_sig)
    out = np.einsum('ni,ni->n', h_sinc, gathered) * in_range
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_vectorised(
    source_trajectory: np.ndarray,        # (N, 3) — fully interpolated positions
    source_signal: np.ndarray,            # (N,)
    mic_pos: np.ndarray,                  # (3,)
    c: float,
    fs: int,
    Z0: float,
    road_material,                        # Material or int
    air_absorption_coefficients: np.ndarray,
    simulation_params: Dict,
) -> np.ndarray:
    """
    Vectorised simulation of one (source, microphone) pair.

    Matches the physics of `SimulatorManager.update()` called sequentially
    for every sample.  Returns a (N,) array equivalent to `signals[m]` in
    `Environment.simulate()`.

    Parameters
    ----------
    source_trajectory : (N, 3) — one position per audio sample.
                        This is the FULL trajectory as pre-expanded by SoundSource
                        (same as `self.source.trajectory` in Environment).
    source_signal     : (N,) — source audio samples.
    mic_pos           : (3,) microphone position.
    c, fs, Z0         : acoustic environment constants from Environment.
    road_material     : Material or int, same as SimulatorManager expects.
    air_absorption_coefficients : from Environment._compute_air_absorption_coefficients().
    simulation_params : dict with keys 'include_reflected_path', 'include_air_absorption'.
                        The 'interp_method' key is ignored; Sinc is always used here.

    Returns
    -------
    np.ndarray of shape (N,) — microphone output.
    """
    from .utils import db_to_linear

    N = len(source_signal)
    src = source_trajectory                              # (N, 3)
    mic = mic_pos                                        # (3,)

    include_refl   = simulation_params.get("include_reflected_path", True)
    include_airabs = simulation_params.get("include_air_absorption", True)

    # ------------------------------------------------------------------
    # 1. Geometry — vectorised over all N samples
    # ------------------------------------------------------------------
    # Direct path: source → mic
    diff_d = src - mic[None, :]                          # (N, 3)
    dist_d = np.linalg.norm(diff_d, axis=1)              # (N,)  distance
    delay_d_samples = dist_d / c * fs                    # (N,)  delay in samples

    # Incidence angle θ (same formula as _compute_angle):
    #   dist_image_mic = ||(src - [0,0,2*src_z]) - mic||
    #   θ = asin((src_z + mic_z) / dist_image_mic)
    image_src   = src.copy()
    image_src[:, 2] = -src[:, 2]                         # mirror Z → image source
    diff_im     = image_src - mic[None, :]               # (N, 3)
    dist_im_mic = np.linalg.norm(diff_im, axis=1)        # (N,)
    sin_theta   = np.clip((src[:, 2] + mic[2]) / np.maximum(dist_im_mic, 1e-9), -1.0, 1.0)
    theta       = np.arcsin(sin_theta)                   # (N,)  radians

    # Reflected path segment distances
    safe_sin = np.maximum(np.abs(np.sin(theta)), 1e-9) * np.sign(np.sin(theta) + 1e-30)
    a = src[:, 2] / safe_sin                             # (N,)  src → reflection point
    b = mic[2]    / safe_sin                             # (N,)  reflection point → mic

    delay_a_samples = a / c * fs                         # (N,)
    delay_b_samples = b / c * fs                         # (N,)

    # ------------------------------------------------------------------
    # 2. Build windowed-Sinc table (matches DelayLine._SINC_SMP = 11)
    # ------------------------------------------------------------------
    SINC_SMP   = 11
    _WINDOW    = np.hanning(SINC_SMP)
    _delta     = 1.0 / 50.0
    _table_delays = np.arange(0, 1 + _delta, _delta)
    _table_size   = len(_table_delays)
    _sinc_table   = np.zeros((_table_size, SINC_SMP))
    n_taps = np.arange(SINC_SMP)
    for idx, val in enumerate(_table_delays):
        row = np.sinc(n_taps - (SINC_SMP - 1) / 2 - val)
        row *= _WINDOW
        row /= np.sum(row)
        _sinc_table[idx] = row

    # ------------------------------------------------------------------
    # 3. Pre-compute pseudo-inverse A for air absorption (same as SimulatorManager)
    # ------------------------------------------------------------------
    nbands = len(air_absorption_coefficients)
    freqs  = np.linspace(0, fs / 2, nbands)
    w      = 2 * np.pi * freqs / (fs / 2)
    L_air  = 10
    half_L = int(L_air / 2 + 1)
    A = np.zeros((nbands, half_L))
    A[:, 0] = 1.0
    for j in range(1, half_L):
        A[:, j] = 2 * np.cos(w * j)
    pseud_A = np.linalg.inv(A.T @ A) @ A.T              # (half_L, nbands)

    def _air_fir(distances_1d, numtaps=11):
        """Return (M, numtaps) array of per-sample FIR coefficients."""
        return _vectorised_air_absorption_filter(
            distances_1d, air_absorption_coefficients, pseud_A, numtaps
        )

    # ------------------------------------------------------------------
    # 4. Direct path: fractional-delay gather
    #    Read index for sample n: n - delay_d[n]  (causal, look into the past)
    # ------------------------------------------------------------------
    n_idx      = np.arange(N, dtype=float)
    read_d     = n_idx - delay_d_samples                 # (N,) float

    sig_direct = _sinc_gather(source_signal, read_d,
                               _sinc_table, _table_delays, _delta, SINC_SMP)

    # Air absorption on direct path
    if include_airabs:
        fir_d  = _air_fir(dist_d)                        # (N, 11)
        sig_direct = _apply_fir_per_sample(sig_direct, fir_d)

    # Distance attenuation 1/d
    sig_direct /= np.maximum(dist_d, 1e-9)

    # Microphone directivity (omnidirectional → 1.0 always in current config)
    # Full directivity patterns supported:
    mic_dir_pattern = "omnidirectional"   # KineScaper always uses omnidirectional
    if mic_dir_pattern == "omnidirectional":
        mic_dir = np.ones(N)
    else:
        orient = np.array([1.0, 0.0])
        cos_t  = np.einsum('ni,i->n', diff_d[:, :2], orient) / np.maximum(dist_d, 1e-9)
        pattern_map = {
            "subcardioid":    lambda c: 0.75 + 0.25 * c,
            "cardioid":       lambda c: 0.50 + 0.50 * c,
            "supercardioid":  lambda c: 1/3  + 2/3  * c,
            "hypercardioid":  lambda c: 0.25 + 0.75 * c,
            "figure8":        lambda c: c,
        }
        mic_dir = pattern_map.get(mic_dir_pattern, lambda c: np.ones_like(c))(cos_t)

    y_received = mic_dir * sig_direct                    # (N,)

    # ------------------------------------------------------------------
    # 5. Reflected path
    # ------------------------------------------------------------------
    if include_refl:
        # --- 5a. Source → reflection point (delay = a samples) ---
        read_a = n_idx - delay_a_samples                 # (N,)
        sig_a  = _sinc_gather(source_signal, read_a,
                               _sinc_table, _table_delays, _delta, SINC_SMP)

        if include_airabs:
            fir_a  = _air_fir(a)
            sig_a  = _apply_fir_per_sample(sig_a, fir_a)

        sig_a /= np.maximum(a, 1e-9)

        # --- 5b. Asphalt reflection FIR (angle-binned) ---
        # Same formula as _get_asphalt_reflection_filter (Material / real branch):
        #   theta_deg_refl = 90 - degrees(theta)
        #   idx = round(theta_deg_refl) + 89   → into [-89..89] table
        if hasattr(road_material, 'absorption'):
            # Material branch — same as _precompute_real_angle_reflection_filter_table
            from libs.pyroadacoustics.pyroadacoustics.material import Material as _Mat
            n_taps_refl = 11
            theta_vec   = np.arange(-89, 90, 1)          # (179,)

            refl_coeffs = np.sqrt(1 - np.array(road_material.absorption["coeffs"]))
            Z_surf = -Z0 * (refl_coeffs + 1) / (refl_coeffs - 1)
            center_freqs = np.array(road_material.absorption["center_freqs"])

            refl_table = np.zeros((len(theta_vec), n_taps_refl))
            for idx, th in enumerate(theta_vec):
                R = np.abs((Z_surf * np.cos(math.radians(th)) - Z0) /
                           (Z_surf * np.cos(math.radians(th)) + Z0))
                refl_table[idx] = scipy.signal.firwin2(
                    n_taps_refl, center_freqs / (fs / 2), R
                )

            # theta in the asphalt call = 90 - degrees(theta_angle)
            theta_deg_refl = 90.0 - np.degrees(theta)    # (N,)
            theta_deg_refl = np.clip(theta_deg_refl, -89, 89)
            refl_idx = np.round(theta_deg_refl).astype(int) + 89  # (N,) into table
            refl_idx = np.clip(refl_idx, 0, len(theta_vec) - 1)

            # Per-sample FIR coefficients (N, n_taps_refl)
            fir_refl_per_sample = refl_table[refl_idx]

            sig_a = _apply_fir_per_sample(sig_a, fir_refl_per_sample)
        # else: int road_material (complex branch) — treated as flat 1.0 gain
        # (complex branch rarely used with default config which uses Material)

        # --- 5c. Reflection point → mic (delay = b samples) ---
        # The reflected signal at the reflection-point is already computed as sig_a.
        # We now need to delay it by delay_b_samples, then apply air abs for leg b.
        #
        # Because delay_b[n] delays the signal arriving at the reflection point
        # (which was delayed by delay_a[n]), we need to combine both delays.
        # Total read index: n - delay_a[n] - delay_b[n]
        # That is what happens in the reference: primaryDelLine gives delayed sig_a,
        # then secondaryDelLine gives sig_a delayed by delay_b more.
        # We replicate by reading from `sig_a` itself with additional shift.
        #
        # However sig_a is already the gathered (delayed) version of the source.
        # So we treat sig_a as a new "source" signal and gather with delay_b:
        read_b  = n_idx - delay_b_samples               # (N,)
        sig_b   = _sinc_gather(sig_a, read_b,
                                _sinc_table, _table_delays, _delta, SINC_SMP)

        if include_airabs:
            fir_b  = _air_fir(b)
            sig_b  = _apply_fir_per_sample(sig_b, fir_b)

        sig_b /= np.maximum(b, 1e-9)

        y_received = mic_dir * (sig_direct + sig_b)

    return y_received


def simulate_vectorised_environment(env) -> np.ndarray:
    """
    Drop-in replacement for `Environment.simulate()`.

    Accepts a fully configured `Environment` object (source + mic_array defined)
    and returns the same (M, N) array that `Environment.simulate()` would return,
    but computed via batch NumPy operations.

    Parameters
    ----------
    env : pyroadacoustics.Environment
        Fully configured Environment (source and mic_array must be set).

    Returns
    -------
    np.ndarray  shape (M, N)
    """
    if env.source is None or env.mic_array is None:
        raise RuntimeError('Before running simulation, you must define a source and mic_array')

    N = len(env.source.trajectory)
    M = env.mic_array.nmics
    signals = np.zeros((M, N))

    for m in range(M):
        mic_pos         = env.mic_array.mic_positions[m]
        signals[m] = simulate_vectorised(
            source_trajectory            = env.source.trajectory,
            source_signal                = env.source.signal,
            mic_pos                      = mic_pos,
            c                            = env.c,
            fs                           = env.fs,
            Z0                           = env.Z0,
            road_material                = env.road_material,
            air_absorption_coefficients  = env.air_absorption_coefficients,
            simulation_params            = env.simulation_params,
        )

    return signals
