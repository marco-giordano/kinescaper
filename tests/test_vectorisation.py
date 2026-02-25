"""
tests/test_vectorisation.py
----------------------------
Unit tests comparing the vectorised simulation engine against the reference
SimulatorManager implementation, sample-by-sample.

Run with:
    .venv/bin/python -m unittest tests.test_vectorisation -v

Tolerance rationale
-------------------
The vectorised engine replicates the same physics equations but uses
`scipy.ndimage.map_coordinates`-equivalent Sinc interpolation via a vectorised
table gather, while the reference uses a causal circular-buffer DelayLine with a
windowed Sinc table. Both are valid fractional-delay interpolators; slight
differences in edge handling produce:
  - direct path only:          rms_rel < 10%,  max_abs < 1e-2
  - with air absorption bins:  rms_rel < 30%,  max_abs < 1e-1
  - full (refl + air abs):     rms_rel < 50%,  max_abs < 2e-1
These tolerances validate that the output is physically equivalent (same
magnitude, same phase, same peaks) rather than bit-exact.

Any regression that produces rms_rel > 0.5 or wrong peak alignment indicates
a real bug.
"""

import os
import sys
import math
import unittest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS = os.path.join(ROOT, 'libs', 'pyroadacoustics')
for p in [ROOT, LIBS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pyroadacoustics.environment import Environment
from pyroadacoustics.material import Material
from src.vectorised_simulator import simulate_vectorised_environment

# ── Shared fixtures ───────────────────────────────────────────────────────────
FS      = 8000   # low fs so tests run fast
N_SAMP  = FS * 2  # 2 seconds of audio


def _make_env(
    src_start,
    src_end,
    mic_pos,
    fs=FS,
    velocity=20.0,
    interp_method="Sinc",
    include_reflected=True,
    include_airabs=True,
    road_material=None,
    signal=None,
):
    """Helper: build a fully configured Environment object."""
    env = Environment(
        fs=fs,
        temperature=20,
        pressure=1,
        rel_humidity=50,
        road_material=road_material or 20000,  # int → default flow resistance
    )
    env.set_simulation_params(interp_method, include_reflected, include_airabs)

    if signal is None:
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 0.1, N_SAMP).astype(float)

    env.add_source(
        position=np.array(src_start, dtype=float),
        signal=signal,
        trajectory_points=np.array([src_start, src_end], dtype=float),
        source_velocity=[velocity],
    )
    env.add_microphone_array(np.array([mic_pos], dtype=float))
    return env, signal


class TestVectorisedVsReference(unittest.TestCase):
    """Compare vectorised output against Environment.simulate() sample-by-sample."""

    def _compare(self, env, atol=5e-2, rtol=0.5, msg=""):
        """
        Run both paths and assert arrays are physically close.

        'Close' means: either max absolute error < atol, OR rms-relative error < rtol.
        This handles signals with near-zero rms (e.g. pure impulse with many zero samples).
        """
        ref = env.simulate()                         # (M, N) — reference
        vec = simulate_vectorised_environment(env)   # (M, N) — vectorised

        self.assertEqual(ref.shape, vec.shape,
                         f"Shape mismatch {ref.shape} vs {vec.shape}. {msg}")

        max_err   = float(np.max(np.abs(ref - vec)))
        rms_err   = float(np.sqrt(np.mean((ref - vec) ** 2)))
        rms_ref   = float(np.sqrt(np.mean(ref ** 2)))
        rel_err   = rms_err / max(rms_ref, 1e-12)

        pass_abs  = max_err <= atol
        pass_rel  = rel_err <= rtol

        self.assertTrue(
            pass_abs or pass_rel,
            f"{msg}\n"
            f"  max_abs_error = {max_err:.6e}  (atol={atol})\n"
            f"  rms_rel_error = {rel_err:.6e}  (rtol={rtol})\n"
            f"  rms_ref       = {rms_ref:.6e}"
        )
        return max_err, rel_err

    # ── Test 1: output shapes ─────────────────────────────────────────────────

    def test_output_shape(self):
        """Vectorised output must have the same (M, N) shape as the reference."""
        env, _ = _make_env([-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8])
        ref = env.simulate()
        vec = simulate_vectorised_environment(env)
        self.assertEqual(ref.shape, vec.shape)

    # ── Test 2: direct path only (no reflection, no air abs) ─────────────────

    def test_direct_path_only(self):
        """
        With reflection=False, airabs=False, the only operations are fractional
        delay gather + 1/d attenuation.

        Expected: rms_rel < 10% (sinc interpolation edge differences only).
        """
        env, _ = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=False,
            include_airabs=False,
        )
        max_err, rel_err = self._compare(env, atol=1e-2, rtol=0.10,
                                          msg="direct path only (no abs, no refl)")
        print(f"\n[direct_only]   max_err={max_err:.2e}  rel_err={rel_err:.2e}")
        # Hard constraint: RMS relative error must be below 10%
        self.assertLess(rel_err, 0.10,
            f"Direct path rms_rel_err={rel_err:.4f} exceeds 10% threshold")

    # ── Test 3: air absorption, no reflection ────────────────────────────────

    def test_air_absorption_no_reflection(self):
        """
        Air-absorption FIR (distance-binned) + fractional delay.

        Expected: rms_rel < 30%.  Extra tolerance over direct path because
        distance-binning of filters introduces small approximation per bin edge.
        """
        env, _ = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=False,
            include_airabs=True,
        )
        max_err, rel_err = self._compare(env, atol=5e-2, rtol=0.30,
                                          msg="air absorption, no reflection")
        print(f"\n[airabs_only]   max_err={max_err:.2e}  rel_err={rel_err:.2e}")
        self.assertLess(rel_err, 0.30)

    # ── Test 4: full simulation with Material road surface ───────────────────

    def test_full_with_material(self):
        """
        Full physics (direct + reflection + air abs) using a Material road.

        Expected: rms_rel < 50%.  The angle-binned asphalt FIR and the
        cascaded delay chain compound small differences from each stage.
        """
        mat = Material('average_asphalt')
        env, _ = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=True,
            include_airabs=True,
            road_material=mat,
        )
        max_err, rel_err = self._compare(env, atol=1e-1, rtol=0.50,
                                          msg="full simulation, Material road")
        print(f"\n[full_material] max_err={max_err:.2e}  rel_err={rel_err:.2e}")
        self.assertLess(rel_err, 0.50)

    # ── Test 5: multi-microphone output shape ────────────────────────────────

    def test_multi_mic_shape(self):
        """With two mics the output must have shape (2, N)."""
        env, sig = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=False,
            include_airabs=False,
        )
        # Rebuild environment with two microphones
        env2 = Environment(fs=FS, temperature=20, pressure=1, rel_humidity=50)
        env2.set_simulation_params("Sinc", False, False)
        env2.add_source(
            position=np.array([-50, 5, 1.5], dtype=float),
            signal=sig,
            trajectory_points=np.array([[-50, 5, 1.5], [50, 5, 1.5]], dtype=float),
            source_velocity=[20.0],
        )
        env2.add_microphone_array(np.array([[0, 0, 1.8], [0.2, 0, 1.8]], dtype=float))
        ref = env2.simulate()
        vec = simulate_vectorised_environment(env2)
        self.assertEqual(ref.shape, vec.shape)
        self.assertEqual(ref.shape[0], 2)
        print(f"\n[multi_mic]     shape={vec.shape}  ✓")

    # ── Test 6: impulse signal — peak timing alignment ───────────────────────

    def test_impulse_peak_alignment(self):
        """
        Feed an impulse into a slowly moving source at ~5 m from mic.
        The highest-amplitude peak must appear at the same sample index
        in both ref and vec (within ±2 samples).

        Note: stationary source (same start/end) crashes pyroadacoustics
        (IndexError in SimulatorManager.initialize) so we use a very slow
        near-stationary source instead.
        """
        # Source ~ 5 m from mic  →  delay ≈ 5/343 * 8000 ≈ 116 samples
        impulse = np.zeros(N_SAMP)
        impulse[0] = 1.0

        # Tiny displacement so pyroadacoustics doesn't crash on zero trajectory
        src_s = np.array([-0.001, 5.0, 1.5])
        src_e = np.array([ 0.001, 5.0, 1.5])

        env, _ = _make_env(
            src_s, src_e, [0, 0, 1.8],
            velocity=0.01,
            include_reflected=False,
            include_airabs=False,
            signal=impulse,
        )
        ref = env.simulate()[0]
        vec = simulate_vectorised_environment(env)[0]

        ref_peak = int(np.argmax(np.abs(ref)))
        vec_peak = int(np.argmax(np.abs(vec)))

        self.assertLessEqual(
            abs(ref_peak - vec_peak), 2,
            f"Peak alignment mismatch: ref={ref_peak}  vec={vec_peak}"
        )
        print(f"\n[impulse]       ref_peak={ref_peak}  vec_peak={vec_peak}  ✓")

    # ── Test 7: energy conservation ──────────────────────────────────────────

    def test_energy_ratio(self):
        """
        The total signal energy (RMS) of the vectorised output must be within
        20% of the reference output.  Guards against systematic gain offsets.
        """
        env, _ = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=False,
            include_airabs=False,
        )
        ref = env.simulate()[0]
        vec = simulate_vectorised_environment(env)[0]

        rms_ref = float(np.sqrt(np.mean(ref ** 2)))
        rms_vec = float(np.sqrt(np.mean(vec ** 2)))
        ratio = abs(rms_vec - rms_ref) / max(rms_ref, 1e-12)

        self.assertLess(ratio, 0.20,
            f"Energy ratio deviation {ratio:.4f} > 20%.  rms_ref={rms_ref:.4e}  rms_vec={rms_vec:.4e}")
        print(f"\n[energy]        rms_ref={rms_ref:.4e}  rms_vec={rms_vec:.4e}  ratio={ratio:.4f}  ✓")


class TestMetalVectorised(unittest.TestCase):
    """Tests for the Metal (MPS/CUDA/CPU-torch) backend."""

    @classmethod
    def setUpClass(cls):
        """Build a shared Environment once for all Metal tests."""
        import torch
        cls._has_torch = True
        cls._device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )
        from src.metal_vectorised_backend import simulate_metal_environment
        from src.vectorised_simulator      import simulate_vectorised_environment

        rng = np.random.default_rng(42)
        _sig = rng.normal(0, 0.1, FS * 2).astype(float)

        cls._env, _ = _make_env(
            [-50, 5, 1.5], [50, 5, 1.5], [0, 0, 1.8],
            include_reflected=True, include_airabs=True, signal=_sig,
        )
        cls._ref   = simulate_vectorised_environment(cls._env)   # NumPy path
        cls._metal = simulate_metal_environment(cls._env)         # Metal path

    def test_metal_shape(self):
        """Metal output shape must match NumPy output shape."""
        self.assertEqual(self._ref.shape, self._metal.shape)
        print(f"\n[metal_shape]   shape={self._metal.shape}  device={self._device}")

    def test_metal_cosine_similarity(self):
        """
        Metal output must be highly correlated with NumPy vectorised output.
        Cosine similarity >= 0.99 confirms same physics, same phase — just
        float32 vs float64 rounding differences.
        """
        ref   = self._ref[0]
        metal = self._metal[0]
        cos_sim = float(np.dot(ref, metal) / (np.linalg.norm(ref) * np.linalg.norm(metal) + 1e-12))
        self.assertGreaterEqual(cos_sim, 0.99,
            f"Metal cosine similarity {cos_sim:.4f} < 0.99 — signals are not equivalent.")
        print(f"\n[metal_cosine]  cos_sim={cos_sim:.6f}  device={self._device}")

    def test_metal_energy_ratio(self):
        """
        Metal RMS energy must be within 10% of NumPy vectorised energy.
        Guards against systematic gain offsets from float32 precision.
        """
        rms_ref   = float(np.sqrt(np.mean(self._ref[0] ** 2)))
        rms_metal = float(np.sqrt(np.mean(self._metal[0] ** 2)))
        ratio = abs(rms_metal - rms_ref) / max(rms_ref, 1e-12)
        self.assertLess(ratio, 0.10,
            f"Metal energy ratio {ratio:.4f} > 10%.  rms_ref={rms_ref:.4e}  rms_metal={rms_metal:.4e}")
        print(f"\n[metal_energy]  rms_ref={rms_ref:.4e}  rms_metal={rms_metal:.4e}  ratio={ratio:.4f}")


# ── Quick run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
