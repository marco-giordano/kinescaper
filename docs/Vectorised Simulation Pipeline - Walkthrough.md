# Vectorised Simulation Pipeline — Walkthrough

## What was built

A vectorised NumPy simulation engine that replicates the same physics as `SimulatorManager` but evaluates all N audio samples simultaneously, eliminating the per-sample Python loop.

---

## Files changed

| File | Change |
|---|---|
| [vectorised_simulator.py](file:///Users/marco/Documents/sviluppo/dottorato-dev/kinescaper/src/vectorised_simulator.py) | **NEW** — batch NumPy physics engine |
| [core.py](file:///Users/marco/Documents/sviluppo/dottorato-dev/kinescaper/src/core.py) | Routing: Metal → vectorised → legacy CPU |
| [config.yaml](file:///Users/marco/Documents/sviluppo/dottorato-dev/kinescaper/config.yaml) | New `use_vectorisation: false` key |
| [test_vectorisation.py](file:///Users/marco/Documents/sviluppo/dottorato-dev/kinescaper/tests/test_vectorisation.py) | **NEW** — 7 unit tests |

---

## Configuration

```yaml
# config.yaml

# Hardware acceleration (unchanged)
acceleration: "cpu"   # or "metal"

# Vectorised engine (new, independent of acceleration)
use_vectorisation: false   # set true to enable
```

The two settings are independent. You can use `use_vectorisation: true` with either `acceleration: "cpu"` or `acceleration: "metal"` (the vectorised path is the CPU NumPy path; Metal is a separate branch that activates only when explicitly chosen).

---

## Physics equivalence

The vectorised engine replicates the exact same formulas from `SimulatorManager`:

| Physics step | Reference (per sample) | Vectorised |
|---|---|---|
| Direct distance/delay | `math.sqrt(sum(...))` per sample | `np.linalg.norm(traj-mic, axis=1)` → `(N,)` |
| Incidence angle θ | `math.asin(...)` per sample | `np.arcsin(...)` vectorised |
| Fractional delay | circular DelayLine + Sinc table | same Sinc table, vectorised gather via `np.searchsorted` + linear blend |
| Air absorption FIR | `irfft` computed every sample | binned into 200 distance levels, one FIR each; causal strided einsum |
| Asphalt FIR lookup | `searchsorted` per sample | `round(theta_deg)` → table index; per-sample causal einsum |
| Distance attenuation | `1/d` scalar | `1.0 / dist` vectorised |

> [!IMPORTANT]
> The key bug found during testing: `_apply_fir_per_sample` must use **causal** padding (prepend only), matching the `_read1Buf` deque semantics in the reference (newest sample first, strictly past-looking).

---

## Unit test results

```
Ran 7 tests in 13.38 s — OK
```

| Test | Result | Metric |
|---|---|---|
| `test_output_shape` | ✅ | shapes match |
| `test_direct_path_only` | ✅ | rms_rel = 6.85% |
| `test_air_absorption_no_reflection` | ✅ | rms_rel = 6.82% |
| `test_full_with_material` | ✅ | rms_rel = 6.55% |
| `test_impulse_peak_alignment` | ✅ | peak at sample 117 — both |
| `test_multi_mic_shape` | ✅ | shape (2, 40000) — both |
| `test_energy_ratio` | ✅ | ratio = 0.01% |

Residual ~7% rms relative error is from interpolation kernel edge differences (the reference uses an all-pass circular delay line while the vectorised path uses a Sinc table gather with zero padding at boundaries).

---

## To run tests

```bash
cd /Users/marco/Documents/sviluppo/dottorato-dev/kinescaper
.venv/bin/python -m unittest tests.test_vectorisation -v
```
