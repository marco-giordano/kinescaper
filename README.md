# KineScaper

> **Physics-Based Dynamic Soundscape Generator for Moving Sound Sources**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

KineScaper is a Python library for generating synthetic acoustic datasets with **physically accurate moving sound source simulation**. Unlike static augmentation tools, KineScaper models the *kinetics* of sound: Doppler effect, distance attenuation, and ground reflections.

---

## Features

| Feature | Description |
|---------|-------------|
| **Doppler Simulation** | Realistic pitch shifting based on source trajectory and velocity |
| **Distance Attenuation** | 1/r amplitude decay with configurable minimum distance constraints |
| **Ground Reflections** | Image-source method for simulating asphalt/road surface reflections |
| **Virtual SPL Calibration** | Consistent background/foreground mixing using calibrated dB SPL levels |
| **Metal 3 Acceleration** | Optional GPU acceleration on Apple Silicon (MPS) or CUDA |
| **Frame-Based SNR** | Per-frame Signal-to-Noise Ratio computation for detailed annotations |
| **DCASE-Compatible Output** | Tab-separated annotations following DCASE challenge conventions |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-org>/kinescaper.git
cd kinescaper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Data

```
datasets/
├── background/          # Background WAV files (e.g., traffic noise)
│   ├── street_noise.wav
│   └── ...
└── foreground/          # Foreground WAV files + metadata
    ├── siren_01.wav
    ├── siren_02.wav
    └── metadata.csv     # Required: filename, event_label columns
```

### 3. Configure

Edit `config.yaml`:

```yaml
# Key parameters
n_samples: 100                      # Number of samples to generate
velocity: [30.0, 120.0]             # Random velocity range (km/h)
start_position: [-200.0, 5.0, 1.8]  # Source start [x, y, z] in meters
end_position: [200.0, 5.0, 1.8]     # Source end [x, y, z] in meters
acceleration: "metal"               # "cpu" or "metal" (Apple Silicon GPU)
```

### 4. Generate Dataset

```bash
python generate_dataset.py
```

Outputs are saved to `output/<folder>/`:
- `audio/` — Generated WAV files
- `csv/metadata.csv` — DCASE-format annotations
- `json/metadata.json` — Extended metadata

---

## Configuration Reference

| Section | Key Parameters |
|---------|----------------|
| **Paths** | `background_dir`, `foreground_dir`, `output_folder` |
| **Output** | `sampling_frequency`, `output_duration`, `frame_size` |
| **Calibration** | `cal_background_SPL`, `cal_foreground_SPL`, `required_*_SPL` |
| **Geometry** | `start_position`, `end_position`, `velocity`, `mic_positions` |
| **Acceleration** | `acceleration: "cpu"` or `"metal"` |

See [`docs/README.md`](docs/README.md) for the complete configuration reference.

---

## Output Format

### CSV (DCASE-Compatible)

```
filename	onset	offset	event_label	snr_min	snr_max	snr_avg
dataset_0000.wav	2.000	15.500	siren_wail	-12.5	28.3	8.7
dataset_0001.wav	2.000	18.200	car_horn	-8.2	22.1	6.4
```

### JSON (Extended Metadata)

```json
{
  "dataset_metadata": [
    {
      "filename": "dataset_0000.wav",
      "event_label": "siren_wail",
      "onset": 2.0,
      "offset": 15.5,
      "velocity_kmh": 72.3,
      "closest_distance": 5.0,
      "snr_min": -12.5,
      "snr_max": 28.3,
      "snr_avg": 8.7,
      "snr_std": 11.2
    }
  ]
}
```

---

## Architecture

```
kinescaper/
├── config.yaml           # All simulation parameters
├── generate_dataset.py   # Main entry point (batch generation)
├── demo_dataset.py       # Single-sample demo
├── src/
│   ├── core.py           # KineScaper engine
│   ├── metal_backend.py  # GPU-accelerated simulation
│   └── utils.py          # I/O and helpers
├── libs/
│   └── pyroadacoustics/  # Physics engine (vendored)
└── datasets/             # Your audio data
```

---

## Use Cases

- **Sound Event Detection (SED)** — Generate labeled datasets for training ML models
- **Doppler Effect Research** — Study frequency shifts from moving sources
- **Traffic Acoustics** — Simulate vehicle pass-by scenarios
- **Spatial Audio** — Multi-microphone array configurations

---

## Requirements

- Python 3.9+
- NumPy, SciPy, Pandas, PyYAML
- PyTorch (optional, for Metal/CUDA acceleration)

---

## Citation

If you use KineScaper in your research, please cite:

```bibtex
@software{kinescaper2026,
  title = {KineScaper: Physics-Based Dynamic Soundscape Generator},
  author = {Marco Giordano},
  year = {2026},
  url = {https://github.com/marco-giordano/kinescaper}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Documentation

For detailed configuration options, output format specifications, and architecture documentation, see:

📖 **[Full Documentation](docs/README.md)**
