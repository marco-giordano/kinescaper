# KineScaper Documentation

**KineScaper** is a Python-based synthetic soundscape generator for creating realistic acoustic datasets with moving sound sources. It simulates Doppler effects, distance attenuation, and ground reflections to produce training data for machine learning models in Sound Event Detection (SED) and related acoustic research.

---

## Table of Contents

1. [Project Scope](#project-scope)
2. [User Reference](#user-reference)
   - [Installation](#installation)
   - [Quick Start](#quick-start)
   - [Running the Demo](#running-the-demo)
   - [Generating Datasets](#generating-datasets)
3. [Configuration Reference](#configuration-reference)
   - [Data Paths](#data-paths)
   - [Output Parameters](#output-parameters)
   - [Acceleration](#acceleration)
   - [Virtual Calibration](#virtual-calibration)
   - [Mixing Targets](#mixing-targets)
   - [Batch Generation Settings](#batch-generation-settings)
   - [Scene Geometry](#scene-geometry)
   - [Microphone Array](#microphone-array)
4. [Output Format](#output-format)
5. [Architecture Overview](#architecture-overview)

---

## Project Scope

KineScaper addresses the need for synthetic acoustic datasets that contain physically accurate simulations of moving sound sources. The project is designed for:

- **Sound Event Detection (SED) research** – Generating labeled datasets for training and evaluating models
- **Doppler effect simulation** – Accurate frequency shifts based on source velocity relative to microphones
- **Physics-based mixing** – SPL (Sound Pressure Level) calibration for realistic background/foreground mixing
- **Multi-microphone support** – Configurable microphone arrays for spatial audio research
- **DCASE-compatible output** – Tab-separated annotation files following DCASE challenge conventions

### Key Features

| Feature | Description |
|---------|-------------|
| **Doppler Simulation** | Realistic pitch shifting based on source trajectory and velocity |
| **Distance Attenuation** | 1/r amplitude decay with minimum distance constraints |
| **Ground Reflections** | Image-source method for simulating asphalt reflections |
| **SPL Calibration** | Virtual calibration system for consistent background/foreground levels |
| **Metal Acceleration** | Optional GPU acceleration via Apple Metal/MPS (or CUDA) |
| **Frame-based SNR** | Per-frame Signal-to-Noise Ratio computation for annotations |

---

## User Reference

### Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd kinescaper
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `numpy`
   - `scipy`
   - `pandas`
   - `pyyaml`
   - `torch` (optional, for Metal/CUDA acceleration)

4. **Prepare datasets**:
   - Place background audio files (`.wav`) in `datasets/background/`
   - Place foreground audio files (`.wav`) in `datasets/foreground/`
   - Create `datasets/foreground/metadata.csv` with columns: `filename`, `event_label` (or `siren_mode`)

### Quick Start

1. Edit `config.yaml` to set your desired parameters
2. Run the dataset generator:
   ```bash
   python generate_dataset.py
   ```
3. Find outputs in the configured `output_folder` with subdirectories:
   - `audio/` – Generated WAV files
   - `csv/` – DCASE-format annotations
   - `json/` – Extended metadata

### Running the Demo

For a single-sample demonstration:

```bash
python demo_dataset.py
```

This generates one audio sample with annotations using the current configuration.

### Generating Datasets

The main dataset generation script (`generate_dataset.py`) produces multiple samples based on `n_samples` in the config:

```bash
python generate_dataset.py
```

Each sample:
1. Loads a random background audio chunk
2. Selects a random foreground event from metadata
3. Generates a random velocity within the configured range
4. Simulates the acoustic scene with physics-based mixing
5. Saves audio + annotations

---

## Configuration Reference

All parameters are defined in `config.yaml`. Below is a comprehensive reference:

### Data Paths

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dataset_root` | `string` | Root directory for all datasets | `"datasets"` |
| `background_dir` | `string` | Directory containing background WAV files | `"datasets/background"` |
| `foreground_dir` | `string` | Directory containing foreground WAV files | `"datasets/foreground"` |
| `metadata_path` | `string` | Path to foreground metadata CSV | `"datasets/foreground/metadata.csv"` |

---

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_frequency` | `integer` | `44100` | Output audio sample rate in Hz |
| `output_duration` | `float` | `30.0` | Duration of each generated audio file in seconds |
| `output_folder` | `string` | `"output/demo_dataset"` | Base output directory (creates `audio/`, `csv/`, `json/` subdirectories) |
| `frame_size` | `integer` | `2048` | Analysis frame size in samples for SNR computation |

---

### Acceleration

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `acceleration` | `string` | `"cpu"` | `"cpu"`, `"metal"` | Simulation backend. Use `"metal"` for GPU acceleration on Apple Silicon (requires PyTorch with MPS support) |

> **Note**: When `metal` is selected but unavailable, the system automatically falls back to CPU.

#### Metal Backend Requirements

- macOS with Apple Silicon (M1/M2/M3)
- PyTorch with MPS support: `pip install torch`
- The Metal backend provides ~2-5x speedup for large datasets

---

### Virtual Calibration

The virtual calibration system establishes a consistent relationship between digital audio levels (dBFS) and physical sound pressure levels (dB SPL).

#### Background Calibration

| Parameter | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `cal_background_norm` | `float` | `-20.0` | dBFS | Target RMS level after normalizing background audio |
| `cal_background_SPL` | `float` | `50.0` | dB SPL | Declared SPL that corresponds to the normalized level |

**Process**:
1. Background audio is loaded and its RMS is measured
2. Gain is applied to reach `cal_background_norm` (e.g., -20 dBFS)
3. This normalized level is *declared* to represent `cal_background_SPL` (e.g., 50 dB)

#### Foreground Calibration

| Parameter | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `cal_foreground_norm` | `float` | `-1.0` | dBFS | Target peak level after normalizing foreground audio |
| `cal_foreground_SPL` | `float` | `100.0` | dB SPL @ 1m | Declared SPL at 1 meter distance |

**Process**:
1. Foreground audio is loaded and its peak is measured
2. Gain is applied to reach `cal_foreground_norm` (e.g., -1 dBFS peak)
3. This normalized level is *declared* to represent `cal_foreground_SPL` at 1 meter

---

### Mixing Targets

These parameters control the final SPL levels in the mixed output:

| Parameter | Type | Default | Unit | Description |
|-----------|------|---------|------|-------------|
| `required_background_SPL` | `float` | `50.0` | dB SPL | Desired background level in the final mix |
| `required_foreground_SPL` | `float` | `90.0` | dB SPL @ 1m | Desired foreground level at 1 meter in the final mix |

**Mixing Logic**:
```
Background Gain = (required_background_SPL - cal_background_SPL) + alignment_offset
Foreground Gain = required_foreground_SPL - cal_foreground_SPL
```

Example: With defaults, foreground at 1m is 40 dB louder than background (90 - 50 = 40 dB SNR at closest approach).

---

### Batch Generation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | `integer` | `5` | Number of audio samples to generate in batch mode |

---

### Scene Geometry

#### Trajectory Definition

| Parameter | Type | Format | Description |
|-----------|------|--------|-------------|
| `start_position` | `list[float]` | `[x, y, z]` | Starting position of the moving source (meters) |
| `end_position` | `list[float]` | `[x, y, z]` | Ending position of the moving source (meters) |

**Coordinate System**:
- `x`: Lateral position (positive = right)
- `y`: Forward/backward position (positive = forward from microphone)
- `z`: Height above ground (positive = up)

**Example**: A vehicle passing from left to right at 5m forward distance and 1.8m height:
```yaml
start_position: [-200.0, 5.0, 1.8]
end_position: [200.0, 5.0, 1.8]
```

#### Velocity

| Parameter | Type | Format | Unit | Description |
|-----------|------|--------|------|-------------|
| `velocity` | `list[float]` | `[min, max]` | km/h | Range for random velocity selection |

**Example**: Random velocity between 30 and 80 km/h:
```yaml
velocity: [30.0, 80.0]
```

> **Note**: Velocity values in the config are in **km/h** and are automatically converted to **m/s** internally for physics calculations.

---

### Microphone Array

| Parameter | Type | Format | Description |
|-----------|------|--------|-------------|
| `mic_positions` | `list[list[float]]` | `[[x, y, z], ...]` | List of microphone positions in meters |

**Examples**:

Single microphone:
```yaml
mic_positions:
  - [0.0, 0.0, 1.8]
```

Stereo pair:
```yaml
mic_positions:
  - [-0.1, 0.0, 1.8]  # Left
  - [0.1, 0.0, 1.8]   # Right
```

Linear array:
```yaml
mic_positions:
  - [-0.3, 0.0, 1.5]
  - [-0.1, 0.0, 1.5]
  - [0.1, 0.0, 1.5]
  - [0.3, 0.0, 1.5]
```

> **Constraint**: The minimum distance between any point on the trajectory and any microphone must be at least **1 meter**. Events that violate this constraint are skipped with a warning.

---

## Output Format

### Directory Structure

```
output_folder/
├── audio/
│   ├── dataset_0000.wav
│   ├── dataset_0001.wav
│   └── ...
├── csv/
│   └── metadata.csv
└── json/
    └── metadata.json
```

### Audio Files

- Format: 32-bit float WAV
- Sample rate: As configured in `sampling_frequency`
- Channels: 1 (mono) or N (multi-channel for N microphones)

### CSV Annotations (DCASE Format)

Tab-separated file with columns:

| Column | Type | Description |
|--------|------|-------------|
| `filename` | string | Audio filename |
| `onset` | float | Event start time (seconds) |
| `offset` | float | Event end time (seconds) |
| `event_label` | string | Event class label |
| `snr_min` | float | Minimum per-frame SNR (dB) |
| `snr_max` | float | Maximum per-frame SNR (dB) |
| `snr_avg` | float | Average per-frame SNR (dB) |
| `snr_std` | float | Standard deviation of per-frame SNR (dB) |
| `velocity_kmh` | float | Source velocity (km/h) |
| `closest_distance` | float | Minimum trajectory-to-mic distance (m) |
| `frame_size` | int | Analysis frame size (samples) |

### JSON Metadata

Extended metadata with all annotation fields plus additional physics details:

```json
{
  "dataset_metadata": [
    {
      "event_label": "siren_wail",
      "filename": "dataset_0000.wav",
      "onset": 2.0,
      "offset": 12.5,
      "snr_min": -5.2,
      "snr_max": 35.1,
      "snr_avg": 18.3,
      "snr_std": 12.4,
      "velocity_kmh": 65.3,
      "closest_distance": 5.0,
      "frame_size": 2048
    }
  ]
}
```

---

## Architecture Overview

```
kinescaper/
├── config.yaml           # Configuration file
├── generate_dataset.py   # Main batch dataset generator
├── demo_dataset.py       # Single-sample demo script
├── requirements.txt      # Python dependencies
├── src/
│   ├── __init__.py
│   ├── core.py           # KineScaper engine (main simulation logic)
│   ├── metal_backend.py  # GPU-accelerated simulation (Metal/MPS/CUDA)
│   └── utils.py          # Utility functions (I/O, conversion, annotations)
├── datasets/
│   ├── background/       # Background audio files (.wav)
│   └── foreground/       # Foreground audio files (.wav) + metadata.csv
├── libs/
│   └── pyroadacoustics/  # External library for road acoustics simulation
├── output/               # Generated datasets
└── docs/                 # Documentation
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `core.py` | Main `KineScaper` class: event management, rendering, physics simulation |
| `metal_backend.py` | GPU-accelerated `MetalSimulator` for Doppler/attenuation computation |
| `utils.py` | File I/O, audio loading, format conversion, annotation saving |

---

## Example Configuration

```yaml
# Data Paths
dataset_root: "datasets"
background_dir: "datasets/background"
foreground_dir: "datasets/foreground"
metadata_path: "datasets/foreground/metadata.csv"

# Output Parameters
sampling_frequency: 44100
output_duration: 30.0
output_folder: "output/siren_dataset"
frame_size: 2048

# Acceleration
acceleration: "metal"  # Use GPU

# Virtual Calibration - Background
cal_background_norm: -20.0
cal_background_SPL: 50.0

# Virtual Calibration - Foreground
cal_foreground_norm: -1.0
cal_foreground_SPL: 100.0

# Mixing Targets
required_background_SPL: 50.0
required_foreground_SPL: 90.0

# Batch Generation
n_samples: 100

# Scene Geometry
start_position: [-200.0, 5.0, 1.8]
end_position: [200.0, 5.0, 1.8]
velocity: [30.0, 120.0]

# Microphone Array
mic_positions:
  - [0.0, 0.0, 1.8]
```

---

*Generated: January 2026*
