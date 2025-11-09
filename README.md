# ApneaEye_Demo

Light demo utilities for capturing and analysing thermal camera video to extract respiration signals. This repository includes simple scripts to access a thermal camera feed (via FFmpeg), record the thermal video, and run a YOLO-based localiser to extract nasal and thoracic signals.

Contents
--------
- `AccessThermalCam.py` — example script that reads camera frames via ffmpeg and shows them using OpenCV.
- `ThermalRecord.py` — records thermal camera frames to `thermal-rec/` as .avi files.
- `DemoApp.py` — demo application that performs YOLO-based localisation (uses `models/Yolov8_Localiser.pt`) to extract respiration signals, filter them, and display a combined visualization.
- `models/` — folder containing model weights (e.g. `Yolov8_Localiser.pt`, `Yolov11_Localiser.pt`, `NoseLoc_200.pt`).

Requirements
------------
- System: `ffmpeg` is required and invoked by the scripts. On macOS install via Homebrew:

```bash
brew install ffmpeg
```

- Python packages: see `requirements.txt`.

Installation (Miniconda, recommended / tested)
--------------------------------------------
1. Create and activate a conda environment (example with Python 3.11):

```bash
conda create -n apneaeye python=3.11 -y
conda activate apneaeye
```

2. Install pip and Python dependencies:

```bash
conda install pip -y
pip install --upgrade pip
pip install -r requirements.txt
# Optionally install an appropriate PyTorch wheel (see REQUIREMENTS_NOTES.md)
```

Installation (venv, untested)
------------------------------
If you prefer a standard venv, use these steps (the developer tested with Miniconda; venv is provided but untested):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# then install the appropriate torch wheel if necessary
```

Usage
-----
Basic examples assume you have a thermal camera connected and accessible via the platform camera index used in the scripts.

- Record thermal video to `thermal-rec/`:

```bash
python ThermalRecord.py
```

- Run the demo application (YOLO-based localisation and respiration plotting):

```bash
python DemoApp.py
```

- Quick camera access example (show frames):

```bash
python AccessThermalCam.py
```

Notes
-----
- Models: the `models/` directory contains pretrained weights used by `DemoApp.py`. If you replace models, update the path in the script or use absolute paths.
- FFmpeg: scripts start ffmpeg as a subprocess. If your OS or camera requires different ffmpeg flags, update the ffmpeg command arrays in the Python files.
- Tested environment: the developer tested installs and basic runs using Miniconda. venv instructions are included but not explicitly tested.

# ApneaEye_Demo
Demo for ApneaEye Respiration Extraction 
