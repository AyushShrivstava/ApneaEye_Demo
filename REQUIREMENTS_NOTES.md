Requirements and install notes
=================================

This project uses the following Python packages (see `requirements.txt`):

- ultralytics
- torch
- opencv-python
- numpy
- matplotlib
- scipy
- pandas

Important notes and assumptions
---------------------------------
- ffmpeg is required as a system dependency and is invoked by the code via subprocess. Install via Homebrew on macOS:

  brew install ffmpeg

- PyTorch (`torch`) must match your platform and CUDA availability. The `requirements.txt` uses a generic `torch>=2.0.0` line but you should install the recommended wheel for macOS (often CPU-only) or for your GPU. Example (CPU-only):

  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade

- If you encounter issues with `opencv-python` (macOS camera backend or conflicts), consider installing `opencv-contrib-python-headless` or building OpenCV from source.

Install commands
------------------
Option A — Miniconda (recommended / tested):

1. Create a new conda environment (example using Python 3.11):

```bash
conda create -n apneaeye python=3.11 -y
conda activate apneaeye
```

2. Install system ffmpeg via Homebrew (macOS) if not already installed:

```bash
brew install ffmpeg
```

3. Install pip and the requirements. Installing PyTorch via the official wheel is recommended — the example below is for CPU-only macOS:

```bash
conda install pip -y
pip install --upgrade pip
pip install -r requirements.txt
# If you need a specific PyTorch wheel (recommended), install it after reading the PyTorch install selector:
# Example (CPU-only wheel):
# python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade
```

Option B — venv (unguided / untested here):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# then install the appropriate torch wheel if necessary (see note above)
```

Optional packages
------------------
- If you want the optional Pillow package (image IO) enable it in `requirements.txt` or install separately:

  pip install pillow

Notes about testing
--------------------
- The developer tested dependency installation and basic runs using Miniconda (the `conda` steps above). The `venv` instructions are provided and should work in general, but they have not been explicitly tested by the developer in this project.

If you want me to pin exact versions or add environment-specific instructions (CPU vs CUDA torch), tell me which platform/GPU you plan to use and I'll update `requirements.txt` accordingly.
