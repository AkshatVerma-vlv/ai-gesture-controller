# Zero-Latency Biomechanical Gesture Controller

## Objective
A zero-latency, AI-powered biomechanical gesture controller that enables fully hands-free gameplay for high-speed games like Subway Surfers and Highway Bike Racer. It uses edge-computed computer vision and custom control theory physics to translate natural head and torso movements into instantaneous hardware-level keystrokes.

## Core Engineering Innovations
* **Velocity-Gated Drift Compensation:** Uses an Exponential Moving Average (EMA) to act as a low-pass filter, seamlessly auto-calibrating baseline posture drift (slouching) without disrupting fast gameplay inputs.
* **Biomechanical Asymmetry Guard:** Decouples vertical thresholds to require minimal effort for upward movements (jumping) while strictly filtering gravity-assisted downward movements (slouching).
* **Recoil-and-Pulse State Machine:** Eliminates accidental double-inputs by enforcing a mandatory 62.5% physical recoil toward the center before rearming sequential moves.
* **Scale-Invariant Kinematics:** Normalizes all raw pixel distances against dynamic shoulder and ear spans, ensuring the controller works perfectly regardless of the user's distance from the webcam.

## Performance
Runs entirely on a local GPU (tested on RTX 4060) utilizing PyTorch and CUDA. It sustains a strict 15ms inference latency and 15 FPS execution loop for real-time, zero-lag actuation.

## Setup Instructions
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Run the controller: `python subwaysurfers.py` or `python bikeracer.py`
