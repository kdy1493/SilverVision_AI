# Real-Time Intrusion Detection Pipeline

This repository implements a four-stage end-to-end system for detecting and responding to unauthorized entry:

1. **CSI-Based Presence Detection**  
   Continuously monitor WiFi CSI (Channel State Information) to detect door openings or human entry.

2. **Intruder Localization**  
   Analyze phase and amplitude shifts to approximate the intruder's position in the room.

3. **Real-Time Human Detection & Tracking**  
   - **YOLOv8** for ultra-fast person bounding-box detection  
   - **SAMURAI** for pixel-precise segmentation, centroid extraction
   - **PTZ Controller** MQTT pan/tilt commands to the Pi
4.  **Video Recording & Behavioral Analysis**  
   - **Automated Recording**: Trigger 5-second video clips when suspicious activity is detected  
   - **DAM (Dynamic Action Model)**: Generate natural language descriptions of human actions and behaviors from recorded video segments  
5. **Logging & Anomaly Alerts**  
   Record intrusion events, track "stationary interaction" behaviors, and push logs/alerts to the dashboard or mobile.

By combining wireless sensing, computer vision, and intelligent logging, this pipeline delivers robust, automated intrusion monitoring in real time.


## Setup Instructions
I recommend using uv venv to create isolated environments, simplifying dependency management and ensuring reproducible setups.

### 1. Create & activate virtualenv
```bash
# Install the 'uv' CLI and create a new venv
pip install uv
uv venv

# On macOS / Linux
source .venv/bin/activate
# On Windows (PowerShell)
source .venv/Scripts/activate
```

### 2. Clone the repository
```bash
git clone https://github.com/NVA-Lab/intrusion-detector.git
```

### 3. Install packages
```bash
cd intrusion-detector

# Install the core package (SAM2 + demo app) in editable mode
uv pip install -e .

# (Skip if CPU only)
# If you want to use GPU acceleration (CUDA), you must install PyTorch with the correct CUDA version manually.
# For example, to install PyTorch with CUDA 12.1, run the following command before installing the rest:
uv pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### 4. Download SAM2 Checkpoints
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

---

## Usage
### Web Application
Run the web application for real-time human detection and tracking:

```bash
python -m demo.app
```

After running, access the web interface at `http://localhost:5000`.

### DAM Model Demonstration
To demonstrate the DAM (Describe Anything Model) functionality:

1. Open two separate terminal windows
2. In the first terminal, run the demo script:
```bash
python scripts/demo.py
```
3. In the second terminal, run the API controller:
```bash
python scripts/api_controller.py
```
4. Press 's' in the API controller terminal to trigger recording and description generation

Note: Currently, the human detection and description systems are not fully integrated. The API controller sends dummy data for demonstration purposes. This integration will be updated in future releases.

### Running the Integrated System
To run the complete intrusion detection system with human detection and DAM analysis:

1. Start the DAM model server first:
```bash
python scripts/demo.py
```

2. In a separate terminal, start the human detection system:
```bash
python demo/app.py
```

The system will now automatically:
- Detect humans in the camera feed
- Track their movements
- Trigger DAM analysis when stationary behavior is detected
- Display results in the web interface

### MQTT Configuration
To use the CSI-based presence detection feature, you need to configure your MQTT broker settings in `demo/config/settings.py`:

```python
BROKER_ADDR = "your_broker_address"
BROKER_PORT = your_broker_port
```

The repository includes default topic configurations for ESP32 devices, but you can modify these settings according to your specific MQTT setup.

### Customization
You can customize various settings in `demo/config/settings.py`:

- **Camera Settings**: Change `CAMERA_INDEX` to use different cameras
- **Model Paths**: Update `YOLO_MODEL_PATH` and `SAM_CHECKPOINT_PATH` to use different model checkpoints
- **Detection Thresholds**: Adjust `STATIONARY_THRESHOLD` and `MASK_THRESHOLD` to fine-tune detection sensitivity

### PTZ Camera Control
This project uses a Raspberry Pi 4 + NoIR v2 camera mounted on a pan‑tilt HAT.  The Pi acts as a micro‑streaming server (MJPEG/HTTP) while also receiving PTZ commands over MQTT

### Acknowledgment
This project leverages:  
- **YOLOv8** by Ultralytics for ultra-fast real-time person detection.  
- **SAM2** by Meta FAIR for pixel-precise segmentation and tracking.  
- **SAMURAI** by the University of Washington's Information Processing Lab for motion-aware memory modeling.  
- **DAM (Describe Anything Model)** by NVIDIA for concise natural language descriptions of subject actions or state changes in videos.

## Citation
```
@article{glenn2024yolov8,
  title={YOLOv8: Next-Generation Real-Time Object Detection},
  author={Glenn Jocher and Ultralytics},
  year={2024},
  url={https://github.com/ultralytics/ultralytics}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi et al.},
  year={2024},
  url={https://arxiv.org/abs/2408.00714}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting SAM for Zero-Shot Visual Tracking with Motion-Aware Memory},
  author={Yang et al.},
  year={2024},
  url={https://arxiv.org/abs/2411.11922}
}

```




---
