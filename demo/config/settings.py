import os
import torch
import cv2



# Path to YOLOv8 model checkpoint
YOLO_MODEL_PATH = os.path.abspath("checkpoints/yolov8n.pt")

# Path to SAM2 configuration file
SAM_CONFIG_PATH = "./configs/samurai/sam2.1_hiera_b+.yaml"

# Path to SAM2 model checkpoint
SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_base_plus.pt")

# Device selection for model inference
# Options: "cuda:0" for GPU, "cpu" for CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Web server settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# Detection and tracking thresholds
STATIONARY_THRESHOLD = 5  # pixels
STATIONARY_TIME_THRESHOLD = 3.0  # seconds
MASK_THRESHOLD = 0.5  # confidence threshold for mask generation

# Camera settings
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_DSHOW

# Camera streaming settingsAdd commentMore actions
STREAM_URL = "your_camera_url"

# MQTT Broker settings
BROKER_ADDR = "your_broker_address"
BROKER_PORT = 1883
# PTZ MQTT settings
MQTT_PTZ_TOPIC = "ptz/control"
MQTT_PTZ_CLIENT_ID = "human_app_ptz"
MQTT_PTZ_KEEPALIVE = 15

# PTZ control settings
PTZ_INIT_PAN = 100
PTZ_INIT_TILT = 140
PTZ_PAN_DIR = -1
PTZ_TILT_DIR = 1
PTZ_DEADZONE_PX = 5
PTZ_MIN_STEP_DEG = 1.0
PTZ_SMOOTH_ALPHA = 0.40

# DAM API settings
DEMO_API = "http://localhost:5100/trigger_recording"

CSI_TOPIC = ["L0382/ESP/8"]
# CSI MQTT settings
CSI_TOPICS = [
    "L0382/ESP/1",
    "L0382/ESP/2",
    "L0382/ESP/3",
    "L0382/ESP/4",
    "L0382/ESP/5",
    "L0382/ESP/6",
    "L0382/ESP/7",
    "L0382/ESP/8",
]
CSI_INDICES_TO_REMOVE = list(range(21, 32))
CSI_SUBCARRIERS = 52
CSI_WINDOW_SIZE = 320
CSI_STRIDE = 40
CSI_SMALL_WIN_SIZE = 64
CSI_FPS_LIMIT = 10

# Zone-based PTZ control settings for CSI mode
ZONE_PTZ_ANGLES = {
    1: {"pan": 80, "tilt": 0},
    2: {"pan": 140, "tilt": 0},
    3: {"pan": 90, "tilt": 20},
    4: {"pan": 140, "tilt": 40},
    5: {"pan": 90, "tilt": 80},
    6: {"pan": 130, "tilt": 100},
}

PERSON_FRAME_MARGIN = 10

# CSI mode settings
CSI_ZONE_TOPIC = "Loc/zone/response"
CSI_PERSON_LOST_FRAMES = 1