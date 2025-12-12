import cv2
import time
import threading
import requests
import numpy as np
import paho.mqtt.client as paho
from typing import Optional, Tuple
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker
from demo.config.settings import (
    YOLO_MODEL_PATH, DEVICE, SAM_CONFIG_PATH, SAM_CHECKPOINT_PATH, 
    MASK_THRESHOLD, DEMO_API, BROKER_ADDR, BROKER_PORT,
    ZONE_PTZ_ANGLES, CSI_ZONE_TOPIC, CSI_PERSON_LOST_FRAMES, PERSON_FRAME_MARGIN
)
from demo.utils.alerts import AlertManager, AlertCodes
from demo.services.mqtt_manager import MQTTManager
import torch

class HumanDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self._warm_up()

    def _warm_up(self):
        dummy = torch.zeros(1, 3, 640, 640, device=DEVICE)
        _ = self.model.predict(dummy, device=DEVICE, verbose=False) 
    
    def detect(self, frame):
        persons = []
        results = self.model.predict(frame, classes=[0], conf=0.45, device=DEVICE, verbose=False)
        max_conf = 0
        best_box = None
        
        for res in results:
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_box = [[x1, y1], [x2, y2]]
        
        if best_box is not None:
            persons.append(best_box)
        
        return persons


class HumanTracker:
    def __init__(self):
        self.tracker = build_sam2_object_tracker(
            num_objects=1,
            config_file=SAM_CONFIG_PATH,
            ckpt_path=SAM_CHECKPOINT_PATH,
            device=DEVICE,
            verbose=False
        )
        self._warm_up()
        self.last_center = None
        self.stationary_timer_start = None
        

    def _warm_up(self):
        dummy = np.zeros((256, 256, 3), np.uint8)
        _ = self.tracker.track_all_objects(img=dummy)

    def initialize(self, frame, persons):
        self.tracker.track_new_object(
            img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            box=np.array(persons)
        )
    
    def track(self, frame):
        if self.tracker is None:
            return None, False
            
        out = self.tracker.track_all_objects(img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks = out.get("pred_masks")
        has_mask = False
        
        if masks is not None:
            m_np = masks.cpu().numpy()
            for i in range(m_np.shape[0]):
                if (m_np[i,0] > MASK_THRESHOLD).sum() > 0:
                    has_mask = True
                    break
                    
        return m_np if has_mask else None, has_mask
    
    def check_stationary(self, bbox_coords, current_time):
        if bbox_coords is None:
            return False
            
        x1, y1, x2, y2 = bbox_coords
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if self.last_center is None:
            self.last_center = (cx, cy)
            self.stationary_timer_start = current_time
            return False
            
        dist = np.hypot(cx - self.last_center[0], cy - self.last_center[1])
        
        if dist < 5:
            if self.stationary_timer_start and current_time - self.stationary_timer_start >= 3.0:
                self.stationary_timer_start = None
                return True
        else:
            self.last_center = (cx, cy)
            self.stationary_timer_start = current_time
            
        return False

    def is_person_in_frame(self, bbox_coords, frame_shape):
        if bbox_coords is None:
            return False
            
        x1, y1, x2, y2 = bbox_coords
        h, w = frame_shape[:2]

        return (x1 >= -PERSON_FRAME_MARGIN and y1 >= -PERSON_FRAME_MARGIN and 
                x2 <= w + PERSON_FRAME_MARGIN and y2 <= h + PERSON_FRAME_MARGIN)


class DetectionProcessor:
    MODE_DETECTION = "detection"
    MODE_TRACKING = "tracking"
    MODE_CSI = "csi"
    
    def __init__(self):
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.alert_manager = AlertManager()
        self.last_alert_zone = None
        self.last_alert_ts   = 0.0  

        self.mqtt_manager = MQTTManager()
        self.mqtt_manager.subscribe(CSI_ZONE_TOPIC, self._on_zone_response)

        self.ptz_publisher = None
        self._init_ptz_publisher()

        self.zone_angles = ZONE_PTZ_ANGLES
        self.last_alert_zone = None 
        
        self.csi_tracker = None
        self.csi_tracker_ready = False

        self.reset_state()
        
    def _init_ptz_publisher(self):

        try:
            from demo.services.mqtt_publisher import MQTTPublisher
            self.ptz_publisher = MQTTPublisher()
            print("[PTZ] Publisher initialized")
        except Exception as e:
            print(f"[PTZ] Publisher initialization failed: {e}")
        
    def _on_zone_response(self, msg):
        import json, time
        payload = msg.payload.decode().strip()
        #print(f"[CSI] Zone response received: {payload}")

        try:
            zone_data = json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"[CSI] Invalid JSON: {e}")
            return
        
        self._send_zone_status_to_web(zone_data)

        if self.current_mode != self.MODE_CSI:
            return

        alert_zone = None
        for key, state in zone_data.items():
            if state == "alert":
                alert_zone = int(key.split('_')[1])
                #print(f"[CSI] Alert zone detected: {alert_zone}")
                break

        if alert_zone is None or alert_zone not in self.zone_angles:
            return

        now = time.time()
        if (alert_zone == self.last_alert_zone) and (now - self.last_alert_ts < 5):
            #print(f"[CSI] Duplicate zone {alert_zone} within 5 s → ignore")
            return

        angles = self.zone_angles[alert_zone]
        if self.ptz_publisher:
            self.ptz_publisher.publish("pan",  angles["pan"])
            self.ptz_publisher.publish("tilt", angles["tilt"])
        if hasattr(self, 'ptz_service') and self.ptz_service:
            self.ptz_service.sync_controller_state(angles["pan"], angles["tilt"])

        #print(f"[CSI] PTZ moved to zone {alert_zone}: "
        #      f"pan={angles['pan']}, tilt={angles['tilt']}")

        self.last_alert_zone = alert_zone
        self.last_alert_ts   = now

    def reset_ptz_on_door_open(self):
        if hasattr(self, 'ptz_service') and self.ptz_service:
            self.ptz_service.reset()

    def _send_zone_status_to_web(self, zone_data):
        try:
            import threading
            def send_async():
                try:
                    requests.post("http://localhost:5000/zone_status", 
                                 json=zone_data, timeout=0.5)
                except Exception as e:
                    pass
            threading.Thread(target=send_async, daemon=True).start()
        except Exception as e:
            pass

    def _send_mode_status_to_web(self, mode):
        try:
            import threading
            def send_async():
                try:
                    requests.post("http://localhost:5000/mode_status", 
                                 json={"mode": mode}, timeout=0.5)
                except Exception as e:
                    pass
                    
            threading.Thread(target=send_async, daemon=True).start()
        except Exception as e:
            pass

    def _update_mode(self, new_mode):
        if self.current_mode != new_mode:
            self.current_mode = new_mode
            self._send_mode_status_to_web(new_mode)
            #print(f"[MODE] Changed to: {new_mode.upper()}")

    def _prepare_csi_tracker(self):
        if not self.csi_tracker_ready:
            #print("[CSI] Preparing tracker for CSI mode...")
            self.csi_tracker = HumanTracker()
            self.csi_tracker_ready = True
            #print("[CSI] Tracker prepared and ready")
        
    def reset_state(self):
        self.current_mode = self.MODE_DETECTION
        self.was_tracking = False
        self.last_timestamp = "--:--:--"
        self.frame_count_without_person = 0
        self.max_frames_without_person = CSI_PERSON_LOST_FRAMES
        
    def force_redetection(self):
        self.current_mode = self.MODE_DETECTION
        self.was_tracking = False
        self.tracker = HumanTracker()
        self.frame_count_without_person = 0
        p#rint("[MODE] Force redetection - tracker reinitialized")
        return True
        
    def post_stationary_bbox(self, bbox: Tuple, frame_size: Tuple[int, int]):
        x1, y1, x2, y2 = bbox
        w, h = frame_size
        bbox_norm = [x1/w, y1/h, x2/w, y2/h]
        payload = {
            "signal_type": "stationary_behavior",
            "bbox_normalized": bbox_norm,
            "metadata": {"source": "detector.py"}
        }

        try:
            requests.post(DEMO_API, json=payload, timeout=1)
            #print(f"[DAM] Sent stationary bbox {bbox_norm}")
        except Exception as e:
            print(f"[DAM] POST failed: {e}")
    
    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, Optional[Tuple]]:
        from demo.utils.viz import draw_timestamp, process_masks, draw_detection_boxes
        
        disp = frame.copy()
        h, w = frame.shape[:2]
        
        cx, cy = w // 2, h // 2
        cv2.line(disp, (cx, 0), (cx, h), (0, 255, 0), 1)
        cv2.line(disp, (0, cy), (w, cy), (0, 255, 0), 1)
        cv2.circle(disp, (cx, cy), 4, (0, 0, 255), -1)

        now = time.time()
        draw_timestamp(disp, time.strftime("%H:%M:%S", time.localtime(now)))
        
        bbox_for_ptz = None

        if self.current_mode == self.MODE_DETECTION:
            persons = self.detector.detect(frame)
            if persons:
                bbox_for_ptz = persons[0]
                self.tracker.initialize(frame, persons)
                self.was_tracking = True
                self._update_mode(self.MODE_TRACKING)
                self.frame_count_without_person = 0
                draw_detection_boxes(disp, persons)
                self.alert_manager.send_alert(AlertCodes.PERSON_DETECTED, "사람 감지 - Tracking 모드로 전환")
                #print(f"[MODE] Detection → Tracking")

        elif self.current_mode == self.MODE_TRACKING:
            if self.tracker.tracker is not None:
                masks, has_mask = self.tracker.track(frame)
                if has_mask:
                    bbox_for_ptz = process_masks(masks, disp, frame)
                    self.frame_count_without_person = 0
                    
                    if not self.tracker.is_person_in_frame(bbox_for_ptz, frame.shape):
                        self.frame_count_without_person += 1
                        if self.frame_count_without_person >= self.max_frames_without_person:
                            self._update_mode(self.MODE_CSI)
                            self._prepare_csi_tracker()
                            self.alert_manager.send_alert(AlertCodes.PERSON_LOST, "대상 이탈 감지 - CSI 모드로 전환")
                            #print(f"[MODE] Tracking → CSI (person left frame)")
                    
                    if self.tracker.check_stationary(bbox_for_ptz, now):
                        self.alert_manager.send_alert(AlertCodes.STATIONARY_BEHAVIOR,
                                                      "이상행동 감지 - 분석 필요")
                        threading.Thread(
                            target=self.post_stationary_bbox, 
                            args=(bbox_for_ptz, (w, h)), 
                            daemon=True
                        ).start()
                else:
                    self.frame_count_without_person += 1
                    if self.frame_count_without_person >= self.max_frames_without_person:
                        self._update_mode(self.MODE_CSI)
                        self._prepare_csi_tracker()
                        self.alert_manager.send_alert(AlertCodes.PERSON_LOST, "대상 이탈 감지 - CSI 모드로 전환")
                        #print(f"[MODE] Tracking → CSI (no mask detected)")

        elif self.current_mode == self.MODE_CSI:
            persons = self.detector.detect(frame)
            if persons:
                bbox_for_ptz = persons[0]
                if self.csi_tracker_ready and self.csi_tracker is not None:
                    self.tracker = self.csi_tracker
                    self.csi_tracker_ready = False
                    #print("[CSI] Using pre-prepared tracker")
                else:
                    self.tracker = HumanTracker()
                    #print("[CSI] Creating new tracker (fallback)")
                
                self.tracker.initialize(frame, persons)
                self.was_tracking = True
                self._update_mode(self.MODE_TRACKING)
                self.frame_count_without_person = 0
                draw_detection_boxes(disp, persons)
                self.alert_manager.send_alert(AlertCodes.PERSON_DETECTED, "사람 감지 - Tracking 모드로 전환")
                #print(f"[MODE] CSI → Tracking (person detected)")

        return disp, bbox_for_ptz 
    
    