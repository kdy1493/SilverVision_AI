import cv2
import threading
import time
import numpy as np
from typing import Optional
from demo.config.settings import STREAM_URL

class FrameGrabber(threading.Thread):
    RETRY_SEC = 0.7

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url  = url
        self.cap  = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()
            print("[Stream] Releasing previous capture")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"[Stream] Failed to open {self.url}")
            return
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                if not self.cap.isOpened():
                    time.sleep(self.RETRY_SEC)
                    continue

            ret, frm = self.cap.read()
            if not ret:

                self.cap.release()
                time.sleep(self.RETRY_SEC)
                continue

            with self.lock:
                self.frame = frm

        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

class StreamManager:
    def __init__(self, stream_url: str = STREAM_URL):
        self.stream_url = stream_url
        self.stream_active = False
        self.grabber: Optional[FrameGrabber] = None
        
    def start_stream(self):
        if not self.stream_active:
            self.grabber = FrameGrabber(self.stream_url)
            self.grabber.start()
            print(f"[Stream] Starting stream from {self.stream_url}")
            self.stream_active = True
            
    def stop_stream(self):
        if self.stream_active:
            self.stream_active = False
            if self.grabber:
                self.grabber.stop()
                self.grabber = None
                
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.stream_active or self.grabber is None:
            return None
        return self.grabber.read()
    
    def is_active(self) -> bool:
        return self.stream_active
    
    def get_blank_frame(self) -> np.ndarray:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for trigger...", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return 
    
    