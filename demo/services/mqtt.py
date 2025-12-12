import time, threading
from demo.services.mqtt_manager import MQTTManager
from demo.config.settings import BROKER_ADDR, BROKER_PORT
from demo.services.mqtt_publisher import MQTTPublisher
import paho.mqtt.client as paho
import json


class MQTTService:
    def __init__(self, stream_manager, detection_processor=None):
        self.publisher = MQTTPublisher()
        self.stream_manager = stream_manager
        self.detection_processor = detection_processor
        self.last_trigger_time = 0
        self.last_loc_msg = None
        self.last_loc = None
        self.mgr = MQTTManager()
        self.mgr.subscribe("ptz/trigger", self._on_message)

    def _on_message(self, msg):
        payload = msg.payload.decode().strip()
        now     = int(time.time())

        #print(f"[MQTT] Received '{payload}' (stream_active: {self.stream_manager.is_active()})")

        if payload == "1":
            if not self.stream_manager.is_active():
                if now - self.last_trigger_time < 3:
                    self.last_trigger_time = now
                    #print(f"[MQTT] Rate limiting: {now - self.last_trigger_time}s since last trigger")
                    return
                #print("[TRIGGER] 1 → stream ON")
                try:
                    threading.Thread(target=self._send_stream_on, daemon=True).start()
                    self.stream_manager.start_stream()
                    #print(f"[TRIGGER] Stream start attempted. Active: {self.stream_manager.is_active()}")
                except Exception as e:
                    print(f"[TRIGGER] Stream start failed: {e}")
            else:
                if now - self.last_trigger_time < 3:
                    self.last_trigger_time = now
                    return
                #print("[TRIGGER] stream OFF")
                threading.Thread(target=self._send_stream_off, daemon=True).start()
                self.stream_manager.stop_stream()
            
            if self.detection_processor:
                self.detection_processor.reset_ptz_on_door_open()

        elif payload == "0" and self.stream_manager.is_active():
            #print("[TRIGGER] 0 → stream OFF")
            self.stream_manager.stop_stream()

        self.last_trigger_time = now

    def _send_stream_on(self):
        try:
            if not hasattr(self, "_fire_client"):
                self._fire_client = paho.Client()
                self._fire_client.connect(BROKER_ADDR, BROKER_PORT, 60)
            self._fire_client.publish("ptz/stream", "on", qos=0, retain=False)
        except Exception as e:
            print("[MQTT] stream-on publish failed:", e)

    def _send_stream_off(self):
        try:
            if not hasattr(self, "_fire_client"):
                self._fire_client = paho.Client()
                self._fire_client.connect(BROKER_ADDR, BROKER_PORT, 60)
            self._fire_client.publish("ptz/stream", "off", qos=0, retain=False)
        except Exception as e:
            print("[MQTT] stream-off publish failed:", e)