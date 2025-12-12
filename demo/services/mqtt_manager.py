import time
import threading
import socket
import uuid
from collections import defaultdict

import paho.mqtt.client as mqtt

from demo.config.settings import BROKER_ADDR, BROKER_PORT, MQTT_PTZ_KEEPALIVE



def _unique_id(prefix="ptz"):
    host = socket.gethostname()
    return f"{prefix}_{host}_{uuid.uuid4().hex[:4]}"


class MQTTManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        broker_addr: str = BROKER_ADDR,
        broker_port: int = BROKER_PORT,
        keepalive: int = MQTT_PTZ_KEEPALIVE,
        client_id: str = None,
    ):
        if hasattr(self, "_init_done"):
            return
        self._init_done = True

        self.client = mqtt.Client(
            client_id=client_id or _unique_id(),
            clean_session=True,
        )
        self.client.reconnect_delay_set(0.2, 2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._callbacks: dict[str, list] = defaultdict(list)

        self.client.connect(broker_addr, broker_port, keepalive)
        self.client.loop_start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def publish(self, topic: str, payload: str, qos: int = 1, retain: bool = False):
        rc = self.client.publish(topic, payload, qos, retain)[0]
        if rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] Publish 실패 (rc={rc})")

    def subscribe(self, topic: str, callback):
        if not self._callbacks[topic]:
            self.client.subscribe(topic)
            self.client.message_callback_add(topic, self._dispatch)
        self._callbacks[topic].append(callback)

    def _on_connect(self, client, userdata, flags, rc):
        print("[MQTT] Connected")
        for topic in self._callbacks:
            client.subscribe(topic)

    def _on_disconnect(self, client, userdata, rc):
        print("[MQTT] Disconnected – auto-reconnect")

    def _dispatch(self, client, userdata, msg):
        for cb in self._callbacks[msg.topic]:
            cb(msg)

    def is_connected(self):
        return self.client.is_connected()
    
    def _watchdog(self):
        while True:
            if not self.client.is_connected():
                try:
                    print("[MQTT] Attempting to reconnect...")
                    self.client.reconnect()
                except Exception as e:
                    print(f"[MQTT] Reconnect failed: {e}")
            time.sleep(5)