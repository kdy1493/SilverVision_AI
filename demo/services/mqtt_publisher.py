from __future__ import annotations
from typing import Union

from demo.config.settings import (
    BROKER_ADDR, BROKER_PORT, MQTT_PTZ_TOPIC, MQTT_PTZ_CLIENT_ID,
)
from demo.services.mqtt_manager import MQTTManager



class MQTTPublisher:
    def __init__(
        self,
        broker_addr: str = BROKER_ADDR,
        broker_port: int = BROKER_PORT,
        topic: str = MQTT_PTZ_TOPIC,
        client_id: str = MQTT_PTZ_CLIENT_ID,
    ) -> None:
        self.topic   = topic
        self._last_sent = {"pan": None, "tilt": None}
        self._pending   = {"pan": None, "tilt": None}

        self.mgr = MQTTManager(
            broker_addr=broker_addr,
            broker_port=broker_port,
            client_id=client_id,
        )

    @staticmethod
    def _clamp(angle: Union[int, float]) -> int:
        if not (0 <= angle <= 180):
            raise ValueError("angle must be 0–180")
        return int(angle)

    @staticmethod
    def _fmt(axis: str, ang: int) -> str:
        if axis not in ("pan", "tilt"):
            raise ValueError("axis must be pan|tilt")
        return f"{axis},{ang}"

    def _raw_publish(self, axis: str, ang: int):
        payload = self._fmt(axis, ang)
        self.mgr.publish(self.topic, payload, qos=1)
        self._last_sent[axis] = ang
        #print(f"[PTZ] PTZ Command: {payload}")

    def publish(self, axis: str, angle: Union[int, float]) -> None:
        ang = self._clamp(angle)
        self._pending[axis] = ang

        if not self.mgr.client.is_connected():
            print(f"[PTZ] MQTT not connected, cannot publish {axis}={ang}")
            return

        if self._last_sent.get(axis) == ang:
            print(f"[PTZ] No change for {axis}={ang}, skipping publish")
            return

        self._raw_publish(axis, ang)

if __name__ == "__main__":
    import time, uuid, os
    pub = MQTTPublisher(client_id=f"human_ptz_{os.getpid()}_{uuid.uuid4().hex[:4]}")
    try:
        while True:
            pub.publish("pan", 120)
            time.sleep(0.3)
            pub.publish("tilt", 110)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
