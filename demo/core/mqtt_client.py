import threading, time
import paho.mqtt.client as mqtt

class BaseMQTTClient:
    def __init__(self, broker, port, keepalive, client_id):
        self.client = mqtt.Client(client_id=client_id, clean_session=True)
        self.client.reconnect_delay_set(0.2, 2)
        self.client.connect(broker, port, keepalive)
        self.client.loop_start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while True:
            if not self.client.is_connected():
                try: self.client.reconnect_async()
                except: pass
            time.sleep(5)

            
