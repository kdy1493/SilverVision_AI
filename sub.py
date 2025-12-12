import sys
import json
import paho.mqtt.client as mqtt

BROKER = "61.252.57.136"
PORT   = 4991
TOPIC  = sys.argv[1] if len(sys.argv) > 1 else "Loc/zone/response"

def on_connect(client, userdata, flags, rc):
    print(f"Connected (rc={rc}), subscribing to {TOPIC!r}")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        print("Invalid JSON:", payload)
        return

    alert_zones = [zone for zone, state in data.items() if state == "alert"]

    if alert_zones:
        print("Alert zones:", ", ".join(alert_zones))

client = mqtt.Client()
client.on_connect  = on_connect
client.on_message  = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
