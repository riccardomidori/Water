import datetime
import time

import paho.mqtt.client
import pandas as pd
import pymongo
import seaborn as sns

import ssl
import json
import paho.mqtt.client as mqtt
from config.Logger import XMLLogger


class MidoriMQTTClient:
    def __init__(
        self,
        devices=None,
        cert_path: str = "data/ELTEK/56f7177dbf29c6261aab663e3ed59754cd3b769a0a90f1a0989353ebece5fa5c-certificate.pem.crt",
        key_path: str = "data/ELTEK/56f7177dbf29c6261aab663e3ed59754cd3b769a0a90f1a0989353ebece5fa5c-private.pem.key",
        ca_path: str = "data/ELTEK/AmazonRootCA1.pem",
        host: str = "ansavq2kpw85y-ats.iot.eu-west-1.amazonaws.com",
        port: int = 8883,
        frequency=1,
        duration=1,
    ):
        if devices is None:
            devices = ["EWCS30065"]
        self.devices = devices
        self.duration = duration
        self.frequency = frequency

        with pymongo.MongoClient() as c:
            db = c.get_database("eltek")
            self.devices_collection = db.get_collection("devices")
            self.water_flow = db.get_collection("water-flow")

        for device_id in devices:
            device = {
                "device_id": device_id,
                "config": {
                    "frequency": frequency,
                    "duration": duration,
                }
            }

            check = self.devices_collection.count_documents(device)
            if check == 0:
                print(f"Inserting {device}")
                self.devices_collection.insert_one(device)

        self.client = mqtt.Client(client_id="midori-mqtt")

        # TLS/SSL configuration
        self.client.tls_set(
            ca_certs=ca_path,
            certfile=cert_path,
            keyfile=key_path,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self.host = host
        self.port = port

        self.logger = XMLLogger(name=f"mqtt-logger",
                                log_file="mqtt.log",
                                path_to_log="log/Eltek/").logger

    def on_disconnect(self, client, userdata, rc):
        self.logger.warning(f"Disconnected with result code {rc}")
        time.sleep(0.2)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            topics = [
                (f"{device_id}/out/datastream", 0)
                for device_id in self.devices
            ]
            self.client.subscribe(topics)
            self.logger.info(f"Subscribed to: {topics}")
            time.sleep(0.1)
        else:
            self.logger.error(f"Failed to connect. Code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.logger.debug(f"Received {payload} from {msg.topic}")
            obj = {
                "device_id": payload["UniqueName"],
                "date": datetime.datetime.fromtimestamp(payload["Timestamp"]["epoch"] / 1000),
                "water-flow": payload["WaterFlow"],
                "water-temperature": payload["WaterTemperature"],
                "water-volume": payload["Volume"],
                "water-pressure": payload["WaterPressure"],
                "ambient-temperature": payload["AmbientTemperature"],
                "volume": payload["Volume"],
                "valve-state": payload["ValveState"]
            }
            self.water_flow.insert_one(obj)
        except Exception:
            self.logger.exception(f"Error decoding message")

    def connect(self):
        self.logger.debug("Connecting to MQTT broker...")
        self.client.connect(self.host, self.port)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.debug("Disconnected")

    def send_set_datastream(self):
        payload = {"frequency": self.frequency, "duration": self.duration}
        for device_id in self.devices:
            topic = f"{device_id}/setdatastream"
            self.logger.debug(f"Sending to {topic}: {payload}")
            self.client.publish(topic, json.dumps(payload))
            time.sleep(0.1)


class WaterSensor:
    def __init__(self):
        pass

    @staticmethod
    def connect():
        # EWCS3A077 not working
        device_list = ["EWCS30065", "EWCS30156", "EWCS30154", "EWCS30143", "EWCS30144"]

        client = MidoriMQTTClient(
            devices=device_list,
            frequency=1,
            duration=-1
        )
        client.connect()
        client.send_set_datastream()
        try:
            while True:
                pass  # Keep the script running
        except KeyboardInterrupt:
            client.disconnect()
        except Exception as e:
            #retry
            time.sleep(10)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    sns.set(font_scale=1.8)
    WaterSensor.connect()
