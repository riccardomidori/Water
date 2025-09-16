import datetime
import time

import pandas as pd
from pathlib import Path

import pymongo
from matplotlib import pyplot as plt
import seaborn as sns

import ssl
import json
import paho.mqtt.client as mqtt
from config.Logger import XMLLogger


class MidoriMQTTClient:
    def __init__(
        self,
        device_id: str = "EWCS30065",
        cert_path: str = "data/ELTEK/56f7177dbf29c6261aab663e3ed59754cd3b769a0a90f1a0989353ebece5fa5c-certificate.pem.crt",
        key_path: str = "data/ELTEK/56f7177dbf29c6261aab663e3ed59754cd3b769a0a90f1a0989353ebece5fa5c-private.pem.key",
        ca_path: str = "data/ELTEK/AmazonRootCA1.pem",
        host: str = "ansavq2kpw85y-ats.iot.eu-west-1.amazonaws.com",
        port: int = 8883,
        frequency=1,
        duration=1,
    ):
        self.duration = duration
        self.frequency = frequency
        self.device_id = device_id
        self.set_topic = f"{device_id}/setdatastream"
        self.out_topic = f"{device_id}/out/datastream"

        with pymongo.MongoClient() as c:
            db = c.get_database("eltek")
            self.devices = db.get_collection("devices")
            self.water_flow = db.get_collection("water-flow")

        device = {
            "device_id": device_id,
            "config": {
                "frequency": frequency,
                "duration": duration,
            }
        }

        check = self.devices.find(device).count()
        if check == 0:
            print(f"Inserting {device}")
            self.devices.insert_one(device)

        self.client = mqtt.Client()

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

        self.host = host
        self.port = port

        self.logger = XMLLogger(name=f"mqtt-logger-{device_id}",
                                log_file="mqtt.log",
                                path_to_log="log/Eltek/").logger

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            self.client.subscribe(self.out_topic)
            self.logger.info(f"Subscribed to: {self.out_topic}")
        else:
            self.logger.error(f"Failed to connect. Code: {rc}")

    def on_message(self, client, userdata, msg):
        self.logger.debug(f"Received message from {msg.topic}")
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            obj = {
                "device_id": payload["UniqueName"],
                "date": datetime.datetime.fromtimestamp(payload["Timestamp"]["epoch"] / 1000),
                "water-flow": payload["WaterFlow"],
                "water-temperature": payload["WaterTemperature"],
                "water-volume": payload["Volume"],
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
        self.logger.debug(f"Sending to {self.set_topic}: {payload}")
        self.client.publish(self.set_topic, json.dumps(payload))


class WaterSensor:
    def __init__(self):
        pass

    @staticmethod
    def connect():
        client = MidoriMQTTClient(
            device_id="EWCS30065",
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
