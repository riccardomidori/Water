import contextlib
import datetime
import logging

import mysql.connector
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursor
from config.ConfigManager import ConfigManager
from config.Logger import XMLLogger
from config.Utils import auto_str


@auto_str
class DatabaseConnector:
    def __init__(self, name="MySQL", logger: logging.Logger = None):
        if logger is None:
            logger = XMLLogger(
                name="DatabaseManager",
                path_to_log="log/DigitalManager",
                level="debug",
                log_file="server.log",
            ).logger
        self.name = name
        self.logger = logger
        cm = ConfigManager()

        self.port = cm.mysql_port
        self.dbname = cm.mysql_dbname
        self.passwd = cm.mysql_password
        self.host = cm.mysql_host
        self.user = cm.mysql_user

        self.connection = None
        self.connector = None
        self.connection_string = (
            f"mysql://{self.user}:{self.passwd}@{self.host}:{self.port}/{self.dbname}"
        )
        self.logger.debug(f"Initialize DatabaseManager-{self.name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"Exiting Database Connection")

        if exc_tb is None:
            self.connector.commit()
        else:
            self.connector.rollback()
        self.connector.close()

    def set_name(self, name):
        self.logger.debug(f"MySQL Connection from {self.name} to {name}")
        self.name = name

    @contextlib.contextmanager
    def __connect__(self) -> MySQLConnection:
        self.logger.info(f"Connecting with MySQL")

        _connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            port=self.port,
            db=self.dbname,
        )
        try:
            yield _connection
        except Exception:
            self.logger.exception(f"Exception on Config {self}")
            _connection.rollback()
            raise
        else:
            _connection.commit()
        finally:
            _connection.close()

    @contextlib.contextmanager
    def __cursor__(self) -> MySQLCursor:
        self.logger.info(f"Creating DatabaseManager cursor")

        with self.__connect__() as conn:
            cursor_ = conn.cursor(buffered=True, dictionary=True)
            try:
                yield cursor_
            finally:
                cursor_.close()

    @staticmethod
    def get_cursor(connection: MySQLConnection):
        return connection.cursor(buffered=True, dictionary=True)

    def save_monitoring(self,
                        software_id: str,
                        utility: str,
                        has_alert: bool,
                        has_data: bool,
                        timestamp_data=None,
                        notes=None):
        self.logger.debug(f"Save monitoring for {utility}")
        with self.__connect__() as c:
            now = int(datetime.datetime.now().timestamp())
            cursor = self.get_cursor(c)
            if has_data and has_alert:
                query = ("INSERT INTO tab_software_monitoring "
                         "(id_software, timestamp_run, timestamp_software, admin_utility, alert, note) "
                         "VALUES ('%s', %s, %s, '%s', %s, '%s')")
                cursor.execute(query % (software_id, now, timestamp_data, utility, True, notes))
            elif has_data:
                query = ("INSERT INTO tab_software_monitoring "
                         "(id_software, timestamp_run, timestamp_software, admin_utility) "
                         "VALUES ('%s', %s, %s, '%s')")
                cursor.execute(query % (software_id, now, timestamp_data, utility))
            elif has_alert:
                query = ("INSERT INTO tab_software_monitoring "
                         "(id_software, timestamp_run, admin_utility, alert, note) "
                         "VALUES ('%s', %s, '%s', %s, '%s')")
                cursor.execute(query % (software_id, now, utility, True, notes))
            else:
                query = ("INSERT INTO tab_software_monitoring "
                         "(id_software, timestamp_run, admin_utility) "
                         "VALUES ('%s', %s, '%s')")
                cursor.execute(query % (software_id, now, utility))

            c.commit()


if __name__ == "__main__":
    pass
