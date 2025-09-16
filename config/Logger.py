import io
import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


class XMLLogger:
    def __init__(
        self,
        name,
        path_to_log="log/DigitalManager/",
        level="debug",
        log_file="server.log",
        filemode="a",
        date_format="'%Y-%m-%d %H:%M:%S'",
        formatter=logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"),
        backup_count=30,
        when="Midnight",
        system_output=sys.stderr,
        with_console=True,
    ):
        self.name = name
        if log_file is None:
            log_file = "server.log"
        if path_to_log != "":
            if not Path(path_to_log).exists():
                Path(path_to_log).mkdir(parents=True)
            self.path = Path(f"{path_to_log}/{log_file}")
        else:
            self.path = Path.cwd() / log_file
        if level == "info":
            level = logging.INFO
        elif level == "debug":
            level = logging.DEBUG
        elif level == "warning" or level == "warn":
            level = logging.WARNING
        elif level == "critical":
            level = logging.CRITICAL
        else:
            level = logging.INFO
        self.with_console = with_console
        self.level = level
        self.filemode = filemode
        self.date_format = date_format
        self.formatter = formatter
        self.backup_count = backup_count
        self.when = when
        self.system_output = system_output

        self.logger = self.create_logger()

    def create_logger(self):
        time_handler = TimedRotatingFileHandler(
            self.path, when=self.when, backupCount=self.backup_count, delay=True
        )
        time_handler.setFormatter(self.formatter)
        logger = logging.getLogger(self.name)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(time_handler)

        if self.with_console:
            console_handler = StreamHandler(self.system_output)
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)

        logger.setLevel(self.level)

        return logger
