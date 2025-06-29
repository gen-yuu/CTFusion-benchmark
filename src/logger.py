import json
import logging
import os
import sys


class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    Handles the 'extra' parameter to include custom fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        # These are the standard attributes from a LogRecord
        standard_keys = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        # Base log object with standard information
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Add any fields passed in the 'extra' parameter
        for key, value in record.__dict__.items():
            if key not in standard_keys:
                log_object[key] = value

        return json.dumps(log_object, ensure_ascii=False)


def setup_logging(log_level: str = "INFO", log_filepath: str = None) -> None:
    """
    Sets up a logger with a JSON formatter.

    Args:
        log_level (str): The logging level to set. Default is "INFO".
        log_filepath (str): The file path to log to. Default is None.
    """
    log_level_upper = log_level.upper()
    level = getattr(logging, log_level_upper, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Prevents adding handlers multiple times in interactive environments
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # フォーマッタの定義
    formatter = JsonFormatter()

    # 標準出力(stdout)へのハンドラを設定
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # ファイルへのハンドラを設定
    if log_filepath:
        # ディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    logging.info(f"Logging setup complete. Level: {log_level_upper}")
    if log_filepath:
        logging.info(f"Logging is also directed to file: {log_filepath}")
