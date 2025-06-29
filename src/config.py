import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    指定されたパスからYAML設定ファイルを読み込み、辞書として返す
    エラーが発生した場合はログに記録する

    Args:
        config_path (str): YAMLファイルのパス

    Returns:
        Dict[str, Any]: 読み込まれた設定内容

    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合
        yaml.YAMLError: ファイルの解析に失敗した場合
    """
    try:
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"Configuration file '{config_path}' is empty")
            return {}

        logger.info("Configuration loaded successfully")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'")
        raise

    except yaml.YAMLError as e:
        logger.error(
            f"Failed to parse YAML file at '{config_path}'",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
