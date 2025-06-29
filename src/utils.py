import logging
import re

import cpuinfo
import torch

logger = logging.getLogger(__name__)


def get_system_identifier() -> str:
    """
    CPUとGPUのモデル名から、ファイル名として使えるシステム識別子を生成する

    Returns:
        str: 生成されたシステム識別子
    """

    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get("brand_raw", "UnknownCPU")
    except Exception as e:
        logger.warning(
            "Could not retrieve CPU info",
            extra={"error": str(e)},
            exc_info=True,
        )
        cpu_name = "UnknownCPU"

    gpu_name = "CPUOnly"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(
                "Could not retrieve GPU info",
                extra={"error": str(e)},
                exc_info=True,
            )
            gpu_name = "UnknownGPU"

    cpu_name_safe = re.sub(r"[^a-zA-Z0-9]", "", cpu_name)
    gpu_name_safe = re.sub(r"[^a-zA-Z0-9]", "", gpu_name)

    identifier = f"{cpu_name_safe}-{gpu_name_safe}"
    logger.info(f"Generated system identifier: {identifier}")
    return identifier
