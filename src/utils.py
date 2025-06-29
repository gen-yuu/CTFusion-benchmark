import logging
import re
from typing import Tuple

import cpuinfo
import torch

logger = logging.getLogger(__name__)


def get_system_identifier(fullname: bool = False) -> Tuple[str, str, str]:
    """
    CPUとGPUのモデル名からシステム識別子を生成する

    Args:
        fullname (bool, optional):
            Trueの場合、識別子として整形済みのフルネームを返す
            Falseの場合、識別子として抽象化されたカテゴリ名を返す
            Defaults to False

    Returns:
        Tuple[str, str, str]: (識別子, 生のCPU名, 生のGPU名) のタプル。
    """
    # --- CPU名の取得 ---
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name_raw = cpu_info.get("brand_raw", "UnknownCPU")
    except Exception as e:
        logger.warning(f"Could not retrieve CPU info: {e}")
        cpu_name_raw = "UnknownCPU"

    # --- GPU名の取得 ---
    gpu_name_raw = "CPUOnly"
    if torch.cuda.is_available():
        try:
            gpu_name_raw = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(f"Could not retrieve GPU info: {e}")
            gpu_name_raw = "UnknownGPU"

    # --- 整形済みフルネームの生成 ---
    cpu_name_full = re.sub(r"\(R\)|\(TM\)|CPU @.*", "", cpu_name_raw).strip()
    cpu_name_full = re.sub(r"[\s-]+", "_", cpu_name_full)
    cpu_name_full = re.sub(r"_+", "_", cpu_name_full)
    gpu_name_full = gpu_name_raw.replace(" ", "_")
    full_identifier = f"{cpu_name_full}-{gpu_name_full}"

    # --- 抽象化識別子の生成 ---
    cpu_abstract_name = "UnknownCPU"
    if "Xeon" in cpu_name_raw:
        match = re.search(r"Xeon(?:_|\s)(?P<series>\w+)", cpu_name_full)
        cpu_abstract_name = (
            f"Intel_Xeon_{match.group('series')}" if match else "Intel_Xeon"
        )
    elif "Core" in cpu_name_raw:
        match = re.search(r"i(\d)[-_]?(\d{1,2})\d{3}", cpu_name_raw)
        if match:
            series, gen = match.groups()
            cpu_abstract_name = f"Intel_i{series}_Gen{gen}"
        else:
            cpu_abstract_name = "Intel_Core"

    gpu_abstract_name = "UnknownGPU"
    if "NVIDIA" in gpu_name_raw:
        temp_name = re.sub(r"NVIDIA(?:_|\s)?(GeForce(?:_|\s)?)?", "", gpu_name_full)
        gpu_abstract_name = temp_name

    abstract_identifier = f"{cpu_abstract_name}-{gpu_abstract_name}"

    if not fullname:
        identifier = abstract_identifier
        logger.info(f"Generated abstract identifier: {identifier}")
    else:
        identifier = full_identifier
        logger.info(f"Generated full identifier: {identifier}")

    return identifier, cpu_name_raw, gpu_name_raw
