import argparse
import datetime
import json
import logging
import os
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

from src.analysis.curve import (
    calculate_data_transfer_features,
    calculate_gpu_compute_features,
    calculate_host_compute_features,
)
from src.benchmarks.data_transfer.core import DataTransferRunner
from src.benchmarks.gpu_compute.core import GpuComputeRunner
from src.benchmarks.host_compute.core import HostComputeRunner
from src.config import load_config
from src.logger import setup_logging
from src.utils import get_system_identifier

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):

    log_level = "DEBUG" if args.debug else "INFO"
    log_filepath = None
    if args.log_to_file:
        log_filepath = "run.log"
    setup_logging(log_level=log_level, log_filepath=log_filepath)

    features_list = []
    output_dir = None

    try:
        # 設定とファイルパスの準備
        config = load_config(args.config)
        abstract_system_id, cpu_raw, gpu_raw = get_system_identifier(
            fullname=args.fullname
        )
        jst_now = datetime.datetime.now(tz=ZoneInfo("Asia/Tokyo"))
        timestamp = jst_now.strftime("%Y%m%d_%H%M%S")

        # ディレクトリ名には抽象化IDを使用
        output_dir = f"results/{timestamp}_{abstract_system_id}"
        raws_dir = os.path.join(output_dir, "raws")
        os.makedirs(raws_dir, exist_ok=True)

        logger.info(f"Created output directory: {output_dir}")

        metadata = {
            "run_id": output_dir,
            "run_timestamp": jst_now.isoformat(),
            "config_file_used": os.path.abspath(args.config),
            "system_info": {
                "cpu_brand_raw": cpu_raw,
                "gpu_brand_raw": gpu_raw,
            },
        }
        with open(
            os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)
        shutil.copy(args.config, os.path.join(output_dir, "config_snapshot.yaml"))

        # データ取得
        logger.info("Starting Raw Data Acquisition")
        # --target 引数に応じて、実行するRunnerを決定
        gpu_compute_raw_list, data_transfer_raw_list = [], []
        if args.target in ["all", "gpu_compute"]:
            logger.info("Starting GPU Compute Benchmarks")
            gpu_runner = GpuComputeRunner(config)
            gpu_compute_raw_list = gpu_runner.run()

        if args.target in ["all", "data_transfer"]:
            logger.info("Starting Data Transfer Benchmarks")
            transfer_runner = DataTransferRunner(config)
            data_transfer_raw_list = transfer_runner.run()

        if args.target in ["all", "host_compute"]:
            logger.info("Starting Host Compute Benchmarks")
            host_runner = HostComputeRunner(config)
            host_compute_raw_list = host_runner.run()

        logger.info("Complete. Acquired raw data.")

        # データ分析
        logger.info("Starting Feature Extraction")
        # gpu_computeの分析

        if gpu_compute_raw_list:
            df = pd.DataFrame(gpu_compute_raw_list)
            for (name, dtype), group_df in df.groupby(["benchmark_name", "data_type"]):
                metrics = calculate_gpu_compute_features(group_df)
                features_list.append(
                    {
                        "group_name": "gpu_compute",
                        "benchmark_name": name,
                        "data_type": dtype,
                        **metrics,
                    }
                )
        # data_transferの分析
        if data_transfer_raw_list:
            df = pd.DataFrame(data_transfer_raw_list)
            for name, group_df in df.groupby("benchmark_name"):
                metrics = calculate_data_transfer_features(group_df)
                params = group_df.iloc[0]
                features_list.append(
                    {
                        "group_name": "data_transfer",
                        "benchmark_name": name,
                        "direction": params["direction"],
                        "use_pinned_memory": params["use_pinned_memory"],
                        **metrics,
                    }
                )
        if host_compute_raw_list:
            df = pd.DataFrame(host_compute_raw_list)
            for name, group_df in df.groupby("benchmark_name"):
                metrics = calculate_host_compute_features(group_df)

                domain = group_df.iloc[0]["domain"]

                features_list.append(
                    {
                        "group_name": "host_compute",
                        "benchmark_name": name,
                        "domain": domain,
                        **metrics,
                    }
                )
        logger.info(f"Complete. Extracted {len(features_list)} feature sets.")

    except Exception as e:
        logger.error(
            "An unhandled exception occurred in the main process",
            extra={"error": str(e)},
            exc_info=True,
        )

    finally:
        logger.info("Finalizing and Saving Results")
        if output_dir:
            # gpu_compute の生データを保存
            if gpu_compute_raw_list:
                pd.DataFrame(gpu_compute_raw_list).to_csv(
                    os.path.join(raws_dir, "gpu_compute.csv"), index=False
                )

            # data_transfer の生データを保存
            if data_transfer_raw_list:
                pd.DataFrame(data_transfer_raw_list).to_csv(
                    os.path.join(raws_dir, "data_transfer.csv"), index=False
                )

            if host_compute_raw_list:
                pd.DataFrame(host_compute_raw_list).to_csv(
                    os.path.join(raws_dir, "host_compute.csv"), index=False
                )

            # 統合された特徴量データを保存
            if features_list:
                pd.DataFrame(features_list).to_csv(
                    os.path.join(output_dir, "features.csv"), index=False
                )

            logger.info(f"All results saved in {output_dir}")
        else:
            logger.error("Output directory was not created. Cannot save results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTFusion Benchmark Runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to the benchmark configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging.",
    )
    parser.add_argument(
        "--fullname",
        action="store_true",
        help="Use the full, detailed system identifier for the output directory name.",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Enable logging to a file inside the run directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "gpu_compute",
            "data_transfer",
            "host_compute",
        ],
        help="Specify which benchmark group to run.",
    )
    args = parser.parse_args()

    main(args)
