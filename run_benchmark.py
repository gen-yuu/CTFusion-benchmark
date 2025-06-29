import argparse
import datetime
import json
import logging
import os
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

from src.analysis.curve import calculate_4_metrics
from src.benchmarks.gpu_compute.core import GpuComputeRunner
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

    original_results_list = []
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
        os.makedirs(output_dir, exist_ok=True)
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
        runner = GpuComputeRunner(config)
        original_results_list = runner.run()
        logger.info(f"Complete. Acquired {len(original_results_list)} data points")

        if not original_results_list:
            logger.warning("No data was generated. Exiting")
            return

        # データ分析
        logger.info("Starting Feature Extraction")
        original_df = pd.DataFrame(original_results_list)
        grouped = original_df.groupby(["benchmark_name", "data_type"])

        for (name, dtype), group_df in grouped:
            logger.debug(f"Analyzing curve for {name} ({dtype})")
            metrics = calculate_4_metrics(group_df)
            features_list.append(
                {"benchmark_name": name, "data_type": dtype, **metrics}
            )
        logger.info(f"Complete. Extracted {len(features_list)} feature sets")

    except Exception as e:
        logger.error(
            "An unhandled exception occurred in the main process",
            extra={"error": str(e)},
            exc_info=True,
        )

    finally:
        logger.info("Finalizing and Saving Results")
        if output_dir:

            if original_results_list:
                try:
                    original_output_path = os.path.join(output_dir, "raw_data.csv")
                    logger.info(f"Saving raw data to {original_output_path}..")
                    pd.DataFrame(original_results_list).to_csv(
                        original_output_path, index=False
                    )
                    logger.info("Raw data saved successfully")
                except Exception as e:
                    logger.error(
                        "Failed to save raw data",
                        extra={"error": str(e)},
                        exc_info=True,
                    )

            if features_list:
                try:
                    features_output_path = os.path.join(output_dir, "features.csv")
                    logger.info(f"Saving feature data to {features_output_path}..")
                    pd.DataFrame(features_list).to_csv(
                        features_output_path, index=False
                    )
                    logger.info("Feature data saved successfully")
                except Exception as e:
                    logger.error(
                        "Failed to save feature data",
                        extra={"error": str(e)},
                        exc_info=True,
                    )
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
    args = parser.parse_args()

    main(args)
