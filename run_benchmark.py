import argparse
import datetime
import logging
import os

import pandas as pd

from src.analysis.curve import calculate_4_metrics
from src.benchmarks.gpu_compute.core import GpuComputeRunner
from src.config import load_config
from src.logger import setup_logging
from src.utils import get_system_identifier

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description="CTFusion Benchmark Runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to the benchmark configuration file.",
    )
    args = parser.parse_args()

    original_results_list = []
    features_list = []

    try:
        # 設定とファイルパスの準備
        config = load_config(args.config)
        system_id = get_system_identifier()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir_original = f"results/original/{system_id}"
        output_dir_features = f"results/features/{system_id}"
        os.makedirs(output_dir_original, exist_ok=True)
        os.makedirs(output_dir_features, exist_ok=True)

        original_output_path = os.path.join(output_dir_original, f"{timestamp}_raw.csv")
        features_output_path = os.path.join(
            output_dir_features, f"{timestamp}_features.csv"
        )

        # データ取得
        logging.info("Starting Raw Data Acquisition")
        runner = GpuComputeRunner(config)
        original_results_list = runner.run()
        logging.info(f"Complete. Acquired {len(original_results_list)} data points")

        if not original_results_list:
            logging.warning("No data was generated. Exiting")
            return

        # データ分析
        logging.info("Starting Feature Extraction")
        original_df = pd.DataFrame(original_results_list)
        grouped = original_df.groupby(["benchmark_name", "data_type"])

        for (name, dtype), group_df in grouped:
            logging.debug(f"Analyzing curve for {name} ({dtype})")
            metrics = calculate_4_metrics(group_df)
            features_list.append(
                {"benchmark_name": name, "data_type": dtype, **metrics}
            )
        logging.info(f"Complete. Extracted {len(features_list)} feature sets")

    except Exception as e:
        logging.error(
            "An unhandled exception occurred in the main process",
            extra={"error": str(e)},
            exc_info=True,
        )

    finally:
        logging.info("Finalizing and Saving Results")

        if original_results_list:
            try:
                logging.info(f"Saving raw data to {original_output_path}..")
                pd.DataFrame(original_results_list).to_csv(
                    original_output_path, index=False
                )
                logging.info("Raw data saved successfully")
            except Exception as e:
                logging.error(
                    "Failed to save raw data",
                    extra={"error": str(e)},
                    exc_info=True,
                )

        if features_list:
            try:
                logging.info(f"Saving feature data to {features_output_path}..")
                pd.DataFrame(features_list).to_csv(features_output_path, index=False)
                logging.info("Feature data saved successfully")
            except Exception as e:
                logging.critical(
                    "Failed to save feature data",
                    extra={"error": str(e)},
                    exc_info=True,
                )


if __name__ == "__main__":
    main()
