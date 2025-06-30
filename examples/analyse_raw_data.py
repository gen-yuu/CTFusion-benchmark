import argparse
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from analysis.curve import calculate_gpu_compute_features
    from logger import setup_logging
except ImportError as e:
    print(f"Error: Failed to import src modules. Error: {e}")
    sys.exit(1)


def main():
    """
    既存のraw_data.csvから特徴量を再計算し、features.csvを生成する。
    """
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Re-calculate features from an existing raw_data.csv file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input raw_data.csv file.",
    )
    args = parser.parse_args()

    input_path = args.input_file

    # --- 1. 入力ファイルの読み込み ---
    try:
        logger.info(f"Loading raw data from: {input_path}")
        original_df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return

    # --- 2. データ分析（特徴量抽出）---
    logger.info("Starting feature extraction...")
    features_list = []

    # 'benchmark_name'と'data_type'でグループ化
    # `dropna=False`は、万が一これらのキーがNaNの場合でもグループ化対象とするため
    grouped = original_df.groupby(["benchmark_name", "data_type"], dropna=False)

    for (name, dtype), group_df in grouped:
        logger.debug(f"Analyzing curve for {name} ({dtype})")
        metrics = calculate_gpu_compute_features(group_df)
        features_list.append({"benchmark_name": name, "data_type": dtype, **metrics})

    if not features_list:
        logger.warning(
            "No features were extracted. The input file might be empty or invalid."
        )
        return

    features_df = pd.DataFrame(features_list)
    logging.info(f"Complete. Extracted {len(features_list)} feature sets.")

    # --- 3. 結果の表示と保存 ---

    # コンソールに結果を表示
    print("\n--- Calculated Features ---")
    print(features_df.to_string())
    print("-------------------------\n")

    # 入力ファイルと同じディレクトリにfeatures.csvを保存
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "features.csv")

    try:
        features_df.to_csv(output_path, index=False)
        logger.info(f"Feature data saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save feature data to {output_path}", exc_info=True)


if __name__ == "__main__":
    main()
