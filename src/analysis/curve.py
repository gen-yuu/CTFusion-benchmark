import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_4_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    単一の性能曲線データ（DataFrame）から4つの指標を計算する

    Args:
        group_df (pd.DataFrame): 'benchmark_name'と'data_type'でグループ化されたDataFrame。
            'batch_size', 'latency_sec', 'throughput_items_per_sec'カラムを持つ

    Returns:
        Dict[str, Any]: 計算された4指標を含む辞書。
            - `latency`: レイテンシ（`batch_size=1`での実行時間）
            - `peak_throughput`: ピークスループット（性能曲線の最大値）
            - `saturation_point`: 飽和点（ピーク性能の90%に達するバッチサイズ）
            - `efficiency_slope`: 効率の傾き（性能曲線の立ち上がり具合）
    """
    # エラーのあった行は除外して計算
    valid_df = group_df[group_df["throughput_items_per_sec"] >= 0].copy()
    valid_df = valid_df.sort_values(by="batch_size").reset_index()

    # --- ストレステスト（データが1点）の場合の特別処理 ---
    if len(valid_df) <= 1:
        if len(valid_df) == 1:
            latency = valid_df.iloc[0]["latency_sec"]
            throughput = valid_df.iloc[0]["throughput_items_per_sec"]
        else:
            latency = -1.0
            throughput = -1.0

        logger.debug("Only one data point found. Treating as stress test.")
        return {
            "latency": latency,
            "peak_throughput": throughput,
            "saturation_point": "N/A",
            "efficiency_slope": "N/A",
        }

    # --- 4指標の計算 ---
    bs1_row = valid_df[valid_df["batch_size"] == 1]
    latency = bs1_row.iloc[0]["latency_sec"] if not bs1_row.empty else -1.0
    peak_throughput = valid_df["throughput_items_per_sec"].max()
    # ピークの90%のスループットを最初に超えたバッチサイズ
    try:
        saturation_threshold = peak_throughput * 0.9
        saturation_row = valid_df[
            valid_df["throughput_items_per_sec"] >= saturation_threshold
        ].iloc[0]
        saturation_point = int(saturation_row["batch_size"])
    except IndexError:
        saturation_point = "N/A"

    # 最初の2点間のスループットの傾きで近似する
    if len(valid_df) >= 2:
        p1 = valid_df.iloc[0]
        p2 = valid_df.iloc[1]
        delta_throughput = (
            p2["throughput_items_per_sec"] - p1["throughput_items_per_sec"]
        )
        delta_batch = p2["batch_size"] - p1["batch_size"]
        efficiency_slope = delta_throughput / delta_batch if delta_batch > 0 else -1.0
    else:
        efficiency_slope = "N/A"

    return {
        "latency": latency,
        "peak_throughput": peak_throughput,
        "saturation_point": saturation_point,
        "efficiency_slope": efficiency_slope,
    }
