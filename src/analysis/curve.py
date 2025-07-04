import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_gpu_compute_features(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    単一の性能曲線データ（DataFrame）から4つの指標を計算する

    Args:
        group_df (pd.DataFrame): 'benchmark_name'と'data_type'でグループ化されたDataFrame。
            'batch_size', 'latency_ms', 'throughput_items_per_sec'カラムを持つ

    Returns:
        Dict[str, Any]: 計算された4指標を含む辞書。
            - "latency": レイテンシ（`batch_size=1`での実行時間）
            - "peak_throughput": ピークスループット（性能曲線の最大値）
            - "saturation_point": 飽和点（ピーク性能の90%に達するバッチサイズ）
            - "efficiency_slope": 効率の傾き（性能曲線の立ち上がり具合）
    """
    # エラーのあった行は除外して計算
    valid_df = group_df[group_df["throughput_items_per_sec"] >= 0].copy()
    valid_df = valid_df.sort_values(by="batch_size").reset_index()

    # --- ストレステスト（データが1点）の場合の特別処理 ---
    if len(valid_df) <= 1:
        if len(valid_df) == 1:
            latency = valid_df.iloc[0]["latency_ms"]
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
    # レイテンシ (latency)
    bs1_row = valid_df[valid_df["batch_size"] == 1]
    latency = bs1_row.iloc[0]["latency_ms"] if not bs1_row.empty else -1.0
    # ピークスループット (peak_throughput)
    peak_throughput = valid_df["throughput_items_per_sec"].max()

    # 飽和点 (saturation_point)
    try:
        saturation_threshold = peak_throughput * 0.9
        saturation_row = valid_df[
            valid_df["throughput_items_per_sec"] >= saturation_threshold
        ].iloc[0]
        saturation_point = int(saturation_row["batch_size"])
    except IndexError:
        saturation_point = "N/A"

    # 効率の傾き (efficiency_slope)
    efficiency_slope = "N/A"
    if len(valid_df) >= 2:
        try:
            # ピークスループットを最初に達成した点のインデックス位置を取得
            # idxmax() はインデックス"ラベル"を返すので、.index.get_loc() で整数位置に変換
            peak_idx_label = valid_df["throughput_items_per_sec"].idxmax()
            peak_idx_pos = valid_df.index.get_loc(peak_idx_label)

            # 最初の点からピークの点までの全てのデータ点を回帰の対象とする
            regression_df = valid_df.iloc[0 : peak_idx_pos + 1]

            # 回帰には最低2つの異なる点が必要
            if (
                len(regression_df) >= 2
                and len(regression_df["batch_size"].unique()) >= 2
            ):
                x = regression_df["batch_size"]
                y = regression_df["throughput_items_per_sec"]

                # 線形回帰で傾きを計算
                slope, _ = np.polyfit(x, y, 1)
                efficiency_slope = slope
            else:
                # ピークが最初の点だった場合は、傾きは実質的に0
                efficiency_slope = 0.0

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(
                "Could not calculate efficiency_slope due to a numerical error.",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )
            efficiency_slope = "N/A"

    return {
        "latency": latency,
        "peak_throughput": peak_throughput,
        "saturation_point": saturation_point,
        "efficiency_slope": efficiency_slope,
    }


def calculate_data_transfer_features(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    データ転送の生データから特徴量を抽出する。

    Args:
        group_df (pd.DataFrame): 'benchmark_name'でグループ化されたDataFrame。
                                 'metric_type', 'value'カラムを持つ。

    Returns:
        Dict[str, Any]: 抽出された2つの特徴量を含む辞書。
        - "peak_bandwidth_gbps": ピーク帯域幅 (GB/s)
        - "small_transfer_latency_ms": 小サイズレイテンシ (ms)
    """
    # エラーのある行は除外
    valid_df = group_df[group_df["error"].isnull()].copy()
    if valid_df.empty:
        return {"peak_bandwidth_gbps": -1.0, "small_transfer_latency_ms": -1.0}

    # ピーク帯域幅の抽出
    bandwidth_df = valid_df[valid_df["metric_type"] == "bandwidth_gbps"]
    peak_bandwidth_gbps = (
        bandwidth_df["value"].max() if not bandwidth_df.empty else -1.0
    )

    # 小サイズレイテンシの抽出
    latency_df = valid_df[valid_df["metric_type"] == "latency_ms"]
    small_transfer_latency_ms = (
        latency_df.iloc[0]["value"] if not latency_df.empty else -1.0
    )

    return {
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
        "small_transfer_latency_ms": small_transfer_latency_ms,
    }


def calculate_host_compute_features(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Host computeの生データから3つの特徴量を抽出する

    Args:
        group_df (pd.DataFrame): 'benchmark_name'でグループ化されたDataFrame
                                 'num_workers', 'value'カラムを持つ

    Returns:
        Dict[str, Any]: 抽出された3つの特徴量を含む辞書
        - "peak_host_throughput": ピークスループット
        - "single_worker_throughput": 単一ワーカー性能
        - "scalability_factor": スケイラビリティ係数
    """
    # エラーのある行（スループットが負の値）は除外
    valid_df = group_df[group_df["value"] >= 0].copy()
    if valid_df.empty:
        return {
            "peak_host_throughput": -1.0,
            "single_worker_throughput": -1.0,
            "scalability_factor": -1.0,
        }

    # ピークスループット (peak_host_throughput)
    peak_throughput = valid_df["value"].max()

    # 単一ワーカー性能 (single_worker_throughput)
    single_worker_row = valid_df[valid_df["num_workers"] == 1]
    if not single_worker_row.empty:
        single_worker_throughput = single_worker_row.iloc[0]["value"]
    else:
        # num_workers=1 のデータがない場合は -1 とする
        single_worker_throughput = -1.0

    # スケーラビリティ係数 (scalability_factor)
    if single_worker_throughput > 0:
        scalability_factor = peak_throughput / single_worker_throughput
    else:
        scalability_factor = 0.0

    return {
        "peak_host_throughput": peak_throughput,
        "single_worker_throughput": single_worker_throughput,
        "scalability_factor": scalability_factor,
    }
