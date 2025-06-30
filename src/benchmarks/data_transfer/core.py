import logging
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataTransferRunner:
    """
    data_transferベンチマークの実行を管理するクラス
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): benchmark_config.yamlから読み込まれた設定
        """
        self.config = config
        self.run_settings = config["run_settings"]
        self.device = torch.device(self.run_settings["device"])
        logger.info(f"DataTransferRunner initialized for device: '{self.device}'")

    def _measure_bandwidth(self, transfer_op: callable, size_mb: int) -> float:
        """
        スループット（帯域幅）測定パターン

        Args:
            transfer_op (callable): データ転送処理
            size_mb (int): 転送するデータサイズ (MB)

        Returns:
            float: 帯域幅 (GB/s)
        """
        # スループット（帯域幅）測定パターン
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)

        start_event.record()
        timed_runs = self.run_settings.get("timed_runs", 50)
        for _ in range(timed_runs):
            transfer_op()
        end_event.record()

        torch.cuda.synchronize(self.device)
        elapsed_time_ms = start_event.elapsed_time(end_event)

        # 1回あたりの平均時間（秒）
        avg_latency_sec = (elapsed_time_ms / 1000) / timed_runs

        # 帯域幅 (GB/s)
        return (size_mb / 1024) / avg_latency_sec if avg_latency_sec > 0 else -1.0

    def _measure_latency(self, transfer_op: callable, timed_runs: int) -> float:
        """
        レイテンシ測定パターン

        Args:
            transfer_op (callable): データ転送処理
            timed_runs (int): 測定回数

        Returns:
            float: レイテンシ (ms)
        """
        # レイテンシ測定パターン
        for _ in range(self.run_settings.get("warmup_runs", 5)):
            transfer_op()
        torch.cuda.synchronize(self.device)

        latencies_ms = []
        for _ in range(timed_runs):
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            transfer_op()
            end_event.record()

            torch.cuda.synchronize(self.device)
            latencies_ms.append(start_event.elapsed_time(end_event))

        return (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else -1.0

    def run(self) -> List[Dict[str, Any]]:
        """
        設定ファイルに基づき、全てのtransferベンチマークを実行し、生データリストを返す

        Returns:
            List[Dict[str, Any]]: _description_
        """
        logger.info("Running data_transfer benchmarks...")
        original_results_list = []
        group_config = next(
            (
                g
                for g in self.config["benchmark_groups"]
                if g["group_name"] == "data_transfer"
            ),
            None,
        )
        if not group_config or not group_config.get("enabled", False):
            return original_results_list

        benchmarks_to_run = group_config.get("benchmarks", [])
        total_steps = 0
        for b_conf in benchmarks_to_run:
            sizes = b_conf.get("parameters", {}).get("transfer_sizes_mb", [])
            if sizes:
                total_steps += len(sizes)  # 帯域幅測定の回数
                total_steps += 1  # レイテンシ測定の回数
        pbar = tqdm(
            total=len(benchmarks_to_run),
            desc="Running Data Transfer Benchmarks",
            mininterval=1,
            file=sys.stdout,
            leave=False,
        )

        for bench_conf in benchmarks_to_run:
            params = bench_conf.get("parameters", {})
            direction = params.get("direction", "H2D")
            use_pinned = params.get("use_pinned_memory", False)
            transfer_sizes = params.get("transfer_sizes_mb", [])

            base_info = {
                "group_name": "data_transfer",
                "benchmark_name": bench_conf["name"],
                "direction": direction,
                "use_pinned_memory": use_pinned,
            }

            # 帯域幅カーブの測定
            max_size_mb = max(transfer_sizes) if transfer_sizes else 0
            if max_size_mb > 0:
                max_elements = (max_size_mb * 1024 * 1024) // 4
                cpu_tensor_full = torch.randn(max_elements, dtype=torch.float32)
                if use_pinned:
                    cpu_tensor_full = cpu_tensor_full.pin_memory()
                gpu_tensor_full = torch.randn_like(cpu_tensor_full, device=self.device)

            for i, size_mb in enumerate(transfer_sizes):
                try:
                    num_elements = (size_mb * 1024 * 1024) // 4
                    cpu_tensor_slice = cpu_tensor_full[:num_elements]
                    gpu_tensor_slice = gpu_tensor_full[:num_elements]

                    transfer_op = (
                        (
                            lambda: gpu_tensor_slice.copy_(
                                cpu_tensor_slice, non_blocking=True
                            )
                        )
                        if direction == "H2D"
                        else (
                            lambda: cpu_tensor_slice.copy_(
                                gpu_tensor_slice, non_blocking=True
                            )
                        )
                    )

                    for _ in range(self.run_settings.get("warmup_runs", 5)):
                        transfer_op()

                    bandwidth = self._measure_bandwidth(transfer_op, size_mb)
                    original_results_list.append(
                        {
                            **base_info,
                            "transfer_size_mb": size_mb,
                            "metric_type": "bandwidth_gbps",
                            "value": bandwidth,
                            "error": None,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(
                        f"Error on '{bench_conf['name']}' "
                        f"bandwidth measurement (size={size_mb}MB)",
                        extra={"error": str(e)},
                    )
                    original_results_list.append(
                        {
                            **base_info,
                            "transfer_size_mb": size_mb,
                            "metric_type": "bandwidth_gbps",
                            "value": -1.0,
                            "error": str(e),
                        }
                    )
                    pbar.update(1)

                    # 残りのステップ数を計算
                    remaining_steps = len(transfer_sizes) - (i + 1)
                    if remaining_steps > 0:
                        pbar.update(remaining_steps)
                    logger.warning(
                        "CUDA OOM detected. Skipping larger sizes for this benchmark."
                    )
                    break

            # 小サイズレイテンシの測定
            if transfer_sizes:
                try:
                    smallest_size_mb = min(transfer_sizes)
                    num_elements = (smallest_size_mb * 1024 * 1024) // 4
                    cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
                    if use_pinned:
                        cpu_tensor = cpu_tensor.pin_memory()
                    gpu_tensor = torch.randn_like(cpu_tensor, device=self.device)
                    transfer_op = (
                        (lambda: gpu_tensor.copy_(cpu_tensor, non_blocking=True))
                        if direction == "H2D"
                        else (lambda: cpu_tensor.copy_(gpu_tensor, non_blocking=True))
                    )

                    latency_ms = self._measure_latency(
                        transfer_op, self.run_settings.get("timed_runs", 50)
                    )
                    original_results_list.append(
                        {
                            **base_info,
                            "transfer_size_mb": smallest_size_mb,
                            "metric_type": "latency_ms",
                            "value": latency_ms,
                            "error": None,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(
                        f"Error on '{bench_conf['name']}' "
                        f"latency measurement (size={min(transfer_sizes)}MB)",
                        extra={"error": str(e)},
                    )
                    original_results_list.append(
                        {
                            **base_info,
                            "transfer_size_mb": min(transfer_sizes),
                            "metric_type": "latency_ms",
                            "value": -1.0,
                            "error": str(e),
                        }
                    )
                    pbar.update(1)

        pbar.close()
        return original_results_list
