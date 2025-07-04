import logging
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from .layers import LayerFactory

logger = logging.getLogger(__name__)


class GpuComputeRunner:
    """
    gpu_computeベンチマークの実行を管理するクラス。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): benchmark_config.yamlから読み込まれた設定
        """
        self.config = config
        self.run_settings = config["run_settings"]
        self.device = torch.device(self.run_settings["device"])
        self.layer_factory = LayerFactory()
        logger.info(f"GpuComputeRunner initialized for device: '{self.device}'")

    def run(self) -> List[Dict[str, Any]]:
        """
        設定ファイルに基づき、全てのGPUベンチマークを実行し、生データリストを返す

        Returns:
            List[Dict[str, Any]]: 全ての測定結果（original_results_list）
        """
        logger.info("Running gpu_compute benchmarks...")
        original_results_list = []

        group_config = next(
            (
                g
                for g in self.config["benchmark_groups"]
                if g["group_name"] == "gpu_compute"
            ),
            None,
        )

        if not group_config or not group_config["enabled"]:
            logger.warning(
                "gpu_compute benchmark group is not defined or disabled. Skipping."
            )
            return original_results_list

        benchmarks_to_run = group_config.get("benchmarks", [])

        # プログレスバーの準備
        total_runs = sum(
            len(b.get("batch_sizes", self.run_settings["batch_sizes"]))
            * len(b["data_types"])
            for b in benchmarks_to_run
        )
        pbar = tqdm(
            total=total_runs,
            desc="Running GPU Compute Benchmarks",
            mininterval=3,
            file=sys.stdout,
            leave=False,
        )

        for bench_conf in benchmarks_to_run:
            for dtype_str in bench_conf["data_types"]:
                # ベンチマーク固有のbatch_sizesがあればそれを使い、なければグローバル設定を使う
                loop_batch_sizes = bench_conf.get(
                    "batch_sizes", self.run_settings["batch_sizes"]
                )

                logger.debug(
                    f"Running {bench_conf['name']} with batch sizes: {loop_batch_sizes}"
                )
                for batch_size in loop_batch_sizes:
                    result = self._measure_single_run(
                        bench_conf,
                        dtype_str,
                        batch_size,
                    )
                    original_results_list.append(result)
                    pbar.update(1)

                    # エラーが発生した場合、この曲線の残りのバッチサイズはスキップ
                    if result["error"] is not None:
                        # 残りのバッチサイズ分、プログレスバーを進めておく
                        remaining_batches = (
                            len(loop_batch_sizes)
                            - loop_batch_sizes.index(batch_size)
                            - 1
                        )
                        pbar.update(remaining_batches)
                        break

        pbar.close()
        logger.info("gpu_compute benchmarks completed.")
        return original_results_list

    def _measure_single_run(
        self, bench_conf: Dict[str, Any], dtype_str: str, batch_size: int
    ) -> Dict[str, Any]:
        """
        1つのベンチマーク設定・データ型・バッチサイズの組み合わせで測定を行う

        Args:
            bench_conf (Dict[str, Any]): ベンチマーク設定
            dtype_str (str): データ型 (fp16 or fp32)
            batch_size (int): バッチサイズ

        Returns:
            Dict[str, Any]: 測定結果
        """
        dtype = torch.float16 if dtype_str == "fp16" else torch.float32

        result_dict = {
            "group_name": "gpu_compute",
            "benchmark_name": bench_conf["name"],
            "data_type": dtype_str,
            "batch_size": batch_size,
            "latency_ms": -1.0,
            "throughput_items_per_sec": -1.0,
            "error": None,
        }

        try:
            # LayerFactoryからモジュールとダミーデータを生成
            module, inputs = self.layer_factory.create(
                layer_type=bench_conf["layer_type"],
                parameters=bench_conf["parameters"],
                batch_size=batch_size,
                dtype=dtype,
                device=self.device,
            )

            # ウォームアップ
            warmup_runs = self.run_settings.get("warmup_runs", 5)
            for _ in range(warmup_runs):
                _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)

            # GPUの非同期性を考慮
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize(self.device)  # 測定前に同期
            start_event.record()
            timed_runs = self.run_settings.get("timed_runs", 20)
            for _ in range(timed_runs):
                _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)

            end_event.record()
            torch.cuda.synchronize(self.device)  # 測定完了を待つ

            # 経過時間をミリ秒で取得し、秒に変換
            elapsed_time_ms = start_event.elapsed_time(end_event)

            # 1回あたりのレイテンシとスループットを計算
            latency_ms = elapsed_time_ms / self.run_settings["timed_runs"]
            latency_sec = latency_ms / 1000
            throughput = batch_size / latency_sec

            result_dict.update(
                {
                    "latency_ms": latency_ms,
                    "throughput_items_per_sec": throughput,
                }
            )

        except Exception as e:
            logger.info(
                f"Limit reached for '{bench_conf['name']}' "
                f"({dtype_str}, bs={batch_size}). This is an expected boundary.",
                extra={"error": str(e)},
            )
            result_dict["error"] = str(e)
            # CUDAメモリをクリア(OOM後の回復のため)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result_dict
