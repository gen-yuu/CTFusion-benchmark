import logging
import os
from typing import Any, Dict, List

import psutil
from tqdm import tqdm

from .factory import create_host_benchmark

logger = logging.getLogger(__name__)


class HostComputeRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_settings = config["run_settings"]
        logger.info("HostComputeRunner initialized.")

    def _generate_auto_worker_list(self) -> List[int]:
        """
        物理コア数に基づき、テストするワーカー数のリストを動的に生成する。
        例: 8コアなら [1, 2, 4, 8]

        Returns:
            List[int]: ワーカー数のリスト
        """
        max_workers = 1

        try:
            # ハイパースレッディングを含まない物理コア数を取得
            max_workers = psutil.cpu_count(logical=False)
            logger.info(f"Physical core count detected: {max_workers}")
        except Exception:
            max_workers = os.cpu_count()
            logger.warning(
                f"Failed to get physical core count. Falling back \
                    to logical cores: {max_workers}"
            )

        # 2のべき乗のリストを作成
        worker_list = [1]
        while worker_list[-1] * 2 <= max_workers:
            worker_list.append(worker_list[-1] * 2)

        # 最大コア数がリストに含まれていなければ追加
        if max_workers not in worker_list:
            worker_list.append(max_workers)

        logger.info(f"Generated auto worker list: {worker_list}")
        return worker_list

    def run(self) -> List[Dict[str, Any]]:
        """
        設定ファイルに基づき、全てのhost_computeベンチマークを実行し、生データリストを返す

        Returns:
            List[Dict[str, Any]]: 全ての測定結果(original_results_list)
        """
        original_results_list = []
        group_config = next(
            (
                g
                for g in self.config["benchmark_groups"]
                if g["group_name"] == "host_compute"
            ),
            None,
        )
        if not group_config or not group_config.get("enabled", False):
            return original_results_list

        benchmarks_to_run = group_config.get("benchmarks", [])
        pbar = tqdm(
            total=len(benchmarks_to_run), desc="Running Host Compute Benchmarks"
        )

        for bench_conf in benchmarks_to_run:
            params = bench_conf.get("parameters", {})
            domain = params.get("domain")
            num_workers_config = params.get("num_workers", [1])
            if num_workers_config == "auto":
                num_workers_list = self._generate_auto_worker_list()
            elif isinstance(num_workers_config, list):
                num_workers_list = num_workers_config
            else:
                logger.warning(
                    f"Invalid num_workers_list format: {num_workers_config}."
                )
                num_workers_list = [1]

            base_info = {
                "group_name": "host_compute",
                "benchmark_name": bench_conf["name"],
                "domain": domain,
            }

            try:
                benchmark_instance = create_host_benchmark(
                    domain, params, self.run_settings
                )

                for n_workers in num_workers_list:
                    logger.info(
                        f"Running {bench_conf['name']} with {n_workers} workers..."
                    )
                    throughput = benchmark_instance.measure_throughput(
                        n_workers=n_workers
                    )
                    original_results_list.append(
                        {
                            **base_info,
                            "num_workers": n_workers,
                            "metric_type": "throughput_images_per_sec",
                            "value": throughput,
                            "error": None,
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Error on host benchmark '{bench_conf['name']}'", exc_info=True
                )
                original_results_list.append({**base_info, "error": str(e)})

            pbar.update(1)
        pbar.close()
        return original_results_list
