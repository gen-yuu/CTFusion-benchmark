import time
from io import BytesIO
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from PIL import Image


# Pool.mapに渡すため、この関数はクラスの外（トップレベル）に定義します
def _worker_process_image(task_args: Tuple[bytes, Tuple[int, int]]) -> bool:
    """
    1枚の画像をデコードし、前処理を行うワーカ関数
    """
    image_bytes, target_size = task_args
    try:
        # メモリ上のバイトデータから画像を読み込む
        with Image.open(BytesIO(image_bytes)) as img:
            img_resized = img.resize(target_size)
            img_array = np.array(img_resized, dtype=np.float32)
            img_array /= 255.0
        return True
    except Exception:
        # エラーが発生した場合は失敗として扱う
        return False


class ImageClassificationPreprocessing:
    """
    画像分類の前処理ベンチマークを管理するクラス。
    """

    def __init__(self, parameters: dict, run_settings: dict):
        self.params = parameters
        self.run_settings = run_settings
        self.is_setup = False
        self.dummy_image_bytes = None
        self.num_images = self.params.get("num_images", 1000)
        self.image_size = tuple(self.params.get("image_size", (224, 224)))

    def setup(self):
        """
        メモリ上に単一のダミーJPEG画像データを準備する
        """
        if self.is_setup:
            return

        dummy_array = np.uint8(np.random.rand(256, 256, 3) * 255)
        img = Image.fromarray(dummy_array)

        bytes_io = BytesIO()
        img.save(bytes_io, format="jpeg")
        self.dummy_image_bytes = bytes_io.getvalue()  # バイト列として保持
        self.is_setup = True

    def measure_throughput(self, n_workers: int) -> float:
        """
        指定されたワーカー数で前処理を実行し、スループット（images/sec）を返す。
        """
        self.setup()

        # 全ワーカーに同じ画像データを同じ回数だけ処理させるタスクリストを作成
        tasks = [(self.dummy_image_bytes, self.image_size)] * self.num_images

        # ウォームアップ
        warmup_tasks = tasks[: n_workers * 2]  # ワーカあたり2タスクでウォームアップ
        if warmup_tasks:
            with Pool(processes=n_workers) as pool:
                pool.map(_worker_process_image, warmup_tasks)

        # 時間測定
        start_time = time.perf_counter()

        with Pool(processes=n_workers) as pool:
            pool.map(_worker_process_image, tasks)

        end_time = time.perf_counter()

        total_time = end_time - start_time
        return self.num_images / total_time if total_time > 0 else -1.0
