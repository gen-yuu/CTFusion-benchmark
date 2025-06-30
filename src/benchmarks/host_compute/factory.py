from typing import Any, Dict

from .domains.image_processing import ImageClassificationPreprocessing


def create_host_benchmark(
    domain_name: str, parameters: Dict[str, Any], run_settings: Dict[str, Any]
):
    """
    ホストコンピューティングベンチマークを作成する

    Args:
        domain_name (str): ベンチマークのドメイン名
        parameters (Dict[str, Any]): ベンチマークのパラメータ
        run_settings (Dict[str, Any]): ベンチマークの実行設定
    """
    if domain_name == "image_classification":
        return ImageClassificationPreprocessing(parameters, run_settings)
    else:
        raise ValueError(f"Unknown host benchmark domain: {domain_name}")
