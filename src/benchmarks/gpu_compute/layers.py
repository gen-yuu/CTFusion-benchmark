import logging
from typing import Any, Callable, Dict, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LayerFactory:
    """
    YAML設定に基づき、ベンチマーク対象のPyTorchモジュールとダミー入力データを生成するファクトリクラス
    """

    @staticmethod
    def create(
        layer_type: str,
        parameters: Dict[str, Any],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Callable, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        指定されたパラメータから、呼び出し可能なモジュールと入力を生成する

        Args:
            layer_type (str): 'MatMul', 'Conv2d' など、YAMLで定義されたレイヤタイプ
            parameters (Dict[str, Any]): YAMLのparametersセクション
            batch_size (int): 現在測定中のバッチサイズ
            dtype (torch.dtype): torch.float32 または torch.float16
            device (torch.device): 'cuda' デバイスオブジェクト

        Returns:
            Tuple[Callable, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
                - callable_module: 実行時間を測定する対象のモジュールまたは関数
                - inputs: モジュールに渡す入力データ（単一テンソルまたはタプル）
        """
        try:
            logger.debug(
                f"Creating layer '{layer_type}'"
                f"with batch_size={batch_size}, dtype={dtype}"
            )

            if layer_type == "MatMul":
                variant = parameters.get("variant")

                if variant == "StressTest_2D":
                    # batch_sizeは1
                    shape_a = parameters["shape_a"]
                    shape_b = parameters["shape_b"]
                    input_a = torch.randn(*shape_a, device=device, dtype=dtype)
                    input_b = torch.randn(*shape_b, device=device, dtype=dtype)

                    logger.debug("Creating 2D MatMul (StressTest) using torch.mm")
                    module = lambda a, b: torch.mm(a, b)  # noqa: E731

                    return module, (input_a, input_b)

                elif variant == "Batched_3D":
                    # batch_sizeを先頭に追加して3Dテンソルを生成
                    shape_a = [batch_size] + parameters["shape_a"]
                    shape_b = [batch_size] + parameters["shape_b"]
                    input_a = torch.randn(*shape_a, device=device, dtype=dtype)
                    input_b = torch.randn(*shape_b, device=device, dtype=dtype)

                    logger.debug("Creating 3D MatMul (Batched) using torch.bmm")
                    module = lambda a, b: torch.bmm(a, b)  # noqa: E731

                    return module, (input_a, input_b)

                else:
                    logger.warning(f"Unknown MatMul variant: {variant}")
                    raise ValueError

            elif layer_type == "Add":
                shape = [batch_size] + parameters["shape"]
                input_tensor = torch.randn(*shape, device=device, dtype=dtype)
                module = lambda x: x + x  # noqa: E731
                return module, input_tensor

            elif layer_type == "Conv2d":
                # ユーザーの決定通り、in_channelsはYAMLから直接読み取る
                c_in, h, w = parameters["input_shape"]
                input_tensor = torch.randn(
                    batch_size, c_in, h, w, device=device, dtype=dtype
                )
                module = nn.Conv2d(
                    in_channels=parameters["in_channels"],
                    out_channels=parameters["out_channels"],
                    kernel_size=parameters["kernel_size"],
                    stride=parameters["stride"],
                    padding=parameters["padding"],
                ).to(
                    device=device,
                    dtype=dtype,
                )
                return module, input_tensor

            elif layer_type == "BatchNorm2d":
                c, h, w = parameters["input_shape"]
                input_tensor = torch.randn(
                    batch_size, c, h, w, device=device, dtype=dtype
                )
                module = nn.BatchNorm2d(num_features=parameters["num_features"]).to(
                    device=device,
                    dtype=dtype,
                )
                return module, input_tensor

            elif layer_type == "LayerNorm":
                shape = [batch_size] + parameters["input_shape"]
                input_tensor = torch.randn(*shape, device=device, dtype=dtype)

                module = nn.LayerNorm(
                    normalized_shape=parameters["normalized_shape"],
                ).to(
                    device=device,
                    dtype=dtype,
                )
                return module, input_tensor

            elif layer_type == "Softmax":
                shape = [batch_size] + parameters["input_shape"]
                input_tensor = torch.randn(*shape, device=device, dtype=dtype)

                module = nn.Softmax(dim=parameters.get("dim", -1)).to(
                    device=device,
                    dtype=dtype,
                )
                return module, input_tensor

            else:
                # 未知のレイヤタイプが指定された場合はエラーを送出
                logger.warning(f"Unknown layer type: {layer_type}")
                raise ValueError
        except Exception as e:
            logger.error(
                f"Failed to create layer '{layer_type}'",
                extra={
                    "error": str(e),
                    "parameters": parameters,
                },
                exc_info=True,
            )
            raise
