# タグは、CUDAバージョン、cuDNNバージョン、OS、Pythonバージョンを考慮して選択する。
# 例: CUDA 12.1, cuDNN 8, Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Pythonの出力をバッファリングしないように設定。
ENV PYTHONUNBUFFERED 1
# apt-getが対話的なプロンプトで停止しないように設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
# Pythonの文字化けを防ぐ
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-venv \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

# CMD ["python", "run_benchmark.py"]