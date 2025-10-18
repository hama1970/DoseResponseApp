# Python 3.11/3.12 が安定（3.13はSciPy等が未対応）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# フォント（Matplotlib用）とビルド基本ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係を先に入れてレイヤーをキャッシュ
COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel && \
    pip install -r requirements.txt

# アプリ本体
COPY dose_response_app_v11.py .

EXPOSE 7860
CMD ["streamlit", "run", "dose_response_app_v11.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]
