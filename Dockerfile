FROM python:3.11-slim

# minimal build deps & fonts so matplotlibが文字化けしない
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係を先に入れてキャッシュを効かせる
COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel && \
    pip install -r requirements.txt

# アプリ本体
COPY dose_response_app_v11.py .

# RenderはPORT環境変数を渡すので、$PORT をそのまま使う
CMD ["sh", "-c", "streamlit run dose_response_app_v11.py --server.port $PORT --server.address 0.0.0.0"]
