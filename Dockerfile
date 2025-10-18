# ---- 固定したい Python を選ぶ（例: 3.11）----
FROM python:3.11-slim

# 推奨の環境設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依存のビルドに最低限あると安心なパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python 依存をインストール
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリ本体
COPY . .

# Render は実行時に $PORT をセットするのでそれを使って起動
# JSON 形式の CMD だと $PORT 展開されないため bash -lc で包みます
CMD ["bash", "-lc", "streamlit run dose_response_app_v11.py --server.address=0.0.0.0 --server.port=$PORT --browser.gatherUsageStats=false"]