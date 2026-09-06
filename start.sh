#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] 檢查 Python 虛擬環境..."
if [[ ! -x ".venv/bin/python" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[錯誤] 找不到 python3，請先安裝 Python 3.11 以上版本。" >&2
        exit 1
    fi
    python3 -m venv .venv
fi

echo "[2/4] 檢查並安裝必要套件..."
if ! ".venv/bin/python" -m pip install -r requirements.txt; then
    echo "[錯誤] 套件安裝失敗，請檢查網路連線與上方錯誤訊息。" >&2
    exit 1
fi

echo "[3/4] 檢查環境設定..."
if [[ ! -f ".env" ]]; then
    cp ".env.example" ".env"
fi

echo "[4/4] 啟動網站..."
echo "網址：http://127.0.0.1:7860"
echo "按 Ctrl+C 可停止服務。"
exec ".venv/bin/python" src/app.py
