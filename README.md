# PDF GraphRAG 測試工具

這是 Gradio 初版介面，已包含連線設定、PDF 文字解析、chunk 參數與預覽、JSON 設定匯出，以及建圖／問答的操作入口。Neo4j 寫入與 GraphRAG 後端會在下一階段接入。

## 安裝與啟動

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

瀏覽器會開啟 `http://127.0.0.1:7860`。應用程式不建立公開分享網址。

## 測試

```bash
. .venv/bin/activate
pytest
```

匯出的設定保存在 `data/`，此目錄不納入版本控制。密碼與 API key 不會寫入設定 JSON。
