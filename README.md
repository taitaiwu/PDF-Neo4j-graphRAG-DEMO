# PDF GraphRAG 測試工具

這是 Gradio 初版介面，已包含連線設定、PyMuPDF4LLM PDF 解析、chunk 參數與預覽、JSON 設定匯出，以及建圖／問答的操作入口。Neo4j 寫入與 GraphRAG 後端會在下一階段接入。

## PDF 解析

PDF 會由 PyMuPDF4LLM 逐頁轉換成適合 RAG 使用的 Markdown，保留來源頁碼並回報無文字頁面。現階段停用 OCR，因此僅支援具有文字層的 PDF；掃描型文件仍會顯示無法解析提示。

解析完成後可使用頁碼滑桿或「上一頁／下一頁」逐頁檢視所有相關 chunk。預覽區採滿版寬度配置，表格最大高度為 750px，並提供內容搜尋。

加密 PDF 不受支援。PyMuPDF4LLM 採 GNU AGPL v3／商業雙重授權，發布或商業使用本專案前應確認所採授權符合使用情境。

## 安裝與啟動

### Linux 一鍵啟動

在 Terminal 執行：

```bash
./start.sh
```

腳本會自動建立虛擬環境、安裝必要套件、建立本機 `.env` 設定檔，然後啟動網站。首次啟動需要網路下載套件，並需先安裝 Python 3.11 以上版本與 `venv` 模組。

### 手動啟動

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python src/app.py
```

瀏覽器會開啟 `http://127.0.0.1:7860`。應用程式不建立公開分享網址。

## 測試

```bash
. .venv/bin/activate
pytest
```

介面啟動時讀取 `.env`，欄位修改後會自動寫回，也可按「重新讀取 .env」。Password 與 API key 會以明文保存在本機 `.env`；該檔案已排除於版本控制，請限制檔案權限並避免分享。

匯出的實驗設定保存在 `data/`，不包含密碼或 API key；`data/` 也不納入版本控制。
