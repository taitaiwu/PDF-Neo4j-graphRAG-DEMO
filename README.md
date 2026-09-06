# PDF GraphRAG 測試工具

這是 Gradio 初版介面，已包含連線設定、PyMuPDF PDF 解析、chunk 參數與預覽、JSON 設定匯出，以及建圖／問答的操作入口。Neo4j 寫入與 GraphRAG 後端會在下一階段接入。

## PDF 解析

PDF 會由原生 PyMuPDF 逐頁擷取排序後的純文字，保留來源頁碼並回報無文字頁面。擷取時預設排除每頁上方與下方各 8% 區域，避免常見頁首、頁尾及頁碼進入內容；特殊排版若將正文放在此區域也會被排除，因此仍需檢查預覽結果。現階段不使用 OCR，僅支援具有文字層的 PDF；掃描型文件仍會顯示無法解析提示。

上傳 PDF 後可設定「解析起始頁」與「解析結束頁」；系統會先讀取總頁數，將結束頁自動設為文件最後一頁，使用者仍可縮小範圍。PyMuPDF 只處理指定範圍，並驗證起訖順序及頁碼是否超出文件範圍。

解析結果會依 `chunk_size` 以固定字元長度切分，相鄰 chunk 依 `chunk_overlap` 保留重疊文字；跨頁 chunk 會保留涉及的所有來源頁碼。

解析完成後可使用頁碼滑桿或「上一頁／下一頁」逐頁檢視所選範圍內的所有相關 chunk。預覽表會顯示來源頁碼、字元數與內容；預覽區採滿版寬度配置，表格最大高度為 750px，並提供內容搜尋。

加密 PDF 不受支援。PyMuPDF 採 GNU AGPL v3／商業雙重授權，發布或商業使用本專案前應確認所採授權符合使用情境。

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

## 實體與關係規劃

完成 PDF chunk 預覽後，前往「建圖」頁選擇抽取 LLM 與後續向量建圖使用的 Embedding 模型。模型服務需提供 OpenAI-compatible `POST /chat/completions` API；API Base URL 例如 `http://localhost:11434/v1`。

1. 按「分析文件並規劃 Schema」：系統在 30,000 字元預算內均勻抽取文件前、中、後段 chunks，產生實體類型與關係類型 JSON。
2. 在 Schema 編輯器檢查或修改 JSON；`entity_types` 與 `relationship_types` 必須是非空陣列，每一項必須有 `name`。
3. 按「確認 Schema 並生成」：系統依確認後的類型批次讀取全部 chunks，抽取、去重並顯示實體與關係，同時列出來源 chunk 與 PDF 頁碼。

目前抽取結果與 chunks 都只保存在本次 Gradio 頁面工作階段，尚未寫入 Neo4j 或磁碟；Embedding 模型選擇會記入結果 state，實際向量生成與 Neo4j 寫入仍屬後續階段。大型文件會產生多次 LLM API 呼叫，執行時間與費用取決於 chunk 數量及所選模型。
