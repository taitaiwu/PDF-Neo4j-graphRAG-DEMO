# PDF GraphRAG 測試工具

這是 Gradio 初版介面，已包含連線設定、PyMuPDF PDF 解析、chunk 參數與預覽、JSON 設定匯出，以及建圖／問答的操作入口。實體與關係抽取後可自動匯入 Neo4j；GraphRAG 問答後端仍待後續接入。

## PDF 解析

PDF 會由原生 PyMuPDF 逐頁擷取排序後的純文字，保留來源頁碼並回報無文字頁面。擷取時預設排除每頁上方與下方各 8% 區域，避免常見頁首、頁尾及頁碼進入內容；特殊排版若將正文放在此區域也會被排除，因此仍需檢查預覽結果。現階段不使用 OCR，僅支援具有文字層的 PDF；掃描型文件仍會顯示無法解析提示。

上傳 PDF 後可設定「解析起始頁」與「解析結束頁」；系統會先讀取總頁數，將結束頁自動設為文件最後一頁，使用者仍可縮小範圍。PyMuPDF 只處理指定範圍，並驗證起訖順序及頁碼是否超出文件範圍。

解析結果會依 `chunk_size` 以固定字元長度切分，相鄰 chunk 依 `chunk_overlap` 保留重疊文字；跨頁 chunk 會保留涉及的所有來源頁碼。

解析完成後可使用頁碼滑桿或「上一頁／下一頁」逐頁檢視所選範圍內的所有相關 chunk。預覽表會顯示來源頁碼、字元數與內容；預覽區採滿版寬度配置，表格最大高度為 750px。

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

完成 PDF chunk 預覽後，前往「建圖」頁操作。PDF 參數頁只保留頁碼範圍、chunk size 與 overlap，不再顯示建圖 LLM、Embedding、Temperature 或最大輸出 tokens，也不提供 chunk 關鍵字搜尋；抽取區可另外選擇後續向量建圖使用的 Embedding 模型。模型服務需提供 OpenAI-compatible `POST /chat/completions` API；API Base URL 例如 `http://localhost:11434/v1`。

1. 在建圖頁設定 Schema 規劃／抽取 LLM、Temperature 與最大輸出 tokens；這兩項生成參數會實際傳給每次模型 API 呼叫。按「分析文件並規劃 Schema」後，系統以每批最多約 30,000 字元分析所有 chunks，各批先產生候選 Schema，再以多輪階層式合併去重、統一同義名稱並產生最終 JSON。畫面會顯示批次、已分析 chunk 數與整合進度；任一批失敗時會中止且不顯示不完整 Schema。
2. 在獨立的 Schema 規劃區檢查或修改 JSON；編輯器固定高度，內容超出時可使用水平與垂直捲動條。`entity_types` 與 `relationship_types` 必須是非空陣列，每一項必須有 `name`。
3. 到獨立的抽取區按「確認 Schema 並生成」：系統依確認後的類型批次讀取全部 chunks，抽取、去重並顯示實體與關係，同時列出來源 chunk 與 PDF 頁碼。抽取完成後會自動以單一交易匯入「連線設定」頁指定的 Neo4j。

Chunks、Schema 與畫面抽取結果仍只保存在本次 Gradio 頁面工作階段；實體與關係會寫入 Neo4j，每次生成使用獨立 `run_id`。Neo4j 以 `GraphDocument`、`ExtractedEntity` 節點及 `EXTRACTED_RELATION` 關係保存資料。Embedding 模型選擇會記入文件節點與結果 state，實際向量生成仍屬後續階段。若 Neo4j 寫入失敗，畫面會保留已抽取結果並顯示錯誤。大型文件會產生多次 LLM API 呼叫，執行時間與費用取決於 chunk 數量及所選模型。

### Schema JSON 解析失敗

模型回覆若在 JSON 前後加入說明文字或不完整的 Markdown code fence，系統會嘗試從內容中擷取第一個完整 JSON 物件。若仍無法解析，會自動以 Temperature 0 要求同一模型修正一次。第二次仍失敗時才中止該批或合併組；若 API 的 `finish_reason` 顯示輸出長度截斷，狀態會提示提高「最大輸出 tokens」或減少 Schema 類型數量。
