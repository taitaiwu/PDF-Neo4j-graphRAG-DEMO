# PDF GraphRAG 測試工具

這是以 Gradio 製作的本機 PDF GraphRAG 工具，包含連線測試、PyMuPDF PDF 解析、chunk 參數與預覽、Schema 規劃、知識圖譜抽取、Neo4j 匯入及問答。實體與關係抽取後由使用者確認並手動匯入；問答頁使用原文 chunk、實體及關係的混合向量檢索與圖譜擴展，顯示答案及檢索證據表格。

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

### Docker：陌生環境從零部署

以下流程不需要先安裝 Python 或專案套件；Python 與相依套件會建置在 Docker 映像內。

#### 1. 安裝必要工具

安裝 Git 與 Docker Desktop（Windows／macOS），或 Docker Engine（Linux）。安裝完成後開啟 Terminal 或 PowerShell，確認 Docker daemon 正在運作：

```text
git --version
docker version
```

`docker version` 必須同時顯示 Client 與 Server；若只有 Client，請先啟動 Docker Desktop 或 Docker 服務。

#### 2. 取得專案

如果專案已放在 Git 倉庫，將下列網址與目錄名稱替換成實際值：

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

目前專案尚未設定公開 Git remote，因此文件無法提供固定 clone URL。也可以下載或複製完整專案資料夾，然後在 Terminal／PowerShell 進入包含 `Dockerfile` 的目錄。

#### 3. 建立本機設定與資料目錄

Linux／macOS：

```bash
cp .env.example .env
mkdir -p data
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force data
```

用文字編輯器開啟 `.env`，填寫 Neo4j、模型端點、模型名稱、密碼與 API key。請勿將 `.env` 提交或傳送給其他人。

如果 Neo4j 或本機模型服務執行在 Docker 主機，而不是另一個容器，容器內不能使用 `localhost`。請改用：

```env
NEO4J_URI="bolt://host.docker.internal:7687"
MODEL_API_BASE="http://host.docker.internal:11434/v1"
```

使用 OpenAI API 時，模型端點可維持 `https://api.openai.com/v1`，並設定 API key。

#### 4. 建置 Docker 映像

在包含 `Dockerfile` 的專案根目錄執行：

```bash
docker build -t pdf-graphrag:latest .
```

首次建置會下載 Python 基礎映像與套件，需要網路連線。完成時應看到映像名稱 `pdf-graphrag:latest`；可用以下指令確認：

```bash
docker image ls pdf-graphrag
```

#### 5. 啟動容器

Linux／macOS：

```bash
docker run -d \
  --name pdf-graphrag \
  --restart unless-stopped \
  -p 7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  -v ./.env:/app/.env \
  -v ./data:/app/data \
  pdf-graphrag:latest
```

Windows PowerShell：

```powershell
docker run -d `
  --name pdf-graphrag `
  --restart unless-stopped `
  -p 7860:7860 `
  --add-host=host.docker.internal:host-gateway `
  -v ./.env:/app/.env `
  -v ./data:/app/data `
  pdf-graphrag:latest
```

開啟 `http://127.0.0.1:7860`，先到「連線設定」分別測試 Neo4j 與模型服務。掛載 `.env` 可保留介面修改的連線設定，掛載 `data/` 則保存匯出與問答紀錄；刪除或重建容器不會刪除這兩個主機檔案。

#### 6. 查看狀態、日誌與停止服務

```bash
docker ps --filter name=pdf-graphrag
docker logs -f pdf-graphrag
docker stop pdf-graphrag
docker start pdf-graphrag
```

若容器啟動後立即停止，使用 `docker logs pdf-graphrag` 查看錯誤。若 7860 連接埠已被占用，可把啟動參數改為 `-p 8080:7860`，然後開啟 `http://127.0.0.1:8080`。

#### 7. 更新版本

Git 取得的專案可依序執行：

```bash
git pull
docker build -t pdf-graphrag:latest .
docker stop pdf-graphrag
docker rm pdf-graphrag
```

然後重新執行第 5 步。因為 `.env` 與 `data/` 位於主機，重建容器後設定與問答紀錄仍會保留。

#### 常見連線問題

- 容器中的 `localhost` 指向容器本身，不是 Docker 主機。主機服務請使用 `host.docker.internal`。
- Linux 若沒有加入 `--add-host=host.docker.internal:host-gateway`，可能無法解析主機名稱。
- 本機 Neo4j 或模型服務若只監聽 `127.0.0.1`，容器可能無法連入；需讓服務監聽 Docker 主機可存取的介面，並確認防火牆允許對應連接埠。
- `.env` 修改後可在介面按「重新讀取 .env」，或執行 `docker restart pdf-graphrag`。
- 若看到掛載權限錯誤，確認目前使用者可讀寫 `.env` 與 `data/`。

## 測試

```bash
. .venv/bin/activate
pytest
```

介面啟動時讀取 `.env`，欄位修改後會自動寫回，也可按「重新讀取 .env」。連線設定頁可分別按「測試 Neo4j 連線」確認指定 Database 可存取，以及按「測試模型服務連線」檢查 OpenAI-compatible `GET /models` 端點。Password 與 API key 會以明文保存在本機 `.env`；該檔案已排除於版本控制，請限制檔案權限並避免分享。

匯出的實驗設定保存在 `data/`，不包含密碼或 API key；`data/` 也不納入版本控制。

## 實體與關係規劃

完成 PDF chunk 預覽後，前往「建圖」頁操作。PDF 參數頁只保留頁碼範圍、chunk size 與 overlap，不再顯示建圖 LLM、Embedding、Temperature 或最大輸出 tokens，也不提供 chunk 關鍵字搜尋；獨立的匯入區可選擇後續向量建圖使用的 Embedding 模型。模型服務需提供 OpenAI-compatible `POST /chat/completions` API；API Base URL 例如 `http://localhost:11434/v1`。

1. 在建圖頁設定 Schema 規劃 LLM、Temperature、最大輸出 tokens（預設 4096）、Schema 粒度、實體／關係類型數量上限、最大並行請求數（預設 3），以及全部頁面或隨機抽取 N 頁的規劃範圍；生成參數會傳給每次模型 API 呼叫，粒度與數量上限則套用於候選規劃和每輪整合。按「分析文件並規劃 Schema」按鈕後，系統以每批最多約 30,000 字元分析所選範圍的 chunks，各規劃批次會受最大並行數限制而同時產生候選 Schema，待全部完成後，再將同一整合輪中的各組並行處理；每輪全部完成後才進入下一輪，整合組每組約 12,000 字元；中間結果只保留名稱、最多 120 字元的簡短說明，以及關係的來源／目標類型，最後去重並統一同義名稱後產生 JSON。畫面會顯示批次、已分析 chunk 數與整合進度；任一批失敗時會中止且不顯示不完整 Schema。
2. 在獨立的 Schema 規劃區檢查或修改 JSON；編輯器固定高度，內容超出時可使用水平與垂直捲動條。`entity_types` 與 `relationship_types` 必須是非空陣列，每一項必須有 `name`。
3. 到獨立的抽取區選擇知識圖譜抽取 LLM 與最大並行請求數，再按「確認 Schema 並抽取」：系統依確認後的類型受控並行讀取全部 chunks，等待所有批次回應後統一去重、整合並顯示實體與關係，同時列出來源 chunk 與 PDF 頁碼；畫面會依已完成的抽取批次顯示進度。確認抽取結果後，在獨立區塊選擇 Embedding 模型與匯入模式，再按「Embedding 並匯入 Neo4j」，才會以單一交易寫入「連線設定」頁指定的 Neo4j，並在該區塊顯示匯入數量或錯誤原因。

問答頁不要求先在本次工作階段建圖；送出問題時會直接連線到「連線設定」指定的 Neo4j，使用最近更新的 `GraphDocument`。向量 RAG 透過 Neo4j Vector Search 同時搜尋原文 chunk、實體與關係；GraphRAG 先取得 Top K 混合證據，再依命中的 chunk 與實體擴展同一建圖結果中的相關實體及關係。答案只能根據檢索證據生成，畫面會以表格顯示證據、來源頁碼、chunk 與相似度；每次結果另存於 `data/qa/`。

Chunks 與畫面抽取結果保存在本次 Gradio 頁面工作階段；匯入時會將原文 chunk、實體與關係證據寫入 Neo4j，每次生成使用獨立 `run_id`。Neo4j 以 `GraphDocument`、`ExtractedEntity` 節點及 `EXTRACTED_RELATION` 關係保存資料。匯入模式可選擇保留既有圖譜、取代最近一次圖譜，或清空本工具建立的所有圖譜；後兩者必須勾選刪除確認，清空不會刪除其他應用的節點。「Embedding 並匯入 Neo4j」會分批預先計算原文、實體與關係證據向量，建立 `GraphEvidence` 節點及 Neo4j Vector Index；問答時只計算問題向量。既有圖譜需重新匯入才會具有原文 chunk 向量。若 Neo4j 寫入失敗，畫面會保留已抽取結果並顯示錯誤。大型文件會產生多次 LLM API 呼叫，執行時間與費用取決於 chunk 數量及所選模型。

### Schema JSON 解析失敗

模型回覆若在 JSON 前後加入說明文字或不完整的 Markdown code fence，系統會嘗試從內容中擷取第一個完整 JSON 物件。若 JSON 無法解析，或雖可解析但缺少非空的 `entity_types`、`relationship_types` 及必要的 `name`，會把實際驗證錯誤提供給模型，並以 Temperature 0 自動修正一次。第二次仍失敗時才中止該批或合併組；若 API 的 `finish_reason` 顯示輸出長度截斷，狀態會提示提高「最大輸出 tokens」或減少 Schema 類型數量。
