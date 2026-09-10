# PDF Neo4j GraphRAG Demo

這是一套將 PDF 文件轉換為 Neo4j 知識圖譜，並透過混合式 GraphRAG 進行問答的本機 Web 工具。介面使用 Gradio，支援 OpenAI 相容的模型服務，可自行設定建圖、Embedding 與回答模型。

> 本專案目前定位為單機、單使用者 Demo，不會建立 Gradio 公開分享網址。

## 目錄

- [專案介紹](#專案介紹)
- [實作原理](#實作原理)
- [基本使用流程](#基本使用流程)
- [Docker：陌生環境從零部署](#docker陌生環境從零部署)
- [部署後如何啟動](#部署後如何啟動)
- [常見問題](#常見問題)
- [非 Docker 啟動方式](#非-docker-啟動方式)
- [測試](#測試)
- [限制與注意事項](#限制與注意事項)

## 專案介紹

本專案提供以下功能：

- 建立與載入獨立專案工作區，保存連線、API、模型、處理參數、Chunk、文件、建圖狀態與問答紀錄。
- 上傳與預覽含文字層的 PDF。
- 從全文或隨機抽取 N 頁規劃實體、關係 Schema。
- 以多請求並行方式抽取知識圖譜，整批完成後再統一去重整合。
- 將文件片段、實體與關係建立 Embedding，寫入 Neo4j 並建立向量與全文索引。
- 支援保留既有資料、取代目前圖譜，以及清空本工具建立的圖譜。
- 使用 Neo4j 官方 HybridCypherRetriever 執行向量與全文混合搜尋，並結合圖譜擴展及原文片段組成 GraphRAG 問答內容。
- 在介面中測試 Neo4j 與模型服務連線。

## 實作原理

### 建圖流程

1. 使用 PyMuPDF 讀取 PDF 文字，並排除每頁頂部與底部約 8% 的常見頁首頁尾區域。
2. 將文字切成可重疊的片段；預設片段大小為 1500 字元、重疊 200 字元，並保留頁碼。
3. 將文件分批送往建圖模型規劃 Schema。各批可同時執行，全部完成後再分層整合。
4. 依確認後的 Schema 並行抽取實體與關係，等待整批完成後統一正規化、去重及整合。
5. 分批建立原文片段、實體與關係的向量，寫入 Neo4j，並建立向量索引。

### 問答流程

送出問題時只會對「問題」建立 Embedding，不會重新計算整個圖譜的向量。系統會：

1. 使用問題向量搜尋相關原文、實體與關係。
2. 使用問題文字從 CJK 全文索引搜尋精確詞彙、錯誤碼與實體名稱。
3. 由 Neo4j 官方 HybridCypherRetriever 使用內建 naive ranker 正規化並融合兩組結果，再選出 Top K。
4. GraphRAG 模式再從命中的節點向外擴展相關圖譜內容，並回查 PDF 原文片段。
5. 將問題、圖譜內容及原文證據交給回答模型生成答案。

兩個問答模式都會使用官方 HybridCypherRetriever 進行向量與全文混合檢索；GraphRAG 會額外執行圖譜擴展。匯入階段預先建立 Embedding、向量索引與全文索引，是後續問答能快速檢索的關鍵。

從舊版升級時，請在介面重新執行一次「Embedding 並匯入 Neo4j」，以建立 `graph_evidence_fulltext` 全文索引；既有資料不會只因更新程式碼而自動建立索引。

## 基本使用流程

1. 在「0. 專案設定」建立新專案，或載入既有專案。
2. 在「連線設定」填入 Neo4j 與 OpenAI 相容模型服務，分別執行連線測試。
3. 在 PDF 頁上傳文件並確認解析結果。
4. 選擇全文或隨機 N 頁，執行「分析文件並規劃 Schema」。
5. 視需要修改 Schema，設定 LLM 與最大並行數，再執行「確認 Schema 並抽取知識圖譜」。
6. 選擇資料處理方式，按下「Embedding 並匯入 Neo4j」。
7. 前往問答頁提問；成功結果會自動加入目前專案的歷史紀錄。
8. 回到「0. 專案設定」按下「保存目前專案設定」，保存完整工作狀態。

專案資料位於 `data/projects/<project-id>/`。`project.json` 保存設定與狀態，`documents/` 保存 PDF 副本。Password 與 API Key 會以明文保存在本機專案檔，請勿提交或分享 `data/`。

問答頁不要求在同一工作階段先建圖；只要指定的 Neo4j 中已有本專案建立的圖譜即可使用。

## Docker：陌生環境從零部署

以下步驟適用於一台尚未下載本專案的新電腦。

### 1. 安裝必要工具

安裝 [Git](https://git-scm.com/downloads) 與 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows／macOS），Linux 則可安裝 Docker Engine。

~~~bash
git --version
docker version
~~~

docker version 應同時顯示 Client 與 Server；若只有 Client，請先啟動 Docker Desktop 或 Docker Engine。

### 2. 下載專案

~~~bash
git clone https://github.com/wakaba0972/PDF-Neo4j-graphRAG-DEMO.git
cd PDF-Neo4j-graphRAG-DEMO
~~~

### 3. 建立設定檔與資料目錄

Linux／macOS：

~~~bash
cp .env.example .env
mkdir -p data
~~~

Windows PowerShell：

~~~powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force data
~~~

編輯 .env，至少確認：

~~~dotenv
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

MODEL_API_BASE=http://host.docker.internal:11434/v1
MODEL_API_KEY=your-api-key
BUILD_MODEL=your-build-model
EMBEDDING_MODEL=your-embedding-model
ANSWER_MODEL=your-answer-model
~~~

- Neo4j 或模型服務若運行於宿主機，容器內不能使用 localhost，請使用 host.docker.internal。
- 使用 OpenAI 官方 API 時，將 MODEL_API_BASE 設為 https://api.openai.com/v1。
- 模型欄位必須填入服務端實際提供的模型名稱。

### 4. 建置映像

~~~bash
docker build -t pdf-graphrag:latest .
~~~

### 5. 建立並啟動容器

Linux／macOS：

~~~bash
docker run -d \
  --name pdf-graphrag \
  --restart unless-stopped \
  -p 7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/data:/app/data" \
  pdf-graphrag:latest
~~~

Windows PowerShell（單行）：

~~~powershell
docker run -d --name pdf-graphrag --restart unless-stopped -p 7860:7860 --add-host=host.docker.internal:host-gateway -v "$PWD/.env:/app/.env" -v "$PWD/data:/app/data" pdf-graphrag:latest
~~~

確認服務：

~~~bash
docker ps --filter name=pdf-graphrag
docker logs -f pdf-graphrag
~~~

看到 Gradio 啟動訊息後，開啟 http://localhost:7860，並先在「連線設定」測試 Neo4j 與模型服務。

## 部署後如何啟動

首次執行 docker run 後，容器名稱會是 pdf-graphrag。日後不需要再次 git clone 或 docker build。

啟動既有容器：

~~~bash
docker start pdf-graphrag
~~~

查看狀態及日誌：

~~~bash
docker ps --filter name=pdf-graphrag
docker logs -f pdf-graphrag
~~~

停止服務：

~~~bash
docker stop pdf-graphrag
~~~

部署命令包含 --restart unless-stopped，Docker 服務重啟後通常會自動啟動容器；若曾手動停止，請執行 docker start pdf-graphrag。

### 更新到新版

~~~bash
git pull
docker build -t pdf-graphrag:latest .
docker stop pdf-graphrag
docker rm pdf-graphrag
~~~

接著重新執行前一節的 docker run。掛載於專案 data 目錄的資料不會因移除容器而消失。

## 常見問題

### 網頁無法開啟

~~~bash
docker ps -a --filter name=pdf-graphrag
docker logs pdf-graphrag
~~~

若 7860 已被占用，將啟動參數改為 -p 8080:7860，再開啟 http://localhost:8080。

### Neo4j 或模型服務連線失敗

- 宿主機服務請使用 host.docker.internal，不要填 localhost 或 127.0.0.1。
- 確認 Neo4j Bolt 連接埠通常為 7687，且帳號、密碼、資料庫名稱正確。
- 確認本機服務已監聽容器可連線的網路介面。
- Linux 無法解析 host.docker.internal 時，確認 docker run 包含 --add-host=host.docker.internal:host-gateway。
- 修改 .env 後可在介面重新讀取，或執行 docker restart pdf-graphrag。

### 設定或資料沒有保留

確認 .env 與 data 掛載路徑存在且有讀寫權限。連線設定會寫回 .env，持久資料則寫入 data 目錄。

## 非 Docker 啟動方式

需要 Python 3.11 以上版本。Linux 可使用一鍵腳本：

~~~bash
./start.sh
~~~

或手動啟動：

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python src/app.py
~~~

## 測試

~~~bash
. .venv/bin/activate
pytest
~~~

## 限制與注意事項

- PDF 必須包含可選取的文字層；目前不提供 OCR。
- 加密 PDF 不支援。
- 建議先以少量頁面驗證 Schema、模型輸出及 Neo4j 寫入結果，再處理大型文件。
- 「清空資料庫」與「取代目前圖譜」會刪除資料，介面會要求再次確認。
- .env 可能包含 API Key 與 Neo4j 密碼，請勿提交至 Git；專案已透過 .gitignore 排除。
- PyMuPDF 採 AGPL／商業雙授權，封裝、散布或商用前請確認授權需求。
