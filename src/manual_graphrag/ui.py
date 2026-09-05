from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .chunking import chunk_pages, preview_rows
from .config import BuildConfig, public_settings
from .env_store import load_env, save_env
from .pdf_service import extract_pdf
from .storage import write_json


def connection_summary(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
) -> tuple[str, dict[str, object]]:
    missing = [
        label
        for label, value in {
            "Neo4j URI": neo4j_uri,
            "Database": neo4j_database,
            "Username": neo4j_username,
            "Password": neo4j_password,
            "模型端點": model_endpoint,
        }.items()
        if not value.strip()
    ]
    if missing:
        return f"⚠️ 尚未填寫：{'、'.join(missing)}", {}
    settings = public_settings(
        {
            "neo4j_uri": neo4j_uri,
            "neo4j_database": neo4j_database,
            "neo4j_username": neo4j_username,
            "neo4j_password": neo4j_password,
            "model_endpoint": model_endpoint,
            "api_key": api_key,
        }
    )
    return "✅ 欄位格式已通過初步檢查；實際連線將於下一版接入。", settings


def persist_env_settings(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
    build_model: str,
    embedding_model: str,
    answer_model: str,
) -> str:
    save_env(
        {
            "NEO4J_URI": neo4j_uri,
            "NEO4J_DATABASE": neo4j_database,
            "NEO4J_USERNAME": neo4j_username,
            "NEO4J_PASSWORD": neo4j_password,
            "MODEL_API_BASE": model_endpoint,
            "MODEL_API_KEY": api_key,
            "BUILD_MODEL": build_model,
            "EMBEDDING_MODEL": embedding_model,
            "ANSWER_MODEL": answer_model,
        }
    )
    return "✅ 已自動儲存至 .env"


def reload_env_settings() -> tuple[str, ...]:
    settings = load_env()
    return (
        settings["NEO4J_URI"],
        settings["NEO4J_DATABASE"],
        settings["NEO4J_USERNAME"],
        settings["NEO4J_PASSWORD"],
        settings["MODEL_API_BASE"],
        settings["MODEL_API_KEY"],
        settings["BUILD_MODEL"],
        settings["EMBEDDING_MODEL"],
        settings["ANSWER_MODEL"],
        "✅ 已重新讀取 .env",
    )


def preview_pdf(
    file_path: str | None,
    build_model: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, list[list[object]], dict[str, Any]]:
    if not file_path:
        return "請先上傳 PDF。", [], {}
    try:
        config = BuildConfig(
            build_model=build_model,
            embedding_model=embedding_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
        pages, empty_pages = extract_pdf(file_path)
        chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap)
        if not chunks:
            return "PDF 沒有可解析文字；掃描文件需在後續版本加入 OCR。", [], {}
        state = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "page_count": len(pages),
            "empty_pages": empty_pages,
            "chunk_count": len(chunks),
            "config": config.to_dict(),
        }
        note = f"已解析 {len(pages)} 頁，產生 {len(chunks)} 個 chunk。"
        if empty_pages:
            note += f" 無文字頁面：{', '.join(map(str, empty_pages))}。"
        return note, preview_rows(chunks), state
    except (ValueError, TypeError) as exc:
        return f"❌ {exc}", [], {}


def save_config(state: dict[str, Any]) -> tuple[str, str | None]:
    if not state:
        return "請先成功預覽 PDF。", None
    output = write_json(Path("data/exports/latest-config.json"), state)
    return f"設定已儲存：{output}", str(output)


def initial_build_status(state: dict[str, Any]) -> str:
    if not state:
        return "請先在「PDF 與參數」完成預覽。"
    return (
        "介面與參數快照已就緒。Neo4j 寫入、實體關係抽取與 embedding "
        "將在下一個開發階段接入。"
    )


def initial_answer(question: str, state: dict[str, Any]) -> tuple[str, str]:
    if not question.strip():
        return "請輸入問題。", ""
    if not state:
        return "請先上傳 PDF 並完成 chunk 預覽。", ""
    return (
        "GraphRAG 後端尚未接入；初版介面已記錄問題與目前設定。",
        f"問題：{question}\n\n目標文件：{state.get('file_name', '未知')}",
    )


def build_app() -> gr.Blocks:
    env = load_env()
    with gr.Blocks(title="PDF GraphRAG 測試工具") as app:
        gr.Markdown(
            "# PDF GraphRAG 測試工具\n"
            "上傳使用手冊、調整建圖參數，並測試 Neo4j GraphRAG。"
        )
        preview_state = gr.State({})

        with gr.Tab("1. 連線設定"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Neo4j")
                    neo4j_uri = gr.Textbox(label="URI", value=env["NEO4J_URI"])
                    neo4j_database = gr.Textbox(label="Database", value=env["NEO4J_DATABASE"])
                    neo4j_username = gr.Textbox(label="Username", value=env["NEO4J_USERNAME"])
                    neo4j_password = gr.Textbox(label="Password", value=env["NEO4J_PASSWORD"], type="password")
                with gr.Column():
                    gr.Markdown("### 模型服務")
                    model_endpoint = gr.Textbox(label="API Base URL", value=env["MODEL_API_BASE"])
                    api_key = gr.Textbox(label="API Key", value=env["MODEL_API_KEY"], type="password")
                    connection_button = gr.Button("檢查設定", variant="primary")
                    reload_button = gr.Button("重新讀取 .env")
            gr.Markdown("⚠️ Password 與 API Key 會以明文寫入本機 `.env`；請勿提交此檔案。")
            env_status = gr.Markdown("啟動時已讀取 .env；欄位修改後會自動儲存。")
            connection_status = gr.Markdown()
            safe_connection = gr.JSON(label="非機密設定預覽")

        with gr.Tab("2. PDF 與參數"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(label="PDF 使用手冊", file_types=[".pdf"], type="filepath")
                    build_model = gr.Textbox(label="建圖 LLM", value=env["BUILD_MODEL"])
                    embedding_model = gr.Textbox(label="Embedding 模型", value=env["EMBEDDING_MODEL"])
                    chunk_size = gr.Slider(100, 10000, value=1500, step=100, label="Chunk size（字元）")
                    chunk_overlap = gr.Slider(0, 2000, value=200, step=50, label="Chunk overlap（字元）")
                    temperature = gr.Slider(0, 2, value=0, step=0.1, label="Temperature")
                    max_output_tokens = gr.Number(value=2048, precision=0, label="最大輸出 tokens")
                    preview_button = gr.Button("解析並預覽 Chunk", variant="primary")
                    export_button = gr.Button("匯出目前設定")
                    export_file = gr.File(label="設定 JSON", interactive=False)
                with gr.Column(scale=2):
                    preview_status = gr.Markdown("尚未解析 PDF。")
                    chunk_table = gr.Dataframe(
                        headers=["編號", "頁碼", "字元數", "內容"],
                        datatype=["number", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                    )

        with gr.Tab("3. 建圖"):
            gr.Markdown("### 建立 Neo4j 知識圖譜")
            build_button = gr.Button("開始建圖", variant="primary")
            build_status = gr.Markdown("請先完成 PDF 與參數設定。")

        with gr.Tab("4. 問答測試"):
            answer_model = gr.Textbox(label="問答 LLM", value=env["ANSWER_MODEL"])
            question = gr.Textbox(label="問題", placeholder="例如：設備出現 E01 時該如何處理？")
            with gr.Row():
                retrieval_mode = gr.Radio(["GraphRAG", "向量 RAG"], value="GraphRAG", label="檢索模式")
                top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
            ask_button = gr.Button("送出問題", variant="primary")
            answer_status = gr.Markdown()
            answer = gr.Markdown()

        with gr.Tab("5. 歷史紀錄"):
            gr.Markdown("建圖與問答紀錄將在後續開發階段顯示於此。")

        connection_button.click(
            connection_summary,
            inputs=[neo4j_uri, neo4j_database, neo4j_username, neo4j_password, model_endpoint, api_key],
            outputs=[connection_status, safe_connection],
        )
        env_inputs = [
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            model_endpoint,
            api_key,
            build_model,
            embedding_model,
            answer_model,
        ]
        for component in env_inputs:
            component.change(persist_env_settings, inputs=env_inputs, outputs=env_status)
        reload_button.click(reload_env_settings, outputs=[*env_inputs, env_status])
        preview_button.click(
            preview_pdf,
            inputs=[pdf_file, build_model, embedding_model, chunk_size, chunk_overlap, temperature, max_output_tokens],
            outputs=[preview_status, chunk_table, preview_state],
        )
        export_button.click(save_config, inputs=[preview_state], outputs=[preview_status, export_file])
        build_button.click(initial_build_status, inputs=[preview_state], outputs=[build_status])
        ask_button.click(initial_answer, inputs=[question, preview_state], outputs=[answer_status, answer])
    return app
