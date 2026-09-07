import os

from manual_graphrag.ui import build_app


def launch_settings() -> dict[str, object]:
    return {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        "share": False,
        "inbrowser": os.getenv("GRADIO_INBROWSER", "true").lower()
        in {"1", "true", "yes"},
    }


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(**launch_settings())
