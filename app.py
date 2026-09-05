from manual_graphrag.ui import build_app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
    )
