"""App de desktop: a mesma interface do app.py, numa janela nativa.

Sobe o servidor Gradio só em 127.0.0.1 e abre uma janela WebKit apontando
para ele. Nada é exposto na rede.
"""

import socket
import threading

import webview

import theme as dc_theme
from app import demo


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    ready = threading.Event()

    def serve():
        demo.queue(default_concurrency_limit=1).launch(
            theme=dc_theme.build_theme(),
            css=dc_theme.CSS,
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=False,
            quiet=True,
            prevent_thread_lock=True,
        )
        ready.set()

    threading.Thread(target=serve, daemon=True).start()
    if not ready.wait(timeout=120):
        raise RuntimeError("O servidor Gradio não subiu a tempo.")

    webview.create_window(
        "Dream Canvas",
        f"http://127.0.0.1:{port}",
        width=1280,
        height=900,
        min_size=(900, 600),
    )
    webview.start()


if __name__ == "__main__":
    main()
