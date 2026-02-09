import threading
import subprocess
import webview
import sys
import os
import time


PORT = "8501"


def start_streamlit():
    script_path = os.path.join(os.path.dirname(__file__), "scatter_plot_single_file.py")

    subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script_path,
        "--server.headless=true",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false"
    ])


def wait_for_server():
    import socket
    while True:
        s = socket.socket()
        try:
            s.connect(("localhost", int(PORT)))
            s.close()
            break
        except:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()
    wait_for_server()

    webview.create_window(
        "Data Tool",
        f"http://localhost:{PORT}",
        width=1200,
        height=800
    )

    webview.start()
