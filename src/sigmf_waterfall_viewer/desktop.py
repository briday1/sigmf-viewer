"""Native pywebview launcher for the focused SigMF application."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from sigvue.web.application import _make_handler, create_app

from .runtime import APPLICATION_NAME, runtime_profile

_NATIVE_FULLSCREEN_SCRIPT = r"""
(() => {
  const button = document.querySelector('#fullscreen-toggle');
  if (!button || button.dataset.nativeFullscreen === 'true') return;
  button.dataset.nativeFullscreen = 'true';

  let active = false;
  const render = value => {
    active = Boolean(value);
    button.setAttribute(
      'aria-label',
      active ? 'Exit fullscreen' : 'Enter fullscreen',
    );
    button.setAttribute('aria-pressed', String(active));
    button.textContent = active ? '×' : '⛶';
    window.dispatchEvent(new Event('resize'));
  };
  window.__sigmfWaterfallSetNativeFullscreen = render;
  const toggle = async () => {
    if (!window.pywebview?.api?.toggle_fullscreen) return;
    button.disabled = true;
    try {
      render(await window.pywebview.api.toggle_fullscreen());
    } finally {
      button.disabled = false;
    }
  };

  button.addEventListener('click', event => {
    event.preventDefault();
    event.stopImmediatePropagation();
    void toggle();
  }, true);
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !active) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void toggle();
  }, true);
  if (window.pywebview?.api?.fullscreen_state) {
    void window.pywebview.api.fullscreen_state().then(render);
  }
})();
"""


class _DesktopApi:
    """Small bridge for native fullscreen and the workspace folder picker."""

    def __init__(self, folder_dialog_type: Any) -> None:
        self._window: Any | None = None
        self._fullscreen = False
        self._lock = Lock()
        self._folder_dialog_type = folder_dialog_type

    def _bind(self, window: Any) -> None:
        self._window = window

    def toggle_fullscreen(self) -> bool:
        with self._lock:
            if self._window is None:
                return False
            self._window.toggle_fullscreen()
            self._fullscreen = not self._fullscreen
            return self._fullscreen

    def fullscreen_state(self) -> bool:
        with self._lock:
            return self._fullscreen

    def choose_directory(self) -> str | None:
        with self._lock:
            window = self._window
        if window is None:
            return None
        selected = window.create_file_dialog(self._folder_dialog_type)
        return str(selected[0]) if selected else None

    def _restored(self) -> None:
        with self._lock:
            if not self._fullscreen:
                return
            self._fullscreen = False
            window = self._window
        if window is not None:
            window.evaluate_js("window.__sigmfWaterfallSetNativeFullscreen?.(false)")


def _install_native_fullscreen(window: Any) -> None:
    window.evaluate_js(_NATIVE_FULLSCREEN_SCRIPT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Open the SigMF Waterfall Viewer in a native desktop window"),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-root", type=Path)
    source.add_argument("--recording", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        import webview
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install desktop support first: "
            'pip install "sigmf-waterfall-viewer[desktop]"'
        ) from exc

    with runtime_profile(
        data_root=args.recording or args.data_root,
        output_root=args.output_root,
        desktop=True,
    ) as profile:
        app = create_app(
            title=APPLICATION_NAME,
            reload_workspaces=False,
            config_path=profile,
        )
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _make_handler(app),
        )
        server.daemon_threads = True
        thread = Thread(
            target=server.serve_forever,
            name="sigmf-waterfall-viewer-server",
            daemon=True,
        )
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        desktop_api = _DesktopApi(webview.FileDialog.FOLDER)
        window = webview.create_window(
            APPLICATION_NAME,
            url,
            width=max(args.width, 900),
            height=max(args.height, 600),
            min_size=(900, 600),
            js_api=desktop_api,
        )
        desktop_api._bind(window)
        window.events.loaded += _install_native_fullscreen
        window.events.restored += desktop_api._restored
        try:
            webview.start(debug=args.debug)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
