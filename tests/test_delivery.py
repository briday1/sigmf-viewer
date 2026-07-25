from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from sigvue.profile import load_browser_profile

from scripts.download_data import download_datasets
from sigmf_viewer import cli, desktop
from sigmf_viewer.runtime import runtime_profile


class FakeEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class FakeWindow:
    def __init__(self):
        self.events = SimpleNamespace(
            loaded=FakeEvent(),
            restored=FakeEvent(),
        )
        self.scripts = []
        self.fullscreen_toggles = 0
        self.selected_directory = "/tmp/sigmf-data"

    def evaluate_js(self, script):
        self.scripts.append(script)

    def toggle_fullscreen(self):
        self.fullscreen_toggles += 1

    def create_file_dialog(self, dialog_type):
        self.dialog_type = dialog_type
        return (self.selected_directory,)


def test_runtime_profile_uses_explicit_paths_and_flat_discovery():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "sigmf data"
        output = root / "rendered output"
        data.mkdir()
        with runtime_profile(
            data_root=data,
            output_root=output,
        ) as profile_path:
            profile = load_browser_profile(profile_path)
            payload = tomllib.loads(profile_path.read_text(encoding="utf-8"))

        assert profile.title == "SigMF Viewer"
        assert len(profile.workspaces) == 1
        workspace = profile.workspaces[0]
        assert workspace.module_name == "sigmf_viewer.workspace"
        assert workspace.attribute == "create_workspace"
        assert workspace.flatten_discovery is True
        assert Path(workspace.configuration["data_root"]) == data.resolve()
        assert Path(workspace.configuration["output_root"]) == output.resolve()
        assert payload["workspaces"][0]["id"] == "sigmf-viewer"
        assert payload["workspaces"][0]["config"]["batch_fft_size"] == 256
        assert output.is_dir()
        assert not profile_path.exists()


def test_cli_accepts_one_recording_and_preserves_sigvue_arguments():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        recording = root / "capture.sigmf-meta"
        observed = {}

        def inspect_invocation():
            arguments = list(sys.argv[1:])
            profile = Path(arguments[arguments.index("--config") + 1])
            observed["arguments"] = arguments
            observed["profile"] = load_browser_profile(profile)

        with (
            patch.object(
                sys,
                "argv",
                [
                    "sigmf-viewer",
                    "batch",
                    "--recording",
                    str(recording),
                    "--output-root",
                    str(root / "outputs"),
                    "--list",
                ],
            ),
            patch.object(
                cli,
                "sigvue_main",
                side_effect=inspect_invocation,
            ),
        ):
            cli.main()

        assert observed["arguments"][2:] == ["batch", "--list"]
        workspace = observed["profile"].workspaces[0]
        assert Path(workspace.configuration["data_root"]) == (recording.resolve())


def test_desktop_hosts_live_private_server_and_native_fullscreen():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        result = {}
        window = FakeWindow()

        def create_window(title, url, **options):
            result.update(title=title, url=url, options=options)
            return window

        def start(**options):
            with urlopen(f"{result['url']}/health", timeout=5) as response:
                result["health"] = json.load(response)
            for handler in window.events.loaded.handlers:
                handler(window)
            result["start_options"] = options

        fake_webview = SimpleNamespace(
            FileDialog=SimpleNamespace(FOLDER=20),
            create_window=create_window,
            start=start,
        )
        with (
            patch.dict(sys.modules, {"webview": fake_webview}),
            patch.object(
                sys,
                "argv",
                [
                    "sigmf-viewer-desktop",
                    "--data-root",
                    str(root / "data"),
                    "--output-root",
                    str(root / "outputs"),
                    "--width",
                    "1200",
                    "--height",
                    "700",
                ],
            ),
        ):
            desktop.main()

        assert result["title"] == "SigMF Viewer"
        assert result["url"].startswith("http://127.0.0.1:")
        assert result["health"] == {"status": "ok"}
        assert result["options"]["width"] == 1200
        assert result["options"]["height"] == 700
        bridge = result["options"]["js_api"]
        assert bridge.choose_directory() == "/tmp/sigmf-data"
        assert bridge.toggle_fullscreen() is True
        for handler in window.events.restored.handlers:
            handler()
        assert bridge.fullscreen_state() is False
        assert window.fullscreen_toggles == 1
        assert len(window.scripts) == 2
        assert "#fullscreen-toggle" in window.scripts[0]
        assert "stopImmediatePropagation" in window.scripts[0]
        assert window.scripts[1] == ("window.__sigmfViewerSetNativeFullscreen?.(false)")
        assert result["start_options"] == {"debug": False}


def test_downloader_writes_standard_collection(tmp_path):
    def fake_download(remote, destination, **options):
        del options
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / remote.filename
        path.write_bytes(b"metadata" if path.suffix == ".sigmf-meta" else b"data")
        return path

    with patch(
        "scripts.download_data.download_file",
        side_effect=fake_download,
    ) as mocked:
        paths = download_datasets(tmp_path, datasets=("lte",))

    assert mocked.call_count == 4
    collection = paths[-1]
    payload = json.loads(collection.read_text(encoding="utf-8"))
    streams = payload["collection"]["core:streams"]
    assert len(streams) == 2
    assert streams[0]["name"].startswith("downlink/")
    assert streams[1]["name"].startswith("uplink/")


def test_package_declares_all_delivery_entry_points_and_dependencies():
    project = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert payload["project"]["name"] == "sigmf-viewer"
    assert scripts["sigmf-viewer"] == "sigmf_viewer.cli:main"
    assert scripts["sigmf-viewer-desktop"] == "sigmf_viewer.desktop:main"
    assert scripts["sigmf-viewer-build"] == "sigmf_viewer._packaging.build:main"
    dependencies = {
        requirement.split(">=", 1)[0]
        for requirement in payload["project"]["dependencies"]
    }
    assert {
        "matplotlib",
        "numpy",
        "plotly",
        "sigvue",
    } == dependencies
    assert {"pyinstaller", "pywebview"} == {
        requirement.split(">=", 1)[0]
        for requirement in payload["project"]["optional-dependencies"]["desktop"]
    }
    assert (
        project / "src" / "sigmf_viewer" / "_packaging" / "sigmf_viewer.spec"
    ).is_file()
    assert "sigmf-viewer-download" not in scripts
    assert not (project / "src" / "sigmf_viewer" / "download.py").exists()
