from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from sigvue.profile import load_browser_profile

from scripts.download_data import download_datasets
from sigmf_viewer import cli
from sigmf_viewer.runtime import runtime_profile


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
        assert (
            payload["workspaces"][0]["config"]["batch_max_native_cells"]
            == 75_000_000
        )
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


def test_package_leaves_delivery_to_sigvue():
    project = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert payload["project"]["name"] == "sigmf-viewer"
    assert scripts["sigmf-viewer"] == "sigmf_viewer.cli:main"
    assert "sigmf-viewer-desktop" not in scripts
    assert "sigmf-viewer-build" not in scripts
    dependencies = {
        requirement.split(">=", 1)[0]
        for requirement in payload["project"]["dependencies"]
    }
    assert {
        "matplotlib",
        "numpy",
        "Pillow",
        "plotly",
        "sigvue",
    } == dependencies
    assert "desktop" not in payload["project"]["optional-dependencies"]
    assert not (project / "src" / "sigmf_viewer" / "desktop.py").exists()
    assert not (
        project / "src" / "sigmf_viewer" / "_packaging" / "build.py"
    ).exists()
    assert not (
        project / "src" / "sigmf_viewer" / "_packaging" / "sigmf_viewer.spec"
    ).exists()
    assert "sigmf-viewer-download" not in scripts
    assert not (project / "src" / "sigmf_viewer" / "download.py").exists()
