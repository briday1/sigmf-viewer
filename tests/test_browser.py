from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.build_pages import CORE_MODULES, fetch_verified
from scripts.download_data import COLDFERRY_FILES, SIGMF_LOGO_FILES
from sigmf_viewer.analysis import analyze
from sigmf_viewer.browser import (
    MAX_CELLS, _reduce_power, browser_analyze, open_browser_recording,
    validate_metadata,
)
from sigmf_viewer.models import WaterfallSettings
from sigmf_viewer.sigmf import read_window

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recording(tmp_path):
    n = np.arange(4096)
    samples = (0.3 * np.exp(2j * np.pi * n / 16)).astype(np.complex64)
    samples.tofile(tmp_path / "tone.sigmf-data")
    metadata = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 32000,
                   "core:offset": 100},
        "captures": [{"core:sample_start": 100, "core:frequency": 1000000}],
        "annotations": [],
    }
    path = tmp_path / "tone.sigmf-meta"
    path.write_text(json.dumps(metadata))
    return open_browser_recording(path)


def request(**changes):
    return {"start": 23, "count": 2048, "fft": 128, "overlap": 50, "channel": 0, **changes}


def test_browser_matches_native_numerical_core(recording):
    actual = browser_analyze(recording, request())
    expected = analyze(read_window(recording, 23, 2048), WaterfallSettings(128, 50, 0))
    np.testing.assert_allclose(actual["spectrum"], expected.spectrum_dbfs, atol=1e-12)
    np.testing.assert_allclose(actual["z"], expected.waterfall_dbfs, atol=1e-12)
    np.testing.assert_array_equal(actual["x"], expected.frequency_mhz)
    np.testing.assert_allclose(actual["y"], (expected.time_edges_ms[:-1] + expected.time_edges_ms[1:]) / 2)
    np.testing.assert_array_equal(actual["yEdges"], expected.time_edges_ms)
    assert len(actual["xEdges"]) == len(actual["x"]) + 1
    assert actual["metrics"]["nativeCells"] <= MAX_CELLS


def test_linear_power_reduction_not_db_average():
    reduced, edges = _reduce_power(np.array([0.0, -20.0]), 0, 1)
    np.testing.assert_allclose(reduced, [10 * np.log10(0.505)])
    np.testing.assert_array_equal(edges, [0, 2])


def test_frequency_zoom_recovers_native_bins(recording):
    whole = browser_analyze(recording, request(fft=1024))
    zoom = browser_analyze(recording, request(fft=1024, frequency=[1.001, 1.003]))
    expected = analyze(read_window(recording, 23, 2048), WaterfallSettings(1024, 50, 0))
    selection = (expected.frequency_mhz >= 1.001) & (expected.frequency_mhz <= 1.003)
    assert len(whole["x"]) == 512
    np.testing.assert_array_equal(zoom["x"], expected.frequency_mhz[selection])
    np.testing.assert_allclose(zoom["spectrum"], expected.spectrum_dbfs[selection])


@pytest.mark.parametrize("changes", [
    {"count": 65537}, {"count": 0}, {"start": -1}, {"start": 4096},
    {"fft": 32768}, {"overlap": 99}, {"overlap": 3.5}, {"channel": 1},
    {"count": True},
])
def test_limits_precede_sample_read(recording, changes):
    with patch("sigmf_viewer.browser.read_window", side_effect=AssertionError("allocated")):
        with pytest.raises(ValueError):
            browser_analyze(recording, request(**changes))


def test_cell_limit_precedes_sample_read(recording):
    from dataclasses import replace

    large = replace(recording, sample_count=100000)
    with patch("sigmf_viewer.browser.read_window", side_effect=AssertionError("allocated")):
        with pytest.raises(ValueError, match="STFT"):
            browser_analyze(large, request(count=65536, fft=64, overlap=90))


def test_capture_boundary_rejected(recording):
    recording.metadata["captures"].append({"core:sample_start": 1100, "core:frequency": 2000000})
    with pytest.raises(ValueError, match="capture boundary"):
        browser_analyze(recording, request())


@pytest.mark.parametrize("field,value", [
    ("core:num_channels", 17), ("core:num_channels", True),
    ("core:trailing_bytes", 4),
    ("core:offset", 2**53), ("core:offset", -1),
])
def test_metadata_limits(recording, field, value):
    recording.metadata["global"][field] = value
    with pytest.raises(ValueError):
        validate_metadata(recording.metadata, 100)
    with pytest.raises(ValueError):
        validate_metadata(recording.metadata, 32 * 1024 * 1024 + 1)


def test_browser_imports_do_not_load_server_or_plot_packages():
    command = """
import sys
import sigmf_viewer
import sigmf_viewer.browser
assert not any(name.split('.')[0] in {'sigvue', 'matplotlib', 'PIL', 'plotly'}
               for name in sys.modules)
assert callable(sigmf_viewer.analyze)
try:
    sigmf_viewer.missing
except AttributeError:
    pass
else:
    raise AssertionError('missing attribute accepted')
"""
    subprocess.run([sys.executable, "-c", command], check=True)


def test_example_manifest_is_seven_pairs_without_lte():
    entries = (*COLDFERRY_FILES, *SIGMF_LOGO_FILES)
    assert len(entries) == 14
    assert sum(e.size for e in entries) < 13 * 1024 * 1024
    assert all("LTE" not in e.filename for e in entries)
    assert all(len(e.checksum.removeprefix("sha256:")) == 64 for e in entries)


def test_download_rejects_wrong_hash_and_size(tmp_path):
    import hashlib

    source = tmp_path / "source"
    source.write_bytes(b"test")
    destination = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_verified(source.as_uri(), destination, "0" * 64, 4)
    assert not destination.exists()
    with pytest.raises(ValueError, match="Size"):
        fetch_verified(source.as_uri(), destination, hashlib.sha256(b"test").hexdigest(), 3)
    fetch_verified(source.as_uri(), destination, hashlib.sha256(b"test").hexdigest(), 4)
    assert destination.read_bytes() == b"test"


def test_static_archive_is_core_only_and_self_contained():
    import hashlib

    artifact = ROOT / "site"
    if not artifact.exists():
        pytest.skip("Run npm run build to validate the built artifact")
    with ZipFile(artifact / "sigmf_viewer.zip") as archive:
        assert set(archive.namelist()) == {f"sigmf_viewer/{name}" for name in CORE_MODULES}
    entries = json.loads((artifact / "examples.json").read_text())
    assert len(entries) == 7
    build = json.loads((artifact / "build-info.json").read_text())
    assert build["numpy"] == "2.0.2"
    assert (artifact / "vendor/pyodide.asm.wasm").is_file()
    assert len(list((artifact / "vendor").glob("numpy*.whl"))) == 1
    assert len(list((artifact / "licenses").glob("*.txt"))) == 4
    for name, digest in build["sha256"].items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == digest
    assert all((artifact / e["data"]).stat().st_size == e["dataBytes"] for e in entries)
    for filename in ("index.html", "app.mjs", "worker.js"):
        content = (artifact / filename).read_text()
        assert "https://" not in content
        assert "http://" not in content
