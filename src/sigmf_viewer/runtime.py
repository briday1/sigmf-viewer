"""Runtime paths and generated profiles for the focused CLI."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

APPLICATION_NAME = "SigMF Viewer"


def _source_checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "browser.toml").is_file() else None


def default_data_root() -> Path:
    configured = os.environ.get("SIGMF_VIEWER_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    working = Path.cwd() / "data"
    if working.is_dir():
        return working.resolve()
    checkout = _source_checkout_root()
    if checkout is not None:
        return (checkout / "data").resolve()
    return working.resolve()


def default_output_root() -> Path:
    configured = os.environ.get("SIGMF_VIEWER_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = _source_checkout_root()
    if checkout is not None:
        return (checkout / "outputs").resolve()
    return (Path.cwd() / "outputs").resolve()


def profile_text(data_root: Path, output_root: Path) -> str:
    """Build the focused one-workspace profile with absolute paths."""
    return f"""[browser]
title = "SigMF Viewer"
subtitle = "Windowed spectrum analysis for recordings and collections"

[[workspaces]]
use = "sigmf_viewer.workspace:create_workspace"
id = "sigmf-viewer"
name = "SigMF Viewer"
description = "Browse SigMF pairs and collections as windowed waterfalls."
category = "spectrum monitoring"
tags = ["SigMF", "windowed", "waterfall", "spectrum"]
flatten_discovery = true

[workspaces.config]
data_root = {json.dumps(str(data_root.expanduser().resolve()))}
output_root = {json.dumps(str(output_root.expanduser().resolve()))}
filename = "*.sigmf-meta"
default_window_seconds = 0.012
minimum_window_seconds = 0.004
window_step_seconds = 0.002
overview_bins = 300
overview_frequency_bins = 48
overview_fft_size = 256
batch_fft_size = 256
batch_overlap_percent = 50
batch_colormap = "turbo"
batch_max_native_cells = 75000000
"""


@contextmanager
def runtime_profile(
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Iterator[Path]:
    resolved_data = (
        default_data_root()
        if data_root is None
        else Path(data_root).expanduser().resolve()
    )
    resolved_output = (
        default_output_root()
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sigmf-viewer-") as directory:
        profile = Path(directory) / "browser.toml"
        profile.write_text(
            profile_text(resolved_data, resolved_output),
            encoding="utf-8",
        )
        yield profile


__all__ = [
    "APPLICATION_NAME",
    "default_data_root",
    "default_output_root",
    "profile_text",
    "runtime_profile",
]
