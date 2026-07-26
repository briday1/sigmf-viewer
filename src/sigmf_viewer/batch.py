"""Durable shareable PNG and tiled whole-recording waterfall rendering."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Iterator
from math import ceil, log2
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from uuid import uuid4

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from sigvue import (
    Batch,
    BatchDestination,
    BatchRequest,
    BatchResult,
    CapabilityChoice,
    DataResource,
)

from .models import SigMFCollection, SigMFRecording, SigMFSource
from .sigmf import open_source

RENDER_WATERFALL = "render-high-resolution-waterfall"

_TILE_SIZE = 512
_HISTOGRAM_MIN = -200.0
_HISTOGRAM_MAX = 20.0
_HISTOGRAM_STEP = 0.01
_HISTOGRAM_EDGES = np.linspace(
    _HISTOGRAM_MIN,
    _HISTOGRAM_MAX,
    int((_HISTOGRAM_MAX - _HISTOGRAM_MIN) / _HISTOGRAM_STEP) + 1,
)


def _validate_render_options(
    *,
    fft_size: int,
    overlap_percent: int,
    max_native_cells: int,
    colormap: str,
) -> None:
    integers = {
        "fft_size": fft_size,
        "max_native_cells": max_native_cells,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integers.values()
    ) or fft_size < 1:
        raise ValueError(
            "batch FFT must be positive and the native-cell limit non-negative"
        )
    if not 0 <= overlap_percent < 100:
        raise ValueError("batch overlap must be between 0 and 99")
    if colormap not in matplotlib.colormaps:
        raise ValueError(f"unknown batch colormap: {colormap}")


def _stft_geometry(
    recording: SigMFRecording,
    fft_size: int,
    overlap_percent: int,
) -> tuple[int, int, int]:
    effective_fft = min(fft_size, recording.sample_count)
    hop = max(
        1,
        round(effective_fft * (1.0 - overlap_percent / 100.0)),
    )
    total_frames = max(1, ceil(recording.sample_count / hop))
    return effective_fft, hop, total_frames


def _recording_power_chunks(
    recording: SigMFRecording,
    *,
    channel: int,
    effective_fft: int,
    total_frames: int,
    chunk_frames: int = _TILE_SIZE,
    cancel: Callable[[], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the same centered STFT cells as the interactive analysis."""
    taper = (
        np.hanning(effective_fft)
        if effective_fft > 2
        else np.ones(effective_fft)
    )
    for first_index in range(0, total_frames, chunk_frames):
        if cancel is not None:
            cancel()
        frame_indexes = np.arange(
            first_index,
            min(total_frames, first_index + chunk_frames),
            dtype=np.int64,
        )
        centers = (
            (frame_indexes.astype(np.float64) + 0.5)
            * recording.sample_count
            / total_frames
        )
        starts = np.rint(centers - effective_fft / 2.0).astype(np.int64)
        read_start = max(0, int(starts[0]))
        read_stop = min(
            recording.sample_count,
            int(starts[-1] + effective_fft),
        )
        samples = recording.read(
            read_start,
            read_stop - read_start,
        )[channel]
        blocks = np.zeros(
            (frame_indexes.size, effective_fft),
            dtype=samples.dtype,
        )
        normalizers = np.empty(frame_indexes.size, dtype=np.float64)
        for row, start in enumerate(starts):
            source_start = max(0, int(start))
            source_stop = min(
                recording.sample_count,
                int(start + effective_fft),
            )
            target_start = source_start - int(start)
            target_stop = target_start + source_stop - source_start
            blocks[row, target_start:target_stop] = samples[
                source_start - read_start : source_stop - read_start
            ]
            normalizers[row] = max(
                float(np.sum(taper[target_start:target_stop])),
                1.0,
            )
        spectra = np.fft.fftshift(
            np.fft.fft(blocks * taper, axis=1),
            axes=1,
        )
        power = (np.abs(spectra) / normalizers[:, None]) ** 2
        yield first_index, 10.0 * np.log10(np.maximum(power, 1e-20))


def _rounded_range(lower: float, upper: float) -> tuple[float, float]:
    low = max(-200.0, 5.0 * np.floor(lower / 5.0))
    high = min(20.0, 5.0 * np.ceil(upper / 5.0))
    if high - low < 20.0:
        low = max(-200.0, high - 20.0)
    if high == 0:
        high = 0.0
    return float(low), float(high)


def _histogram_percentile(counts: np.ndarray, percentile: float) -> float:
    total = int(np.sum(counts))
    if total <= 0:
        raise ValueError("cannot discover a display range from an empty waterfall")
    rank = percentile / 100.0 * max(0, total - 1)
    cumulative = np.cumsum(counts, dtype=np.uint64)
    index = min(
        counts.size - 1,
        int(np.searchsorted(cumulative, rank, side="right")),
    )
    preceding = 0 if index == 0 else int(cumulative[index - 1])
    within = (
        0.0
        if counts[index] == 0
        else (rank - preceding) / int(counts[index])
    )
    return float(
        _HISTOGRAM_EDGES[index]
        + within * (_HISTOGRAM_EDGES[index + 1] - _HISTOGRAM_EDGES[index])
    )


def _discover_dbfs_range(
    recording: SigMFRecording,
    *,
    channel: int,
    effective_fft: int,
    total_frames: int,
    cancel: Callable[[], None] | None = None,
) -> tuple[float, float]:
    """Apply the interactive robust range rule across the complete recording."""
    histogram = np.zeros(_HISTOGRAM_EDGES.size - 1, dtype=np.uint64)
    spectrum_power_sum = np.zeros(effective_fft, dtype=np.float64)
    histogram_ceiling = np.nextafter(_HISTOGRAM_MAX, _HISTOGRAM_MIN)
    for _, dbfs in _recording_power_chunks(
        recording,
        channel=channel,
        effective_fft=effective_fft,
        total_frames=total_frames,
        cancel=cancel,
    ):
        clipped = np.clip(
            dbfs,
            _HISTOGRAM_MIN,
            histogram_ceiling,
        )
        histogram += np.histogram(
            clipped,
            bins=_HISTOGRAM_EDGES,
        )[0].astype(np.uint64)
        spectrum_power_sum += np.sum(
            np.power(10.0, dbfs / 10.0),
            axis=0,
        )
    spectrum_dbfs = 10.0 * np.log10(
        np.maximum(spectrum_power_sum / total_frames, 1e-20)
    )
    signal_top = max(
        _histogram_percentile(histogram, 99.9),
        float(np.percentile(spectrum_dbfs, 99.5)),
    )
    return _rounded_range(
        _histogram_percentile(histogram, 10.0) - 3.0,
        signal_top + 3.0,
    )


def _frequency_bounds_mhz(
    recording: SigMFRecording,
    effective_fft: int,
) -> tuple[float, float]:
    centers = (
        recording.center_frequency_at(0)
        + np.fft.fftshift(
            np.fft.fftfreq(
                effective_fft,
                1.0 / recording.sample_rate,
            )
        )
    ) / 1e6
    spacing = recording.sample_rate / effective_fft / 1e6
    return (
        float(centers[0] - spacing / 2.0),
        float(centers[-1] + spacing / 2.0),
    )


def _validate_png_options(
    *,
    time_bins: int,
    width_pixels: int,
    height_pixels: int,
) -> None:
    values = {
        "time_bins": time_bins,
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values.values()
    ):
        raise ValueError("PNG bins, width, and height must be positive")
    if width_pixels < 640 or height_pixels < 480:
        raise ValueError("PNG dimensions must be at least 640 by 480")


def _shareable_recording_summary(
    recording: SigMFRecording,
    *,
    channel: int,
    effective_fft: int,
    total_frames: int,
    time_bins: int,
    cancel: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Summarize every STFT frame into a bounded max-hold PNG raster."""
    row_count = min(time_bins, total_frames)
    waterfall = np.full(
        (row_count, effective_fft),
        -np.inf,
        dtype=np.float32,
    )
    spectrum_power_sum = np.zeros(effective_fft, dtype=np.float64)
    histogram = np.zeros(_HISTOGRAM_EDGES.size - 1, dtype=np.uint64)
    histogram_ceiling = np.nextafter(_HISTOGRAM_MAX, _HISTOGRAM_MIN)
    for first_index, dbfs in _recording_power_chunks(
        recording,
        channel=channel,
        effective_fft=effective_fft,
        total_frames=total_frames,
        cancel=cancel,
    ):
        clipped = np.clip(
            dbfs,
            _HISTOGRAM_MIN,
            histogram_ceiling,
        )
        histogram += np.histogram(
            clipped,
            bins=_HISTOGRAM_EDGES,
        )[0].astype(np.uint64)
        spectrum_power_sum += np.sum(
            np.power(10.0, dbfs / 10.0),
            axis=0,
        )
        indexes = np.arange(
            first_index,
            first_index + dbfs.shape[0],
            dtype=np.int64,
        )
        rows = np.minimum(
            row_count - 1,
            indexes * row_count // total_frames,
        )
        for row in np.unique(rows):
            selected = rows == row
            waterfall[int(row)] = np.maximum(
                waterfall[int(row)],
                np.max(dbfs[selected], axis=0),
            )
    if not np.all(np.isfinite(waterfall)):
        raise ValueError("shareable PNG raster contains an empty time bin")
    spectrum = 10.0 * np.log10(
        np.maximum(spectrum_power_sum / total_frames, 1e-20)
    )
    signal_top = max(
        _histogram_percentile(histogram, 99.9),
        float(np.percentile(spectrum, 99.5)),
    )
    dbfs_min, dbfs_max = _rounded_range(
        _histogram_percentile(histogram, 10.0) - 3.0,
        signal_top + 3.0,
    )
    frequency_lower, frequency_upper = _frequency_bounds_mhz(
        recording,
        effective_fft,
    )
    return (
        np.linspace(
            frequency_lower,
            frequency_upper,
            effective_fft + 1,
        ),
        np.linspace(0.0, recording.duration_seconds, row_count + 1),
        waterfall,
        spectrum,
        dbfs_min,
        dbfs_max,
    )


def render_recording_png(
    recording: SigMFRecording,
    target: str | Path,
    *,
    channel: int = 0,
    fft_size: int = 256,
    overlap_percent: int = 50,
    colormap: str = "turbo",
    time_bins: int = 1600,
    width_pixels: int = 2400,
    height_pixels: int = 1600,
    cancel: Callable[[], None] | None = None,
) -> Path:
    """Render a fixed-size, full-recording PNG without skipping intervals."""
    _validate_render_options(
        fft_size=fft_size,
        overlap_percent=overlap_percent,
        max_native_cells=0,
        colormap=colormap,
    )
    _validate_png_options(
        time_bins=time_bins,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
    )
    if not 0 <= channel < recording.channel_count:
        raise ValueError("batch channel is outside the recording")
    effective_fft, _, total_frames = _stft_geometry(
        recording,
        fft_size,
        overlap_percent,
    )
    (
        frequency_edges,
        time_edges,
        waterfall,
        spectrum,
        dbfs_min,
        dbfs_max,
    ) = _shareable_recording_summary(
        recording,
        channel=channel,
        effective_fft=effective_fft,
        total_frames=total_frames,
        time_bins=time_bins,
        cancel=cancel,
    )
    spectrum_min, spectrum_max = _rounded_range(
        float(np.percentile(spectrum, 1.0)) - 3.0,
        float(np.percentile(spectrum, 99.9)) + 3.0,
    )
    dpi = 160
    figure = Figure(
        figsize=(width_pixels / dpi, height_pixels / dpi),
        dpi=dpi,
        layout="constrained",
        facecolor="#0d1d24",
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(0.10, 0.875, 0.025),
        wspace=0.03,
    )
    spectrum_axes = figure.add_subplot(grid[0, 0])
    waterfall_axes = figure.add_subplot(grid[0, 1], sharey=spectrum_axes)
    colorbar_axes = figure.add_subplot(grid[0, 2])
    axes = (spectrum_axes, waterfall_axes, colorbar_axes)
    for axis in axes:
        axis.set_facecolor("#10252d")
        axis.tick_params(colors="#d8e7ea")
        for spine in axis.spines.values():
            spine.set_color("#54717a")
    frequency_centers = (
        frequency_edges[:-1] + frequency_edges[1:]
    ) / 2.0
    spectrum_axes.plot(
        spectrum,
        frequency_centers,
        color="#50c8d3",
        linewidth=0.9,
    )
    spectrum_axes.set_xlabel("dBFS", color="#e7f1f3")
    spectrum_axes.set_ylabel("RF frequency (MHz)", color="#e7f1f3")
    spectrum_axes.set_xlim(spectrum_min, spectrum_max)
    spectrum_axes.grid(color="white", alpha=0.12, linewidth=0.45)
    image = waterfall_axes.pcolormesh(
        time_edges,
        frequency_edges,
        waterfall.T,
        shading="flat",
        cmap=colormap,
        vmin=dbfs_min,
        vmax=dbfs_max,
        rasterized=True,
    )
    waterfall_axes.set_xlabel("Recording time (s)", color="#e7f1f3")
    waterfall_axes.tick_params(labelleft=False)
    waterfall_axes.grid(color="white", alpha=0.08, linewidth=0.35)
    waterfall_axes.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    waterfall_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar = figure.colorbar(image, cax=colorbar_axes)
    colorbar.set_label("Power (dBFS)", color="#e7f1f3")
    colorbar.outline.set_edgecolor("#54717a")
    metadata = recording.metadata["global"]
    title = str(
        metadata.get("core:description")
        or recording.metadata_path.name.removesuffix(".sigmf-meta")
    )
    if recording.channel_count > 1:
        title = f"{title} · {recording.channel_labels[channel]}"
    figure.suptitle(
        (
            f"{title}\n"
            f"{recording.sample_rate / 1e6:g} MS/s · FFT {effective_fft} · "
            f"{overlap_percent}% overlap · complete recording max-hold"
        ),
        x=0.07,
        ha="left",
        color="#e7f1f3",
        fontsize=11,
    )

    destination = Path(target).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".png",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        if cancel is not None:
            cancel()
        figure.savefig(
            temporary,
            format="png",
            dpi=dpi,
            facecolor=figure.get_facecolor(),
            metadata={
                "Title": title,
                "Description": (
                    "Complete recording; every centered STFT frame contributes "
                    "to a bounded time bin using power-domain max-hold."
                ),
            },
        )
        if cancel is not None:
            cancel()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        figure.clear()
    return destination


def _save_tile(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        target,
        format="PNG",
        compress_level=6,
    )


def _color_lut(colormap: str) -> np.ndarray:
    """Return the exact 8-bit lookup table used by Matplotlib colormaps."""
    return np.asarray(
        matplotlib.colormaps[colormap](
            np.arange(256, dtype=np.float64) / 256.0,
            bytes=True,
        ),
        dtype=np.uint8,
    )[:, :3]


def _color_indexes(
    dbfs: np.ndarray,
    *,
    dbfs_min: float,
    dbfs_max: float,
) -> np.ndarray:
    normalized = np.clip(
        (dbfs - dbfs_min) / (dbfs_max - dbfs_min),
        0.0,
        1.0,
    )
    return np.minimum(
        np.floor(normalized * 256.0),
        255.0,
    ).astype(np.uint8)


def _render_native_tiles(
    recording: SigMFRecording,
    *,
    channel: int,
    effective_fft: int,
    total_frames: int,
    level: int,
    directory: Path,
    scalar_directory: Path,
    dbfs_min: float,
    dbfs_max: float,
    colormap: str,
    cancel: Callable[[], None] | None = None,
) -> None:
    level_directory = directory / str(level)
    scalar_level_directory = scalar_directory / str(level)
    color_lut = _color_lut(colormap)
    for first_index, dbfs in _recording_power_chunks(
        recording,
        channel=channel,
        effective_fft=effective_fft,
        total_frames=total_frames,
        cancel=cancel,
    ):
        indexes = _color_indexes(
            dbfs,
            dbfs_min=dbfs_min,
            dbfs_max=dbfs_max,
        )
        scalar_pixels = np.transpose(indexes[:, ::-1], (1, 0))
        pixels = color_lut[scalar_pixels]
        tile_x = first_index // _TILE_SIZE
        for tile_y, first_row in enumerate(
            range(0, effective_fft, _TILE_SIZE)
        ):
            tile = Image.fromarray(
                pixels[
                    first_row : first_row + _TILE_SIZE,
                    :,
                ],
                mode="RGB",
            )
            try:
                _save_tile(
                    tile,
                    level_directory / f"{tile_x}_{tile_y}.png",
                )
            finally:
                tile.close()
            scalar_tile = Image.fromarray(
                scalar_pixels[
                    first_row : first_row + _TILE_SIZE,
                    :,
                ],
                mode="L",
            )
            try:
                _save_tile(
                    scalar_tile,
                    scalar_level_directory / f"{tile_x}_{tile_y}.png",
                )
            finally:
                scalar_tile.close()


def _build_time_pyramid(
    *,
    native_width: int,
    native_height: int,
    max_level: int,
    directory: Path,
    scalar_directory: Path,
    color_lut: np.ndarray,
    cancel: Callable[[], None] | None = None,
) -> None:
    child_width = native_width
    vertical_tiles = ceil(native_height / _TILE_SIZE)
    for child_level in range(max_level, 0, -1):
        if cancel is not None:
            cancel()
        parent_level = child_level - 1
        parent_width = ceil(child_width / 2)
        horizontal_tiles = ceil(parent_width / _TILE_SIZE)
        for tile_x in range(horizontal_tiles):
            child_start = tile_x * _TILE_SIZE * 2
            region_width = min(
                _TILE_SIZE * 2,
                child_width - child_start,
            )
            for tile_y in range(vertical_tiles):
                if cancel is not None:
                    cancel()
                region_height = min(
                    _TILE_SIZE,
                    native_height - tile_y * _TILE_SIZE,
                )
                canvas = Image.new(
                    "L",
                    (region_width, region_height),
                    0,
                )
                try:
                    for child_offset in range(2):
                        child_x = tile_x * 2 + child_offset
                        child_path = (
                            scalar_directory
                            / str(child_level)
                            / f"{child_x}_{tile_y}.png"
                        )
                        if not child_path.is_file():
                            continue
                        child = Image.open(child_path)
                        try:
                            canvas.paste(
                                child,
                                (child_offset * _TILE_SIZE, 0),
                            )
                        finally:
                            child.close()
                    source = np.asarray(canvas, dtype=np.uint8)
                    even = source[:, 0::2]
                    odd_source = source[:, 1::2]
                    odd = np.zeros_like(even)
                    odd[:, : odd_source.shape[1]] = odd_source
                    parent_indexes = np.maximum(even, odd)
                    parent_scalar = Image.fromarray(
                        parent_indexes,
                        mode="L",
                    )
                    try:
                        _save_tile(
                            parent_scalar,
                            scalar_directory
                            / str(parent_level)
                            / f"{tile_x}_{tile_y}.png",
                        )
                    finally:
                        parent_scalar.close()
                    parent_color = Image.fromarray(
                        color_lut[parent_indexes],
                        mode="RGB",
                    )
                    try:
                        _save_tile(
                            parent_color,
                            directory
                            / str(parent_level)
                            / f"{tile_x}_{tile_y}.png",
                        )
                    finally:
                        parent_color.close()
                finally:
                    canvas.close()
        shutil.rmtree(scalar_directory / str(child_level))
        child_width = parent_width


def _colormap_gradient(colormap: str) -> str:
    function = matplotlib.colormaps[colormap]
    stops = []
    for index in range(17):
        fraction = index / 16.0
        red, green, blue, _ = function(fraction)
        stops.append(
            f"rgb({round(red * 255)} {round(green * 255)} "
            f"{round(blue * 255)}) {fraction * 100:g}%"
        )
    return ", ".join(stops)


_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SigMF waterfall</title>
<style>
:root{color-scheme:dark;--bg:#0d1d24;--panel:#132a34;--ink:#e7f1f3;--muted:#9eb5bc;--line:#31505b;--accent:#50c8d3}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--ink);font:13px system-ui,sans-serif}
body{display:grid;grid-template-rows:auto minmax(0,1fr)}
header{display:flex;min-width:0;align-items:center;gap:18px;min-height:58px;overflow:hidden;padding:9px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
.title{min-width:0;flex:1;overflow:hidden}.title h1{display:block;overflow:hidden;margin:0;font-size:15px;text-overflow:ellipsis;white-space:nowrap}.title p{display:block;overflow:hidden;margin:3px 0 0;color:var(--muted);font-size:11px;text-overflow:ellipsis;white-space:nowrap}
.controls{display:flex;flex:none;align-items:center;gap:6px;white-space:nowrap}.controls button,.controls a{flex:none;min-height:30px;padding:5px 9px;border:1px solid var(--line);border-radius:5px;color:var(--ink);background:#183640;font:600 11px system-ui;text-decoration:none;cursor:pointer}.controls button:hover,.controls a:hover{border-color:var(--accent);color:var(--accent)}
#plot{position:relative;min-width:0;min-height:0}
#stage{position:absolute;inset:12px 116px 55px 82px;overflow:hidden;border:1px solid var(--line);background:#07161c;cursor:grab;touch-action:none}
#stage.dragging{cursor:grabbing}#waterfall{display:block;width:100%;height:100%}
.axis-title{position:absolute;color:var(--muted);font-size:11px}.axis-title.x{right:116px;bottom:9px;left:82px;text-align:center}.axis-title.y{top:50%;left:12px;transform:translate(-35%,-50%) rotate(-90deg)}
#x-ticks{position:absolute;right:116px;bottom:30px;left:82px;height:18px}#y-ticks{position:absolute;top:12px;bottom:55px;left:5px;width:68px}
.tick{position:absolute;color:var(--muted);font:10px ui-monospace,monospace;white-space:nowrap}.x-tick{top:0;transform:translateX(-50%)}.y-tick{right:0;transform:translateY(-50%)}
#colorbar{position:absolute;top:12px;right:70px;bottom:55px;width:18px;border:1px solid var(--line);background:linear-gradient(to top,__COLOR_STOPS__)}
#colorbar-label{position:absolute;top:12px;right:12px;color:var(--muted);font:10px ui-monospace,monospace}
#colorbar-ticks{position:absolute;top:12px;right:8px;bottom:55px;width:54px}.dbfs-tick{right:0;transform:translateY(-50%)}
#status{position:absolute;top:18px;right:124px;padding:3px 6px;border-radius:4px;color:var(--muted);background:#07161ccc;font:10px ui-monospace,monospace;pointer-events:none}
</style>
</head>
<body>
<header>
  <div class="title"><h1 id="title"></h1><p id="subtitle"></p></div>
  <div class="controls">
    <a id="png" hidden target="_blank" rel="noopener">Full PNG</a>
    <button id="full" type="button">Full recording</button>
    <button id="zoom-out" type="button" aria-label="Zoom out">−</button>
    <button id="zoom-in" type="button" aria-label="Zoom in">+</button>
    <button id="native" type="button">1:1 time</button>
  </div>
</header>
<main id="plot">
  <div class="axis-title y">RF frequency (MHz)</div>
  <div id="y-ticks"></div>
  <div id="stage"><canvas id="waterfall"></canvas></div>
  <div id="x-ticks"></div>
  <div class="axis-title x">Recording time (s)</div>
  <div id="colorbar"></div>
  <div id="colorbar-label">dBFS</div>
  <div id="colorbar-ticks"></div>
  <div id="status"></div>
</main>
<script>
const config=__CONFIG__;
const stage=document.querySelector('#stage'),canvas=document.querySelector('#waterfall'),context=canvas.getContext('2d');
const cache=new Map(),cacheLimit=96;let left=0,span=config.width,drag=null,drawQueued=false;
document.querySelector('#title').textContent=config.title;
document.querySelector('#title').title=config.title;
document.querySelector('#subtitle').textContent=config.subtitle;
if(config.shareablePng){const png=document.querySelector('#png');png.href=config.shareablePng;png.hidden=false}
function clampView(){span=Math.max(1,Math.min(config.width,span));left=Math.max(0,Math.min(config.width-span,left))}
function levelForView(){const px=stage.clientWidth/span;if(px>=1)return config.maxLevel;const reduction=Math.max(0,Math.round(Math.log2(1/px)));return Math.max(0,config.maxLevel-reduction)}
function requestDraw(){if(drawQueued)return;drawQueued=true;requestAnimationFrame(()=>{drawQueued=false;draw()})}
function tile(level,x,y){const key=`${level}/${x}_${y}`;if(cache.has(key)){const hit=cache.get(key);cache.delete(key);cache.set(key,hit);return hit}const image=new Image();image.onload=requestDraw;image.src=`${config.assetRoot}/${level}/${x}_${y}.png`;cache.set(key,image);while(cache.size>cacheLimit)cache.delete(cache.keys().next().value);return image}
function labels(){const xTicks=document.querySelector('#x-ticks'),yTicks=document.querySelector('#y-ticks'),dbfs=document.querySelector('#colorbar-ticks');xTicks.innerHTML='';yTicks.innerHTML='';dbfs.innerHTML='';for(let i=0;i<5;i++){const f=i/4,x=document.createElement('span');x.className='tick x-tick';x.style.left=`${f*100}%`;x.textContent=((left+span*f)/config.width*config.durationSeconds).toFixed(2);xTicks.append(x);const y=document.createElement('span');y.className='tick y-tick';y.style.top=`${f*100}%`;y.textContent=(config.frequencyUpperMHz-(config.frequencyUpperMHz-config.frequencyLowerMHz)*f).toFixed(3);yTicks.append(y);const d=document.createElement('span');d.className='tick dbfs-tick';d.style.bottom=`${f*100}%`;d.textContent=(config.dbfsMin+(config.dbfsMax-config.dbfsMin)*f).toFixed(0);dbfs.append(d)}}
function draw(){clampView();const ratio=window.devicePixelRatio||1,width=Math.max(1,stage.clientWidth),height=Math.max(1,stage.clientHeight);if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio)}context.setTransform(ratio,0,0,ratio,0,0);context.imageSmoothingEnabled=false;context.fillStyle='#07161c';context.fillRect(0,0,width,height);const level=levelForView(),down=2**(config.maxLevel-level),levelWidth=Math.ceil(config.width/down),levelLeft=left/down,levelSpan=span/down,first=Math.max(0,Math.floor(levelLeft/config.tileSize)),last=Math.min(Math.ceil(levelWidth/config.tileSize)-1,Math.floor((levelLeft+levelSpan)/config.tileSize)),vertical=Math.ceil(config.height/config.tileSize);for(let x=first;x<=last;x++)for(let y=0;y<vertical;y++){const image=tile(level,x,y);if(!image.complete||!image.naturalWidth)continue;const dx=(x*config.tileSize-levelLeft)/levelSpan*width,dw=image.naturalWidth/levelSpan*width,dy=y*config.tileSize/config.height*height,dh=image.naturalHeight/config.height*height;context.drawImage(image,dx,dy,dw,dh)}labels();const nativePerPixel=span/width;document.querySelector('#status').textContent=`${(left/config.width*config.durationSeconds).toFixed(2)}–${((left+span)/config.width*config.durationSeconds).toFixed(2)} s · level ${level}/${config.maxLevel} · ${nativePerPixel<=1?'native STFT cells':nativePerPixel.toFixed(1)+' frames/screen px'}`;}
function zoom(factor,anchor=.5){const point=left+span*anchor,newSpan=Math.max(1,Math.min(config.width,span*factor));left=point-newSpan*anchor;span=newSpan;requestDraw()}
document.querySelector('#full').onclick=()=>{left=0;span=config.width;requestDraw()};
document.querySelector('#native').onclick=()=>{const center=left+span/2;span=Math.min(config.width,stage.clientWidth);left=center-span/2;requestDraw()};
document.querySelector('#zoom-in').onclick=()=>zoom(.5);
document.querySelector('#zoom-out').onclick=()=>zoom(2);
stage.onwheel=event=>{event.preventDefault();const rect=stage.getBoundingClientRect(),horizontal=Math.abs(event.deltaX)>Math.abs(event.deltaY);if(horizontal||event.shiftKey){const delta=horizontal?event.deltaX:event.deltaY;left+=delta/stage.clientWidth*span;requestDraw();return}zoom(Math.exp(event.deltaY*.0015),(event.clientX-rect.left)/rect.width)};
stage.onpointerdown=event=>{stage.setPointerCapture(event.pointerId);stage.classList.add('dragging');drag={x:event.clientX,left}};
stage.onpointermove=event=>{if(!drag)return;left=drag.left-(event.clientX-drag.x)/stage.clientWidth*span;requestDraw()};
stage.onpointerup=stage.onpointercancel=()=>{drag=null;stage.classList.remove('dragging')};
stage.ondblclick=()=>{left=0;span=config.width;requestDraw()};
new ResizeObserver(requestDraw).observe(stage);requestDraw();
</script>
</body>
</html>
"""


def _viewer_html(
    *,
    configuration: dict[str, object],
    colormap: str,
) -> str:
    return (
        _VIEWER_HTML
        .replace("__COLOR_STOPS__", _colormap_gradient(colormap))
        .replace(
            "__CONFIG__",
            json.dumps(
                configuration,
                ensure_ascii=True,
                separators=(",", ":"),
            ).replace("</", "<\\/"),
        )
    )


def render_recording_viewer(
    recording: SigMFRecording,
    target: str | Path,
    *,
    channel: int = 0,
    fft_size: int = 256,
    overlap_percent: int = 50,
    colormap: str = "turbo",
    max_native_cells: int = 75_000_000,
    shareable_png: str | None = None,
    cancel: Callable[[], None] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Render a bounded tiled viewer with exact cells at maximum zoom."""
    _validate_render_options(
        fft_size=fft_size,
        overlap_percent=overlap_percent,
        max_native_cells=max_native_cells,
        colormap=colormap,
    )
    if not 0 <= channel < recording.channel_count:
        raise ValueError("batch channel is outside the recording")
    if cancel is not None:
        cancel()

    destination = Path(target).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    effective_fft, _, total_frames = _stft_geometry(
        recording,
        fft_size,
        overlap_percent,
    )
    native_cells = total_frames * effective_fft
    if max_native_cells and native_cells > max_native_cells:
        pyramid_gib = native_cells * 6 / 1024**3
        raise ValueError(
            f"full-resolution waterfall requires {native_cells:,} native cells "
            f"(about {pyramid_gib:.1f} GiB for the uncompressed RGB tile "
            "pyramid); configured "
            f"limit is {max_native_cells:,}. Increase batch_max_native_cells "
            "or set it to 0 only when that portable output size is intentional. "
            "For long recordings, use the interactive windowed viewer: it "
            "computes the exact native cells for only the visible interval."
        )
    dbfs_min, dbfs_max = _discover_dbfs_range(
        recording,
        channel=channel,
        effective_fft=effective_fft,
        total_frames=total_frames,
        cancel=cancel,
    )
    max_level = max(0, ceil(log2(total_frames)))
    frequency_lower, frequency_upper = _frequency_bounds_mhz(
        recording,
        effective_fft,
    )
    metadata = recording.metadata["global"]
    title = str(
        metadata.get("core:description")
        or recording.metadata_path.name.removesuffix(".sigmf-meta")
    )
    if recording.channel_count > 1:
        title = f"{title} · {recording.channel_labels[channel]}"
    subtitle = (
        f"{recording.sample_rate / 1e6:g} MS/s · FFT {effective_fft} · "
        f"{overlap_percent}% overlap · {total_frames:,} exact STFT frames · "
        f"{dbfs_min:g} to {dbfs_max:g} dBFS"
    )

    final_assets = destination.with_name(f"{destination.stem}.assets")
    temporary_assets = Path(
        mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.stem}.assets.",
        )
    )
    scalar_directory = Path(
        mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.stem}.scalars.",
        )
    )
    temporary_html: Path | None = None
    backup_assets: Path | None = None
    try:
        _render_native_tiles(
            recording,
            channel=channel,
            effective_fft=effective_fft,
            total_frames=total_frames,
            level=max_level,
            directory=temporary_assets,
            scalar_directory=scalar_directory,
            dbfs_min=dbfs_min,
            dbfs_max=dbfs_max,
            colormap=colormap,
            cancel=cancel,
        )
        _build_time_pyramid(
            native_width=total_frames,
            native_height=effective_fft,
            max_level=max_level,
            directory=temporary_assets,
            scalar_directory=scalar_directory,
            color_lut=_color_lut(colormap),
            cancel=cancel,
        )
        configuration = {
            "title": title,
            "subtitle": subtitle,
            "assetRoot": final_assets.name,
            "width": total_frames,
            "height": effective_fft,
            "durationSeconds": recording.duration_seconds,
            "frequencyLowerMHz": frequency_lower,
            "frequencyUpperMHz": frequency_upper,
            "dbfsMin": dbfs_min,
            "dbfsMax": dbfs_max,
            "tileSize": _TILE_SIZE,
            "maxLevel": max_level,
            "nativeCells": native_cells,
            "shareablePng": shareable_png,
        }
        (temporary_assets / "metadata.json").write_text(
            json.dumps(configuration, indent=2) + "\n",
            encoding="utf-8",
        )
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".html",
            delete=False,
        ) as stream:
            temporary_html = Path(stream.name)
            stream.write(
                _viewer_html(
                    configuration=configuration,
                    colormap=colormap,
                ).encode("utf-8")
            )
        if cancel is not None:
            cancel()
        if final_assets.exists():
            backup_assets = final_assets.with_name(
                f".{final_assets.name}.previous.{uuid4().hex}"
            )
            final_assets.replace(backup_assets)
        try:
            temporary_assets.replace(final_assets)
            temporary_html.replace(destination)
        except BaseException:
            if final_assets.exists():
                shutil.rmtree(final_assets)
            if backup_assets is not None:
                backup_assets.replace(final_assets)
                backup_assets = None
            raise
        if backup_assets is not None:
            shutil.rmtree(backup_assets)
            backup_assets = None
    finally:
        if temporary_assets.exists():
            shutil.rmtree(temporary_assets)
        if scalar_directory.exists():
            shutil.rmtree(scalar_directory)
        if temporary_html is not None:
            temporary_html.unlink(missing_ok=True)
        if backup_assets is not None and backup_assets.exists():
            if final_assets.exists():
                shutil.rmtree(backup_assets)
            else:
                backup_assets.replace(final_assets)

    assets = tuple(
        path
        for path in sorted(final_assets.rglob("*"))
        if path.is_file()
    )
    return destination, assets


class SigMFWaterfallBatch(Batch[SigMFSource]):
    """Durable whole-recording PNGs and exact tiled waterfall viewers."""

    item_actions = (
        CapabilityChoice(
            RENDER_WATERFALL,
            "Render full-recording PNG and tiled viewer",
        ),
    )
    workspace_actions = (
        CapabilityChoice(
            RENDER_WATERFALL,
            "Render all full-recording PNGs and tiled viewers",
        ),
    )

    def __init__(
        self,
        output_root: str | Path,
        *,
        fft_size: int = 256,
        overlap_percent: int = 50,
        colormap: str = "turbo",
        max_native_cells: int = 75_000_000,
        png_time_bins: int = 1600,
        png_width_pixels: int = 2400,
        png_height_pixels: int = 1600,
    ) -> None:
        _validate_render_options(
            fft_size=fft_size,
            overlap_percent=overlap_percent,
            max_native_cells=max_native_cells,
            colormap=colormap,
        )
        _validate_png_options(
            time_bins=png_time_bins,
            width_pixels=png_width_pixels,
            height_pixels=png_height_pixels,
        )
        self.output_root = Path(output_root).expanduser().resolve()
        self.fft_size = fft_size
        self.overlap_percent = overlap_percent
        self.colormap = colormap
        self.max_native_cells = max_native_cells
        self.png_time_bins = png_time_bins
        self.png_width_pixels = png_width_pixels
        self.png_height_pixels = png_height_pixels

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "-",
            value.lower(),
        ).strip("-")

    def _filename_stem(
        self,
        resource: DataResource,
        recording: SigMFRecording,
        channel: int,
        *,
        collection: SigMFCollection | None,
    ) -> str:
        slug = self._slug(resource.identifier)
        if collection is not None:
            member = recording.metadata_path.relative_to(
                collection.collection_path.parent
            ).as_posix()
            slug = f"{slug}-{self._slug(member.removesuffix('.sigmf-meta'))}"
        channel_slug = self._slug(recording.channel_labels[channel])
        return (
            f"{slug}-{channel_slug}-waterfall-"
            f"fft{self.fft_size}-{self.overlap_percent}pct-"
            f"{self._slug(self.colormap)}"
        )

    def _filename(
        self,
        resource: DataResource,
        recording: SigMFRecording,
        channel: int,
        *,
        collection: SigMFCollection | None,
        kind: str,
    ) -> str:
        stem = self._filename_stem(
            resource,
            recording,
            channel,
            collection=collection,
        )
        if kind == "png":
            return f"{stem}-{self.png_width_pixels}x{self.png_height_pixels}.png"
        if kind == "viewer":
            return f"{stem}-tiled.html"
        raise ValueError(f"unknown waterfall output kind: {kind}")

    @staticmethod
    def _recordings(source: SigMFSource) -> tuple[SigMFRecording, ...]:
        return source.members if isinstance(source, SigMFCollection) else (source,)

    def _supports_tiled_viewer(self, recording: SigMFRecording) -> bool:
        effective_fft, _, total_frames = _stft_geometry(
            recording,
            self.fft_size,
            self.overlap_percent,
        )
        return (
            self.max_native_cells == 0
            or total_frames * effective_fft <= self.max_native_cells
        )

    def _filenames(
        self,
        resource: DataResource,
        source: SigMFSource | None = None,
    ) -> tuple[str, ...]:
        opened = open_source(resource.source) if source is None else source
        collection = opened if isinstance(opened, SigMFCollection) else None
        filenames = []
        for recording in self._recordings(opened):
            for channel in range(recording.channel_count):
                filenames.append(
                    self._filename(
                        resource,
                        recording,
                        channel,
                        collection=collection,
                        kind="png",
                    )
                )
                if self._supports_tiled_viewer(recording):
                    filenames.append(
                        self._filename(
                            resource,
                            recording,
                            channel,
                            collection=collection,
                            kind="viewer",
                        )
                    )
        return tuple(filenames)

    def item_destination(
        self,
        resource: DataResource,
        request: BatchRequest,
    ) -> BatchDestination:
        return BatchDestination(
            self.output_root,
            self._filenames(resource),
            "Full-recording waterfall results are ready",
        )

    def workspace_destination(
        self,
        resources: tuple[DataResource, ...],
        request: BatchRequest,
    ) -> BatchDestination:
        return BatchDestination(
            self.output_root,
            tuple(
                filename
                for resource in resources
                for filename in self._filenames(resource)
            ),
            "All full-recording waterfall results are ready",
        )

    def _render(
        self,
        resource: DataResource,
        source: SigMFSource,
        directory: Path,
        request: BatchRequest,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        collection = source if isinstance(source, SigMFCollection) else None
        files = []
        assets = []
        for recording in self._recordings(source):
            for channel in range(recording.channel_count):
                png_target = render_recording_png(
                    recording,
                    directory
                    / self._filename(
                        resource,
                        recording,
                        channel,
                        collection=collection,
                        kind="png",
                    ),
                    channel=channel,
                    fft_size=self.fft_size,
                    overlap_percent=self.overlap_percent,
                    colormap=self.colormap,
                    time_bins=self.png_time_bins,
                    width_pixels=self.png_width_pixels,
                    height_pixels=self.png_height_pixels,
                    cancel=request.raise_if_cancelled,
                )
                files.append(png_target)
                if self._supports_tiled_viewer(recording):
                    target, support = render_recording_viewer(
                        recording,
                        directory
                        / self._filename(
                            resource,
                            recording,
                            channel,
                            collection=collection,
                            kind="viewer",
                        ),
                        channel=channel,
                        fft_size=self.fft_size,
                        overlap_percent=self.overlap_percent,
                        colormap=self.colormap,
                        max_native_cells=self.max_native_cells,
                        shareable_png=png_target.name,
                        cancel=request.raise_if_cancelled,
                    )
                    files.append(target)
                    assets.extend(support)
        return tuple(files), tuple(assets)

    def run_item(
        self,
        resource: DataResource,
        source_data: SigMFSource,
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        files, assets = self._render(
            resource,
            source_data,
            directory,
            request,
        )
        return BatchResult(
            files,
            f"Rendered {len(files)} full-recording waterfall output(s)",
            assets,
        )

    def run_workspace(
        self,
        resources: tuple[DataResource, ...],
        open_resource,
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        groups = request.each(
            resources,
            lambda resource: self._render(
                resource,
                open_resource(resource),
                directory,
                request,
            ),
        )
        files = tuple(
            path
            for group_files, _ in groups
            for path in group_files
        )
        assets = tuple(
            path
            for _, group_assets in groups
            for path in group_assets
        )
        return BatchResult(
            files,
            f"Rendered {len(files)} full-recording waterfall output(s)",
            assets,
        )


__all__ = [
    "RENDER_WATERFALL",
    "SigMFWaterfallBatch",
    "render_recording_png",
    "render_recording_viewer",
]
