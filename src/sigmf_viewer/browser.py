"""Bounded browser adapter; all sample decoding and DSP use the native core."""

from __future__ import annotations

import json
from math import ceil, isfinite

import numpy as np

from .analysis import analyze
from .models import WaterfallSettings
from .sigmf import SIGMF_DATATYPES, open_recording, read_window

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_SAMPLES = 65536
MAX_CHANNELS = 16
MAX_CELLS = 524288
MAX_RASTER_ROWS = 192
MAX_RASTER_COLUMNS = 512


def validate_metadata(metadata: dict, payload_bytes: int) -> None:
    """Reject unsupported layouts and large inputs before decoding samples."""
    if not 0 < payload_bytes <= MAX_FILE_BYTES:
        raise ValueError("Sample payload must be between 1 byte and 32 MiB")
    if not isinstance(metadata, dict):
        raise ValueError("SigMF metadata must be an object")
    if len(json.dumps(metadata).encode()) > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds 1 MiB")
    global_metadata = metadata.get("global")
    if not isinstance(global_metadata, dict):
        raise ValueError("Missing global metadata object")
    channels = global_metadata.get("core:num_channels", 1)
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise ValueError("Channel count must be an integer")
    if not 1 <= channels <= MAX_CHANNELS:
        raise ValueError("Browser supports 1–16 channels")
    for key in ("core:trailing_bytes",):
        if global_metadata.get(key, 0):
            raise ValueError(f"{key} is not supported in the browser")
    captures = metadata.get("captures", [])
    if not isinstance(captures, list):
        raise ValueError("Captures must be an array")
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("Captures must be objects")
        if capture.get("core:header_bytes", 0):
            raise ValueError("Capture headers are not supported in the browser")
        if not isfinite(float(capture.get("core:frequency", 0))):
            raise ValueError("Capture frequency must be finite")
    entries = metadata.get("annotations", [])
    if not isinstance(entries, list) or any(not isinstance(a, dict) for a in entries):
        raise ValueError("Annotations must be an array of objects")


def open_browser_recording(path):
    recording = open_recording(path)
    validate_metadata(recording.metadata, recording.data_path.stat().st_size)
    return recording


def validate_request(recording, start, count, fft_size, overlap, channel):
    for name, value in (
        ("start", start), ("count", count), ("FFT size", fft_size),
        ("overlap", overlap), ("channel", channel),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if not 0 <= start < recording.sample_count:
        raise ValueError("Start sample is outside the recording")
    if not 1 <= count <= min(MAX_SAMPLES, recording.sample_count - start):
        raise ValueError("Window must contain 1–65,536 available samples")
    if fft_size not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError("FFT size must be a power of two from 64 through 4096")
    if not 0 <= overlap <= 90:
        raise ValueError("Overlap must be 0–90 percent")
    if not 0 <= channel < recording.channel_count <= MAX_CHANNELS:
        raise ValueError("Channel is outside the recording")
    size = min(fft_size, count)
    hop = max(1, round(size * (1 - overlap / 100)))
    cells = max(1, ceil(count / hop)) * size
    if cells > MAX_CELLS:
        raise ValueError("STFT exceeds 524,288 cells; reduce window or overlap")
    # A tuning change inside one STFT would make its frequency axis misleading.
    for capture in recording.metadata.get("captures", []):
        capture_start = capture.get("core:sample_start", recording.sample_offset)
        if not isinstance(capture_start, int):
            raise ValueError("Capture sample starts must be integers")
        local = capture_start - recording.sample_offset
        if start < local < start + count:
            raise ValueError(f"Window crosses capture boundary at sample {local}")
    return cells


def _reduce_power(db, axis, limit):
    """Average linear power in nonempty contiguous bins, never average dB."""
    length = db.shape[axis]
    edges = np.linspace(0, length, min(length, limit) + 1, dtype=int)
    power = np.power(10.0, db / 10.0)
    reduced = np.add.reduceat(power, edges[:-1], axis=axis)
    shape = [1] * db.ndim
    shape[axis] = edges.size - 1
    reduced /= np.diff(edges).reshape(shape)
    return 10.0 * np.log10(np.maximum(reduced, 1e-20)), edges


def browser_analyze(recording, request):
    start = request["start"]
    count = request["count"]
    fft_size = request["fft"]
    overlap = request["overlap"]
    channel = request["channel"]
    cells = validate_request(recording, start, count, fft_size, overlap, channel)
    window = read_window(recording, start, count)
    products = analyze(window, WaterfallSettings(fft_size, overlap, channel))
    frequency = products.frequency_mhz
    selection = np.arange(frequency.size)
    limits = request.get("frequency")
    if limits is not None:
        if len(limits) != 2 or not all(isfinite(float(v)) for v in limits):
            raise ValueError("Frequency limits must be finite")
        selection = selection[(frequency >= min(limits)) & (frequency <= max(limits))]
        if not selection.size:
            raise ValueError("Zoom contains no FFT bins; reset the frequency view")
    waterfall = products.waterfall_dbfs[:, selection]
    waterfall, y_edges = _reduce_power(waterfall, 0, MAX_RASTER_ROWS)
    waterfall, x_edges = _reduce_power(waterfall, 1, MAX_RASTER_COLUMNS)
    spectrum, _ = _reduce_power(products.spectrum_dbfs[selection], 0, MAX_RASTER_COLUMNS)
    x = [float(np.mean(frequency[selection[a:b]])) for a, b in zip(x_edges, x_edges[1:])]
    times = products.time_edges_ms
    y = [(float(times[a]) + float(times[b])) / 2 for a, b in zip(y_edges, y_edges[1:])]
    fmt = SIGMF_DATATYPES[recording.datatype]
    return {
        "x": x, "y": y, "z": waterfall.tolist(), "spectrum": spectrum.tolist(),
        "start": start, "count": count, "channel": channel,
        "metrics": {
            "inputBytes": recording.data_path.stat().st_size,
            "windowBytes": window.buffer_nbytes,
            "nativeCells": cells,
            "rasterCells": waterfall.size,
            "rawWindowBytes": count * recording.channel_count * fmt.components
            * np.dtype(fmt.dtype).itemsize,
        },
    }
