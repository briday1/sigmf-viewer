"""Framework-independent spectral processing for exact SigMF sample windows."""

from __future__ import annotations

from math import ceil

import numpy as np

from .models import SigMFWindow, WaterfallProducts, WaterfallSettings

DEFAULT_SETTINGS = WaterfallSettings()


def cell_edges(
    centers: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    if centers.size == 1:
        return np.asarray([lower, upper], dtype=np.float64)
    return np.concatenate(([lower], (centers[:-1] + centers[1:]) / 2.0, [upper]))


def analyze(
    data: SigMFWindow,
    settings: WaterfallSettings = DEFAULT_SETTINGS,
) -> WaterfallProducts:
    """Compute a normalized STFT waterfall and average spectrum."""
    if (
        isinstance(settings.fft_size, bool)
        or not isinstance(settings.fft_size, int)
        or settings.fft_size < 1
    ):
        raise ValueError("fft_size must be a positive integer")
    if not 0 <= settings.overlap_percent < 100:
        raise ValueError("overlap_percent must be between 0 and 99")
    if not 0 <= settings.channel < data.recording.channel_count:
        raise ValueError("channel is outside the recording")
    samples = data.channel(settings.channel)
    if not samples.size:
        raise ValueError("the selected window does not contain samples")
    fft_size = min(settings.fft_size, samples.size)
    hop = max(
        1,
        round(fft_size * (1.0 - settings.overlap_percent / 100.0)),
    )
    frame_count = max(1, ceil(samples.size / hop))
    time_edges = np.linspace(0.0, float(samples.size), frame_count + 1)
    centers = (time_edges[:-1] + time_edges[1:]) / 2.0
    taper = np.hanning(fft_size) if fft_size > 2 else np.ones(fft_size)
    blocks = np.zeros((frame_count, fft_size), dtype=samples.dtype)
    normalizers = np.empty(frame_count, dtype=np.float64)
    for index, center in enumerate(centers):
        start = int(np.rint(center - fft_size / 2.0))
        source_start = max(0, start)
        source_stop = min(samples.size, start + fft_size)
        target_start = source_start - start
        target_stop = target_start + source_stop - source_start
        blocks[index, target_start:target_stop] = samples[source_start:source_stop]
        normalizers[index] = max(
            float(np.sum(taper[target_start:target_stop])),
            1.0,
        )
    spectra = np.fft.fftshift(
        np.fft.fft(blocks * taper, axis=1),
        axes=1,
    )
    power = (np.abs(spectra) / normalizers[:, None]) ** 2
    waterfall = 10.0 * np.log10(np.maximum(power, 1e-20))
    average = 10.0 * np.log10(np.maximum(np.mean(power, axis=0), 1e-20))
    frequency = (
        data.recording.center_frequency_at(data.start_sample)
        + np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / data.recording.sample_rate))
    ) / 1e6
    return WaterfallProducts(
        recording=data.recording,
        start_sample=data.start_sample,
        channel=settings.channel,
        spectrum_dbfs=average,
        waterfall_dbfs=waterfall,
        frequency_mhz=frequency,
        time_edges_ms=(
            (data.start_sample + time_edges) / data.recording.sample_rate * 1e3
        ),
        buffer_nbytes=data.buffer_nbytes,
    )


__all__ = ["analyze", "cell_edges"]
