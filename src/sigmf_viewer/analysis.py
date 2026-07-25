"""Framework-independent spectral processing for exact SigMF sample windows."""

from __future__ import annotations

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
    starts = np.arange(
        0,
        max(1, samples.size - fft_size + 1),
        hop,
        dtype=np.int64,
    )
    blocks = np.asarray([samples[start : start + fft_size] for start in starts])
    if blocks.shape[1] < fft_size:
        blocks = np.pad(
            blocks,
            ((0, 0), (0, fft_size - blocks.shape[1])),
        )
    taper = np.hanning(fft_size)
    spectra = np.fft.fftshift(
        np.fft.fft(blocks * taper, axis=1),
        axes=1,
    )
    power = (np.abs(spectra) / max(float(np.sum(taper)), 1.0)) ** 2
    waterfall = 10.0 * np.log10(np.maximum(power, 1e-20))
    average = 10.0 * np.log10(np.maximum(np.mean(power, axis=0), 1e-20))
    frequency = (
        data.recording.center_frequency_at(data.start_sample)
        + np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / data.recording.sample_rate))
    ) / 1e6
    centers = starts + fft_size / 2.0
    time_edges = cell_edges(
        centers,
        0.0,
        float(data.sample_count),
    )
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
