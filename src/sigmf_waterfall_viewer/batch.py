"""Durable, bounded-memory high-resolution waterfall rendering."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sigvue import (
    Batch,
    BatchDestination,
    BatchRequest,
    BatchResult,
    CapabilityChoice,
    DataResource,
)

from .analysis import cell_edges
from .models import SigMFCollection, SigMFRecording, SigMFSource
from .sigmf import open_source

RENDER_WATERFALL = "render-high-resolution-waterfall"


def _validate_render_options(
    *,
    fft_size: int,
    overlap_percent: int,
    time_bins: int,
    width_pixels: int,
    height_pixels: int,
) -> None:
    integers = {
        "fft_size": fft_size,
        "time_bins": time_bins,
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in integers.values()
    ):
        raise ValueError("batch FFT, bins, width, and height must be positive")
    if not 0 <= overlap_percent < 100:
        raise ValueError("batch overlap must be between 0 and 99")
    if width_pixels < 640 or height_pixels < 480:
        raise ValueError("batch image dimensions must be at least 640 by 480")


def _recording_power_raster(
    recording: SigMFRecording,
    *,
    channel: int,
    fft_size: int,
    overlap_percent: int,
    time_bins: int,
    chunk_frames: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read every STFT frame and aggregate power into bounded output rows."""
    if not 0 <= channel < recording.channel_count:
        raise ValueError("batch channel is outside the recording")
    effective_fft = min(fft_size, recording.sample_count)
    hop = max(
        1,
        round(effective_fft * (1.0 - overlap_percent / 100.0)),
    )
    total_frames = max(
        1,
        1 + (recording.sample_count - effective_fft) // hop,
    )
    row_count = min(time_bins, total_frames)
    accumulated = np.zeros((row_count, effective_fft), dtype=np.float64)
    row_counts = np.zeros(row_count, dtype=np.int64)
    spectrum_sum = np.zeros(effective_fft, dtype=np.float64)
    taper = np.hanning(effective_fft)
    normalization = max(float(np.sum(taper)), 1.0)

    for first_index in range(0, total_frames, chunk_frames):
        frame_indexes = np.arange(
            first_index,
            min(total_frames, first_index + chunk_frames),
            dtype=np.int64,
        )
        starts = frame_indexes * hop
        read_start = int(starts[0])
        read_stop = int(starts[-1] + effective_fft)
        samples = recording.read(
            read_start,
            read_stop - read_start,
        )[channel]
        blocks = np.asarray(
            [
                samples[
                    int(start - read_start) : int(start - read_start) + effective_fft
                ]
                for start in starts
            ]
        )
        spectra = np.fft.fftshift(
            np.fft.fft(blocks * taper, axis=1),
            axes=1,
        )
        power = (np.abs(spectra) / normalization) ** 2
        spectrum_sum += np.sum(power, axis=0)
        rows = np.minimum(
            row_count - 1,
            frame_indexes * row_count // total_frames,
        )
        for row in np.unique(rows):
            selected = rows == row
            accumulated[int(row)] += np.sum(power[selected], axis=0)
            row_counts[int(row)] += int(np.count_nonzero(selected))

    power_raster = accumulated / row_counts[:, None]
    average_power = spectrum_sum / total_frames
    center_frequency = recording.center_frequency_at(0)
    frequency_centers = (
        center_frequency
        + np.fft.fftshift(
            np.fft.fftfreq(
                effective_fft,
                1.0 / recording.sample_rate,
            )
        )
    ) / 1e6
    frequency_edges = cell_edges(
        frequency_centers,
        float(frequency_centers[0] - recording.sample_rate / effective_fft / 2.0 / 1e6),
        float(
            frequency_centers[-1] + recording.sample_rate / effective_fft / 2.0 / 1e6
        ),
    )
    time_edges = np.linspace(
        0.0,
        recording.duration_seconds,
        row_count + 1,
    )
    return (
        frequency_edges,
        time_edges,
        10.0 * np.log10(np.maximum(power_raster, 1e-20)),
        10.0 * np.log10(np.maximum(average_power, 1e-20)),
    )


def _rounded_range(lower: float, upper: float) -> tuple[float, float]:
    low = max(-200.0, 5.0 * np.floor(lower / 5.0))
    high = min(20.0, 5.0 * np.ceil(upper / 5.0))
    if high - low < 20.0:
        low = max(-200.0, high - 20.0)
    return float(low), float(high)


def render_recording_png(
    recording: SigMFRecording,
    target: str | Path,
    *,
    channel: int = 0,
    fft_size: int = 2048,
    overlap_percent: int = 50,
    time_bins: int = 1200,
    width_pixels: int = 2400,
    height_pixels: int = 1600,
) -> Path:
    """Render the complete recording to a deterministic PNG atomically."""
    _validate_render_options(
        fft_size=fft_size,
        overlap_percent=overlap_percent,
        time_bins=time_bins,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
    )
    if not 0 <= channel < recording.channel_count:
        raise ValueError("batch channel is outside the recording")
    (
        frequency_edges,
        time_edges,
        waterfall_dbfs,
        spectrum_dbfs,
    ) = _recording_power_raster(
        recording,
        channel=channel,
        fft_size=fft_size,
        overlap_percent=overlap_percent,
        time_bins=time_bins,
    )
    finite_waterfall = waterfall_dbfs[np.isfinite(waterfall_dbfs)]
    finite_spectrum = spectrum_dbfs[np.isfinite(spectrum_dbfs)]
    zmin, zmax = _rounded_range(
        float(np.percentile(finite_waterfall, 10.0)) - 3.0,
        max(
            float(np.percentile(finite_waterfall, 99.9)),
            float(np.percentile(finite_spectrum, 99.5)),
        )
        + 3.0,
    )
    spectrum_min, spectrum_max = _rounded_range(
        float(np.percentile(finite_spectrum, 1.0)) - 3.0,
        float(np.percentile(finite_spectrum, 99.9)) + 3.0,
    )
    dpi = 160
    figure = plt.figure(
        figsize=(width_pixels / dpi, height_pixels / dpi),
        dpi=dpi,
        layout="constrained",
    )
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.025),
        height_ratios=(0.10, 0.90),
        hspace=0.04,
        wspace=0.02,
    )
    spectrum_axes = figure.add_subplot(grid[0, 0])
    waterfall_axes = figure.add_subplot(
        grid[1, 0],
        sharex=spectrum_axes,
    )
    colorbar_axes = figure.add_subplot(grid[1, 1])
    frequency_centers = (frequency_edges[:-1] + frequency_edges[1:]) / 2.0
    spectrum_axes.plot(
        frequency_centers,
        spectrum_dbfs,
        color="#087e8b",
        linewidth=0.9,
    )
    spectrum_axes.set_ylabel("dBFS")
    spectrum_axes.set_ylim(spectrum_min, spectrum_max)
    spectrum_axes.grid(alpha=0.18, linewidth=0.5)
    spectrum_axes.tick_params(labelbottom=False)
    image = waterfall_axes.pcolormesh(
        frequency_edges,
        time_edges,
        waterfall_dbfs,
        shading="flat",
        cmap="turbo",
        vmin=zmin,
        vmax=zmax,
        rasterized=True,
    )
    waterfall_axes.set_xlabel("RF frequency (MHz)")
    waterfall_axes.set_ylabel("Recording time (s)")
    waterfall_axes.grid(
        color="white",
        alpha=0.10,
        linewidth=0.35,
    )
    waterfall_axes.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    waterfall_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar = figure.colorbar(image, cax=colorbar_axes)
    colorbar.set_label("Power (dBFS)")
    metadata = recording.metadata["global"]
    title = str(
        metadata.get("core:description")
        or recording.metadata_path.name.removesuffix(".sigmf-meta")
    )
    if recording.channel_count > 1:
        title = f"{title} · Channel {channel + 1}"
    figure.suptitle(
        (
            f"{title}\n"
            f"{recording.sample_rate / 1e6:g} MS/s · "
            f"FFT {min(fft_size, recording.sample_count)} · "
            f"{overlap_percent}% overlap"
        ),
        x=0.07,
        ha="left",
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
        figure.savefig(
            temporary,
            format="png",
            dpi=dpi,
            metadata={
                "Title": title,
                "Description": (
                    "Every STFT frame was read; power is averaged only when "
                    "multiple frames share one output-pixel row."
                ),
            },
        )
        temporary.replace(destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return destination


class SigMFWaterfallBatch(Batch[SigMFSource]):
    """Durable per-recording and whole-workspace PNG rendering."""

    item_actions = (
        CapabilityChoice(
            RENDER_WATERFALL,
            "Render high-resolution waterfall PNGs",
        ),
    )
    workspace_actions = (
        CapabilityChoice(
            RENDER_WATERFALL,
            "Render all high-resolution waterfall PNGs",
        ),
    )

    def __init__(
        self,
        output_root: str | Path,
        *,
        fft_size: int = 2048,
        overlap_percent: int = 50,
        time_bins: int = 1200,
        width_pixels: int = 2400,
        height_pixels: int = 1600,
    ) -> None:
        _validate_render_options(
            fft_size=fft_size,
            overlap_percent=overlap_percent,
            time_bins=time_bins,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
        )
        self.output_root = Path(output_root).expanduser().resolve()
        self.fft_size = fft_size
        self.overlap_percent = overlap_percent
        self.time_bins = time_bins
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "-",
            value.lower(),
        ).strip("-")

    def _filename(
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
        return (
            f"{slug}-ch{channel + 1}-waterfall-"
            f"fft{self.fft_size}-{self.overlap_percent}pct-"
            f"{self.width_pixels}x{self.height_pixels}.png"
        )

    @staticmethod
    def _recordings(source: SigMFSource) -> tuple[SigMFRecording, ...]:
        return source.members if isinstance(source, SigMFCollection) else (source,)

    def _filenames(
        self,
        resource: DataResource,
        source: SigMFSource | None = None,
    ) -> tuple[str, ...]:
        opened = open_source(resource.source) if source is None else source
        collection = opened if isinstance(opened, SigMFCollection) else None
        return tuple(
            self._filename(
                resource,
                recording,
                channel,
                collection=collection,
            )
            for recording in self._recordings(opened)
            for channel in range(recording.channel_count)
        )

    def item_destination(
        self,
        resource: DataResource,
        request: BatchRequest,
    ) -> BatchDestination:
        return BatchDestination(
            self.output_root,
            self._filenames(resource),
            "High-resolution waterfall PNGs are ready",
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
            "All high-resolution waterfall PNGs are ready",
        )

    def _render(
        self,
        resource: DataResource,
        source: SigMFSource,
        directory: Path,
    ) -> tuple[Path, ...]:
        collection = source if isinstance(source, SigMFCollection) else None
        return tuple(
            render_recording_png(
                recording,
                directory
                / self._filename(
                    resource,
                    recording,
                    channel,
                    collection=collection,
                ),
                channel=channel,
                fft_size=self.fft_size,
                overlap_percent=self.overlap_percent,
                time_bins=self.time_bins,
                width_pixels=self.width_pixels,
                height_pixels=self.height_pixels,
            )
            for recording in self._recordings(source)
            for channel in range(recording.channel_count)
        )

    def run_item(
        self,
        resource: DataResource,
        source_data: SigMFSource,
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        outputs = self._render(resource, source_data, directory)
        return BatchResult(
            outputs,
            f"Rendered {len(outputs)} high-resolution waterfall PNG(s)",
        )

    def run_workspace(
        self,
        resources: tuple[DataResource, ...],
        open_resource,
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        outputs = tuple(
            output
            for resource in resources
            for output in self._render(
                resource,
                open_resource(resource),
                directory,
            )
        )
        return BatchResult(
            outputs,
            f"Rendered {len(outputs)} high-resolution waterfall PNG(s)",
        )


__all__ = [
    "RENDER_WATERFALL",
    "SigMFWaterfallBatch",
    "render_recording_png",
]
