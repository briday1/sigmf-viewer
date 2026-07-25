"""Format-native buffers and small waterfall analysis values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SigMFRecording:
    """A validated SigMF recording whose samples remain on disk."""

    metadata_path: Path
    data_path: Path
    sample_rate: float
    channel_count: int
    sample_count: int
    metadata: dict[str, object]
    datatype: str
    sample_offset: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def center_frequency(self) -> float:
        return self.center_frequency_at(0)

    @property
    def channel_labels(self) -> tuple[str, ...]:
        """Return metadata-provided channel names or honest zero-based identities."""
        global_metadata = self.metadata.get("global", {})
        if not isinstance(global_metadata, dict):
            global_metadata = {}
        raw_labels = next(
            (
                global_metadata[key]
                for key in (
                    "sigmf_viewer:channel_names",
                    "core:channel_names",
                    "core:channel_labels",
                    "channel_names",
                    "channel_labels",
                    "channels",
                )
                if key in global_metadata
            ),
            (),
        )
        labels: list[str] = []
        if isinstance(raw_labels, (list, tuple)):
            for entry in raw_labels:
                if isinstance(entry, str):
                    label = entry.strip()
                elif isinstance(entry, dict):
                    label = str(
                        entry.get("core:label")
                        or entry.get("label")
                        or entry.get("name")
                        or entry.get("core:description")
                        or ""
                    ).strip()
                else:
                    label = ""
                labels.append(label)
        resolved = [
            (
                labels[index]
                if index < len(labels) and labels[index]
                else f"Channel {index} (unlabeled)"
            )
            for index in range(self.channel_count)
        ]
        seen: dict[str, int] = {}
        unique = []
        for index, label in enumerate(resolved):
            count = seen.get(label, 0)
            seen[label] = count + 1
            unique.append(label if count == 0 else f"{label} [{index}]")
        return tuple(unique)

    def center_frequency_at(self, sample: int) -> float:
        """Return the most recent capture tuning at a local sample index."""
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise TypeError("sample must be an integer")
        if not 0 <= sample <= self.sample_count:
            raise ValueError("sample is outside the recording")
        absolute_sample = self.sample_offset + sample
        captures = self.metadata.get("captures", [])
        if not isinstance(captures, list):
            raise TypeError("SigMF captures must be an array")
        eligible: list[tuple[int, dict[str, object]]] = []
        for capture in captures:
            if not isinstance(capture, dict):
                raise TypeError("SigMF captures must be objects")
            raw_start = capture.get("core:sample_start", self.sample_offset)
            if isinstance(raw_start, bool) or not isinstance(raw_start, int):
                raise TypeError("Capture sample starts must be integers")
            if raw_start <= absolute_sample:
                eligible.append((raw_start, capture))
        if not eligible:
            return 0.0
        _, selected = max(eligible, key=lambda item: item[0])
        return float(selected.get("core:frequency") or 0.0)

    def read(
        self,
        start: int,
        count: int,
        *,
        normalized: bool = True,
    ) -> np.ndarray:
        """Read exact frames as a channel-first complex64 array."""
        from .sigmf import read_samples

        return read_samples(self, start, count, normalized=normalized)


@dataclass(frozen=True)
class SigMFCollection:
    """One collection manifest and its validated member recordings."""

    collection_path: Path
    metadata: dict[str, object]
    members: tuple[SigMFRecording, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("A SigMF collection must contain at least one recording")
        paths = [member.metadata_path.resolve() for member in self.members]
        if len(paths) != len(set(paths)):
            raise ValueError("A SigMF collection cannot repeat a recording")


SigMFSource = SigMFRecording | SigMFCollection


@dataclass(frozen=True)
class SigMFWindow:
    """One exact channel-first sample window and its source coordinates."""

    recording: SigMFRecording
    start_sample: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.start_sample, bool) or not isinstance(
            self.start_sample,
            int,
        ):
            raise TypeError("start_sample must be an integer")
        if not 0 <= self.start_sample <= self.recording.sample_count:
            raise ValueError("window start is outside the recording")
        if not isinstance(self.samples, np.ndarray) or self.samples.ndim != 2:
            raise ValueError(
                "window samples must be a channel-first two-dimensional array"
            )
        if self.samples.shape[0] != self.recording.channel_count:
            raise ValueError("window samples do not match the recording channel count")
        if self.start_sample + self.samples.shape[-1] > self.recording.sample_count:
            raise ValueError("window samples extend beyond the recording")

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def stop_sample(self) -> int:
        return self.start_sample + self.sample_count

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.recording.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.recording.sample_rate

    @property
    def buffer_nbytes(self) -> int:
        return int(self.samples.nbytes)

    def channel(self, index: int = 0) -> np.ndarray:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("channel index must be an integer")
        if not 0 <= index < self.recording.channel_count:
            raise ValueError("channel index is outside the recording")
        return self.samples[index]


@dataclass(frozen=True)
class WaterfallSettings:
    """Explicit spectral settings usable from scripts and from the browser."""

    fft_size: int = 256
    overlap_percent: int = 50
    channel: int = 0


@dataclass(frozen=True)
class WaterfallProducts:
    recording: SigMFRecording
    start_sample: int
    channel: int
    spectrum_dbfs: np.ndarray
    waterfall_dbfs: np.ndarray
    frequency_mhz: np.ndarray
    time_edges_ms: np.ndarray
    buffer_nbytes: int


__all__ = [
    "SigMFCollection",
    "SigMFRecording",
    "SigMFSource",
    "SigMFWindow",
    "WaterfallProducts",
    "WaterfallSettings",
]
