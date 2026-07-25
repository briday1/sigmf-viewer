"""Fault-tolerant SigMF catalog discovery and exact window delivery."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from sigvue import DataResource, DiscoveryColumn, Reader
from sigvue.helpers import WorkspaceConfig

from .models import SigMFCollection, SigMFRecording, SigMFSource, SigMFWindow
from .sigmf import (
    open_collection,
    open_recording,
    open_source,
    paired_metadata_path,
    read_window,
)

DISCOVERY_COLUMNS = (
    DiscoveryColumn("date", "Date", "datetime"),
    DiscoveryColumn("recordings", "Recordings", "number"),
    DiscoveryColumn("duration", "Total duration", "si", unit="s"),
    DiscoveryColumn("sample_rate", "Sampling rate", "si", unit="sample/s"),
    DiscoveryColumn("rf_frequency", "RF frequency", "si", unit="Hz"),
)


def resolve_source(
    source: str | Path,
    pattern: str = "*.sigmf-meta",
) -> tuple[Path, str, bool]:
    """Resolve a directory, pair member, or collection manifest."""
    candidate = Path(source).expanduser().resolve()
    if candidate.name.endswith((".sigmf-meta", ".sigmf-data")):
        metadata_path = paired_metadata_path(candidate)
        return metadata_path.parent, metadata_path.name, False
    if candidate.name.endswith(".sigmf-collection"):
        return candidate.parent, candidate.name, False
    return candidate, pattern, True


def _first_capture(recording: SigMFRecording) -> dict[str, object]:
    captures = recording.metadata.get("captures", [])
    if isinstance(captures, list) and captures and isinstance(captures[0], dict):
        return captures[0]
    return {}


def _recording_title(recording: SigMFRecording) -> str:
    global_metadata = recording.metadata["global"]
    return str(
        global_metadata.get("core:description")
        or recording.metadata_path.name.removesuffix(".sigmf-meta")
    )


def _recording_date(recording: SigMFRecording) -> object:
    global_metadata = recording.metadata["global"]
    return _first_capture(recording).get("core:datetime") or global_metadata.get(
        "core:datetime"
    )


def _recording_frequency(recording: SigMFRecording) -> object:
    return _first_capture(recording).get("core:frequency")


def _common(values: tuple[object, ...]) -> object | None:
    return values[0] if values and all(value == values[0] for value in values) else None


def _recording_resource(
    metadata_path: Path,
    recording: SigMFRecording,
    root: Path,
) -> DataResource:
    relative = metadata_path.resolve().relative_to(root)
    channel_label = (
        "1 channel"
        if recording.channel_count == 1
        else f"{recording.channel_count} channels"
    )
    return DataResource(
        identifier=relative.as_posix().removesuffix(".sigmf-meta").replace("/", "::"),
        title=_recording_title(recording),
        source=metadata_path,
        subtitle=(
            f"{recording.sample_rate / 1e6:g} MS/s · "
            f"{recording.datatype} · {channel_label}"
        ),
        timestamp=datetime.fromtimestamp(
            metadata_path.stat().st_mtime,
            tz=timezone.utc,
        ),
        tags=("SigMF", "recording", recording.datatype, channel_label),
        summary={
            "date": _recording_date(recording),
            "recordings": 1,
            "duration": recording.duration_seconds,
            "sample_rate": recording.sample_rate,
            "rf_frequency": _recording_frequency(recording),
            "channels": recording.channel_count,
        },
        navigation_path=(),
    )


def _collection_resource(
    collection_path: Path,
    collection: SigMFCollection,
    root: Path,
) -> DataResource:
    relative = collection_path.resolve().relative_to(root)
    details = collection.metadata["collection"]
    title = str(
        details.get("core:description")
        or collection_path.name.removesuffix(".sigmf-collection")
    )
    rates = tuple(member.sample_rate for member in collection.members)
    frequencies = tuple(_recording_frequency(member) for member in collection.members)
    dates = tuple(
        date for member in collection.members if (date := _recording_date(member))
    )
    datatypes = tuple(sorted({member.datatype for member in collection.members}))
    return DataResource(
        identifier=relative.as_posix().replace("/", "::"),
        title=title,
        source=collection_path,
        subtitle=(f"{len(collection.members)} recordings · {', '.join(datatypes)}"),
        timestamp=datetime.fromtimestamp(
            collection_path.stat().st_mtime,
            tz=timezone.utc,
        ),
        tags=("SigMF", "collection", *datatypes),
        summary={
            "date": min(dates, default=None),
            "recordings": len(collection.members),
            "duration": sum(member.duration_seconds for member in collection.members),
            "sample_rate": _common(rates),
            "rf_frequency": _common(frequencies),
            "channels": max(member.channel_count for member in collection.members),
        },
        navigation_path=(),
    )


def create_files(
    source: str | Path,
    *,
    pattern: str = "*.sigmf-meta",
) -> Reader[Path, SigMFSource]:
    """Create a flat catalog of standalone pairs and collection items.

    Each candidate is validated independently. A broken pair or collection is
    omitted, while all other usable entries remain discoverable.
    """
    root, selected_pattern, recursive = resolve_source(source, pattern)
    descriptions: dict[Path, DataResource] = {}

    def metadata_candidates() -> tuple[Path, ...]:
        paths = (
            root.rglob(selected_pattern) if recursive else root.glob(selected_pattern)
        )
        return tuple(sorted(path.resolve() for path in paths if path.is_file()))

    def collection_candidates() -> tuple[Path, ...]:
        if not recursive:
            selected = root / selected_pattern
            return (
                (selected.resolve(),)
                if selected.name.endswith(".sigmf-collection") and selected.is_file()
                else ()
            )
        return tuple(
            sorted(path.resolve() for path in root.rglob("*.sigmf-collection"))
        )

    def discover() -> tuple[Path, ...]:
        descriptions.clear()
        valid_collections: list[Path] = []
        referenced: set[Path] = set()
        for path in collection_candidates():
            try:
                collection = open_collection(path)
                resource = _collection_resource(path, collection, root)
            except (KeyError, OSError, TypeError, ValueError):
                continue
            descriptions[path] = resource
            valid_collections.append(path)
            referenced.update(
                member.metadata_path.resolve() for member in collection.members
            )

        valid_recordings: list[Path] = []
        for path in metadata_candidates():
            if path in referenced:
                continue
            try:
                recording = open_recording(path)
                resource = _recording_resource(path, recording, root)
            except (KeyError, OSError, TypeError, ValueError):
                continue
            descriptions[path] = resource
            valid_recordings.append(path)
        return tuple(
            sorted(
                (*valid_collections, *valid_recordings),
                key=lambda path: descriptions[path].title.casefold(),
            )
        )

    def describe(path: Path) -> DataResource:
        return descriptions[path]

    return Reader(discover, open_source, describe=describe)


def _read_interval(
    recording: SigMFRecording,
    start_seconds: float,
    stop_seconds: float,
) -> SigMFWindow:
    start_sample = round(start_seconds * recording.sample_rate)
    sample_count = max(
        1,
        round((stop_seconds - start_seconds) * recording.sample_rate),
    )
    return read_window(recording, start_sample, sample_count)


def power_overview(
    recording: SigMFRecording,
    *,
    bins: int = 300,
) -> np.ndarray:
    """Compute exact mean power over bounded source intervals."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("overview bins must be a positive integer")
    count = min(bins, recording.sample_count)
    edges = np.linspace(
        0,
        recording.sample_count,
        count + 1,
        dtype=np.int64,
    )
    values = np.empty(count, dtype=np.float64)
    for index, (start, stop) in enumerate(pairwise(edges)):
        samples = recording.read(int(start), int(stop - start))
        power = float(np.mean(np.abs(samples) ** 2))
        values[index] = 10.0 * np.log10(max(power, 1e-20))
    return values


def power_spectrum_overview(
    recording: SigMFRecording,
    *,
    time_bins: int = 300,
    frequency_bins: int = 48,
    fft_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Read every sample once into an exact-power line and compact spectrum.

    The returned heatmap is frequency-by-time: columns place recording time
    left-to-right in the window bar and rows place frequency bottom-to-top.
    This is the same time-x/frequency-y orientation as the main waterfall.
    Source intervals are complete and non-overlapping; any partial FFT block
    is zero padded, then weighted by its actual sample count through
    Parseval-consistent linear-power sums.
    """
    settings = {
        "time_bins": time_bins,
        "frequency_bins": frequency_bins,
        "fft_size": fft_size,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in settings.values()
    ):
        raise ValueError("overview bin counts and FFT size must be positive integers")
    effective_fft = min(fft_size, recording.sample_count)
    frequency_count = min(frequency_bins, effective_fft)
    time_count = min(time_bins, recording.sample_count)
    time_edges = np.linspace(
        0,
        recording.sample_count,
        time_count + 1,
        dtype=np.int64,
    )
    frequency_edges = np.linspace(
        0,
        effective_fft,
        frequency_count + 1,
        dtype=np.int64,
    )
    line_power = np.empty(time_count, dtype=np.float64)
    raster_power = np.empty(
        (frequency_count, time_count),
        dtype=np.float64,
    )

    for time_index, (start, stop) in enumerate(pairwise(time_edges)):
        sample_count = int(stop - start)
        samples = recording.read(int(start), sample_count)
        line_power[time_index] = float(np.mean(np.abs(samples) ** 2))
        frame_count = (sample_count + effective_fft - 1) // effective_fft
        padded = np.zeros(
            (recording.channel_count, frame_count * effective_fft),
            dtype=samples.dtype,
        )
        padded[:, :sample_count] = samples
        blocks = padded.reshape(
            recording.channel_count,
            frame_count,
            effective_fft,
        )
        spectra = np.fft.fftshift(
            np.fft.fft(blocks, axis=-1),
            axes=-1,
        )
        frequency_power = (
            np.sum(np.abs(spectra) ** 2, axis=(0, 1))
            / effective_fft
            / sample_count
            / recording.channel_count
        )
        for frequency_index, (lower, upper) in enumerate(pairwise(frequency_edges)):
            raster_power[frequency_index, time_index] = float(
                np.sum(frequency_power[int(lower) : int(upper)])
            )

    line_dbfs = 10.0 * np.log10(np.maximum(line_power, 1e-20))
    raster_dbfs = 10.0 * np.log10(np.maximum(raster_power, 1e-20))
    return line_dbfs, raster_dbfs


@lru_cache(maxsize=32)
def _cached_overview(
    metadata_path: str,
    metadata_modified_ns: int,
    data_modified_ns: int,
    time_bins: int,
    frequency_bins: int,
    fft_size: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    del metadata_modified_ns, data_modified_ns
    line, heatmap = power_spectrum_overview(
        open_recording(metadata_path),
        time_bins=time_bins,
        frequency_bins=frequency_bins,
        fft_size=fft_size,
    )
    return (
        tuple(round(float(value), 3) for value in line),
        tuple(tuple(round(float(value), 2) for value in row) for row in heatmap),
    )


def _overview(
    recording: SigMFRecording,
    *,
    time_bins: int,
    frequency_bins: int,
    fft_size: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    return _cached_overview(
        str(recording.metadata_path),
        recording.metadata_path.stat().st_mtime_ns,
        recording.data_path.stat().st_mtime_ns,
        time_bins,
        frequency_bins,
        fft_size,
    )


def _member_key(
    collection: SigMFCollection,
    recording: SigMFRecording,
) -> str:
    return recording.metadata_path.relative_to(
        collection.collection_path.parent
    ).as_posix()


def _member_options(collection: SigMFCollection) -> dict[str, str]:
    return {
        _member_key(collection, recording): _recording_title(recording)
        for recording in collection.members
    }


def _recording_for(
    source: SigMFSource,
    member: str | int | None = None,
) -> SigMFRecording:
    if isinstance(source, SigMFRecording):
        if member not in (None, 0):
            raise ValueError("Standalone SigMF recordings do not have members")
        return source
    if member is None:
        return source.members[0]
    if isinstance(member, bool):
        raise TypeError("Collection member must be a path key or integer index")
    if isinstance(member, int):
        try:
            return source.members[member]
        except IndexError as error:
            raise ValueError(
                "Collection member index is outside the collection"
            ) from error
    requested = str(member)
    for recording in source.members:
        if _member_key(source, recording) == requested:
            return recording
    raise ValueError(f"Unknown collection recording: {requested}")


def create_reader(config):
    """Create exact headless reads plus browser collection/window controls."""
    values = WorkspaceConfig(config)
    default_window = values.floating("default_window_seconds", 0.012)
    minimum_window = values.floating("minimum_window_seconds", 0.004)
    window_step = values.floating("window_step_seconds", 0.002)
    overview_bins = values.integer("overview_bins", 300)
    overview_frequency_bins = values.integer("overview_frequency_bins", 48)
    overview_fft_size = values.integer("overview_fft_size", 256)
    files = create_files(
        values.path("data_root"),
        pattern=values.string("filename", "*.sigmf-meta"),
    )

    def read(
        source: SigMFSource,
        start: float = 0.0,
        stop: float | None = None,
        *,
        member: str | int | None = None,
    ) -> SigMFWindow:
        recording = _recording_for(source, member)
        selected_stop = (
            min(recording.duration_seconds, start + default_window)
            if stop is None
            else stop
        )
        if not 0 <= start < selected_stop <= recording.duration_seconds:
            raise ValueError("Window bounds must stay within the recording")
        return _read_interval(recording, start, selected_stop)

    def select(source: SigMFSource, ui) -> SigMFWindow:
        if isinstance(source, SigMFCollection):
            options = _member_options(source)
            member = str(
                ui.select(
                    "collection_recording",
                    default=next(iter(options)),
                    options=options,
                    label="Recording",
                    group="Input recording",
                )
            )
        else:
            member = None
        recording = _recording_for(source, member)
        _overview_line, overview_heatmap = _overview(
            recording,
            time_bins=overview_bins,
            frequency_bins=overview_frequency_bins,
            fft_size=overview_fft_size,
        )
        start, stop = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(default_window, recording.duration_seconds),
            overview_heatmap=overview_heatmap,
            overview_colormap_control="colormap",
            overview_limits_control="dbfs_limits",
            overview_label="Full-recording spectrum",
            minimum_window=min(minimum_window, recording.duration_seconds),
            step=min(window_step, recording.duration_seconds),
            time_unit="ms",
        )
        return read(source, start, stop, member=member)

    return files.buffered(read, select)


__all__ = [
    "DISCOVERY_COLUMNS",
    "create_files",
    "create_reader",
    "power_overview",
    "power_spectrum_overview",
    "resolve_source",
]
