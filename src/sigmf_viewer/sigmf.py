"""Framework-independent SigMF metadata, collection, and ranged sample I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import RLock

import numpy as np

from .models import SigMFCollection, SigMFRecording, SigMFSource, SigMFWindow


@dataclass(frozen=True)
class _SampleFormat:
    dtype: str
    components: int
    scale: float
    offset: float = 0.0


def _formats() -> dict[str, _SampleFormat]:
    formats: dict[str, _SampleFormat] = {}
    for kind, components in (("c", 2), ("r", 1)):
        for bits in (32, 64):
            for endian, marker in (("le", "<"), ("be", ">")):
                formats[f"{kind}f{bits}_{endian}"] = _SampleFormat(
                    f"{marker}f{bits // 8}",
                    components,
                    1.0,
                )
        for signedness, letter in (("i", "i"), ("u", "u")):
            for bits in (8, 16, 32, 64):
                suffixes = (
                    (("", "|"),)
                    if bits == 8
                    else (
                        ("_le", "<"),
                        ("_be", ">"),
                    )
                )
                for suffix, marker in suffixes:
                    half_range = float(2 ** (bits - 1))
                    formats[f"{kind}{signedness}{bits}{suffix}"] = _SampleFormat(
                        f"{marker}{letter}{bits // 8}",
                        components,
                        1.0 / half_range,
                        0.0 if signedness == "i" else half_range,
                    )
    # sc16_le is a long-standing alias found in deployed recordings.
    formats["sc16_le"] = formats["ci16_le"]
    return formats


SIGMF_DATATYPES = _formats()
_metadata_lock = RLock()


def load_metadata(metadata_path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("SigMF metadata must be a JSON object")
    return payload


def annotations(
    metadata_path: str | Path,
) -> tuple[dict[str, object], ...]:
    """Return the current standard SigMF annotations."""
    entries = load_metadata(metadata_path).get("annotations", ())
    if not isinstance(entries, list):
        raise TypeError("SigMF annotations must be an array")
    if any(not isinstance(entry, dict) for entry in entries):
        raise TypeError("SigMF annotations must be objects")
    return tuple(entries)


def append_annotation(
    metadata_path: str | Path,
    annotation: dict[str, object],
) -> None:
    """Atomically append and sample-sort one standard annotation."""
    path = Path(metadata_path)
    with _metadata_lock:
        metadata = load_metadata(path)
        entries = list(metadata.get("annotations", ()))
        entries.append(dict(annotation))
        entries.sort(key=lambda entry: int(entry["core:sample_start"]))
        metadata["annotations"] = entries
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def paired_metadata_path(path: str | Path) -> Path:
    """Resolve either member of a `.sigmf-meta`/`.sigmf-data` pair."""
    candidate = Path(path).expanduser().resolve()
    if candidate.name.endswith(".sigmf-meta"):
        return candidate
    if candidate.name.endswith(".sigmf-data"):
        return candidate.with_name(
            candidate.name.removesuffix(".sigmf-data") + ".sigmf-meta"
        )
    raise ValueError("SigMF input must end in .sigmf-meta or .sigmf-data")


def _dataset_path(path: Path, global_metadata: dict[str, object]) -> Path:
    declared = global_metadata.get("core:dataset")
    if declared:
        candidate = Path(str(declared))
        return (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )
    return path.with_name(path.name.removesuffix(".sigmf-meta") + ".sigmf-data")


def open_recording(
    metadata_path: str | Path,
    *,
    sample_rate_fallback: float | None = None,
) -> SigMFRecording:
    """Validate a SigMF pair without loading its sample payload."""
    path = paired_metadata_path(metadata_path)
    metadata = load_metadata(path)
    global_metadata = metadata.get("global")
    if not isinstance(global_metadata, dict):
        raise TypeError(f"{path.name} must define a global metadata object")
    datatype = str(global_metadata.get("core:datatype"))
    if datatype not in SIGMF_DATATYPES:
        raise ValueError(f"Unsupported SigMF datatype: {datatype}")
    channel_count = int(global_metadata.get("core:num_channels") or 1)
    if channel_count < 1:
        raise ValueError(f"{path.name} must define at least one channel")
    raw_sample_rate = global_metadata.get("core:sample_rate")
    if raw_sample_rate is None and sample_rate_fallback is None:
        raise ValueError(f"{path.name} does not define core:sample_rate")
    sample_rate = float(
        raw_sample_rate if raw_sample_rate is not None else sample_rate_fallback
    )
    if not isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"{path.name} must have a finite, positive sample rate")
    raw_offset = global_metadata.get("core:offset", 0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int):
        raise TypeError(f"{path.name} core:offset must be an integer")
    if raw_offset < 0:
        raise ValueError(f"{path.name} core:offset cannot be negative")
    data_path = _dataset_path(path, global_metadata)
    if not data_path.is_file():
        raise ValueError(f"Missing SigMF sample data: {data_path}")
    sample_format = SIGMF_DATATYPES[datatype]
    scalar_bytes = np.dtype(sample_format.dtype).itemsize
    frame_bytes = channel_count * sample_format.components * scalar_bytes
    sample_bytes = data_path.stat().st_size
    if sample_bytes % frame_bytes:
        raise ValueError(f"{data_path.name} contains a partial sample frame")
    if sample_bytes == 0:
        raise ValueError(f"{data_path.name} does not contain any samples")
    return SigMFRecording(
        metadata_path=path,
        data_path=data_path,
        sample_rate=sample_rate,
        channel_count=channel_count,
        sample_count=sample_bytes // frame_bytes,
        metadata=metadata,
        datatype=datatype,
        sample_offset=raw_offset,
    )


def read_samples(
    recording: SigMFRecording,
    start: int,
    count: int,
    *,
    normalized: bool = True,
) -> np.ndarray:
    """Read one exact sample range as channel-first complex64."""
    start = min(recording.sample_count, max(0, int(start)))
    count = min(max(0, int(count)), recording.sample_count - start)
    sample_format = SIGMF_DATATYPES[recording.datatype]
    scalar_dtype = np.dtype(sample_format.dtype)
    scalars_per_frame = recording.channel_count * sample_format.components
    with recording.data_path.open("rb") as stream:
        stream.seek(start * scalars_per_frame * scalar_dtype.itemsize)
        scalars = np.fromfile(
            stream,
            dtype=scalar_dtype,
            count=count * scalars_per_frame,
        )
    expected = count * scalars_per_frame
    if scalars.size != expected:
        raise ValueError(f"{recording.data_path.name} ended during a ranged read")
    frames = scalars.reshape(
        -1,
        recording.channel_count,
        sample_format.components,
    )
    values = frames.astype(np.float64, copy=False)
    if normalized:
        values = (values - sample_format.offset) * sample_format.scale
    if sample_format.components == 2:
        complex_frames = values[..., 0] + 1j * values[..., 1]
    else:
        complex_frames = values[..., 0].astype(np.complex128)
    output_dtype = (
        np.complex128
        if (
            scalar_dtype.itemsize > 4
            or (scalar_dtype.kind in {"i", "u"} and scalar_dtype.itemsize > 2)
        )
        else np.complex64
    )
    return np.asarray(complex_frames, dtype=output_dtype).T


def read_window(
    recording: SigMFRecording,
    start_sample: int,
    sample_count: int,
) -> SigMFWindow:
    return SigMFWindow(
        recording,
        int(start_sample),
        recording.read(int(start_sample), int(sample_count)),
    )


def collection_streams(path: str | Path) -> tuple[Path, ...]:
    """Return metadata paths referenced by one standard collection manifest."""
    collection_path = Path(path).expanduser().resolve()
    payload = load_metadata(collection_path)
    collection = payload.get("collection")
    if not isinstance(collection, dict):
        raise TypeError(f"{collection_path.name} must define a collection object")
    streams = collection.get("core:streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"{collection_path.name} does not define any core:streams")
    paths = []
    for stream in streams:
        if not isinstance(stream, dict) or not stream.get("name"):
            raise ValueError("SigMF collection streams require a name")
        declared = collection_path.parent / str(stream["name"])
        if declared.name.endswith((".sigmf-meta", ".sigmf-data")):
            metadata_path = paired_metadata_path(declared)
        else:
            metadata_path = declared.with_name(f"{declared.name}.sigmf-meta").resolve()
        paths.append(metadata_path)
    return tuple(paths)


def open_collection(path: str | Path) -> SigMFCollection:
    """Validate a collection and every referenced recording without payload I/O."""
    collection_path = Path(path).expanduser().resolve()
    metadata = load_metadata(collection_path)
    members = tuple(
        open_recording(metadata_path)
        for metadata_path in collection_streams(collection_path)
    )
    return SigMFCollection(collection_path, metadata, members)


def open_source(path: str | Path) -> SigMFSource:
    """Open one recording pair or one collection catalog item."""
    candidate = Path(path).expanduser().resolve()
    if candidate.name.endswith(".sigmf-collection"):
        return open_collection(candidate)
    return open_recording(candidate)


__all__ = [
    "SIGMF_DATATYPES",
    "annotations",
    "append_annotation",
    "collection_streams",
    "load_metadata",
    "open_collection",
    "open_recording",
    "open_source",
    "paired_metadata_path",
    "read_samples",
    "read_window",
]
