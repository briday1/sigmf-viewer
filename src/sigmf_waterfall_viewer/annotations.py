"""Standard SigMF persistence for hoverable waterfall annotations."""

from __future__ import annotations

from functools import lru_cache
from math import isfinite
from uuid import uuid4

from sigvue import (
    Annotation,
    AnnotationField,
    AnnotationPlotBinding,
    AnnotationRequest,
    Annotator,
)

from .models import SigMFCollection, SigMFRecording, SigMFSource, SigMFWindow
from .sigmf import annotations, append_annotation


@lru_cache(maxsize=64)
def _read_cached(
    metadata_path: str,
    modified_ns: int,
    size: int,
    sample_rate: float,
    sample_offset: int,
) -> tuple[Annotation, ...]:
    del modified_ns, size
    result = []
    for index, entry in enumerate(annotations(metadata_path)):
        raw_start = entry["core:sample_start"]
        if isinstance(raw_start, bool) or not isinstance(raw_start, int):
            raise TypeError("Annotation sample starts must be integers")
        start = raw_start - sample_offset
        if start < 0:
            raise ValueError("Annotation sample starts cannot precede core:offset")
        raw_count = entry.get("core:sample_count")
        if raw_count is not None and (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise ValueError("Annotation sample counts must be non-negative integers")
        result.append(
            Annotation(
                identifier=str(entry.get("core:uuid") or f"{raw_start}:{index}"),
                start_seconds=start / sample_rate,
                duration_seconds=(
                    None if raw_count is None else raw_count / sample_rate
                ),
                label=(str(entry["core:label"]) if entry.get("core:label") else None),
                comment=str(entry.get("core:comment") or "") or None,
                frequency_lower_hz=(
                    float(entry["core:freq_lower_edge"])
                    if entry.get("core:freq_lower_edge") is not None
                    else None
                ),
                frequency_upper_hz=(
                    float(entry["core:freq_upper_edge"])
                    if entry.get("core:freq_upper_edge") is not None
                    else None
                ),
            )
        )
    return tuple(result)


def read_sigmf_annotations(
    recording: SigMFRecording,
) -> tuple[Annotation, ...]:
    stat = recording.metadata_path.stat()
    return _read_cached(
        str(recording.metadata_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        recording.sample_rate,
        recording.sample_offset,
    )


def add_sigmf_annotation(
    recording: SigMFRecording,
    start_sample: int,
    sample_count: int,
    request: AnnotationRequest,
    *,
    frequency_lower_hz: float,
    frequency_upper_hz: float,
) -> Annotation:
    if (
        start_sample < 0
        or sample_count < 1
        or start_sample + sample_count > recording.sample_count
    ):
        raise ValueError("Annotation samples must stay within the recording")
    comment = request.values.get("comment", "").strip()
    if not comment:
        raise ValueError("An annotation description/comment is required")
    if (
        not isfinite(frequency_lower_hz)
        or not isfinite(frequency_upper_hz)
        or frequency_lower_hz >= frequency_upper_hz
    ):
        raise ValueError("Annotation frequencies must be finite and increasing")
    identifier = str(uuid4())
    entry = {
        "core:sample_start": recording.sample_offset + start_sample,
        "core:sample_count": sample_count,
        "core:comment": comment,
        "core:generator": "SigMF Waterfall Viewer",
        "core:uuid": identifier,
        "core:freq_lower_edge": float(frequency_lower_hz),
        "core:freq_upper_edge": float(frequency_upper_hz),
    }
    append_annotation(recording.metadata_path, entry)
    return Annotation(
        identifier=identifier,
        start_seconds=start_sample / recording.sample_rate,
        duration_seconds=sample_count / recording.sample_rate,
        comment=comment,
        frequency_lower_hz=frequency_lower_hz,
        frequency_upper_hz=frequency_upper_hz,
    )


def _fields() -> tuple[AnnotationField, ...]:
    view = "sigmf-waterfall"
    return (
        AnnotationField(
            "start_seconds",
            "Recording start (s)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view,
                "yaxis2",
                "lower",
                scale=1e-3,
                selection_policy="box_preferred",
            ),
        ),
        AnnotationField(
            "stop_seconds",
            "Recording stop (s)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view,
                "yaxis2",
                "upper",
                scale=1e-3,
                selection_policy="box_preferred",
            ),
        ),
        AnnotationField(
            "frequency_lower_hz",
            "Lower RF frequency (Hz)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view,
                "xaxis2",
                "lower",
                scale=1e6,
                selection_policy="box_preferred",
            ),
        ),
        AnnotationField(
            "frequency_upper_hz",
            "Upper RF frequency (Hz)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view,
                "xaxis2",
                "upper",
                scale=1e6,
                selection_policy="box_preferred",
            ),
        ),
        AnnotationField(
            "comment",
            "Description / comment",
            "textarea",
            required=True,
        ),
    )


class WaterfallAnnotator(Annotator[SigMFSource, SigMFWindow]):
    """Box-select time/frequency bounds and persist standard metadata."""

    timeline_color_control = "annotation_region_color"

    @property
    def fields(self) -> tuple[AnnotationField, ...]:
        return _fields()

    def discover(
        self,
        source: SigMFSource,
    ) -> tuple[Annotation, ...]:
        if isinstance(source, SigMFCollection):
            # The selected collection member is a buffer control, so a static
            # source-level list would mix coordinates from other recordings.
            # The waterfall view reads the selected member's annotations.
            return ()
        return read_sigmf_annotations(source)

    def annotate(
        self,
        source: SigMFSource,
        delivered: SigMFWindow,
        request: AnnotationRequest,
    ) -> Annotation:
        del source
        recording = delivered.recording
        try:
            start_seconds = float(request.values["start_seconds"])
            stop_seconds = float(request.values["stop_seconds"])
            lower = float(request.values["frequency_lower_hz"])
            upper = float(request.values["frequency_upper_hz"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Waterfall annotation bounds must be numeric") from error
        if (
            not all(
                isfinite(value) for value in (start_seconds, stop_seconds, lower, upper)
            )
            or start_seconds < 0
            or stop_seconds <= start_seconds
        ):
            raise ValueError("Annotation times must be finite and increasing")
        start_sample = min(
            recording.sample_count,
            round(start_seconds * recording.sample_rate),
        )
        stop_sample = min(
            recording.sample_count,
            round(stop_seconds * recording.sample_rate),
        )
        return add_sigmf_annotation(
            recording,
            start_sample,
            stop_sample - start_sample,
            request,
            frequency_lower_hz=lower,
            frequency_upper_hz=upper,
        )


__all__ = [
    "WaterfallAnnotator",
    "add_sigmf_annotation",
    "read_sigmf_annotations",
]
