"""Bind one exact SigMF reader, one view callback, and one batch capability."""

from __future__ import annotations

from sigvue import Workspace
from sigvue.helpers import WorkspaceConfig

from .annotations import WaterfallAnnotator
from .batch import SigMFWaterfallBatch
from .reader import DISCOVERY_COLUMNS, create_reader
from .view import view


def create_workspace(config) -> Workspace:
    values = WorkspaceConfig(config)
    return Workspace(
        identifier="sigmf-waterfall",
        name="SigMF Waterfall",
        description=(
            "Browse standalone SigMF pairs and collections, then "
            "inspect exact ranged sample windows as spectrum/waterfall views."
        ),
        reader=create_reader(config),
        view=view,
        annotator=WaterfallAnnotator(),
        batch=SigMFWaterfallBatch(
            values.path("output_root"),
            fft_size=values.integer("batch_fft_size", 2048),
            overlap_percent=values.integer(
                "batch_overlap_percent",
                50,
            ),
            time_bins=values.integer("batch_time_bins", 1200),
            width_pixels=values.integer(
                "batch_width_pixels",
                2400,
            ),
            height_pixels=values.integer(
                "batch_height_pixels",
                1600,
            ),
        ),
        category="spectrum monitoring",
        tags=("SigMF", "windowed", "waterfall", "spectrum"),
        discovery_columns=DISCOVERY_COLUMNS,
        lazy_views=True,
        flatten_discovery=True,
    )


__all__ = ["create_workspace"]
