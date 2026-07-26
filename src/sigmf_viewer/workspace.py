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
        identifier="sigmf-viewer",
        name="SigMF Viewer",
        description=(
            "Browse standalone SigMF pairs and collections, then "
            "inspect exact ranged sample windows as spectrum/waterfall views."
        ),
        reader=create_reader(config),
        view=view,
        annotator=WaterfallAnnotator(),
        batch=SigMFWaterfallBatch(
            values.path("output_root"),
            fft_size=values.integer("batch_fft_size", 256),
            overlap_percent=values.integer(
                "batch_overlap_percent",
                50,
            ),
            colormap=values.string("batch_colormap", "turbo"),
            max_native_cells=values.integer(
                "batch_max_native_cells",
                75_000_000,
            ),
        ),
        category="spectrum monitoring",
        tags=("SigMF", "windowed", "waterfall", "spectrum"),
        discovery_columns=DISCOVERY_COLUMNS,
        lazy_views=True,
        flatten_discovery=True,
    )


__all__ = ["create_workspace"]
