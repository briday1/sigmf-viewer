"""Headless-first SigMF waterfall analysis and its focused Sigvue workspace."""

from .analysis import analyze
from .models import (
    SigMFCollection,
    SigMFRecording,
    SigMFSource,
    SigMFWindow,
    WaterfallProducts,
    WaterfallSettings,
)
from .sigmf import (
    load_metadata,
    open_collection,
    open_recording,
    open_source,
    read_window,
)


def __getattr__(name):
    """Load optional plotting and server integrations only when requested."""
    from importlib import import_module

    modules = {
        "plot": ".plots",
        "plot_waterfall": ".plots",
        "create_reader": ".reader",
        "power_spectrum_overview": ".reader",
        "create_workspace": ".workspace",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "SigMFCollection",
    "SigMFRecording",
    "SigMFSource",
    "SigMFWindow",
    "WaterfallProducts",
    "WaterfallSettings",
    "analyze",
    "create_reader",
    "create_workspace",
    "load_metadata",
    "open_collection",
    "open_recording",
    "open_source",
    "plot",
    "plot_waterfall",
    "power_spectrum_overview",
    "read_window",
]
