"""Pure Plotly figure builders for analyzed waterfall products."""

from __future__ import annotations

from html import escape
from math import isfinite

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sigvue import Annotation, add_viewport_heatmap

from .models import WaterfallProducts
from .style import TEAL


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _rounded_range(lower: float, upper: float) -> tuple[float, float]:
    lower = max(-200.0, 5.0 * np.floor(lower / 5.0))
    upper = min(20.0, 5.0 * np.ceil(upper / 5.0))
    if upper - lower < 20.0:
        lower = max(-200.0, upper - 20.0)
    return float(lower), float(upper)


def automatic_dbfs_ranges(
    products: WaterfallProducts,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return robust waterfall and spectrum ranges rounded to 5 dB."""
    waterfall = _finite(products.waterfall_dbfs)
    spectrum = _finite(products.spectrum_dbfs)
    if not waterfall.size or not spectrum.size:
        return (-90.0, -20.0), (-90.0, -20.0)
    signal_top = max(
        float(np.percentile(waterfall, 99.9)),
        float(np.percentile(spectrum, 99.5)),
    )
    return (
        _rounded_range(
            float(np.percentile(waterfall, 10.0)) - 3.0,
            signal_top + 3.0,
        ),
        _rounded_range(
            float(np.percentile(spectrum, 1.0)) - 3.0,
            float(np.percentile(spectrum, 99.9)) + 3.0,
        ),
    )


def waterfall_figure(
    products: WaterfallProducts,
    *,
    viewport: object = None,
    colormap: str = "Portland",
    zmin: float | None = None,
    zmax: float | None = None,
    spectrum_min: float | None = None,
    spectrum_max: float | None = None,
    spectrum_style: object | None = None,
    show_spectrum: bool = True,
    show_colorbar: bool = True,
    progressive_render: bool = True,
    render_width: int = 1024,
    render_height: int = 512,
    annotations: tuple[Annotation, ...] = (),
    annotation_color: str = "#ffffff",
    annotation_width: float = 1.5,
    annotation_opacity: float = 0.8,
) -> go.Figure:
    """Build a left average-PSD strip sharing the waterfall frequency axis."""
    automatic_waterfall, automatic_spectrum = automatic_dbfs_ranges(products)
    zmin, zmax = automatic_waterfall if zmin is None or zmax is None else (zmin, zmax)
    spectrum_min, spectrum_max = (
        automatic_spectrum
        if spectrum_min is None or spectrum_max is None
        else (spectrum_min, spectrum_max)
    )
    spectrum_mode = getattr(spectrum_style, "mode", "lines")
    spectrum_line = getattr(
        spectrum_style,
        "line",
        {"color": TEAL, "width": 1.4},
    )
    spectrum_marker = getattr(spectrum_style, "plotly_marker", None)
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=(0.10, 0.90) if show_spectrum else (1e-6, 1.0),
        horizontal_spacing=0.025 if show_spectrum else 0.0,
    )
    if show_spectrum:
        figure.add_trace(
            go.Scatter(
                x=products.spectrum_dbfs,
                y=products.frequency_mhz,
                mode=spectrum_mode,
                line=spectrum_line,
                marker=spectrum_marker,
                name="Average spectrum",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    heatmap = {
        "x": products.time_edges_ms,
        "y": products.frequency_mhz,
        "z": products.waterfall_dbfs.T,
        "zmin": zmin,
        "zmax": zmax,
        "colorscale": colormap,
        "showscale": show_colorbar,
        "colorbar": {"title": "dBFS"},
        "hovertemplate": (
            "Recording time: %{x:.2f} ms"
            "<br>RF frequency: %{y:.6f} MHz"
            "<br>dBFS: %{z:.2f}"
            "<extra></extra>"
        ),
    }
    if progressive_render:
        add_viewport_heatmap(
            figure,
            viewport=viewport,
            render_width=render_width,
            render_height=render_height,
            aggregation="mean",
            row=1,
            col=2,
            **heatmap,
        )
    else:
        figure.add_trace(go.Heatmap(**heatmap), row=1, col=2)
    frequency_step = (
        float(abs(products.frequency_mhz[1] - products.frequency_mhz[0]))
        if products.frequency_mhz.size > 1
        else 1.0
    )
    frequency_range = [
        float(products.frequency_mhz[0] - frequency_step / 2.0),
        float(products.frequency_mhz[-1] + frequency_step / 2.0),
    ]
    _add_annotation_regions(
        figure,
        annotations,
        time_range=(
            float(products.time_edges_ms[0]),
            float(products.time_edges_ms[-1]),
        ),
        frequency_range=(frequency_range[0], frequency_range[1]),
        color=annotation_color,
        width=annotation_width,
        opacity=annotation_opacity,
    )
    if show_spectrum:
        figure.update_xaxes(
            title_text="Power (dBFS)",
            range=[spectrum_min, spectrum_max],
            autorange=False,
            row=1,
            col=1,
        )
        figure.update_yaxes(
            title_text="RF frequency (MHz)",
            range=frequency_range,
            autorange=False,
            tickformat=".2f",
            row=1,
            col=1,
        )
    figure.update_xaxes(
        title_text="Recording time (ms)",
        range=[
            float(products.time_edges_ms[0]),
            float(products.time_edges_ms[-1]),
        ],
        autorange=False,
        tickformat=".2f",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text=None if show_spectrum else "RF frequency (MHz)",
        range=frequency_range,
        autorange=False,
        tickformat=".2f",
        showticklabels=not show_spectrum,
        row=1,
        col=2,
    )
    figure.update_layout(
        uirevision=(f"sigmf-viewer:{products.recording.metadata_path}"),
        showlegend=False,
    )
    return figure


def _add_annotation_regions(
    figure: go.Figure,
    annotations: tuple[Annotation, ...],
    *,
    time_range: tuple[float, float],
    frequency_range: tuple[float, float],
    color: str,
    width: float,
    opacity: float,
) -> None:
    """Draw exact vector outlines and separate hover targets."""
    if not isfinite(width) or width <= 0:
        raise ValueError("annotation line width must be positive")
    if not isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("annotation opacity must be between zero and one")
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    time_lower, time_upper = time_range
    frequency_lower, frequency_upper = frequency_range
    for annotation in annotations:
        start = annotation.start_seconds * 1e3
        stop = (annotation.start_seconds + (annotation.duration_seconds or 0.0)) * 1e3
        lower = (
            annotation.frequency_lower_hz / 1e6
            if annotation.frequency_lower_hz is not None
            else frequency_lower
        )
        upper = (
            annotation.frequency_upper_hz / 1e6
            if annotation.frequency_upper_hz is not None
            else frequency_upper
        )
        if (
            stop < time_lower
            or start > time_upper
            or upper < frequency_lower
            or lower > frequency_upper
        ):
            continue
        if stop > start:
            line_x.extend((start, stop, stop, start, start, None))
            line_y.extend((lower, lower, upper, upper, lower, None))
        else:
            line_x.extend((start, start, None))
            line_y.extend((lower, upper, None))
        hover_x.append((max(start, time_lower) + min(stop, time_upper)) / 2.0)
        hover_y.append(
            (max(lower, frequency_lower) + min(upper, frequency_upper)) / 2.0
        )
        details = [
            f"<b>{escape(annotation.label or 'Annotation')}</b>",
            f"Time: {start:.9g}–{stop:.9g} ms",
            f"Frequency: {lower:.9g}–{upper:.9g} MHz",
        ]
        if annotation.comment:
            details.append(escape(annotation.comment))
        hover_text.append("<br>".join(details))
    if not line_x:
        return
    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": color, "width": width},
            opacity=opacity,
            name="Annotations",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={
                "color": color,
                "size": max(8.0, width * 4.0),
                "opacity": max(0.15, min(0.45, opacity)),
                "symbol": "square-open",
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Annotation details",
            showlegend=False,
        ),
        row=1,
        col=2,
    )


def plot_waterfall(
    products: WaterfallProducts,
    **display_options: object,
) -> go.Figure:
    return waterfall_figure(products, **display_options)


def plot(
    products: WaterfallProducts,
    **display_options: object,
) -> go.Figure:
    return plot_waterfall(products, **display_options)


__all__ = [
    "automatic_dbfs_ranges",
    "plot",
    "plot_waterfall",
    "waterfall_figure",
]
