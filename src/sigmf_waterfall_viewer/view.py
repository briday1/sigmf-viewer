"""The complete Sigvue UI expression for one delivered SigMF window."""

from __future__ import annotations

from sigvue import UI
from sigvue.helpers import format_bytes

from .analysis import analyze
from .annotations import read_sigmf_annotations
from .models import SigMFWindow, WaterfallSettings
from .plots import automatic_dbfs_ranges, waterfall_figure
from .style import TEAL, heatmap_grid_color, style_figure

COLORMAPS = (
    "Portland",
    "Plasma",
    "Viridis",
    "Cividis",
    "Inferno",
    "Magma",
    "Turbo",
    "Electric",
    "Hot",
    "IceFire",
)


def view(data: SigMFWindow, ui: UI) -> None:
    """Configure, process, and present one exact selected sample window."""
    defaults = WaterfallSettings()
    channel_options = {
        index: f"Channel {index + 1}" for index in range(data.recording.channel_count)
    }
    with ui.details_group("Spectrogram processing"):
        channel = int(
            ui.select(
                "channel",
                label="Input channel",
                default=defaults.channel,
                options=channel_options,
            )
        )
        fft_size = int(
            ui.select(
                "fft_size",
                label="Fast-time FFT size (samples)",
                default=defaults.fft_size,
                options=(256, 512, 1024, 2048, 4096, 8192),
            )
        )
        overlap_percent = int(
            ui.select(
                "overlap_percent",
                label="Slow-time overlap (%)",
                default=defaults.overlap_percent,
                options=(0, 25, 50, 75),
            )
        )
    products = ui.compute(
        "sigmf-waterfall-analysis",
        lambda: analyze(
            data,
            WaterfallSettings(
                fft_size=fft_size,
                overlap_percent=overlap_percent,
                channel=channel,
            ),
        ),
    )

    colormap = ui.colormap(
        "colormap",
        label="Waterfall colormap",
        default="Portland",
        options=COLORMAPS,
        group="Display",
    )
    automatic_waterfall, automatic_spectrum = automatic_dbfs_ranges(products)
    zmin, zmax = ui.limits(
        "dbfs_limits",
        label="Waterfall dBFS limits",
        default=automatic_waterfall,
        minimum=-200.0,
        maximum=20.0,
        step=1.0,
        group="Display",
    )
    spectrum_ymin, spectrum_ymax = ui.limits(
        "spectrum_dbfs_limits",
        label="Average PSD limits (dBFS)",
        default=automatic_spectrum,
        minimum=-200.0,
        maximum=20.0,
        step=1.0,
        group="Display",
    )
    spectrum_style = ui.trace_style(
        "spectrum_style",
        label="Average PSD",
        color=TEAL,
        width=1.4,
        group="Display",
    )
    show_colorbar = ui.toggle(
        "show_colorbar",
        label="Show dBFS colorbar",
        default=True,
        group="Display",
    )
    show_annotations = ui.toggle(
        "show_annotations",
        label="Show annotations",
        default=True,
        group="Annotations",
    )
    annotation_color = ui.color(
        "annotation_region_color",
        label="Annotation color",
        default="#ffffff",
        group="Annotations",
    )
    annotation_width = float(
        ui.number(
            "annotation_region_width",
            label="Line weight",
            default=1.5,
            minimum=0.5,
            maximum=8.0,
            step=0.5,
            group="Annotations",
        )
    )
    annotation_opacity = float(
        ui.number(
            "annotation_region_opacity",
            label="Opacity",
            default=0.8,
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            group="Annotations",
        )
    )
    with ui.details_group("Raster rendering"):
        render_width = int(
            ui.select(
                "render_width",
                label="Heatmap render width",
                default=1024,
                options=(256, 512, 1024, 2048),
            )
        )
        render_height = int(
            ui.select(
                "render_height",
                label="Heatmap render height",
                default=512,
                options=(128, 256, 512, 1024),
            )
        )
        aggregation = str(
            ui.select(
                "render_aggregation",
                label="Display-cell aggregation",
                default="mean",
                options=("max", "mean", "median"),
            )
        )

    global_metadata = data.recording.metadata["global"]
    title = str(
        global_metadata.get("core:description")
        or data.recording.metadata_path.name.removesuffix(".sigmf-meta")
    )
    if data.recording.channel_count > 1:
        title = f"{title} · Channel {channel + 1}"

    def figure():
        rendered = waterfall_figure(
            products,
            viewport=ui.plot_viewport("sigmf-waterfall"),
            colormap=colormap,
            zmin=zmin,
            zmax=zmax,
            spectrum_ymin=spectrum_ymin,
            spectrum_ymax=spectrum_ymax,
            spectrum_style=spectrum_style,
            show_colorbar=show_colorbar,
            render_width=render_width,
            render_height=render_height,
            aggregation=aggregation,
            annotations=(
                read_sigmf_annotations(data.recording) if show_annotations else ()
            ),
            annotation_color=annotation_color,
            annotation_width=annotation_width,
            annotation_opacity=annotation_opacity,
        )
        styled = style_figure(rendered, ui.theme, title)
        styled.update_xaxes(
            gridcolor=heatmap_grid_color(ui.theme),
            gridwidth=0.35,
            row=2,
            col=1,
        )
        styled.update_yaxes(
            gridcolor=heatmap_grid_color(ui.theme),
            gridwidth=0.35,
            row=2,
            col=1,
        )
        return styled

    center_frequency = data.recording.center_frequency_at(data.start_sample)
    ui.stat("Sampling rate", f"{data.recording.sample_rate / 1e6:g} MS/s")
    ui.stat("Center frequency", f"{center_frequency / 1e6:g} MHz")
    ui.stat("Displayed channel", f"{channel + 1}/{data.recording.channel_count}")
    ui.stat("Buffer memory", format_bytes(products.buffer_nbytes))
    with ui.tab("Spectrum + waterfall"):
        ui.plot(
            figure,
            key="sigmf-waterfall",
            axis_navigation="bounded",
        )


__all__ = ["COLORMAPS", "view"]
