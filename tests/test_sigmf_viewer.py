from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image
from sigvue import AnnotationRequest, BatchRequest

from sigmf_viewer.analysis import analyze
from sigmf_viewer.annotations import (
    WaterfallAnnotator,
    read_sigmf_annotations,
)
from sigmf_viewer.batch import (
    RENDER_WATERFALL,
    SigMFWaterfallBatch,
    render_recording_png,
    render_recording_viewer,
)
from sigmf_viewer.models import SigMFCollection, WaterfallSettings
from sigmf_viewer.plots import automatic_dbfs_ranges, waterfall_figure
from sigmf_viewer.reader import (
    create_files,
    create_reader,
    power_overview,
    power_spectrum_overview,
)
from sigmf_viewer.sigmf import open_recording, read_window
from sigmf_viewer.workspace import create_workspace


def test_default_fft_size_is_256(tmp_path):
    assert WaterfallSettings().fft_size == 256
    assert SigMFWaterfallBatch(tmp_path).fft_size == 256
    assert SigMFWaterfallBatch(tmp_path).max_native_cells == 75_000_000
    assert SigMFWaterfallBatch(tmp_path).png_time_bins == 1600
    assert SigMFWaterfallBatch(tmp_path).png_width_pixels == 2400
    assert SigMFWaterfallBatch(tmp_path).png_height_pixels == 1600


def write_ci16(
    root: Path,
    stem: str,
    *,
    sample_rate: float = 100_000.0,
    channels: int = 1,
    samples: int = 4096,
    description: str | None = None,
    channel_names: tuple[str, ...] | None = None,
) -> tuple[Path, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples) / sample_rate
    signals = np.asarray(
        [
            0.55 * np.exp(2j * np.pi * (8_000.0 + channel * 3_000.0) * time)
            for channel in range(channels)
        ],
        dtype=np.complex64,
    )
    frames = np.empty((samples, channels, 2), dtype="<i2")
    frames[..., 0] = np.rint(signals.T.real * 32767)
    frames[..., 1] = np.rint(signals.T.imag * 32767)
    metadata = root / f"{stem}.sigmf-meta"
    frames.tofile(root / f"{stem}.sigmf-data")
    metadata.write_text(
        json.dumps(
            {
                "global": {
                    "core:version": "1.2.6",
                    "core:datatype": "ci16_le",
                    "core:sample_rate": sample_rate,
                    "core:num_channels": channels,
                    "core:description": description or stem,
                    **(
                        {
                            "core:extensions": [
                                {
                                    "name": "sigmf_viewer",
                                    "version": "0.1.0",
                                    "optional": True,
                                }
                            ],
                            "sigmf_viewer:channel_names": list(channel_names),
                        }
                        if channel_names is not None
                        else {}
                    ),
                },
                "captures": [
                    {
                        "core:sample_start": 0,
                        "core:frequency": 806_000_000.0,
                        "core:datetime": "2022-04-09T11:09:33Z",
                    }
                ],
                "annotations": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata, signals


def test_exact_multichannel_read_and_direct_pair_selection(tmp_path):
    metadata, expected = write_ci16(
        tmp_path,
        "capture",
        channels=2,
        channel_names=("Reference antenna", "Monitor antenna"),
    )
    recording = open_recording(metadata.with_suffix(".sigmf-data"))
    selected = recording.read(50, 100)

    assert recording.channel_count == 2
    assert recording.channel_labels == (
        "Reference antenna",
        "Monitor antenna",
    )
    assert selected.shape == (2, 100)
    assert np.allclose(selected, expected[:, 50:150], atol=2 / 32768)

    files = create_files(metadata)
    assert len(files.resources()) == 1
    assert files.resources()[0].identifier == "capture"


def test_cf64_source_precision_is_not_downcast(tmp_path):
    metadata = tmp_path / "precise.sigmf-meta"
    data = tmp_path / "precise.sigmf-data"
    expected = np.asarray(
        [
            complex(0.123456789012345, -0.234567890123456),
            complex(0.123456789012346, -0.234567890123457),
        ],
        dtype=np.complex128,
    )
    np.column_stack((expected.real, expected.imag)).astype("<f8").tofile(data)
    metadata.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "cf64_le",
                    "core:sample_rate": 1_000.0,
                },
                "captures": [],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    actual = open_recording(metadata).read(0, 2)

    assert actual.dtype == np.complex128
    assert np.array_equal(actual[0], expected)


def test_standard_collection_members_are_grouped_without_duplication(tmp_path):
    collection_root = tmp_path / "campaign"
    first, _ = write_ci16(collection_root / "streams", "first")
    second, _ = write_ci16(collection_root / "streams", "second")
    manifest = collection_root / "trial.sigmf-collection"
    manifest.write_text(
        json.dumps(
            {
                "collection": {
                    "core:version": "1.2.6",
                    "core:streams": [
                        {"name": "streams/first.sigmf-meta"},
                        {"name": "streams/second.sigmf-meta"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    standalone, _ = write_ci16(tmp_path / "loose", "standalone")

    resources = create_files(tmp_path).resources()
    by_source = {Path(item.source).resolve(): item for item in resources}

    assert len(resources) == 2
    assert set(by_source) == {manifest.resolve(), standalone.resolve()}
    assert all(resource.navigation_path == () for resource in resources)
    assert by_source[manifest.resolve()].summary["recordings"] == 2
    reader = create_files(tmp_path)
    collection = reader.open(
        next(
            resource.source
            for resource in reader.resources()
            if Path(resource.source).resolve() == manifest.resolve()
        )
    )
    assert isinstance(collection, SigMFCollection)
    assert tuple(member.metadata_path.resolve() for member in collection.members) == (
        first.resolve(),
        second.resolve(),
    )
    buffered = create_reader(
        {
            "data_root": tmp_path,
            "default_window_seconds": 0.01,
            "minimum_window_seconds": 0.002,
            "window_step_seconds": 0.001,
            "overview_bins": 10,
        }
    )
    opened = buffered.open(manifest)
    selected = buffered.read(
        opened,
        0.0,
        0.01,
        member="streams/second.sigmf-meta",
    )
    assert selected.recording.metadata_path.resolve() == second.resolve()
    assert selected.sample_count == 1000
    collection_resource = next(
        resource
        for resource in buffered.resources()
        if Path(resource.source).resolve() == manifest.resolve()
    )
    batch = SigMFWaterfallBatch(
        tmp_path / "outputs",
        fft_size=256,
        overlap_percent=50,
    )
    destination = batch.item_destination(
        collection_resource,
        type("Request", (), {"action": RENDER_WATERFALL})(),
    )
    assert len(destination.files) == 4
    assert any("first" in filename for filename in destination.files)
    assert any("second" in filename for filename in destination.files)
    assert sum(filename.endswith(".png") for filename in destination.files) == 2
    assert sum(filename.endswith(".html") for filename in destination.files) == 2


def test_invalid_pairs_and_collection_manifests_are_dropped_individually(
    tmp_path,
):
    valid, _ = write_ci16(tmp_path / "good", "valid")
    broken = tmp_path / "bad" / "broken.sigmf-meta"
    broken.parent.mkdir()
    broken.write_text("{not json", encoding="utf-8")
    (tmp_path / "bad.sigmf-collection").write_text(
        json.dumps(
            {
                "collection": {
                    "core:streams": [
                        {"name": "good/valid.sigmf-meta"},
                        {"name": "missing.sigmf-meta"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    resources = create_files(tmp_path).resources()

    assert [Path(resource.source).resolve() for resource in resources] == [
        valid.resolve()
    ]


def test_window_reader_analysis_and_plot_keep_absolute_axes(tmp_path):
    write_ci16(tmp_path, "tone", samples=8192)
    reader = create_reader(
        {
            "data_root": tmp_path,
            "filename": "*.sigmf-meta",
            "default_window_seconds": 0.02,
            "minimum_window_seconds": 0.005,
            "window_step_seconds": 0.001,
            "overview_bins": 20,
        }
    )
    recording = reader.open(reader.discover()[0])
    window = reader.read(recording, 0.01, 0.03)
    products = analyze(
        window,
        WaterfallSettings(
            fft_size=256,
            overlap_percent=50,
            channel=0,
        ),
    )
    figure = waterfall_figure(products)

    assert window.start_sample == 1000
    assert window.samples.shape == (1, 2000)
    assert products.waterfall_dbfs.shape[1] == 256
    assert np.allclose(products.time_edges_ms[[0, -1]], [10.0, 30.0])
    assert np.allclose(
        np.diff(products.time_edges_ms),
        np.diff(products.time_edges_ms)[0],
    )
    frame_peaks = np.max(products.waterfall_dbfs, axis=1)
    assert frame_peaks[0] > np.max(frame_peaks) - 6.0
    assert frame_peaks[-1] > np.max(frame_peaks) - 6.0
    assert len(power_overview(recording, bins=20)) == 20
    overview_line, overview_heatmap = power_spectrum_overview(
        recording,
        time_bins=20,
        frequency_bins=16,
        fft_size=64,
    )
    assert overview_line.shape == (20,)
    assert overview_heatmap.shape == (16, 20)
    assert np.all(np.isfinite(overview_heatmap))
    assert np.allclose(
        overview_line,
        power_overview(recording, bins=20),
    )
    assert np.allclose(figure.data[0].x, products.spectrum_dbfs)
    assert np.allclose(figure.data[0].y, products.frequency_mhz)
    assert figure.data[0].showlegend is False
    assert np.asarray(figure.data[1].z).shape == products.waterfall_dbfs.T.shape
    assert figure.data[1].hovertemplate == (
        "Recording time: %{x:.2f} ms"
        "<br>RF frequency: %{y:.6f} MHz"
        "<br>dBFS: %{z:.2f}"
        "<extra></extra>"
    )
    assert figure.layout.xaxis.title.text == "Power (dBFS)"
    assert figure.layout.yaxis.title.text == "RF frequency (MHz)"
    assert figure.layout.xaxis2.title.text == "Recording time (ms)"
    assert figure.layout.yaxis2.matches == "y"
    assert figure.layout.yaxis2.showticklabels is False
    assert figure.layout.xaxis2.tickformat == ".2f"
    assert figure.layout.yaxis2.tickformat == ".2f"
    assert figure.layout.xaxis2.autorange is False
    assert figure.layout.yaxis2.autorange is False
    assert "sigmf-viewer" in figure.layout.uirevision


def test_progressive_raster_dimensions_and_switches_follow_time_frequency_axes(
    tmp_path,
):
    metadata, _ = write_ci16(tmp_path, "raster", samples=8192)
    recording = open_recording(metadata)
    products = analyze(
        read_window(recording, 1000, 3000),
        WaterfallSettings(fft_size=256, overlap_percent=50, channel=0),
    )

    time_base = [
        float(products.time_edges_ms[0]),
        float(products.time_edges_ms[-1]),
    ]
    frequency_step = float(products.frequency_mhz[1] - products.frequency_mhz[0])
    frequency_base = [
        float(products.frequency_mhz[0] - frequency_step / 2),
        float(products.frequency_mhz[-1] + frequency_step / 2),
    ]
    rastered = waterfall_figure(
        products,
        render_width=5,
        render_height=7,
    )
    assert not rastered.layout.images
    assert np.asarray(rastered.data[1].z).shape == (7, 5)
    assert "Recording time" in rastered.data[1].hovertemplate
    assert "RF frequency" in rastered.data[1].hovertemplate
    assert "dBFS" in rastered.data[1].hovertemplate
    assert rastered.data[1].xaxis == "x2"
    assert rastered.data[1].yaxis == "y2"
    assert np.allclose(
        [rastered.data[1].x[0], rastered.data[1].x[-1]],
        time_base,
    )
    assert np.allclose(
        [rastered.data[1].y[0], rastered.data[1].y[-1]],
        frequency_base,
    )

    def displayed_extent(figure):
        if figure.layout.images:
            image = figure.layout.images[0]
            return float(image.sizex), float(image.sizey)
        heatmap = figure.data[1]
        return (
            abs(float(heatmap.x[-1]) - float(heatmap.x[0])),
            abs(float(heatmap.y[-1]) - float(heatmap.y[0])),
        )

    first_zoom = waterfall_figure(
        products,
        viewport={
            "xaxis2": {"range": [14.0, 24.0], "base": time_base},
            "yaxis2": {
                "range": [frequency_base[0], sum(frequency_base) / 2],
                "base": frequency_base,
            },
        },
        render_width=2,
        render_height=2,
    )
    second_zoom = waterfall_figure(
        products,
        viewport={
            "xaxis2": {"range": [16.0, 20.0], "base": time_base},
            "yaxis2": {
                "range": [
                    frequency_base[0],
                    frequency_base[0] + (frequency_base[1] - frequency_base[0]) / 4,
                ],
                "base": frequency_base,
            },
        },
        render_width=2,
        render_height=2,
    )
    first_width, first_height = displayed_extent(first_zoom)
    second_width, second_height = displayed_extent(second_zoom)
    assert first_width > second_width
    assert first_height > second_height

    direct = waterfall_figure(
        products,
        progressive_render=False,
        show_spectrum=False,
        render_width=5,
        render_height=7,
    )
    assert not direct.layout.images
    assert len(direct.data) == 1
    assert direct.data[0].xaxis == "x2"
    assert direct.data[0].yaxis == "y2"
    assert np.allclose(direct.data[0].z, products.waterfall_dbfs.T)
    assert direct.layout.xaxis2.title.text == "Recording time (ms)"
    assert direct.layout.yaxis2.title.text == "RF frequency (MHz)"
    assert direct.layout.yaxis2.domain[1] - direct.layout.yaxis2.domain[0] > 0.99


def test_annotations_are_persisted_and_rendered_as_vector_traces(tmp_path):
    metadata, _ = write_ci16(tmp_path, "annotated")
    recording = open_recording(metadata)
    window = read_window(recording, 0, 2048)
    annotator = WaterfallAnnotator()
    assert [field.plot_binding.axis for field in annotator.fields[:4]] == [
        "xaxis2",
        "xaxis2",
        "yaxis2",
        "yaxis2",
    ]
    created = annotator.annotate(
        recording,
        window,
        AnnotationRequest(
            position_seconds=0.0,
            values={
                "start_seconds": "0.002",
                "stop_seconds": "0.006",
                "frequency_lower_hz": "805990000",
                "frequency_upper_hz": "806010000",
                "comment": "LTE allocation",
            },
        ),
    )
    products = analyze(window, WaterfallSettings(256, 50, 0))
    figure = waterfall_figure(
        products,
        annotations=read_sigmf_annotations(recording),
    )

    assert created.comment == "LTE allocation"
    persisted = json.loads(metadata.read_text(encoding="utf-8"))
    assert persisted["annotations"][0]["core:comment"] == "LTE allocation"
    assert [trace.name for trace in figure.data[-2:]] == [
        "Annotations",
        "Annotation details",
    ]
    assert list(figure.data[-2].x[:5]) == [2.0, 6.0, 6.0, 2.0, 2.0]
    assert np.allclose(
        figure.data[-2].y[:5],
        [805.99, 805.99, 806.01, 806.01, 805.99],
    )


def test_batch_viewer_preserves_native_stft_cells_and_is_zoomable(tmp_path):
    metadata, _ = write_ci16(
        tmp_path / "data",
        "multi",
        channels=2,
        samples=4096,
        channel_names=("Primary antenna", "Reference antenna"),
    )
    recording = open_recording(metadata)
    png_output = tmp_path / "short.png"
    rendered_png = render_recording_png(
        recording,
        png_output,
        channel=1,
        fft_size=256,
        overlap_percent=50,
        time_bins=16,
        width_pixels=800,
        height_pixels=600,
    )
    assert rendered_png == png_output.resolve()
    with Image.open(png_output) as image:
        assert image.format == "PNG"
        assert image.size == (800, 600)

    output = tmp_path / "short.html"
    rendered, assets = render_recording_viewer(
        recording,
        output,
        channel=1,
        fft_size=256,
        overlap_percent=50,
        shareable_png=png_output.name,
    )
    short_frames = (4096 + 127) // 128
    assert rendered == output.resolve()
    html = output.read_text(encoding="utf-8")
    assert "Full recording" in html
    assert "1:1 time" in html
    assert "frames/screen px" in html
    assert "event.deltaX" in html
    assert "event.shiftKey" in html
    assert "Full PNG" in html
    assert '"shareablePng":"short.png"' in html
    metadata_path = next(path for path in assets if path.name == "metadata.json")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    discovered_min = metadata_payload["dbfsMin"]
    discovered_max = metadata_payload["dbfsMax"]
    assert metadata_payload["width"] == short_frames
    assert metadata_payload["height"] == 256
    assert metadata_payload["nativeCells"] == short_frames * 256
    assert discovered_min % 5.0 == 0.0
    assert discovered_max % 5.0 == 0.0
    assert discovered_max - discovered_min >= 20.0
    products = analyze(
        read_window(recording, 0, recording.sample_count),
        WaterfallSettings(256, 50, 1),
    )
    assert products.waterfall_dbfs.shape == (short_frames, 256)
    assert (discovered_min, discovered_max) == automatic_dbfs_ranges(products)[0]
    dbfs = products.waterfall_dbfs[0]
    expected_colors = np.asarray(
        matplotlib.colormaps["turbo"](
            np.clip(
                (dbfs - discovered_min)
                / (discovered_max - discovered_min),
                0.0,
                1.0,
            ),
            bytes=True,
        ),
        dtype=np.uint8,
    )[:, :3]
    native_tile = (
        output.with_name("short.assets")
        / str(metadata_payload["maxLevel"])
        / "0_0.png"
    )
    with Image.open(native_tile) as tile:
        assert tile.size == (short_frames, 256)
        pixels = tile.load()
        for frequency_bin in (0, 63, 127, 255):
            x = 0
            y = 255 - frequency_bin
            expected = tuple(int(value) for value in expected_colors[frequency_bin])
            assert pixels[x, y] == expected
    reduced_tile = (
        output.with_name("short.assets")
        / str(metadata_payload["maxLevel"] - 1)
        / "0_0.png"
    )
    normalized_pair = np.clip(
        (products.waterfall_dbfs[:2] - discovered_min)
        / (discovered_max - discovered_min),
        0.0,
        1.0,
    )
    max_hold_indexes = np.max(
        np.minimum(np.floor(normalized_pair * 256.0), 255.0).astype(
            np.uint8
        ),
        axis=0,
    )
    color_lut = np.asarray(
        matplotlib.colormaps["turbo"](
            np.arange(256) / 256.0,
            bytes=True,
        ),
        dtype=np.uint8,
    )[:, :3]
    with Image.open(reduced_tile) as tile:
        assert tile.size == (short_frames // 2, 256)
        pixels = tile.load()
        for frequency_bin in (0, 63, 127, 255):
            expected = tuple(
                int(value)
                for value in color_lut[max_hold_indexes[frequency_bin]]
            )
            assert pixels[0, 255 - frequency_bin] == expected

    resource = create_files(metadata).resources()[0]
    batch = SigMFWaterfallBatch(
        tmp_path / "outputs",
        fft_size=256,
        overlap_percent=50,
    )
    destination = batch.item_destination(
        resource,
        type("Request", (), {"action": RENDER_WATERFALL})(),
    )
    assert destination.directory == (tmp_path / "outputs").resolve()
    assert len(destination.files) == 4
    assert sum(name.endswith(".png") for name in destination.files) == 2
    assert sum(name.endswith(".html") for name in destination.files) == 2
    assert all(
        "-tiled.html" in name
        for name in destination.files
        if name.endswith(".html")
    )
    assert any("primary-antenna" in name for name in destination.files)
    assert any("reference-antenna" in name for name in destination.files)


def test_tiled_batch_rejects_unbounded_native_output_before_rendering(tmp_path):
    metadata, _ = write_ci16(
        tmp_path / "data",
        "capture",
        samples=4096,
    )
    recording = open_recording(metadata)
    output = tmp_path / "bounded.html"

    with pytest.raises(ValueError, match="interactive windowed viewer"):
        render_recording_viewer(
            recording,
            output,
            fft_size=256,
            overlap_percent=50,
            max_native_cells=8191,
        )

    assert not output.exists()
    assert not output.with_name("bounded.assets").exists()

    resource = create_files(metadata).resources()[0]
    batch = SigMFWaterfallBatch(
        tmp_path / "outputs",
        max_native_cells=8191,
        png_time_bins=16,
        png_width_pixels=800,
        png_height_pixels=600,
    )
    destination = batch.item_destination(
        resource,
        type("Request", (), {"action": RENDER_WATERFALL})(),
    )
    assert len(destination.files) == 1
    assert destination.files[0].endswith(".png")
    result = batch.run_item(
        resource,
        recording,
        BatchRequest(RENDER_WATERFALL),
        destination.directory,
    )
    assert len(result.files) == 1
    assert result.files[0].suffix == ".png"
    assert result.files[0].is_file()
    assert result.assets == ()


def test_workspace_is_one_lazy_windowed_pipeline(tmp_path):
    write_ci16(
        tmp_path / "data",
        "capture",
        channels=2,
        channel_names=("Reference antenna", "Monitor antenna"),
    )
    workspace = create_workspace(
        {
            "data_root": tmp_path / "data",
            "output_root": tmp_path / "outputs",
            "filename": "*.sigmf-meta",
            "default_window_seconds": 0.012,
            "minimum_window_seconds": 0.004,
            "window_step_seconds": 0.002,
            "overview_bins": 12,
            "batch_fft_size": 256,
            "batch_overlap_percent": 50,
            "batch_colormap": "turbo",
            "batch_max_native_cells": 75_000_000,
            "batch_png_time_bins": 1600,
            "batch_png_width_pixels": 2400,
            "batch_png_height_pixels": 1600,
        }
    )

    assert workspace.metadata.identifier == "sigmf-viewer"
    assert workspace.lazy_views is True
    assert workspace.flatten_discovery is True
    assert len(workspace.reader.resources()) == 1
    assert workspace.annotator is not None
    assert workspace.batch is not None
    assert workspace.batch.png_time_bins == 1600
    assert workspace.batch.png_width_pixels == 2400
    assert workspace.batch.png_height_pixels == 1600
    resource = workspace.discover_items()[0]
    opened = workspace.open_item(resource.identifier)
    assert opened.page.playback.overview_values == ()
    assert len(opened.page.playback.overview_heatmap) == 48
    assert all(len(row) == 12 for row in opened.page.playback.overview_heatmap)
    assert opened.page.playback.overview_colormap_control == "colormap"
    assert opened.page.playback.overview_limits_control == "dbfs_limits"
    assert all(control.name != "channel" for control in opened.page.controls)
    stack = [opened.page.layout]
    switchers = []
    while stack:
        node = stack.pop()
        if node.kind == "view_switcher":
            switchers.append(node)
        stack.extend(node.children)
    assert len(switchers) == 1
    assert switchers[0].props["label"] == "Channel"
    assert switchers[0].props["selector"] == "dropdown"
    assert switchers[0].props["options"] == (("Reference antenna", "Monitor antenna"),)


def test_single_channel_workspace_has_no_channel_ui(tmp_path):
    write_ci16(tmp_path / "data", "single")
    workspace = create_workspace(
        {
            "data_root": tmp_path / "data",
            "output_root": tmp_path / "outputs",
        }
    )

    opened = workspace.open_item(workspace.discover_items()[0].identifier)
    assert all(control.name != "channel" for control in opened.page.controls)
    stack = [opened.page.layout]
    while stack:
        node = stack.pop()
        assert node.kind != "view_switcher"
        stack.extend(node.children)
    assert [view.name for view in opened.page.views] == ["sigmf-viewer"]
