from __future__ import annotations

import json
import struct
from base64 import b64decode
from pathlib import Path

import numpy as np
from sigvue import AnnotationRequest

from sigmf_viewer.analysis import analyze
from sigmf_viewer.annotations import (
    WaterfallAnnotator,
    read_sigmf_annotations,
)
from sigmf_viewer.batch import (
    RENDER_WATERFALL,
    SigMFWaterfallBatch,
    render_recording_png,
)
from sigmf_viewer.models import SigMFCollection, WaterfallSettings
from sigmf_viewer.plots import waterfall_figure
from sigmf_viewer.reader import (
    create_files,
    create_reader,
    power_overview,
    power_spectrum_overview,
)
from sigmf_viewer.sigmf import open_recording, read_window
from sigmf_viewer.workspace import create_workspace


def write_ci16(
    root: Path,
    stem: str,
    *,
    sample_rate: float = 100_000.0,
    channels: int = 1,
    samples: int = 4096,
    description: str | None = None,
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
    )
    recording = open_recording(metadata.with_suffix(".sigmf-data"))
    selected = recording.read(50, 100)

    assert recording.channel_count == 2
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
        time_bins=64,
        width_pixels=640,
        height_pixels=480,
    )
    destination = batch.item_destination(
        collection_resource,
        type("Request", (), {"action": RENDER_WATERFALL})(),
    )
    assert len(destination.files) == 2
    assert any("first" in filename for filename in destination.files)
    assert any("second" in filename for filename in destination.files)


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
    assert np.allclose(figure.data[0].x, products.frequency_mhz)
    assert np.allclose(figure.data[0].y, products.spectrum_dbfs)
    assert np.asarray(figure.data[1].z).shape == products.waterfall_dbfs.T.shape
    assert figure.layout.yaxis.title.text == "Power (dBFS)"
    assert figure.layout.xaxis2.title.text == "Recording time (ms)"
    assert figure.layout.yaxis2.title.text == "RF frequency (MHz)"
    assert figure.layout.xaxis2.tickformat == ".2f"
    assert figure.layout.yaxis2.tickformat == ".2f"
    assert figure.layout.xaxis2.autorange is False
    assert figure.layout.yaxis2.autorange is False
    assert "sigmf-viewer" in figure.layout.uirevision


def test_progressive_raster_dimensions_and_switches_follow_rotated_axes(tmp_path):
    metadata, _ = write_ci16(tmp_path, "raster", samples=8192)
    recording = open_recording(metadata)
    products = analyze(
        read_window(recording, 1000, 3000),
        WaterfallSettings(fft_size=256, overlap_percent=50, channel=0),
    )

    rastered = waterfall_figure(
        products,
        render_width=5,
        render_height=7,
    )
    assert len(rastered.layout.images) == 1
    encoded = str(rastered.layout.images[0].source).split(",", 1)[1]
    raster = b64decode(encoded)
    assert raster[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", raster[16:24]) == (5, 7)
    assert rastered.layout.images[0].xref == "x2"
    assert rastered.layout.images[0].yref == "y2"

    time_base = [
        float(products.time_edges_ms[0]),
        float(products.time_edges_ms[-1]),
    ]
    frequency_step = float(products.frequency_mhz[1] - products.frequency_mhz[0])
    frequency_base = [
        float(products.frequency_mhz[0] - frequency_step / 2),
        float(products.frequency_mhz[-1] + frequency_step / 2),
    ]
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
    assert first_zoom.layout.images[0].sizex > second_zoom.layout.images[0].sizex
    assert first_zoom.layout.images[0].sizey > second_zoom.layout.images[0].sizey

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


def test_batch_png_is_exact_size_and_has_durable_contract(tmp_path):
    metadata, _ = write_ci16(
        tmp_path / "data",
        "multi",
        channels=2,
        samples=4096,
    )
    recording = open_recording(metadata)
    output = tmp_path / "rendered.png"
    render_recording_png(
        recording,
        output,
        channel=1,
        fft_size=256,
        overlap_percent=50,
        time_bins=64,
        width_pixels=640,
        height_pixels=480,
    )
    with output.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        length = struct.unpack(">I", stream.read(4))[0]
        assert stream.read(4) == b"IHDR"
        width, height = struct.unpack(">II", stream.read(length)[:8])
    assert (width, height) == (640, 480)

    resource = create_files(metadata).resources()[0]
    batch = SigMFWaterfallBatch(
        tmp_path / "outputs",
        fft_size=256,
        overlap_percent=50,
        time_bins=64,
        width_pixels=640,
        height_pixels=480,
    )
    destination = batch.item_destination(
        resource,
        type("Request", (), {"action": RENDER_WATERFALL})(),
    )
    assert destination.directory == (tmp_path / "outputs").resolve()
    assert len(destination.files) == 2
    assert all(name.endswith(".png") for name in destination.files)


def test_workspace_is_one_lazy_windowed_pipeline(tmp_path):
    write_ci16(tmp_path / "data", "capture")
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
            "batch_time_bins": 64,
            "batch_width_pixels": 640,
            "batch_height_pixels": 480,
        }
    )

    assert workspace.metadata.identifier == "sigmf-viewer"
    assert workspace.lazy_views is True
    assert workspace.flatten_discovery is True
    assert len(workspace.reader.resources()) == 1
    assert workspace.annotator is not None
    assert workspace.batch is not None
    resource = workspace.discover_items()[0]
    opened = workspace.open_item(resource.identifier)
    assert opened.page.playback.overview_values == ()
    assert len(opened.page.playback.overview_heatmap) == 48
    assert all(len(row) == 12 for row in opened.page.playback.overview_heatmap)
