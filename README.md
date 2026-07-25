# SigMF Viewer

A focused local browser for SigMF recordings, built as one Sigvue
`Workspace`. The application opens directly onto a flat recording catalog and
supports:

- any directory containing `.sigmf-meta`/`.sigmf-data` pairs;
- one pair selected by either its metadata or data path;
- standard `.sigmf-collection` manifests; and
- interleaved multi-channel recordings.

Each standalone pair is one catalog item. A collection is also one catalog
item: its member recordings are selected from a dropdown after it is opened.
Collection members are not duplicated as standalone rows. Discovery validates
every candidate independently, so a missing payload, malformed metadata file,
or broken collection is omitted without hiding the remaining usable data.

```text
src/sigmf_viewer/
├── sigmf.py       format validation, collections, and exact ranged sample I/O
├── reader.py      fault-tolerant flat discovery and window selection
├── models.py      recordings, collections, windows, and analysis values
├── analysis.py    framework-independent STFT and average PSD
├── plots.py       pure Plotly waterfall figure construction
├── view.py        controls, statistics, tabs, and layout
├── annotations.py standard SigMF annotation discovery and persistence
├── batch.py       durable high-resolution whole-recording PNGs
├── workspace.py   one reader + one view callback + one workspace
├── cli.py         application-specific browser command
├── desktop.py     native pywebview window and server lifetime
├── runtime.py     durable data/output roots and generated profiles
└── _packaging/    installed PyInstaller build support

scripts/
└── download_data.py   repository-only example-data downloader
```

## Install and download

For development alongside Sigvue:

```bash
python -m pip install -e ../Scientific-Workspace-Browser
python -m pip install -e .
python scripts/download_data.py
```

The repository-only download script retrieves:

- a real 806 MHz LTE downlink and 847 MHz LTE uplink recorded by Daniel
  Estévez, grouped into one standard collection;
- six compact synthetic spectrum-monitoring scenes from Operation Coldferry:
  CW, OOK, FSK, LFM, FHSS, and a contested-band scene; and
- the compact two-channel official SigMF logo recording.

All remote files have fixed sizes and SHA-256 digests. Existing metadata is
preserved so locally added annotations are not overwritten. Choose smaller
groups without downloading the roughly 221 MiB LTE collection:

```bash
python scripts/download_data.py --datasets coldferry sigmf-logo
python scripts/download_data.py --datasets lte
```

The downloader is test/development tooling. It is intentionally absent from
the installed package, wheel, and console entry points.

## Run

```bash
sigmf-viewer
```

Open <http://127.0.0.1:8000>. With all example data present, discovery shows
eight flat items: one two-recording LTE collection and seven compact
standalone recordings.

Point the application at another directory, pair, or collection:

```bash
sigmf-viewer --data-root /path/to/sigmf
sigmf-viewer --recording /path/to/capture.sigmf-meta
sigmf-viewer --recording /path/to/capture.sigmf-data
sigmf-viewer --recording /path/to/campaign.sigmf-collection
```

An explicit browser profile remains supported:

```bash
sigmf-viewer --config browser.toml
```

The reader supports standard real and complex signed, unsigned, and
floating-point 8/16/32/64-bit datatypes in their defined endianness, plus the
deployed `sc16_le` alias. Integer samples are normalized to full scale before
dBFS analysis. Float64 and wide-integer inputs retain complex128 precision.
Reads are exact, ranged, and channel-aware.

## Interactive view

The viewer uses continuous window mode rather than segmented playback.
Opening an item provides:

- a collection recording dropdown when applicable;
- a low-resolution full-recording spectrum, plus a draggable exact sample
  window and full-extent icon;
- a metadata-labeled channel view switcher for multi-channel recordings, with
  honest zero-based “unlabeled” fallbacks and no redundant channel UI for
  single-channel data;
- explicit fast-time FFT size (256 samples by default) and slow-time overlap
  controls;
- a visual picker with ten colormaps;
- fixed user-controlled waterfall and average-PSD dBFS ranges;
- a compact average PSD occupying the left 10% of the plot width and sharing
  the waterfall's vertical frequency axis;
- independent switches for the average PSD and progressive raster rendering;
- viewport-aware heatmap rendering for the currently visible time/frequency
  bounds after every zoom;
- stable zoom limits when the window, FFT size, or style controls change;
- a quiet heatmap grid and two-decimal time/frequency axes; and
- hoverable standard SigMF annotations with box selection plus color, weight,
  opacity, and visibility controls.

Viewport rasterization changes display cells only. It does not alter samples,
STFT values, annotation coordinates, or the headless pipeline. Plotly Home
and double-click reset always restore the complete current buffer.
Slow-time cells have equal width over the exact selected buffer. Boundary FFTs
are centered, padded only outside the buffer, and normalized by the taper
support that contains real samples, so the first and last heatmap cells are
neither stretched nor artificially attenuated.

Both the main waterfall and its compact timeline spectrum place recording
time left-to-right and frequency bottom-to-top. The timeline covers the
complete source: every sample is read once into non-overlapping time bins,
FFT power is combined in linear units, and only the small frequency-by-time
result is sent to the browser.
`overview_bins`, `overview_frequency_bins`, and `overview_fft_size` in the
workspace configuration control that summary without changing the 30 px bar.

## Batch rendering

Each catalog row has a **Render high-resolution waterfall PNGs** action. A
standalone recording produces one PNG per channel. A collection action
renders every member and every channel. The workspace action renders all
catalog items. Deterministic results live under `outputs/`, so completed
outputs are recognized after restart.

The default image is 2400×1600 with the same left-side 10% average-PSD strip.
Rendering walks every STFT frame across the complete recording in
bounded-memory chunks. If there are more slow-time frames than output
columns, linear power is averaged only among frames assigned to the same
output column; no source interval is skipped.

Batch actions also run without starting the server:

```bash
sigmf-viewer batch --list
sigmf-viewer batch \
  --workspace sigmf-viewer \
  --item 'lte::public-lte.sigmf-collection' \
  --action render-high-resolution-waterfall
```

Use the identifiers printed by `batch --list`.

## Use the pipeline without the UI

The same format reader, exact window operation, analysis, and plotting
functions work in an ordinary script:

```python
from sigmf_viewer import (
    WaterfallSettings,
    analyze,
    open_recording,
    plot_waterfall,
    read_window,
)

recording = open_recording("capture.sigmf-meta")
window = read_window(recording, start_sample=0, sample_count=262_144)
products = analyze(
    window,
    WaterfallSettings(fft_size=256, overlap_percent=50, channel=0),
)
plot_waterfall(products).show()
```

The reusable reader also exposes the exact browser buffering policy:

```python
from sigmf_viewer import create_reader

reader = create_reader(
    {
        "data_root": "data",
        "default_window_seconds": 0.012,
        "minimum_window_seconds": 0.004,
        "window_step_seconds": 0.002,
        "overview_bins": 300,
    }
)
reference = reader.discover()[0]
source = reader.open(reference)
window = reader.read(source, start=0.0, stop=0.012)
```

For a collection, pass the member's manifest-relative metadata path through
the optional `member=` keyword.

## Native desktop application

```bash
python -m pip install -e ".[desktop]"
sigmf-viewer-desktop
sigmf-viewer-build
```

On macOS the build produces `dist/SigMF Viewer.app`; Windows and
Linux produce a platform-specific `dist/sigmf-viewer` executable.
When `data/` exists, the default build embeds it. Build a smaller application
that uses external recordings with:

```bash
sigmf-viewer-build --without-data
sigmf-viewer-desktop --data-root /path/to/sigmf
```

The frozen app writes batch PNGs to the operating system's application data
location, never into the bundle. Sigvue's fullscreen control toggles the
native window in desktop mode and browser fullscreen otherwise.

## Test and package

```bash
python -m pip install -e ".[test,release]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

The LTE recordings come from Daniel Estévez's
[public LTE directory](http://nas.destevez.net/~daniel/LTE/). The compact
training scenes come from the MIT-licensed
[Operation Coldferry](https://github.com/soniccidr/Operation-Coldferry)
repository. The logo recording comes from the official
[SigMF repository](https://github.com/sigmf/SigMF). The data remains outside
Git and retains its source metadata and licensing.
