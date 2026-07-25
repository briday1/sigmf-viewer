# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the native SigMF Waterfall Viewer."""

import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import copy_metadata


package_spec = importlib.util.find_spec("sigmf_waterfall_viewer")
if package_spec is None or not package_spec.submodule_search_locations:
    raise RuntimeError(
        "sigmf-waterfall-viewer must be installed before building"
    )
package_root = Path(
    next(iter(package_spec.submodule_search_locations))
).resolve()
source_root = package_root.parent
checkout_root = source_root.parent

datas = []
binaries = []
hiddenimports = ["sigmf_waterfall_viewer.workspace"]

for distribution in (
    "sigmf-waterfall-viewer",
    "sigvue",
    "plotly",
    "matplotlib",
    "numpy",
    "certifi",
    "pywebview",
):
    try:
        datas += copy_metadata(distribution, recursive=True)
    except Exception:
        pass

bundle_mode = os.environ.get(
    "SIGMF_WATERFALL_VIEWER_BUNDLE_DATA",
    "auto",
)
configured_data = os.environ.get(
    "SIGMF_WATERFALL_VIEWER_DATA_ROOT"
)
checkout_data = checkout_root / "data"
working_data = Path.cwd() / "data"
data_root = (
    Path(configured_data).expanduser().resolve()
    if configured_data
    else working_data if working_data.is_dir() else checkout_data
)
include_data = bundle_mode == "1" or (
    bundle_mode == "auto" and data_root.is_dir()
)
if include_data:
    if not data_root.is_dir():
        raise RuntimeError(
            f"SigMF data directory does not exist: {data_root}"
        )
    datas.append((str(data_root), "data"))

a = Analysis(
    [str(package_root / "desktop.py")],
    pathex=[str(source_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="sigmf-waterfall-viewer",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    collected = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="sigmf-waterfall-viewer",
    )
    app = BUNDLE(
        collected,
        name="SigMF Waterfall Viewer.app",
        version=version("sigmf-waterfall-viewer"),
        bundle_identifier="com.sigvue.sigmf-waterfall-viewer",
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="sigmf-waterfall-viewer",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
