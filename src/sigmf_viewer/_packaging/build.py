"""Build the SigMF Viewer desktop artifact with PyInstaller."""

from __future__ import annotations

import argparse
import os
from importlib.resources import as_file, files
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Build the native SigMF Viewer desktop application"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Bundle this data directory into the application",
    )
    parser.add_argument(
        "--without-data",
        action="store_true",
        help="Build a smaller application without embedded recordings",
    )
    args, pyinstaller_args = parser.parse_known_args()
    if args.data_root is not None and args.without_data:
        parser.error("--data-root and --without-data cannot be combined")

    try:
        from PyInstaller.__main__ import run
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            'Install desktop support first: pip install "sigmf-viewer[desktop]"'
        ) from exc

    if args.data_root is not None:
        root = args.data_root.expanduser().resolve()
        if not root.is_dir():
            parser.error(f"data directory does not exist: {root}")
        os.environ["SIGMF_VIEWER_BUNDLE_DATA"] = "1"
        os.environ["SIGMF_VIEWER_DATA_ROOT"] = str(root)
    elif args.without_data:
        os.environ["SIGMF_VIEWER_BUNDLE_DATA"] = "0"

    resource = files("sigmf_viewer._packaging").joinpath("sigmf_viewer.spec")
    with as_file(resource) as spec_path:
        run(
            [
                "--clean",
                "--noconfirm",
                *pyinstaller_args,
                str(spec_path),
            ]
        )


if __name__ == "__main__":
    main()
