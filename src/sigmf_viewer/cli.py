"""Application-specific console wrapper around the Sigvue CLI."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

from sigvue.web.application import main as sigvue_main

from .runtime import runtime_profile


def _print_application_options() -> None:
    print(
        "SigMF Viewer defaults:\n"
        "  --data-root PATH    Recursively discover a SigMF directory\n"
        "  --recording PATH    Open one pair or .sigmf-collection\n"
        "  --output-root PATH  Durable PNG output directory\n"
        "\n"
        "All Sigvue server and batch options are also available.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-root", type=Path)
    source.add_argument("--recording", type=Path)
    parser.add_argument("--output-root", type=Path)
    options, remaining = parser.parse_known_args(sys.argv[1:])

    if "-h" in remaining or "--help" in remaining:
        _print_application_options()
    if options.config is not None and (
        options.data_root is not None
        or options.recording is not None
        or options.output_root is not None
    ):
        parser.error("--config cannot be combined with data or output overrides")
    if options.config is not None:
        profile_context = nullcontext(options.config)
    else:
        profile_context = runtime_profile(
            data_root=options.recording or options.data_root,
            output_root=options.output_root,
        )

    with profile_context as profile:
        original = sys.argv[1:]
        sys.argv[1:] = ["--config", str(profile), *remaining]
        try:
            sigvue_main()
        finally:
            sys.argv[1:] = original


if __name__ == "__main__":
    main()
