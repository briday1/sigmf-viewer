"""Download the public example recordings used to exercise this repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sigvue.helpers import RemoteFile, download_file

USER_AGENT = "SigMF-Waterfall-Viewer-Examples/0.1"
LTE_BASE_URL = "http://nas.destevez.net/~daniel/LTE"
COLDFERRY_BASE_URL = (
    "https://raw.githubusercontent.com/soniccidr/Operation-Coldferry/main/captures"
)
SIGMF_BASE_URL = "https://raw.githubusercontent.com/sigmf/SigMF/main/logo"


def _remote(
    base_url: str,
    filename: str,
    size: int,
    checksum: str,
) -> RemoteFile:
    return RemoteFile(
        f"{base_url}/{filename}",
        filename,
        size=size,
        checksum=f"sha256:{checksum}",
    )


LTE_MANIFEST = {
    "downlink": (
        _remote(
            LTE_BASE_URL,
            "LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta",
            1_022,
            "2f591862e15ef67f4f7aceda3457320db839e4de424d793edf3d30a971479b45",
        ),
        _remote(
            LTE_BASE_URL,
            "LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-data",
            122_880_000,
            "d2dfecfec0cdf346d2264ae61ddc80ceba373893b0e8a2d6ebafc87b7215f26c",
        ),
    ),
    "uplink": (
        _remote(
            LTE_BASE_URL,
            "LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-meta",
            799,
            "4593e878261b5f9040195f854d906b9197f1dc0ccf8a84b2d6634b871051eb91",
        ),
        _remote(
            LTE_BASE_URL,
            "LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-data",
            108_871_680,
            "85d3cf17552581eae161491e9a633cce056a9019495805c526f3975592d96e2a",
        ),
    ),
}

COLDFERRY_FILES = (
    _remote(
        COLDFERRY_BASE_URL,
        "cap01_cw_carrier.sigmf-meta",
        679,
        "265b7b22c8d01d84035d57238992e3476f09902b86a66b4693c75ac7108bff3c",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap01_cw_carrier.sigmf-data",
        819_200,
        "4c4481a0f4273aef19a1add0d70ffded2455f50b8df2b1e2d906807b907e69dd",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap02_ook_beacon.sigmf-meta",
        711,
        "3e23245e7eda2dd423b26ea5f4a599f490913b318c1394767b53b97c2f26b909",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap02_ook_beacon.sigmf-data",
        1_638_400,
        "84328ae1f9aed06624f7e01f110105be8a5d17409c0c0d335d9b378b7b4c1349",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap03_fsk_telemetry.sigmf-meta",
        696,
        "031d64a65592a1ffed021d948612ce9554253bef54b984173b859ffe26a4ca63",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap03_fsk_telemetry.sigmf-data",
        1_638_400,
        "f956cb0633a00daf0b36224164b71a06f93a4232ce1764e4a6d3bd975c688214",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap04_lfm_chirp.sigmf-meta",
        717,
        "048d43debbe2a58a16379e54f88f980630d0903cfd170e734e6cf3a0b00ed2c8",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap04_lfm_chirp.sigmf-data",
        1_638_400,
        "c1ff3828b2936ef26edcda37a04da6c960555537ed5befe7f4236fd92105caa7",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap05_fhss_hopper.sigmf-meta",
        747,
        "d1cfa0995ae9c3b8bcf82c4b4f56e21b7818168caf255fbfc9f22a8c416b0858",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap05_fhss_hopper.sigmf-data",
        2_457_600,
        "391cc76dc132e6a6393d870de307eecb7f8b413797e54655b972eb31fe63c509",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap06_contested_scene.sigmf-meta",
        1_484,
        "61bb1bcf7476036a9b186b16e353c2dab9456f1cc20e63e3d360ed0225bf8197",
    ),
    _remote(
        COLDFERRY_BASE_URL,
        "cap06_contested_scene.sigmf-data",
        3_276_800,
        "c3a058d6d629469abb064a2f24965b56ce49412b3e315114b31696ada29c75d4",
    ),
)

SIGMF_LOGO_FILES = (
    _remote(
        SIGMF_BASE_URL,
        "sigmf_logo.sigmf-meta",
        1_409,
        "e33dea79466bb69386ecc121c9a755381c17f83516ab5003557b9a4eb5ac64e8",
    ),
    _remote(
        SIGMF_BASE_URL,
        "sigmf_logo.sigmf-data",
        1_152_000,
        "50eab162b487d0214c1e57de042606051e836b114592ec5f141a949caf106c22",
    ),
)

DATASETS = ("lte", "coldferry", "sigmf-logo")


def _progress(filename: str):
    def report(received: int, total: int | None) -> None:
        status = (
            f"{received / total:6.1%}" if total else f"{received / 1_000_000:.1f} MB"
        )
        print(f"\r{filename}: {status}", end="", flush=True)

    return report


def _download(
    files: tuple[RemoteFile, ...],
    destination: Path,
) -> tuple[Path, ...]:
    downloaded = []
    for remote in files:
        print(f"Preparing {remote.filename}")
        path = download_file(
            remote,
            destination,
            user_agent=USER_AGENT,
            progress=_progress(remote.filename),
            preserve_existing=remote.filename.endswith(".sigmf-meta"),
        )
        print(f"\rReady {path}")
        downloaded.append(path)
    return tuple(downloaded)


def _write_lte_collection(data_root: Path) -> Path:
    collection = data_root / "lte" / "public-lte.sigmf-collection"
    collection.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": {
            "core:version": "1.2.6",
            "core:description": (
                "Public LTE downlink and uplink recordings by Daniel Estévez"
            ),
            "core:streams": [
                {
                    "name": (
                        "downlink/LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta"
                    ),
                },
                {
                    "name": (
                        "uplink/LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-meta"
                    ),
                },
            ],
        },
    }
    collection.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Ready {collection}")
    return collection


def download_datasets(
    data_root: Path,
    *,
    datasets: tuple[str, ...] = DATASETS,
) -> tuple[Path, ...]:
    """Download selected example groups with size and SHA-256 validation."""
    unknown = set(datasets).difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    downloaded: list[Path] = []
    for dataset in datasets:
        if dataset == "lte":
            for case, files in LTE_MANIFEST.items():
                downloaded.extend(_download(files, data_root / "lte" / case))
            downloaded.append(_write_lte_collection(data_root))
        elif dataset == "coldferry":
            downloaded.extend(_download(COLDFERRY_FILES, data_root / "coldferry"))
        else:
            downloaded.extend(_download(SIGMF_LOGO_FILES, data_root / "sigmf-logo"))
    return tuple(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Dataset groups to retrieve (default: all)",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    download_datasets(output, datasets=tuple(args.datasets))
    print(f"All requested SigMF example data is available under {output}")


if __name__ == "__main__":
    main()
