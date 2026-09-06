"""Build a same-origin Pages artifact with integrity-checked runtime and examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_data import COLDFERRY_FILES, SIGMF_LOGO_FILES  # noqa: E402

PYODIDE_VERSION = "0.27.7"
CORE_MODULES = ("__init__.py", "models.py", "sigmf.py", "analysis.py", "browser.py")
RUNTIME_LICENSES = (
    (
        "pyodide.txt",
        "https://raw.githubusercontent.com/pyodide/pyodide/0.27.7/LICENSE",
        "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5",
    ),
    (
        "python.txt",
        "https://raw.githubusercontent.com/python/cpython/v3.12.7/LICENSE",
        "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf",
    ),
)


def fetch_verified(url, destination, checksum, size=None):
    """Use a verified build cache, bounding downloads before writing the artifact."""
    if destination.is_file():
        data = destination.read_bytes()
        if hashlib.sha256(data).hexdigest() == checksum and (
            size is None or len(data) == size
        ):
            return destination
    limit = size if size is not None else 32 * 1024 * 1024
    with urlopen(Request(url, headers={"User-Agent": "SigMF-Viewer-Pages/1"}), timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit or (size is not None and len(data) != size):
        raise ValueError(f"Size mismatch for {url}")
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"SHA-256 mismatch for {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def numpy_wheel(runtime_lock, cache):
    package = runtime_lock["packages"]["numpy"]
    filename = package["file_name"]
    destination = cache / filename
    try:
        return fetch_verified(
            f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/{filename}",
            destination, package["sha256"],
        )
    except URLError:
        # The official release is an alternative for networks without CDN access.
        # Stream only through the requested member; never extract archive paths.
        print("CDN unavailable; reading NumPy from the official Pyodide release")
        url = f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/pyodide-{PYODIDE_VERSION}.tar.bz2"
        with urlopen(url, timeout=120) as response:
            with tarfile.open(fileobj=response, mode="r|bz2") as archive:
                for member in archive:
                    if Path(member.name).name != filename or not member.isfile():
                        continue
                    if member.size > 32 * 1024 * 1024:
                        raise ValueError("Unexpected NumPy wheel size")
                    data = archive.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() != package["sha256"]:
                        raise ValueError("NumPy release wheel SHA-256 mismatch")
                    cache.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    return destination
        raise ValueError("NumPy wheel missing from the official release")


def build(output=ROOT / "site"):
    output = output.resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("Build output must be a child directory of the project")
    if output.exists() and not (output / ".sigmf-pages-build").is_file():
        raise ValueError("Refusing to replace an unmarked output directory")
    runtime = ROOT / "node_modules" / "pyodide"
    package = json.loads((runtime / "package.json").read_text())
    if package["version"] != PYODIDE_VERSION:
        raise ValueError("Run npm ci to restore the pinned Pyodide runtime")
    lock = json.loads((runtime / "pyodide-lock.json").read_text())
    cache = ROOT / ".pages-cache"
    numpy = lock["packages"]["numpy"]
    wheel = numpy_wheel(lock, cache)
    for filename, url, checksum in RUNTIME_LICENSES:
        fetch_verified(url, cache / filename, checksum)
    examples = []
    files = (*COLDFERRY_FILES, *SIGMF_LOGO_FILES)
    for remote in files:
        fetch_verified(
            remote.url, cache / remote.filename,
            remote.checksum.removeprefix("sha256:"), remote.size,
        )
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(ROOT / "web", output)
    (output / ".sigmf-pages-build").touch()
    (output / ".nojekyll").touch()
    vendor = output / "vendor"
    vendor.mkdir()
    for filename in (
        "pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
        "python_stdlib.zip", "pyodide-lock.json",
    ):
        shutil.copy2(runtime / filename, vendor / filename)
    shutil.copy2(wheel, vendor / wheel.name)
    shutil.copy2(
        ROOT / "node_modules/plotly.js-dist-min/plotly.min.js",
        vendor / "plotly.min.js",
    )
    licenses = output / "licenses"
    licenses.mkdir()
    for filename, _, _ in RUNTIME_LICENSES:
        shutil.copy2(cache / filename, licenses / filename)
    shutil.copy2(ROOT / "node_modules/plotly.js-dist-min/LICENSE", licenses / "plotly.txt")
    with ZipFile(wheel) as numpy_archive:
        (licenses / "numpy.txt").write_bytes(
            numpy_archive.read(f"numpy-{numpy['version']}.dist-info/LICENSE.txt")
        )
    with ZipFile(output / "sigmf_viewer.zip", "w", ZIP_DEFLATED) as archive:
        for filename in CORE_MODULES:
            info = ZipInfo(f"sigmf_viewer/{filename}", date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, (ROOT / "src/sigmf_viewer" / filename).read_bytes())
    data_directory = output / "examples"
    data_directory.mkdir()
    for remote in files:
        shutil.copy2(cache / remote.filename, data_directory / remote.filename)
        if remote.filename.endswith(".sigmf-meta"):
            data = next(r for r in files if r.filename == remote.filename.replace(".sigmf-meta", ".sigmf-data"))
            examples.append({
                "name": remote.filename.removesuffix(".sigmf-meta"),
                "metadata": f"examples/{remote.filename}",
                "data": f"examples/{data.filename}",
                "metadataBytes": remote.size, "dataBytes": data.size,
                "metadataSha256": remote.checksum.removeprefix("sha256:"),
                "dataSha256": data.checksum.removeprefix("sha256:"),
            })
    (output / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    assets = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    (output / "build-info.json").write_text(json.dumps({
        "pyodide": PYODIDE_VERSION, "numpy": numpy["version"],
        "plotly": "3.1.0", "sha256": assets,
    }, indent=2) + "\n")
    print(f"Built {output}: {len(examples)} examples, "
          f"{sum(p.stat().st_size for p in output.rglob('*') if p.is_file()):,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args().output)
