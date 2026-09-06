"""Boot a built static site in the real Pyodide runtime and open every item.

The published page runs Python in the browser, so curl checks cannot see
startup failures (missing bundled packages, stale local state, or dataset
discovery crashes). This harness runs the exact worker startup sequence from
``python-worker.js`` under Node.js plus the same Pyodide build the site
serves, then exercises the catalog and item routes the UI calls on load.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Mirrors the startup block in sigvue's static_assets/python-worker.js.
_HARNESS = r"""
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const pyodideDir = process.env.PYODIDE_DIR;
const { loadPyodide } = await import(pathToFileURL(`${pyodideDir}/pyodide.mjs`).href);

const site = process.env.SITE_DIR;
// Convert any Python or JS failure into one readable line; Node would
// otherwise dump the whole minified Pyodide source around the throw site.
process.on('uncaughtException', error => fail(error));
process.on('unhandledRejection', error => fail(error));
function fail(error) {
  console.error(`BOOT CHECK ERROR: ${error?.message || error}`);
  process.exit(1);
}

const manifest = JSON.parse(await readFile(`${site}/manifest.json`, 'utf8'));
const python = await loadPyodide({ indexURL: `${pyodideDir}/` });
await python.loadPackage(manifest.packages);
const unpack = (name, directory) =>
  python.unpackArchive(new Uint8Array(readFileSync(`${site}/${name}`)), 'zip', { extractDir: directory });
unpack(manifest.pythonPackages, '/packages');
unpack(manifest.framework, '/packages');
python.FS.mkdirTree('/project');

// Returning visitors reload onto browser-local state from earlier builds; the
// startup code must refresh stale files instead of crashing.
if (process.env.STALE_ARCHIVE) {
  python.globals.set('_stale_archive', new Uint8Array(readFileSync(process.env.STALE_ARCHIVE)));
  python.runPython(`
import io, zipfile
from pathlib import Path
with zipfile.ZipFile(io.BytesIO(_stale_archive.to_py())) as archive:
    archive.extractall("/project")
Path("/project/.sigvue-build").write_text("stale-build")
`);
}

python.globals.set('_project_archive', new Uint8Array(readFileSync(`${site}/${manifest.project}`)));
python.globals.set('_build_id', manifest.buildId);
python.globals.set('_config_path', `/project/${manifest.config}`);
await python.runPythonAsync(`
import io, os, sys, zipfile
from pathlib import Path
sys.path.insert(0, "/packages")
sys.path.insert(0, "/project")
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.chdir("/project")
_version_path = Path("/project/.sigvue-build")
_refresh_code = not _version_path.exists() or _version_path.read_text() != _build_id
with zipfile.ZipFile(io.BytesIO(_project_archive.to_py())) as archive:
    for entry in archive.infolist():
        target = Path("/project", entry.filename)
        if not target.resolve().is_relative_to(Path("/project").resolve()):
            raise ValueError("Unsafe project archive path")
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not target.exists() or (_refresh_code and target.suffix == ".py"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
_version_path.write_text(_build_id)
from sigvue.web.browser import BrowserRuntime
_runtime = BrowserRuntime(_config_path)
for workspace in _runtime.app.registry.list():
    if getattr(workspace, "batch", None) is not None:
        workspace_id = workspace.metadata.identifier
        _runtime.app._batch_capability(workspace, workspace_id)
        for item in workspace.discover_items():
            _runtime.app._batch_capability(workspace, workspace_id, item.identifier)
`);
console.log('startup ok');
const runtime = python.globals.get('_runtime');

async function request(method, path, body = '') {
  const proxy = runtime.request(method, path, body);
  let result;
  try { result = proxy.toJs({ dict_converter: Object.fromEntries }); }
  finally { proxy.destroy(); }
  await python.runPythonAsync('await _runtime.drain()');
  return result;
}

function expectOk(method, path) {
  return request(method, path).then(result => {
    console.log(method, path.slice(0, 100), '->', result.status);
    if (result.status >= 400) {
      const detail = Buffer.from(result.body, 'base64').toString('utf8');
      throw new Error(`${method} ${path} returned ${result.status}: ${detail.slice(0, 500)}`);
    }
    return result;
  });
}

await expectOk('GET', '/');
await expectOk('GET', '/health');
const listing = await expectOk('GET', '/workspaces');
const workspaces = JSON.parse(Buffer.from(listing.body, 'base64').toString('utf8'));
if (!workspaces.workspaces?.length) throw new Error('No workspaces registered');
for (const workspace of workspaces.workspaces) {
  const items = await expectOk('GET', `/workspaces/${workspace.id}/items`);
  const catalog = JSON.parse(Buffer.from(items.body, 'base64').toString('utf8'));
  if (!catalog.items?.length) throw new Error(`No items discovered for ${workspace.id}`);
  for (const item of catalog.items) {
    const id = encodeURIComponent(item.id);
    await expectOk('GET', `/workspaces/${workspace.id}/items/${id}?__include_static_views=true`);
  }
}
console.log('static site boot check ok');
"""


def _stale_archive(site: Path, workspace: Path) -> Path:
    """Repack the project archive without bundled Python code (pre-fix shape)."""
    manifest = json.loads((site / "manifest.json").read_text(encoding="utf-8"))
    stale = workspace / "stale-project.zip"
    with (
        zipfile.ZipFile(site / manifest["project"]) as source,
        zipfile.ZipFile(stale, "w") as target,
    ):
        for name in source.namelist():
            if name.endswith(".py"):
                continue
            target.writestr(name, source.read(name))
    return stale


def check(site: Path, pyodide: Path) -> None:
    site = site.resolve()
    pyodide = pyodide.resolve()
    for name in ("manifest.json", "index.html"):
        if not (site / name).is_file():
            raise ValueError(f"{site} is not a built static site (missing {name})")
    if not (pyodide / "pyodide.mjs").is_file():
        raise ValueError(f"{pyodide} is not an extracted Pyodide distribution")
    if shutil.which("node") is None:
        raise ValueError("Node.js executable not found: node")

    with tempfile.TemporaryDirectory(prefix="sigmf-viewer-boot-") as directory:
        workspace = Path(directory)
        harness = workspace / "check.mjs"
        harness.write_text(_HARNESS, encoding="utf-8")
        for label, stale in (("clean", None), ("stale", _stale_archive(site, workspace))):
            print(f"Boot check ({label} browser state)…", flush=True)
            environment = {
                **os.environ,
                "SITE_DIR": str(site),
                "PYODIDE_DIR": str(pyodide),
            }
            if stale is not None:
                environment["STALE_ARCHIVE"] = str(stale)
            else:
                environment.pop("STALE_ARCHIVE", None)
            try:
                result = subprocess.run(
                    ["node", str(harness)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as error:
                sys.stderr.write(
                    f"{(error.stderr or '')[-4000:]}\n"
                    if isinstance(error.stderr, str)
                    else ""
                )
                raise SystemExit(f"Boot check timed out ({label} browser state)")
            for line in result.stdout.splitlines():
                if not line.startswith(("Loading ", "Loaded ")):
                    print(f"  {line}")
            if result.returncode:
                # Node echoes the minified Pyodide source line before the real
                # traceback; only short lines carry the actual failure.
                detail = "\n".join(
                    line
                    for line in result.stderr.splitlines()
                    if 0 < len(line) < 500 and not line.startswith(("at ", "file://"))
                )
                sys.stderr.write(f"{detail[-4000:]}\n")
                raise SystemExit(f"Boot check failed ({label} browser state)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path, help="Built static site directory")
    parser.add_argument("--pyodide", required=True, type=Path, help="Extracted Pyodide distribution")
    arguments = parser.parse_args()
    check(arguments.site, arguments.pyodide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
