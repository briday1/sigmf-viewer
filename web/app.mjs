import {LatestJob} from "./scheduler.mjs";

const $ = id => document.getElementById(id);
const MAX_FILE = 32 * 1024 * 1024;
const MAX_META = 1024 * 1024;
let info = null, metadata = null, recordingName = "", result = null;
let frequency = null, localMembers = [], fetchController = null, rendering = false;
let runtimeMilliseconds = null, requestBegin = 0, examples = [];
const exampleDrafts = new Map();
let draftTarget = null;
const worker = new Worker(new URL("./worker.js", import.meta.url));
const jobs = new LatestJob(job => worker.postMessage(job), receive);

function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
}
function fail(error) { status(String(error.message || error), true); }
function disable(disabled) {
  $("analysis-controls").disabled = disabled;
  $("annotation-controls").disabled = disabled;
  $("export-spectrum").disabled = disabled || !result;
  $("export-waterfall").disabled = disabled || !result;
}
function beginOpen(name) {
  const generation = jobs.invalidate();
  fetchController?.abort();
  fetchController = new AbortController();
  info = null; metadata = null; result = null; frequency = null;
  recordingName = name;
  draftTarget = null;
  disable(true);
  $("recording-info").textContent = "Opening…";
  $("annotations").replaceChildren();
  $("annotation-summary").textContent = "No recording open.";
  $("metrics").textContent = "Waiting for analysis.";
  for (const id of ["spectrum", "waterfall"]) {
    if ($(id).data) Plotly.purge(id);
    $(id).replaceChildren();
  }
  status(`Opening ${name}…`);
  return generation;
}
async function checkedFetch(path, size, checksum, signal) {
  const url = new URL(path, document.baseURI);
  if (url.origin !== location.origin || size > MAX_FILE) throw new Error("Invalid example manifest");
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
  const reader = response.body.getReader();
  const parts = [];
  let received = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    received += value.length;
    if (received > size) { await reader.cancel(); throw new Error("Example exceeds its declared size"); }
    parts.push(value);
  }
  if (received !== size) throw new Error("Example download is incomplete");
  const blob = new Blob(parts);
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  const actual = [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, "0")).join("");
  if (actual !== checksum) throw new Error("Example integrity check failed");
  return blob;
}
async function openExample(index) {
  if (index === "") return;
  const example = examples[Number(index)];
  const generation = beginOpen(example.name);
  try {
    const signal = fetchController.signal;
    const meta = await checkedFetch(example.metadata, example.metadataBytes, example.metadataSha256, signal);
    const parsed = JSON.parse(await meta.text());
    const file = await checkedFetch(example.data, example.dataBytes, example.dataSha256, signal);
    if (generation !== jobs.generation) return;
    metadata = structuredClone(exampleDrafts.get(example.name) || parsed);
    draftTarget = {example: example.name};
    jobs.submit("open", {metadata, file});
  } catch (error) {
    if (generation === jobs.generation) fail(error);
  }
}
function safeName(name) {
  if (!name || /[/\\\0]/.test(name) || name === "." || name === "..") {
    throw new Error("Only flat local collections and plain dataset filenames are supported");
  }
  return name;
}
async function readMetadata(file) {
  if (file.size > MAX_META) throw new Error(`${file.name}: metadata exceeds 1 MiB`);
  return JSON.parse(await file.text());
}
async function selectFiles(files) {
  const generation = beginOpen("local files");
  $("examples").value = "";
  $("members").disabled = true;
  localMembers = [];
  try {
    const byName = new Map();
    for (const file of files) {
      if (byName.has(file.name)) throw new Error("Duplicate filenames are ambiguous");
      byName.set(file.name, file);
    }
    if (files.length > 256) throw new Error("Select at most 256 files at once");
    if (files.filter(f => !f.name.endsWith(".sigmf-data")).reduce((sum, file) => sum + file.size, 0) > 4 * MAX_META) {
      throw new Error("Combined local metadata and collections exceed 4 MiB");
    }
    const collections = files.filter(f => f.name.endsWith(".sigmf-collection"));
    if (collections.length > 1) throw new Error("Select at most one collection");
    let names = files.filter(f => f.name.endsWith(".sigmf-meta")).map(f => f.name);
    if (collections.length) {
      const collection = await readMetadata(collections[0]);
      const streams = collection.collection?.["core:streams"];
      if (!Array.isArray(streams) || !streams.length) throw new Error("Collection contains no core:streams");
      names = streams.map(stream => {
        let name = safeName(stream.name);
        if (name.endsWith(".sigmf-data")) name = name.replace(/\.sigmf-data$/, ".sigmf-meta");
        if (!name.endsWith(".sigmf-meta")) name += ".sigmf-meta";
        return name;
      });
      if (names.length > 128 || new Set(names).size !== names.length) throw new Error("Collection is too large or repeats members");
    }
    const members = [];
    for (const name of names) {
      const file = byName.get(name);
      if (!file) throw new Error(`Select the missing metadata file: ${name}`);
      const meta = await readMetadata(file);
      const dataName = safeName(meta.global?.["core:dataset"] || name.replace(/\.sigmf-meta$/, ".sigmf-data"));
      const data = byName.get(dataName);
      if (!data) throw new Error(`Select the missing sample file: ${dataName}`);
      if (!data.size || data.size > MAX_FILE) throw new Error(`${dataName}: sample file must be 1 byte–32 MiB`);
      members.push({name, metadata: meta, file: data});
    }
    if (!members.length) throw new Error("Select at least one .sigmf-meta and its .sigmf-data file");
    if (generation !== jobs.generation) return;
    localMembers = members;
    $("members").replaceChildren(...members.map((member, i) => new Option(member.name, String(i))));
    $("members").disabled = false;
    openLocal(0);
  } catch (error) { if (generation === jobs.generation) fail(error); }
}
function openLocal(index) {
  const member = localMembers[index];
  beginOpen(member.name);
  metadata = structuredClone(member.metadata);
  draftTarget = {member};
  jobs.submit("open", {metadata, file: member.file});
}
function normalizeWindow() {
  if (!info) return;
  const start = Math.max(0, Math.min(info.sampleCount - 1, Math.trunc(Number($("start").value) || 0)));
  const count = Math.max(1, Math.min(65536, info.sampleCount - start, Math.trunc(Number($("count").value) || 1)));
  $("start").value = start;
  $("count").value = count;
  $("scrub").value = start;
  $("annotation-start").value = start + info.offset;
  $("annotation-count").value = count;
}
function analyze() {
  if (!info) return;
  normalizeWindow();
  requestBegin = performance.now();
  $("export-spectrum").disabled = true;
  $("export-waterfall").disabled = true;
  jobs.submit("analyze", {
    start: Number($("start").value), count: Number($("count").value),
    fft: Number($("fft").value), overlap: Number($("overlap").value),
    channel: Number($("channel").value), frequency,
  });
  status(`Analyzing ${recordingName}… (latest scrub wins)`);
}
function receive(message) {
  if (message.error) { fail(message.error); return; }
  if (message.type === "open") {
    info = message.result;
    $("channel").replaceChildren(...info.channelLabels.map((label, i) => new Option(label, String(i))));
    $("start").max = info.sampleCount - 1;
    $("scrub").max = info.sampleCount - 1;
    $("start").value = 0;
    $("count").value = Math.min(32768, info.sampleCount);
    $("recording-info").textContent = `${recordingName} · ${info.datatype} · ${info.sampleCount.toLocaleString()} samples · ${info.sampleRate.toLocaleString()} samples/s · ${(info.sampleCount / info.sampleRate).toFixed(4)} s`;
    disable(false);
    refreshAnnotations();
    analyze();
  } else if (message.type === "analyze") {
    result = message.result;
    result.metrics.workerMilliseconds = Math.round(message.milliseconds);
    result.metrics.requestMilliseconds = Math.round(performance.now() - requestBegin);
    result.metrics.runtimeStartupMilliseconds = Math.round(runtimeMilliseconds || 0);
    $("metrics").textContent = JSON.stringify(result.metrics, null, 2);
    const generation = jobs.generation;
    const text = `${recordingName} · ${result.count.toLocaleString()} samples · ${result.metrics.nativeCells.toLocaleString()} native cells · worker ${Math.round(message.milliseconds)} ms`;
    render().then(() => {
      if (generation === jobs.generation && message.id === jobs.latest) {
        disable(false);
        status(text);
      }
    }).catch(error => { if (generation === jobs.generation && message.id === jobs.latest) fail(error); });
  }
}
worker.onmessage = ({data}) => {
  if (data.type === "ready") {
    runtimeMilliseconds = data.milliseconds;
    if (!metadata && !jobs.active) status("Python ready. Choose an example or select a local SigMF pair.");
  } else if (data.type === "fatal") {
    disable(true); fail(data.error);
  } else jobs.complete(data);
};
worker.onerror = event => { disable(true); fail(`Worker failed: ${event.message}. Reload to retry.`); };

function visibleAnnotations() {
  if (!result) return [];
  return (metadata.annotations || []).filter(a => {
    const start = Number(a["core:sample_start"]) - info.offset;
    const count = Number(a["core:sample_count"]);
    return Number.isFinite(start) && count > 0 && start < result.start + result.count && start + count > result.start;
  }).slice(0, 200);
}
function shapes(waterfall) {
  return visibleAnnotations().map(a => {
    const start = Number(a["core:sample_start"]) - info.offset;
    const low = a["core:freq_lower_edge"], high = a["core:freq_upper_edge"];
    return {
      type: "rect", xref: low == null || high == null ? "paper" : "x",
      yref: waterfall ? "y" : "paper",
      x0: low == null || high == null ? 0 : Number(low) / 1e6,
      x1: low == null || high == null ? 1 : Number(high) / 1e6,
      y0: waterfall ? Math.max(result.start, start) / info.sampleRate * 1000 : 0,
      y1: waterfall ? Math.min(result.start + result.count, start + Number(a["core:sample_count"])) / info.sampleRate * 1000 : 1,
      line: {color: "#ffce73", width: 1}, fillcolor: "#ffce73", opacity: 0.18,
    };
  });
}
function layout(waterfall) {
  return {
    paper_bgcolor: "#151f30", plot_bgcolor: "#101a2b", font: {color: "#bac9dc"},
    margin: {l: 65, r: waterfall ? 85 : 30, t: 10, b: 55}, dragmode: "zoom",
    xaxis: {title: {text: "Frequency (MHz)"}, gridcolor: "#29384f",
      range: frequency || [result.xEdges[0], result.xEdges.at(-1)]},
    yaxis: {title: {text: waterfall ? "Local time (ms)" : "Power (dBFS)"}, gridcolor: "#29384f",
      ...(waterfall ? {range: [(result.start + result.count) / info.sampleRate * 1000, result.start / info.sampleRate * 1000]} : {})},
    shapes: shapes(waterfall), showlegend: false,
  };
}
async function render() {
  if (!result || !info) return;
  const floor = Number($("floor").value), ceiling = Number($("ceiling").value);
  if (!Number.isFinite(floor) || !Number.isFinite(ceiling) || floor >= ceiling) {
    throw new Error("Display floor must be below ceiling");
  }
  const generation = jobs.generation;
  rendering = true;
  const config = {responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d", "autoScale2d"]};
  try {
    await Promise.all([
      Plotly.react("spectrum", [{x: result.x, y: result.spectrum, type: "scatter", mode: "lines", line: {color: "#5be2d1", width: 1.5}, hovertemplate: "%{x:.6f} MHz<br>%{y:.2f} dBFS<extra></extra>"}], layout(false), config),
      Plotly.react("waterfall", [{x: result.xEdges, y: result.yEdges, z: result.z, type: "heatmap",
        colorscale: $("colormap").value, zmin: floor, zmax: ceiling, zsmooth: false,
        colorbar: {title: {text: "dBFS"}}, hovertemplate: "%{x:.6f} MHz<br>%{y:.4f} ms<br>%{z:.2f} dBFS<extra></extra>"}], layout(true), config),
    ]);
    if (generation !== jobs.generation) return;
    for (const id of ["spectrum", "waterfall"]) {
      $(id).removeAllListeners("plotly_relayout");
      $(id).on("plotly_relayout", event => zoom(event, id === "waterfall"));
      $(id).removeAllListeners("plotly_doubleclick");
      $(id).on("plotly_doubleclick", () => { reset(); return false; });
    }
  } finally { rendering = false; }
}
function zoom(event, waterfall) {
  if (rendering || !info) return;
  let changed = false;
  const range = axis => event[`${axis}.range`] || (
    event[`${axis}.range[0]`] !== undefined
      ? [event[`${axis}.range[0]`], event[`${axis}.range[1]`]] : null);
  const x = range("xaxis"), y = range("yaxis");
  if (x && x.every(Number.isFinite)) { frequency = x; changed = true; }
  if (event["xaxis.autorange"]) { frequency = null; changed = true; }
  if (waterfall && y && y.every(Number.isFinite)) {
    const start = Math.max(0, Math.floor(Math.min(...y) / 1000 * info.sampleRate));
    const stop = Math.min(info.sampleCount, Math.ceil(Math.max(...y) / 1000 * info.sampleRate));
    $("start").value = start;
    $("count").value = Math.max(1, stop - start);
    changed = true;
  }
  if (changed) analyze();
}
function reset() {
  if (!info) return;
  frequency = null;
  $("start").value = 0; $("count").value = Math.min(32768, info.sampleCount);
  analyze();
}
function refreshAnnotations() {
  const entries = metadata?.annotations || [];
  $("annotations").replaceChildren(...entries.slice(0, 200).map(a => {
    const row = document.createElement("tr");
    for (const value of [a["core:sample_start"], a["core:sample_count"],
      `${a["core:freq_lower_edge"] ?? "—"} – ${a["core:freq_upper_edge"] ?? "—"}`,
      a["core:label"] || a["core:description"] || a["core:comment"] || ""]) {
      const cell = document.createElement("td");
      cell.textContent = String(value ?? "—"); row.append(cell);
    }
    return row;
  }));
  $("annotation-summary").textContent = `${entries.length} annotations. Table shows first 200; plots show up to 200 intersecting this window. Export preserves all metadata fields.`;
}
function download(blob, name) {
  const url = URL.createObjectURL(blob), anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("annotation-form").addEventListener("submit", event => {
  event.preventDefault();
  if (!info) return;
  try {
    const start = Number($("annotation-start").value), count = Number($("annotation-count").value);
    const lowText = $("annotation-low").value, highText = $("annotation-high").value;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || count < 1 ||
        start < info.offset || start + count > info.offset + info.sampleCount) {
      throw new Error("Annotation samples must lie inside this recording");
    }
    const annotation = {"core:sample_start": start, "core:sample_count": count,
      "core:label": $("annotation-label").value.trim()};
    if (!annotation["core:label"]) throw new Error("Annotation label is required");
    if (lowText || highText) {
      const low = Number(lowText), high = Number(highText);
      if (!lowText || !highText || !Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
        throw new Error("Supply both frequency edges, with low < high");
      }
      annotation["core:freq_lower_edge"] = low; annotation["core:freq_upper_edge"] = high;
    }
    const updated = [...(metadata.annotations || []), annotation].sort((a, b) => a["core:sample_start"] - b["core:sample_start"]);
    const candidate = {...metadata, annotations: updated};
    if (new TextEncoder().encode(JSON.stringify(candidate)).length > MAX_META) throw new Error("Edited metadata exceeds 1 MiB");
    metadata = candidate;
    if (draftTarget?.member) draftTarget.member.metadata = metadata;
    if (draftTarget?.example) exampleDrafts.set(draftTarget.example, metadata);
    refreshAnnotations(); render().catch(fail);
    status("Annotation added in memory. Export metadata to save it.");
  } catch (error) { fail(error); }
});
$("export-metadata").onclick = () => {
  if (metadata) download(new Blob([JSON.stringify(metadata, null, 2) + "\n"], {type: "application/json"}),
    recordingName.replace(/\.sigmf-meta$/, "") + ".sigmf-meta");
};
for (const id of ["spectrum", "waterfall"]) $("export-" + id).onclick = () => {
  if (result) Plotly.downloadImage(id, {format: "png", filename: `${recordingName}-${id}-${result.start}`, width: 1400, height: id === "waterfall" ? 850 : 500}).catch(fail);
};
$("examples").onchange = event => openExample(event.target.value);
$("files").onchange = event => selectFiles([...event.target.files]);
$("members").onchange = event => { $("examples").value = ""; openLocal(Number(event.target.value)); };
for (const id of ["fft", "overlap", "channel", "start", "count"]) $(id).onchange = analyze;
$("scrub").oninput = event => { $("start").value = event.target.value; analyze(); };
for (const [id, sign] of [["previous", -1], ["next", 1]]) $(id).onclick = () => {
  $("start").value = Number($("start").value) + sign * Number($("count").value); analyze();
};
$("reset").onclick = reset;
for (const id of ["colormap", "floor", "ceiling"]) $(id).onchange = () => render().catch(fail);
fetch("./examples.json").then(response => {
  if (!response.ok) throw new Error("Cannot load example catalog");
  return response.json();
}).then(catalog => {
  examples = catalog;
  $("examples").append(...catalog.map((example, i) => new Option(example.name, String(i))));
}).catch(fail);
