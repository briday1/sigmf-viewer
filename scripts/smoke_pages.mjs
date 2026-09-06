/** Real Chromium smoke test using its native DevTools protocol, no test packages. */
import {spawn, spawnSync} from "node:child_process";
import {mkdir, readFile, rm, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cache = path.join(root, ".pages-cache", "browser-smoke");
const base = "http://127.0.0.1:8765/sigmf-viewer/";
const nativeArgument = process.argv.indexOf("--native-python");
const nativePython = nativeArgument >= 0 ? process.argv[nativeArgument + 1] : null;
if (nativeArgument >= 0 && !nativePython) throw new Error("--native-python requires a Python executable with the native project installed");
await mkdir(cache, {recursive: true});
const server = spawn(process.env.PYTHON || "python", [path.join(root, "scripts/serve_pages.py"), "--port", "8765"], {stdio: "ignore"});
let browser, socket;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, description, timeout = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await predicate()) return;
    await delay(100);
  }
  throw new Error(`Timed out: ${description}`);
}
try {
  await until(async () => { try { return (await fetch(base)).ok; } catch { return false; } }, "preview server", 10000);
  browser = spawn(process.env.CHROME || "google-chrome", [
    "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--no-first-run",
    "--no-default-browser-check", "--disable-background-networking",
    "--remote-debugging-port=0", `--user-data-dir=${path.join(cache, "profile")}`,
    "about:blank",
  ], {env: {...process.env, TMPDIR: path.dirname(cache)}, stdio: ["ignore", "ignore", "pipe"]});
  let stderr = "", endpoint;
  browser.stderr.on("data", chunk => {
    stderr += chunk;
    endpoint = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/)?.[1];
  });
  await until(() => {
    if (browser.exitCode !== null || browser.signalCode !== null) throw new Error(`Chromium exited: ${stderr}`);
    return endpoint;
  }, "Chromium DevTools", 20000);
  const debugOrigin = endpoint.replace(/^ws:/, "http:").split("/devtools")[0];
  const target = await (await fetch(`${debugOrigin}/json/new?about:blank`, {method: "PUT"})).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let serial = 0;
  const pending = new Map(), requests = [], violations = [], exceptions = [], downloads = [];
  const origin = new URL(base).origin;
  function cdp(method, params = {}, sessionId) {
    const id = ++serial;
    return new Promise((resolve, reject) => {
      pending.set(id, {resolve, reject});
      socket.send(JSON.stringify({id, method, params, sessionId}));
    });
  }
  socket.onmessage = async ({data}) => {
    const message = JSON.parse(data);
    if (message.id) {
      const task = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) task?.reject(new Error(JSON.stringify(message.error)));
      else task?.resolve(message.result);
    } else if (message.method === "Fetch.requestPaused") {
      const {requestId, request} = message.params;
      const allowed = new URL(request.url).origin === origin || /^(data|blob):/.test(request.url);
      if (!allowed) violations.push(request.url);
      try {
        await cdp(allowed ? "Fetch.continueRequest" : "Fetch.failRequest",
          allowed ? {requestId} : {requestId, errorReason: "BlockedByClient"}, message.sessionId);
      } catch (error) {
        // Rapid file changes deliberately abort in-flight fetches.
        if (!error.message.includes("Invalid InterceptionId")) exceptions.push(error.message);
      }
    } else if (message.method === "Target.attachedToTarget") {
      const {sessionId} = message.params;
      try {
        await cdp("Network.enable", {}, sessionId);
        await cdp("Runtime.enable", {}, sessionId);
      } catch (error) { exceptions.push(error.message); }
      finally { await cdp("Runtime.runIfWaitingForDebugger", {}, sessionId); }
    } else if (message.method === "Network.requestWillBeSent") {
      const url = message.params.request.url;
      requests.push(url);
      if (new URL(url).origin !== origin && !/^(data|blob):/.test(url)) violations.push(url);
    } else if (message.method === "Runtime.exceptionThrown") {
      exceptions.push(message.params.exceptionDetails);
    } else if (message.method === "Page.downloadWillBegin") {
      downloads.push(message.params.suggestedFilename);
    }
  };
  await cdp("Page.enable");
  await cdp("Runtime.enable");
  await cdp("Network.enable");
  await cdp("Fetch.enable", {patterns: [{urlPattern: "*"}]});
  await cdp("Target.setAutoAttach", {autoAttach: true, waitForDebuggerOnStart: true, flatten: true});
  await cdp("Page.setDownloadBehavior", {behavior: "allow", downloadPath: path.join(cache, "downloads")});
  async function evaluate(expression) {
    const response = await cdp("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true});
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  const status = () => evaluate("document.getElementById('status')?.textContent");
  async function analyzed(name) {
    await until(async () => {
      const text = await status();
      if (await evaluate("document.getElementById('status')?.classList.contains('error')")) throw new Error(text);
      return text?.includes(name) && text.includes("native cells") &&
        await evaluate("!!document.getElementById('waterfall').data");
    }, `analysis ${name}`);
  }
  await cdp("Page.navigate", {url: base});
  await until(async () => {
    const text = await status();
    if (await evaluate("document.getElementById('status')?.classList.contains('error')")) throw new Error(text);
    if (exceptions.length) throw new Error(JSON.stringify(exceptions));
    return text?.includes("Python ready");
  }, "Python runtime startup");
  assert.ok(requests.some(url => url.includes("numpy-2.0.2")), "worker runtime requests must be observed");
  assert.equal(requests.filter(url => url.endsWith(".sigmf-data")).length, 0, "startup must not fetch sample payloads");
  const catalog = await (await fetch(new URL("examples.json", base))).json();
  assert.equal(catalog.length, 7);
  const measurements = [];
  const parity = [];
  for (let i = 0; i < catalog.length; i++) {
    const began = performance.now();
    await evaluate(`document.getElementById('examples').value='${i}'; document.getElementById('examples').dispatchEvent(new Event('change'));`);
    await analyzed(catalog[i].name);
    const metrics = await evaluate("JSON.parse(document.getElementById('metrics').textContent)");
    measurements.push({name: catalog[i].name, openThroughPlotMilliseconds: Math.round(performance.now() - began), ...metrics});
    console.log(`Analyzed ${catalog[i].name}: ${metrics.workerMilliseconds} ms`);
    if (nativePython) {
      const reference = spawnSync(nativePython, ["-c", `
import json, sys
from sigmf_viewer.browser import browser_analyze, open_browser_recording
recording = open_browser_recording(sys.argv[1])
print(json.dumps(browser_analyze(recording, {
    "start": 0, "count": 32768, "fft": 256, "overlap": 50, "channel": 0,
})))
`, path.join(root, "site", catalog[i].metadata)], {encoding: "utf8", maxBuffer: 16 * 1024 * 1024});
      if (reference.status !== 0) throw new Error(reference.stderr || String(reference.error));
      const expected = JSON.parse(reference.stdout);
      const actual = await evaluate(`({
        x: document.getElementById('spectrum').data[0].x,
        spectrum: document.getElementById('spectrum').data[0].y,
        xEdges: document.getElementById('waterfall').data[0].x,
        yEdges: document.getElementById('waterfall').data[0].y,
        z: document.getElementById('waterfall').data[0].z,
      })`);
      let maximumAbsoluteError = 0, comparedValues = 0;
      function compare(a, b) {
        if (Array.isArray(b)) {
          assert.equal(a.length, b.length);
          for (let n = 0; n < b.length; n++) compare(a[n], b[n]);
        } else {
          const difference = Math.abs(a - b);
          assert.ok(Number.isFinite(a) && difference <= 1e-7 + 1e-8 * Math.abs(b),
            `${catalog[i].name}: browser ${a} != native ${b}`);
          maximumAbsoluteError = Math.max(maximumAbsoluteError, difference);
          comparedValues += 1;
        }
      }
      for (const key of Object.keys(actual)) compare(actual[key], expected[key]);
      parity.push({name: catalog[i].name, comparedValues, maximumAbsoluteError});
    }
  }
  assert.equal(requests.filter(url => url.endsWith(".sigmf-data")).length, 7);
  assert.equal(await evaluate("document.getElementById('channel').options.length"), 2);
  const firstChannel = await evaluate("JSON.stringify(document.getElementById('spectrum').data[0].y)");
  await evaluate("document.getElementById('channel').value='1'; document.getElementById('channel').dispatchEvent(new Event('change'));");
  await analyzed(catalog[6].name);
  assert.notEqual(await evaluate("JSON.stringify(document.getElementById('spectrum').data[0].y)"), firstChannel);
  // Re-select a known recording and exercise numerical controls.
  await evaluate("document.getElementById('examples').value='3'; document.getElementById('examples').dispatchEvent(new Event('change'));");
  await analyzed(catalog[3].name);
  await evaluate(`
document.getElementById('fft').value='1024';
document.getElementById('overlap').value='75';
document.getElementById('count').value='16384';
document.getElementById('count').dispatchEvent(new Event('change'));
`);
  await analyzed(catalog[3].name);
  const beforeZoom = await evaluate("document.getElementById('waterfall').data[0].x");
  await evaluate(`Plotly.relayout('waterfall', {'xaxis.range': [${beforeZoom[200]}, ${beforeZoom[260]}], 'yaxis.range': [5, 15]}).then(() => null)`);
  await analyzed(catalog[3].name);
  const afterZoom = await evaluate("document.getElementById('waterfall').data[0].x");
  assert.ok(afterZoom.length < beforeZoom.length, "zoom must crop and rerasterize FFT bins");
  assert.notEqual(await evaluate("document.getElementById('count').value"), "16384");
  await evaluate("document.getElementById('colormap').value='Inferno'; document.getElementById('colormap').dispatchEvent(new Event('change'));");
  // Scrubs must converge to the latest request rather than replay every intermediate.
  await evaluate(`
for(let i=0;i<100;i++){
 document.getElementById('scrub').value=String(1000+i);
 document.getElementById('scrub').dispatchEvent(new Event('input'));
}
`);
  await analyzed(catalog[3].name);
  assert.equal(await evaluate("document.getElementById('start').value"), "1099");
  await evaluate("document.getElementById('annotation-label').value='Browser smoke annotation'; document.getElementById('annotation-form').requestSubmit();");
  assert.ok((await status()).includes("Annotation added"));
  assert.ok(await evaluate("document.getElementById('annotations').textContent.includes('Browser smoke annotation')"));
  await evaluate("document.getElementById('export-metadata').click(); document.getElementById('export-spectrum').click(); document.getElementById('export-waterfall').click();");
  await until(() => downloads.length >= 3, "metadata and PNG downloads", 30000);
  await until(async () => {
    try {
      const exported = JSON.parse(await readFile(path.join(cache, "downloads", downloads.find(n => n.endsWith(".sigmf-meta"))), "utf8"));
      assert.ok(exported.annotations.some(a => a["core:label"] === "Browser smoke annotation"));
      for (const name of downloads.filter(n => n.endsWith(".png"))) {
        const bytes = await readFile(path.join(cache, "downloads", name));
        assert.equal(bytes.subarray(1, 4).toString(), "PNG");
        assert.ok(bytes.length > 1000);
      }
      return true;
    } catch (error) {
      if (error.code === "ENOENT") return false;
      throw error;
    }
  }, "completed metadata and PNG files", 30000);
  // Opening several files rapidly must not display an older response.
  await evaluate(`
for (const index of [0,1,2,5]) {
 document.getElementById('examples').value=String(index);
 document.getElementById('examples').dispatchEvent(new Event('change'));
}
`);
  await analyzed(catalog[5].name);
  const collection = path.join(cache, "smoke.sigmf-collection");
  await writeFile(collection, JSON.stringify({collection: {"core:streams": [
    {name: catalog[0].name}, {name: catalog[1].name},
  ]}}));
  const document = await cdp("DOM.getDocument");
  const input = await cdp("DOM.querySelector", {nodeId: document.root.nodeId, selector: "#files"});
  await cdp("DOM.setFileInputFiles", {nodeId: input.nodeId, files: [
    collection, ...[catalog[0], catalog[1]].flatMap(e => [path.join(root, "site", e.metadata), path.join(root, "site", e.data)]),
  ]});
  await analyzed(catalog[0].name);
  assert.equal(await evaluate("document.getElementById('members').options.length"), 2);
  await evaluate("document.getElementById('members').value='1'; document.getElementById('members').dispatchEvent(new Event('change'));");
  await analyzed(catalog[1].name);
  assert.equal(violations.length, 0, `Cross-origin attempts: ${violations}`);
  assert.ok(requests.filter(url => /^https?:/.test(url)).every(url => new URL(url).pathname.startsWith("/sigmf-viewer/")),
    "all deployed requests must respect the project subpath");
  assert.equal(exceptions.length, 0, `Browser exceptions: ${JSON.stringify(exceptions)}`);
  console.log(JSON.stringify({passed: true, base, browser: await cdp("Browser.getVersion"),
    sameOriginOnly: true, requests: requests.length, downloads, measurements,
    nativeBrowserParity: parity}, null, 2));
} finally {
  socket?.close();
  if (browser && browser.exitCode === null && browser.signalCode === null) {
    const exited = new Promise(resolve => browser.once("exit", resolve));
    browser.kill();
    await exited;
  }
  server.kill();
  await rm(cache, {recursive: true, force: true, maxRetries: 10, retryDelay: 200});
}
