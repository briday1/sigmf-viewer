/* All Python, sample decoding, and numerical work stays off the UI thread. */
importScripts("./vendor/pyodide.js");
let runtime;
const ready = (async () => {
  const begin = performance.now();
  runtime = await loadPyodide({indexURL: new URL("./vendor/", self.location).href});
  await runtime.loadPackage("numpy");
  const response = await fetch("./sigmf_viewer.zip");
  if (!response.ok) throw new Error("Cannot load the analysis package");
  runtime.unpackArchive(await response.arrayBuffer(), "zip");
  runtime.runPython(`
import json
from pathlib import Path
from sigmf_viewer.browser import open_browser_recording, browser_analyze, validate_metadata
Path("/recording").mkdir(exist_ok=True)
recording = None
`);
  self.postMessage({type: "ready", milliseconds: performance.now() - begin});
})();
let queue = Promise.resolve();
self.onmessage = ({data: job}) => {
  queue = queue.then(async () => {
    const begin = performance.now();
    try {
      await ready;
      let result;
      if (job.type === "open") {
        const {metadata, file} = job.payload;
        if (!(file instanceof Blob) || file.size < 1 || file.size > 32 * 1024 * 1024) {
          throw new Error("Sample payload must be between 1 byte and 32 MiB");
        }
        if (new TextEncoder().encode(JSON.stringify(metadata)).length > 1024 * 1024) {
          throw new Error("Metadata exceeds 1 MiB");
        }
        runtime.globals.set("metadata_json", JSON.stringify(metadata));
        runtime.globals.set("payload_bytes", file.size);
        runtime.runPython("validate_metadata(json.loads(metadata_json), payload_bytes)");
        // This isolated fixed path never follows user-supplied filesystem paths.
        const safeMetadata = structuredClone(metadata);
        safeMetadata.global["core:dataset"] = "samples.sigmf-data";
        for (const name of ["samples.sigmf-data", "samples.sigmf-meta"]) {
          try { runtime.FS.unlink(`/recording/${name}`); } catch { /* first open */ }
        }
        runtime.FS.writeFile("/recording/samples.sigmf-data", new Uint8Array(await file.arrayBuffer()), {canOwn: true});
        runtime.FS.writeFile("/recording/samples.sigmf-meta", JSON.stringify(safeMetadata));
        result = JSON.parse(runtime.runPython(`
recording = open_browser_recording("/recording/samples.sigmf-meta")
json.dumps({"sampleCount": recording.sample_count, "sampleRate": recording.sample_rate,
            "channelLabels": recording.channel_labels, "offset": recording.sample_offset,
            "centerFrequency": recording.center_frequency, "datatype": recording.datatype})
`));
      } else if (job.type === "analyze") {
        runtime.globals.set("request_json", JSON.stringify(job.payload));
        result = JSON.parse(runtime.runPython("json.dumps(browser_analyze(recording, json.loads(request_json)), allow_nan=False)"));
      } else {
        throw new Error("Unknown worker operation");
      }
      self.postMessage({...job, payload: undefined, result, milliseconds: performance.now() - begin});
    } catch (error) {
      self.postMessage({id: job.id, generation: job.generation, type: job.type, error: String(error)});
    }
  });
};
ready.catch(error => self.postMessage({type: "fatal", error: String(error)}));
