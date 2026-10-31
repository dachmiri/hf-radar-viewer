// Runs beamforming_sort.py inside the browser with Pyodide (Python compiled to
// WebAssembly). Lives in a Web Worker so the page stays responsive while the
// numbers are crunched. The .SORT file never leaves the visitor's computer.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
importScripts(PYODIDE_URL + "pyodide.js");

// Plain-language reasons for the script's false-alarm flags.
const HELPER = String.raw`
import csv, json, re, os, numpy as np

FLAG_TEXT = {
    "all-bearings": "about as strong in most directions at this distance, so not one object "
                    "(interference or the radar itself)",
    "mains-hum": "at a distance where electrical 50 Hz hum lands",
    "near-site": "within 5 km of the antennas (coast and harbour)",
}

def parse_log_header(log):
    out = []
    for line in log.splitlines():
        m = re.match(r"^([a-zA-Z][\w ~./()-]{2,30}?)\s+:\s+(.+)$", line)
        if m and not line.startswith(" "):
            out.append([m.group(1).strip(), m.group(2).strip()])
    return out

def read_objects(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            row = {k: (r[k] if k in ("kind", "flag") else float(r[k])) for k in r}
            row["why_flagged"] = "; ".join(FLAG_TEXT.get(f, f) for f in row["flag"].split("+") if f)
            rows.append(row)
    return rows

def radar_latlon(path):
    try:
        d = np.load(path)
        return [float(d["lat"]), float(d["lon"])]
    except Exception:
        return None

def summary(log):
    return json.dumps({"header": parse_log_header(log),
                       "objects": read_objects("run_objects.csv"),
                       "radar_latlon": radar_latlon("run.npz")})
`;

const RUNNER = String.raw`
import io, os, sys, contextlib, traceback, importlib
os.chdir("/work")
for f in os.listdir("."):
    if f.startswith("run"):
        os.remove(f)
import beamforming_sort as b
buf = io.StringIO()
ok = True
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        b.main(ARGV.to_py())
    except SystemExit as e:
        ok = e.code in (None, 0)
    except Exception:
        traceback.print_exc()
        ok = False
import matplotlib.pyplot as plt
plt.close("all")
[ok, buf.getvalue()]
`;

let pyodide = null, ready = null;

// Loaded once per tab; the page starts this as soon as it opens.
function init() {
  if (!ready) ready = load().catch(err => { ready = null; throw err; });
  return ready;
}

async function load() {
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  await py.loadPackage(["numpy", "matplotlib"]);
  pyodide = py;
  pyodide.FS.mkdirTree("/work");
  const src = await (await fetch("beamforming_sort.py")).text();
  pyodide.FS.writeFile("/work/beamforming_sort.py", src);
  pyodide.FS.writeFile("/work/viewer_helper.py", HELPER);
  await pyodide.runPythonAsync(`
import sys, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "/work")`);
}

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

self.onmessage = async (e) => {
  if (e.data.warmup) { init().catch(() => {}); return; }
  const { sortBytes, siteText, args } = e.data;
  try {
    if (!pyodide) post({ type: "status",
      text: "Loading Python into your browser (about 30 MB the first time, then cached)..." });
    await init();
    post({ type: "status", text: "Processing the file. This takes about half a minute..." });
    const FS = pyodide.FS;
    FS.writeFile("/work/input.SORT", new Uint8Array(sortBytes));
    FS.writeFile("/work/site.json", siteText);
    const argv = ["input.SORT", "site.json", "--out", "run", ...args];
    pyodide.globals.set("ARGV", argv);
    const t0 = performance.now();
    const res = await pyodide.runPythonAsync(RUNNER);
    const [ok, log] = res.toJs();
    res.destroy();
    const seconds = Math.round((performance.now() - t0) / 100) / 10;
    FS.unlink("/work/input.SORT");

    const files = {}, transfer = [];
    for (const f of ["run_objects.png", "run_max_doppler.png", "run.png", "run_cube.png",
                     "run_objects.csv", "run.npz"]) {
      try {
        const b = FS.readFile("/work/" + f);
        files[f] = b; transfer.push(b.buffer);
      } catch (_) { /* not produced */ }
    }
    let summary = { header: [], objects: [], radar_latlon: null };
    if (ok) {
      pyodide.globals.set("LOG", log);
      summary = JSON.parse(await pyodide.runPythonAsync(
        "import importlib, viewer_helper; importlib.reload(viewer_helper); viewer_helper.summary(LOG)"));
    }
    post({ type: "done", ok, log, seconds, files, ...summary }, transfer);
  } catch (err) {
    post({ type: "error", text: String(err && err.message || err) });
  }
};
