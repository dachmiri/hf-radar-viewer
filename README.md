# HF radar beamforming viewer (browser-only version)

A free, public version of the viewer. It is a plain static website: the
beamforming script runs inside the visitor's browser with Pyodide (Python
compiled to WebAssembly), so there is no server to pay for, and uploaded
`.SORT` files never leave the visitor's computer.

Files: `index.html` (the page), `worker.js` (loads Python, runs the script),
`beamforming_sort.py` (copy of ../beamforming_sort.py), `israel2/` (the
provisional Israel_2 antenna layout), `.nojekyll` (tells GitHub Pages to serve
files as they are).

The first visit downloads about 30 MB (Python, numpy, matplotlib) from the
jsDelivr CDN; browsers cache it after that. The Israel_2 file takes about 10 s.

## Host it for free on GitHub Pages

1. Create a new **public** repository on GitHub (for example `hf-radar-viewer`).
2. Put the contents of this folder at its root and push (or use
   "Add file > Upload files" on GitHub and drag them in).
3. In the repository: Settings > Pages > Source "Deploy from a branch",
   branch `main`, folder `/ (root)`, Save.
4. After a minute the site is at `https://<user>.github.io/<repository>/`.

## Try it locally

    python -m http.server 8000      # in this folder, then open http://localhost:8000

(Opening index.html by double-click does not work: browsers block the worker
from file:// pages.)

When `../beamforming_sort.py` changes, copy it here again.
