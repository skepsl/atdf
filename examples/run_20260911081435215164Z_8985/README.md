# Recorded example: run 8985

![Six recorded localization measurements](progress.gif)

The GIF shows the six original measurement plots at **1 second per step, with the final step held for 3 seconds**. This is presentation timing, **not real-time simulation playback**. Images are rasterizations of the original saved PDFs; trajectories and particles were not recalculated.

[Final plot (PNG)](final.png) · [Final plot (PDF)](pdf/8985_006.pdf) · [Initial prior (PNG)](initial_prior.png) · [Run manifest](run_manifest.json)

The recorded setup used Sionna measurements, the neural ray predictor, 1,000 particles, an initial requested robot base pose of `(-3, 3, 90°)`, and source ground truth `(4, 3.5)` m. Particle-filter bounds were `x=[-4, 5]`, `y=[-1.5, 8]` m. Exact stored settings are in [metadata.json](snapshots/metadata.json).

The last saved step has a source position error of approximately **0.0693 m** and uncertainty of **0.9112 m**. The final frame is the last saved measurement; this alone does not establish algorithm convergence.

| Contents | Files |
| --- | --- |
| Original prior PDF | [8985_000.pdf](pdf/8985_000.pdf) |
| Original measurement PDFs | `pdf/8985_001.pdf` through `pdf/8985_006.pdf` |
| Original mask PDF | [8985_mask.pdf](pdf/8985_mask.pdf) |
| Measurement PNGs | `frames/step_000001.png` through `frames/step_000006.png` |
| Original numerical snapshots | `snapshots/step_000000.npz` through `snapshots/step_000006.npz` |
| Snapshot metadata | `snapshots/step_000000.json` through `snapshots/step_000006.json` |

NPZ files include saved particle positions, weights, component covariances, and source bounds. Measurement snapshots also contain robot and antenna pose histories, source estimate histories, measurements, and metrics. The initial prior snapshot has no measured pose history.

PDFs, NPZ files, and step JSON files are byte-for-byte copies. Only the checkpoint and plot-directory strings in the top-level run metadata were changed from local absolute paths to package-relative paths. [run_manifest.json](run_manifest.json) records these transformations and both original and packaged hashes. No numerical experiment data were changed.

To regenerate PNGs and the GIF, install `poppler-utils` and Pillow, then run from the repository root:

```bash
python3 examples/run_20260911081435215164Z_8985/render_media.py
```

The script uses the included PDFs and overwrites only this example's generated media and media manifest entries. GIF encoding uses a shared 256-color palette.
