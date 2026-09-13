#!/usr/bin/env python3
"""Rasterize the included original PDFs and create a presentation-timed GIF.

Requirements: Poppler's pdftoppm executable and Pillow. This script reads the
original plot PDFs only; it neither runs localization nor changes saved data.
"""

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
RUN_ID = "8985"
WIDTH = 1000
DURATIONS_MS = [1000, 1000, 1000, 1000, 1000, 3000]


def file_record(path):
    return {"path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rasterize(pdf, output):
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(WIDTH),
                    "-scale-to-y", "-1", "-W", str(WIDTH),
                    str(pdf), str(output.with_suffix(""))],
                   check=True)


def main():
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required; install your system's poppler-utils package")
    frame_dir = ROOT / "frames"
    frame_dir.mkdir(exist_ok=True)
    prior = ROOT / "initial_prior.png"
    rasterize(ROOT / "pdf" / (RUN_ID + "_000.pdf"), prior)
    paths = []
    for step in range(1, 7):
        path = frame_dir / ("step_{:06d}.png".format(step))
        rasterize(ROOT / "pdf" / ("{}_{:03d}.pdf".format(RUN_ID, step)), path)
        paths.append(path)
    final = ROOT / "final.png"
    shutil.copyfile(paths[-1], final)

    frames = [Image.open(path).convert("RGB") for path in paths]
    if len({frame.size for frame in frames}) != 1:
        raise ValueError("Original measurement plots must have matching page dimensions")
    if any(frame.width != WIDTH for frame in frames):
        raise ValueError("Rasterized frames must have the requested pixel width")
    # A shared palette prevents colors from changing between GIF frames.
    samples = [frame.resize((250, round(frame.height * 250 / frame.width)))
               for frame in frames]
    sheet = Image.new("RGB", (250, sum(frame.height for frame in samples)), "white")
    offset = 0
    for sample in samples:
        sheet.paste(sample, (0, offset))
        offset += sample.height
    # Preserve frequent exact colors, including thin green ROI lines and the
    # small magenta source star that thumbnail-only quantization can discard.
    color_counts = Counter()
    for frame in frames:
        color_counts.update({color: count for count, color in
                             frame.getcolors(frame.width * frame.height)})
    exact_colors = [color for color, _ in color_counts.most_common(96)]
    adaptive = sheet.quantize(colors=160).getpalette()[:160 * 3]
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in exact_colors for channel in color] + adaptive)
    indexed = [frame.quantize(palette=palette, dither=Image.NONE) for frame in frames]
    gif = ROOT / "progress.gif"
    indexed[0].save(gif, save_all=True, append_images=indexed[1:], loop=0,
                    duration=DURATIONS_MS, disposal=2, optimize=False)
    with Image.open(gif) as check:
        assert check.n_frames == 6
        durations = []
        for step in range(check.n_frames):
            check.seek(step)
            durations.append(check.info["duration"])
        assert durations == DURATIONS_MS, durations

    manifest_path = ROOT / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["media"] = {
            "source": "Original included step PDFs; plots were not recalculated",
            "width_px": WIDTH,
            "measurement_frames": 6,
            "frame_durations_ms": DURATIONS_MS,
            "timing_note": "Presentation timing, not real-time simulation playback",
            "gif_color_conversion": "Shared 256-color palette required by GIF format",
            "artifacts": [file_record(path) for path in [prior, *paths, final, gif]],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered 6 measurement PNGs, initial_prior.png, final.png, and progress.gif")


if __name__ == "__main__":
    main()
