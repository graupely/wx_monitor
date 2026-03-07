"""
wx_monitor.manifest
===================
Generates ``manifest.json`` alongside the viewer HTML so GitHub Pages
(and any other static host that doesn't serve directory listings) can
auto-discover output files.

Schema
------
::

    {
      "generated": "2026-02-28T19:00:00Z",   # UTC ISO-8601
      "output_dir": "output",                  # path relative to manifest
      "goes":   ["output/goes19_conus_visible_mrms.png"],
      "qpe":    ["output/mrms_qpe_vs_hrrr_01hr_...png", ...],
      "frames": ["output/frames/frame_f06_00.png", ...]
    }

All paths are relative to the directory that contains ``manifest.json``
(i.e. the project root), so the viewer HTML can use them directly as
``<img src>``.

Usage
-----
From Python::

    from wx_monitor.manifest import write_manifest
    write_manifest(output_dir="output", manifest_path="manifest.json")

From the CLI (called automatically after every product run)::

    python -m wx_monitor            # writes manifest.json at the end
    python -m wx_monitor --manifest # write manifest without running products
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


# ── filename patterns ─────────────────────────────────────────────────────────

_RE_GOES   = re.compile(r'^goes19[^/]*\.png$', re.IGNORECASE)
_RE_QPE    = re.compile(
    r'^(?:mrms_qpe_vs_hrrr|stageiv_vs_hrrr_precip)_\d{2}hr[^/]*\.png$',
    re.IGNORECASE,
)
_RE_FRAME  = re.compile(r'^frame_f\d+_\d+\.png$', re.IGNORECASE)
_RE_REFL_M10 = re.compile(r'^refl_m10[^/]*\.png$', re.IGNORECASE)


def _classify(name: str) -> str | None:
    """Return 'goes', 'qpe', 'frames', or 'refl_m10' for a known filename."""
    if _RE_GOES.match(name):
        return 'goes'
    if _RE_QPE.match(name):
        return 'qpe'
    if _RE_FRAME.match(name):
        return 'frames'
    if _RE_REFL_M10.match(name):
        return 'refl_m10'
    return None


def _scan_output_dir(output_dir: str) -> dict[str, list[str]]:
    """
    Walk *output_dir* and return a dict with keys 'goes', 'qpe', 'frames'.
    Every value is a sorted list of paths **relative to the project root**
    (i.e. relative to the directory that will contain manifest.json).
    """
    root = Path(output_dir)
    buckets: dict[str, list[str]] = {'goes': [], 'qpe': [], 'frames': [], 'refl_m10': []}

    # Root-level PNGs (goes, qpe)
    for p in sorted(root.glob('*.png')):
        cat = _classify(p.name)
        if cat:
            buckets[cat].append(p.as_posix())   # e.g. "output/goes19_…png"

    # frames/ subdirectory
    frames_dir = root / 'frames'
    if frames_dir.is_dir():
        for p in sorted(frames_dir.glob('*.png')):
            cat = _classify(p.name)
            if cat == 'frames':
                buckets['frames'].append(p.as_posix())   # "output/frames/frame_f…png"

    return buckets


def write_manifest(
    output_dir: str = 'output',
    manifest_path: str = 'manifest.json',
) -> Path:
    """
    Scan *output_dir* and write *manifest_path*.

    Parameters
    ----------
    output_dir:
        Directory that contains the PNG outputs.  Relative to CWD.
    manifest_path:
        Where to write the JSON file.  Relative to CWD.
        Should sit next to ``wx_monitor_viewer.html``.

    Returns
    -------
    Path
        Absolute path of the written manifest file.
    """
    buckets = _scan_output_dir(output_dir)
    total   = sum(len(v) for v in buckets.values())

    manifest = {
        'generated':  datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'output_dir': str(output_dir),
        'goes':       buckets['goes'],
        'qpe':        buckets['qpe'],
        'frames':     buckets['frames'],
        'refl_m10':   buckets['refl_m10'],
    }

    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    print(
        f"  manifest → {out.resolve()}\n"
        f"             goes:{len(buckets['goes'])}  "
        f"qpe:{len(buckets['qpe'])}  "
        f"frames:{len(buckets['frames'])}  "
        f"refl_m10:{len(buckets['refl_m10'])}  "
        f"(total:{total})"
    )
    return out.resolve()
