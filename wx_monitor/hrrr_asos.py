"""
wx_monitor.hrrr_asos
====================
HRRR precipitation-type forecast verified against ASOS METAR observations.

Renders one PNG frame per (forecast-hour, valid-time) pair and writes a
POD/FAR summary CSV.  Valid times span the 12 hours ending at
``config.VALID_TIME``.

Valid-time anchor
-----------------
``valid_times[-1]  ==  config.VALID_TIME``      ← most recent hour
``valid_times[0]   ==  VALID_TIME − 11 h``      ← oldest hour

For each forecast hour *fxx*, the HRRR run that *verifies* at a given
valid time was initialised at ``valid_time − fxx``.

Category conventions
--------------------
+---------------+----------------------------------+
| Canonical     | METAR / HRRR codes merged        |
+===============+==================================+
| Rain          | Rain + Drizzle                   |
| Snow          | Snow + Mixed                     |
| Freezing Rain | Freezing Rain / Freezing Drizzle |
| Ice Pellets   | PL / SHPL / TSPL                 |
| (Hail)        | excluded from verification       |
+---------------+----------------------------------+
"""

from __future__ import annotations

import io
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D
from scipy.spatial import KDTree

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _USE_CARTOPY = True
except ImportError:
    _USE_CARTOPY = False
    warnings.warn("Cartopy not found — falling back to plain lat/lon axes.")

from herbie import Herbie
import xarray as xr

from wx_monitor.config import (
    ASOS_WINDOW_MINUTES,
    CONUS_STATES,
    FRAMES_DIR,
    HRRR_FXX,
    IEM_URL,
    VALID_TIME,
)
from wx_monitor.utils import ensure_dir

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATS = ["Rain", "Snow", "Freezing Rain", "Ice Pellets"]

HRRR_CODE  = {"Rain": 1, "Snow": 2, "Freezing Rain": 3, "Ice Pellets": 4}
CODE_TO_CAT = {v: k for k, v in HRRR_CODE.items()}
CODE_TO_CAT[0] = "None"

_RAW_TOKENS: dict[str, list[str]] = {
    "Rain":          ["-RA","+RA","RA","SHRA","-SHRA","+SHRA","TSRA","-TSRA","+TSRA"],
    "Snow":          ["SN","-SN","+SN","SHSN","-SHSN","BLSN","DRSN","TSSN"],
    "Freezing Rain": ["FZRA","-FZRA","+FZRA","FZDZ","-FZDZ","+FZDZ"],
    "Ice Pellets":   ["PL","-PL","+PL","SHPL","TSPL"],
    "Mixed":         ["RASN","-RASN","+RASN","SNRA","-SNRA"],
    "Drizzle":       ["DZ","-DZ","+DZ"],
    "Hail":          ["GR","GS","-GR","-GS"],
}
TOKEN_MAP: dict[str, str] = {
    tok: cat for cat, toks in _RAW_TOKENS.items() for tok in toks
}
REMAP = {"Drizzle": "Rain", "Mixed": "Snow", "Hail": "None"}
_PRIORITY = ["Hail","Freezing Rain","Ice Pellets","Mixed","Snow","Rain","Drizzle"]


def _remap(cat: str) -> str:
    return REMAP.get(cat, cat)


def classify_asos(wxstr: str) -> str:
    """Raw METAR wx string → remapped canonical category."""
    if not wxstr:
        return "None"
    found = {TOKEN_MAP[t] for t in wxstr.upper().split() if t in TOKEN_MAP}
    if not found:
        return "None"
    for cat in _PRIORITY:
        if cat in found:
            return _remap(cat)
    return "None"


# ---------------------------------------------------------------------------
# Visual styling
# ---------------------------------------------------------------------------

STYLE: dict[str, dict] = {
    "Rain":          dict(color="#1f78b4", marker="o",  s=60,  zorder=8),
    "Snow":          dict(color="#a6cee3", marker="*",  s=110, zorder=8),
    "Freezing Rain": dict(color="#e31a1c", marker="D",  s=65,  zorder=9),
    "Ice Pellets":   dict(color="#ff7f00", marker="^",  s=65,  zorder=9),
}
_HRRR_RGBA = {
    0: (1, 1, 1, 0),
    1: "#1f78b4",
    2: "#a6cee3",
    3: "#e31a1c",
    4: "#ff7f00",
}
hrrr_cmap = mcolors.ListedColormap([_HRRR_RGBA[k] for k in range(5)])
hrrr_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)


# ---------------------------------------------------------------------------
# HRRR fetch helpers
# ---------------------------------------------------------------------------

_SEARCH = r":(CRAIN|CSNOW|CFRZR|CICEP):surface:"

_lats_h:  Optional[np.ndarray] = None
_lons_h:  Optional[np.ndarray] = None
_kdtree:  Optional[KDTree]     = None


def _get_var(ds, *names) -> Optional[np.ndarray]:
    lower = {k.lower(): k for k in ds.data_vars}
    for n in names:
        if n in ds:
            return ds[n].values.astype(float)
        if n.lower() in lower:
            return ds[lower[n.lower()]].values.astype(float)
    return None


def _build_ptype(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crain = _get_var(ds, "crain", "CRAIN")
    csnow = _get_var(ds, "csnow", "CSNOW")
    cfrzr = _get_var(ds, "cfrzr", "CFRZR")
    cicep = _get_var(ds, "cicep", "CICEP")
    for nm, arr in [("CRAIN", crain), ("CSNOW", csnow),
                    ("CFRZR", cfrzr), ("CICEP", cicep)]:
        if arr is None:
            raise RuntimeError(f"{nm} missing in dataset. "
                               f"Available: {list(ds.data_vars)}")
    pt              = np.zeros_like(crain, dtype=np.int8)
    pt[crain == 1]  = 1
    pt[csnow == 1]  = 2
    pt[cfrzr == 1]  = 3
    pt[cicep == 1]  = 4
    lats = ds["latitude"].values  if "latitude"  in ds else ds["lat"].values
    lons = ds["longitude"].values if "longitude" in ds else ds["lon"].values
    if lons.max() > 180:
        lons = lons - 360.0
    return pt, lats, lons


def _fetch_hrrr_frames(
    fxx: int,
    valid_times: list[datetime],
) -> list[Optional[np.ndarray]]:
    """Fetch N ptype grids for *fxx*; returns list (None on failure)."""
    global _lats_h, _lons_h, _kdtree

    run_times = [vt - timedelta(hours=fxx) for vt in valid_times]
    frames: list[Optional[np.ndarray]] = []

    print(f"\n── Fetching HRRR F{fxx:02d} ({len(valid_times)} runs) ─────────────────")
    for vt, rt in zip(valid_times, run_times):
        label = vt.strftime("%Y-%m-%d %H UTC")
        try:
            H  = Herbie(rt.strftime("%Y-%m-%d %H:%M"), model="hrrr",
                        product="sfc", fxx=fxx, verbose=False)
            ds = H.xarray(_SEARCH, remove_grib=True)
            if isinstance(ds, list):
                ds = xr.merge(ds)
            pt, lats, lons = _build_ptype(ds)
            frames.append(pt)
            if _lats_h is None:
                _lats_h, _lons_h = lats, lons
                flat_coords = np.column_stack([_lats_h.ravel(), _lons_h.ravel()])
                _kdtree = KDTree(flat_coords)
                print(f"  KDTree built from grid {_lats_h.shape}")
            print(f"  OK  {label}")
        except Exception as exc:
            print(f"  FAIL {label}: {exc}")
            frames.append(None)

    return frames


# ---------------------------------------------------------------------------
# ASOS fetch helpers
# ---------------------------------------------------------------------------

_ASOS_HEADERS = {"User-Agent": "wx-monitor/1.0 (educational use)"}


def _fetch_asos_network(
    network: str,
    bulk_start: datetime,
    bulk_end: datetime,
    retries: int = 3,
) -> pd.DataFrame:
    params = dict(
        data="all", latlon="yes", elev="no", format="comma",
        tz="UTC", missing="M", trace="T",
        year1=bulk_start.strftime("%Y"), month1=bulk_start.strftime("%m"),
        day1=bulk_start.strftime("%d"),  hour1=bulk_start.strftime("%H"),
        minute1=bulk_start.strftime("%M"),
        year2=bulk_end.strftime("%Y"),   month2=bulk_end.strftime("%m"),
        day2=bulk_end.strftime("%d"),    hour2=bulk_end.strftime("%H"),
        minute2=bulk_end.strftime("%M"),
        network=network,
    )
    for attempt in range(retries):
        try:
            r = requests.get(IEM_URL, params=params,
                             headers=_ASOS_HEADERS, timeout=90)
            r.raise_for_status()
            text = r.text.strip()
            if not text or text.startswith("ERROR") or len(text) < 10:
                return pd.DataFrame()
            lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
            return (
                pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)
                if lines else pd.DataFrame()
            )
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return pd.DataFrame()


def _fetch_all_asos(
    valid_times: list[datetime],
) -> list[pd.DataFrame]:
    """
    Bulk-fetch ASOS data for a window covering all valid times.

    Returns a list of per-hour DataFrames (one per entry in *valid_times*).
    """
    window = timedelta(minutes=ASOS_WINDOW_MINUTES)
    bulk_start = valid_times[0]  - window
    bulk_end   = valid_times[-1] + window

    print(f"\n── Fetching ASOS "
          f"({bulk_start.strftime('%Y-%m-%d %H:%M')} – "
          f"{bulk_end.strftime('%Y-%m-%d %H:%M UTC')}) ─────")

    raw_frames: list[pd.DataFrame] = []
    for i, state in enumerate(CONUS_STATES):
        df_net = _fetch_asos_network(f"{state}_ASOS", bulk_start, bulk_end)
        if not df_net.empty:
            raw_frames.append(df_net)
        if i % 10 == 9:
            time.sleep(1)

    if not raw_frames:
        print("  No ASOS data retrieved.")
        return [pd.DataFrame()] * len(valid_times)

    df_bulk = pd.concat(raw_frames, ignore_index=True)
    df_bulk["valid"] = pd.to_datetime(df_bulk["valid"], utc=True, errors="coerce")
    df_bulk["lat"]   = pd.to_numeric(df_bulk["lat"], errors="coerce")
    df_bulk["lon"]   = pd.to_numeric(df_bulk["lon"], errors="coerce")
    df_bulk = df_bulk.dropna(subset=["valid", "lat", "lon"])
    df_bulk = df_bulk[
        (df_bulk["lat"] >= 24) & (df_bulk["lat"] <= 50) &
        (df_bulk["lon"] >= -123) & (df_bulk["lon"] <= -70)
    ]

    wx_col = next(
        (c for c in ["wxcodes", "presentwx", "present_wx", "wxcode"]
         if c in df_bulk.columns),
        None,
    )
    if wx_col is None:
        cands = [c for c in df_bulk.columns if "wx" in c.lower()]
        wx_col = cands[0] if cands else None

    print(f"  Bulk ASOS rows: {len(df_bulk)}  |  wx column: {wx_col}")

    df_bulk["_wx"] = (
        df_bulk[wx_col].fillna("").astype(str).replace("M", "")
        if wx_col else ""
    )
    df_bulk["precip_type"] = df_bulk["_wx"].apply(classify_asos)

    def _for_hour(vt: datetime) -> pd.DataFrame:
        lo  = vt - window
        hi  = vt + window
        sub = df_bulk[(df_bulk["valid"] >= lo) & (df_bulk["valid"] <= hi)].copy()
        if sub.empty:
            return pd.DataFrame(columns=df_bulk.columns)
        sub["_dt"] = (sub["valid"] - vt).abs()
        return sub.sort_values("_dt").groupby("station", as_index=False).first()

    return [_for_hour(vt) for vt in valid_times]


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _compute_stats(df_a: pd.DataFrame, pt: Optional[np.ndarray]) -> dict:
    """
    Compute POD and FAR for each category at stations where at least one
    side (ASOS or HRRR) reports precipitation.

    Returns
    -------
    dict  cat → {pod, far, tp, fn, fp}
    """
    empty = {c: {"pod": None, "far": None, "tp": 0, "fn": 0, "fp": 0}
             for c in CATS}
    if df_a.empty or pt is None or _kdtree is None:
        return empty

    query_pts = np.column_stack([df_a["lat"].values, df_a["lon"].values])
    _, idxs   = _kdtree.query(query_pts)

    hrrr_flat = pt.ravel()
    hrrr_cats = np.array([CODE_TO_CAT[hrrr_flat[i]] for i in idxs])
    asos_cats = df_a["precip_type"].values

    active    = (asos_cats != "None") | (hrrr_cats != "None")
    asos_cats = asos_cats[active]
    hrrr_cats = hrrr_cats[active]

    stats = {}
    for cat in CATS:
        obs_yes  = asos_cats == cat
        fcst_yes = hrrr_cats == cat
        tp = int(( obs_yes &  fcst_yes).sum())
        fn = int(( obs_yes & ~fcst_yes).sum())
        fp = int((~obs_yes &  fcst_yes).sum())
        pod = tp / (tp + fn) if (tp + fn) > 0 else None
        far = fp / (tp + fp) if (tp + fp) > 0 else None
        stats[cat] = {"pod": pod, "far": far, "tp": tp, "fn": fn, "fp": fp}
    return stats


def _fmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else " N/A"


# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def _render_frame(
    ax,
    fxx: int,
    valid_time: datetime,
    run_time: datetime,
    pt: Optional[np.ndarray],
    df_a: pd.DataFrame,
    xform,
) -> dict:
    """
    Draw one frame onto a *freshly-created* ax and return verification stats.

    The caller is responsible for creating a new figure/axes before each call.
    This avoids clearing ax.collections, which would inadvertently remove
    Cartopy feature artists (coastlines, borders, states) stored in the same
    collection list.
    """
    # HRRR background
    if pt is not None and _lats_h is not None:
        masked = np.ma.masked_where(pt == 0, pt)
        kw = dict(cmap=hrrr_cmap, norm=hrrr_norm,
                  shading="nearest", alpha=0.75, zorder=2)
        if _USE_CARTOPY:
            ax.pcolormesh(_lons_h, _lats_h, masked, transform=xform, **kw)
        else:
            ax.pcolormesh(_lons_h, _lats_h, masked, **kw)

    # ASOS stations
    if not df_a.empty:
        kw_bg = dict(c="#555555", s=6, alpha=0.35, zorder=6)
        if _USE_CARTOPY:
            ax.scatter(df_a["lon"], df_a["lat"], transform=xform, **kw_bg)
        else:
            ax.scatter(df_a["lon"], df_a["lat"], **kw_bg)

        df_p = df_a[df_a["precip_type"] != "None"]
        for cat in CATS:
            sub = df_p[df_p["precip_type"] == cat]
            if sub.empty:
                continue
            st = STYLE[cat]
            kw_sc = dict(c=st["color"], marker=st["marker"], s=st["s"],
                         edgecolors="k", linewidths=0.5, alpha=0.95,
                         zorder=st["zorder"])
            if _USE_CARTOPY:
                ax.scatter(sub["lon"], sub["lat"], transform=xform, **kw_sc)
            else:
                ax.scatter(sub["lon"], sub["lat"], **kw_sc)

    stats = _compute_stats(df_a, pt)

    # Legend
    n_active = int((df_a["precip_type"] != "None").sum()) if not df_a.empty else 0
    handles = [
        Line2D([], [], ls="none",
               label=f"HRRR F{fxx:02d} (filled) / ASOS obs (markers)"),
        Line2D([], [], ls="none",
               label=f"{'Category':<16}  {'POD':>5}"),
        Line2D([], [], ls="none", label="─" * 28),
    ]
    for cat in CATS:
        st  = STYLE[cat]
        s   = stats[cat]
        n   = s["tp"] + s["fn"]
        handles.append(
            Line2D(
                [0], [0], linestyle="none",
                marker=st["marker"], color=st["color"],
                markeredgecolor="k", markeredgewidth=0.6, markersize=11,
                label=f"{cat:<16}  {_fmt(s['pod']):>5}  (n={n})",
            )
        )
    handles += [
        Line2D([], [], ls="none", label=""),
        Line2D([], [], ls="none",
               label=f"POD = hits/(hits+misses)\n"
                     f"Only stations with obs or fcst ≠ None\n"
                     f"Precip-reporting ASOS this hour: {n_active}"),
    ]

    ax.legend(
        handles=handles,
        loc="lower left", framealpha=0.90,
        fontsize=11,
        prop={"family": "monospace", "size": 11},
        title="Precip Type Verification",
        title_fontsize=12,
        handletextpad=0.5, borderpad=1.0,
    )

    n_total = len(df_a)
    ax.set_title(
        f"HRRR F{fxx:02d} + ASOS Precipitation Type  |  "
        f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"HRRR run: {run_time.strftime('%Y-%m-%d %H UTC')}   |   "
        f"ASOS: {n_active} reporting / {n_total} total stations",
        fontsize=13, fontweight="bold", pad=12,
    )

    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    n_hours: int = 12,
    output_dir: str = FRAMES_DIR,
) -> tuple[list[str], str]:
    """
    Render PNG frames for each (fxx, valid_time) pair and write a POD CSV.

    Parameters
    ----------
    n_hours:
        How many hours back from VALID_TIME to render (max 12).
    output_dir:
        Directory for PNG frames.

    Returns
    -------
    (frame_paths, csv_path)
    """
    target = VALID_TIME
    print(f"\n{'─'*70}")
    print(f"  wx_monitor.hrrr_asos  →  valid time: {target.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*70}")

    ensure_dir(output_dir)

    # Build list of valid times (oldest → newest)
    valid_times = [target - timedelta(hours=h)
                   for h in range(n_hours - 1, -1, -1)]
    print(f"\n  Valid times ({n_hours} hours): "
          f"{valid_times[0].strftime('%H UTC')} – "
          f"{valid_times[-1].strftime('%H UTC %d %b %Y')}")

    # Fetch ASOS once (covers all valid times)
    asos_frames = _fetch_all_asos(valid_times)

    # Determine transform for Cartopy (constant across all frames)
    xform = ccrs.PlateCarree() if _USE_CARTOPY else None

    pod_records:  list[dict] = []
    frame_paths:  list[str]  = []

    for fxx in HRRR_FXX:
        hrrr_frames = _fetch_hrrr_frames(fxx, valid_times)
        run_times   = [vt - timedelta(hours=fxx) for vt in valid_times]

        print(f"\n── Saving F{fxx:02d} frames ─────────────────────────────────────")
        for i, vt in enumerate(valid_times):
            # ── Fresh figure + axes per frame ─────────────────────────────
            # A new figure is created each iteration so that Cartopy map
            # features (coastlines, borders, states) are never accidentally
            # removed when data collections are updated.
            if _USE_CARTOPY:
                proj_lcc = ccrs.LambertConformal(
                    central_longitude=-96, central_latitude=39
                )
                fig, ax = plt.subplots(
                    figsize=(17, 10), subplot_kw={"projection": proj_lcc}
                )
                ax.set_extent([-123, -70, 24, 50], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND,
                               facecolor="#f0efe9", zorder=0)
                ax.add_feature(cfeature.OCEAN,
                               facecolor="#cde5f0", zorder=0)
                ax.add_feature(cfeature.LAKES,
                               facecolor="#cde5f0", zorder=1, alpha=0.6)
                ax.add_feature(cfeature.STATES,
                               linewidth=0.5, edgecolor="#777777", zorder=5)
                ax.add_feature(cfeature.BORDERS,
                               linewidth=0.8, edgecolor="#333333", zorder=5)
                ax.add_feature(cfeature.COASTLINE,
                               linewidth=0.8, zorder=5)
            else:
                fig, ax = plt.subplots(figsize=(17, 10))
                ax.set_xlim(-123, -70)
                ax.set_ylim(24, 50)
                ax.set_facecolor("#cde5f0")

            stats = _render_frame(
                ax, fxx, vt, run_times[i],
                hrrr_frames[i], asos_frames[i], xform,
            )

            out_path = Path(output_dir) / f"frame_f{fxx:02d}_{i:02d}.png"
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)   # release memory immediately
            print(f"  {out_path}  (valid {vt.strftime('%Y-%m-%d %H UTC')})")
            frame_paths.append(str(out_path.resolve()))

            for cat in CATS:
                s = stats[cat]
                pod_records.append({
                    "valid_time": vt.strftime("%Y-%m-%d %H:%M UTC"),
                    "fxx":        fxx,
                    "category":   cat,
                    "pod":        round(s["pod"], 4) if s["pod"] is not None else "",
                    "tp":         s["tp"],
                    "fn":         s["fn"],
                    "n_obs":      s["tp"] + s["fn"],
                })

    # Write POD CSV
    csv_name = f"pod_stats_{target.strftime('%Y%m%d_%H%M')}.csv"
    csv_path = Path(output_dir) / csv_name
    pd.DataFrame(pod_records).to_csv(str(csv_path), index=False)
    print(f"\n  POD stats → {csv_path}")

    return frame_paths, str(csv_path.resolve())
