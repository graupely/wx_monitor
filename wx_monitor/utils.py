"""
wx_monitor.utils
================
Shared helper functions used across all product modules.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def floor_hour(dt: datetime) -> datetime:
    """Return *dt* truncated to the whole hour (UTC-aware)."""
    return dt.replace(minute=0, second=0, microsecond=0)


def round_to_hour(dt: datetime) -> datetime:
    """Round *dt* to the nearest whole hour."""
    discard = timedelta(
        minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond
    )
    dt -= discard
    if discard >= timedelta(minutes=30):
        dt += timedelta(hours=1)
    return dt


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: int = 60, **kwargs) -> requests.Response:
    """GET *url*, raise on HTTP error, return Response."""
    resp = requests.get(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


def download_bytes(url: str, timeout: int = 60) -> bytes:
    """Download *url* and return raw bytes."""
    return http_get(url, timeout=timeout).content


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create *path* if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# GOES filename parsing
# ---------------------------------------------------------------------------

def parse_goes_scan_time(filename: str) -> Optional[datetime]:
    """Parse scan-start datetime from GOES ABI filename token ``s{YYYYDDDHHMMSST}``."""
    basename = filename.split("/")[-1]
    for token in basename.split("_"):
        if token.startswith("s") and len(token) == 15:
            try:
                return datetime.strptime(token[1:14], "%Y%j%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# MRMS filename parsing
# ---------------------------------------------------------------------------

def parse_mrms_time(filename: str) -> Optional[datetime]:
    """Parse datetime from MRMS filename e.g. ``…_20260228-190036.grib2.gz``."""
    m = re.search(r"(\d{8})-(\d{6})", filename)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    return None


# ---------------------------------------------------------------------------
# NWS precipitation colormap (shared between mrms_hrrr, goes_mrms, and any other caller)
# ---------------------------------------------------------------------------

def make_nws_precip_cmap():
    """Return ``(cmap, norm)`` for the NWS precipitation color scale (inches)."""
    import matplotlib.colors as mcolors
    from wx_monitor.config import NWS_LEVELS, NWS_COLORS

    cmap = mcolors.ListedColormap(NWS_COLORS)
    norm = mcolors.BoundaryNorm(NWS_LEVELS, cmap.N)
    return cmap, norm


# ---------------------------------------------------------------------------
# Map feature helpers
# ---------------------------------------------------------------------------

def add_standard_features(ax, zorder: int = 3) -> None:
    """Add coastlines, borders, states and lakes to a Cartopy GeoAxes."""
    import cartopy.feature as cfeature

    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.6, edgecolor="white", zorder=zorder,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.5, edgecolor="white", zorder=zorder,
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.35, edgecolor="white", zorder=zorder,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        linewidth=0.3, edgecolor="white", facecolor="none", zorder=zorder,
    )
