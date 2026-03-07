"""
wx_monitor.config
=================
Single source of truth for the shared *valid time* and all
product-level constants used across the package.

Valid time rule
---------------
All products are anchored to::

    valid_time = floor(utcnow, hour) - 1 hour

This guarantees that GOES, MRMS reflectivity, MRMS QPE, HRRR, and ASOS are all
verified/fetched for the same UTC hour, regardless of which module
is called first or how long data retrieval takes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Shared valid time  (computed once at import time)
# ---------------------------------------------------------------------------

def _compute_valid_time() -> datetime:
    """Return floor(utcnow, hour) - 1h, always timezone-aware (UTC)."""
    now = datetime.now(timezone.utc)
    floored = now.replace(minute=0, second=0, microsecond=0)
    return floored - timedelta(hours=1)


VALID_TIME: datetime = _compute_valid_time()
"""The shared valid time for **all** products in this run.

Set once at package import so every module that does
``from wx_monitor.config import VALID_TIME`` sees the identical value.
"""

# ---------------------------------------------------------------------------
# GOES / MRMS
# ---------------------------------------------------------------------------
GOES_BUCKET  = "noaa-goes19"
GOES_PRODUCT = "ABI-L1b-RadC"
# GeoColor uses C01+C02+C03 (day) and C07+C13 (night); C02 is the scan-time anchor
GOES_BAND    = "C02"   # reference band for backward compatibility
MRMS_URL        = "https://mrms.ncep.noaa.gov/2D/MergedReflectivityQCComposite/"
MRMS_REFL_M10_URL = "https://mrms.ncep.noaa.gov/2D/Reflectivity_-10C/"
CONUS_EXTENT = [-125, -66.5, 20, 50]   # [W, E, S, N] degrees

# ---------------------------------------------------------------------------
# MRMS Multi-Sensor QPE
# ---------------------------------------------------------------------------
MRMS_QPE_BASE = "https://mrms.ncep.noaa.gov/2D/"

# Subdirectory name for each accumulation period (hours → directory)
MRMS_QPE_DIRS = {
    1:  "MultiSensor_QPE_01H_Pass2",
    3:  "MultiSensor_QPE_03H_Pass2",
    6:  "MultiSensor_QPE_06H_Pass2",
    12: "MultiSensor_QPE_12H_Pass2",
}

# Expected filename prefix per period (used in directory listing regex)
MRMS_QPE_PREFIXES = {
    1:  "MRMS_MultiSensor_QPE_01H_Pass2",
    3:  "MRMS_MultiSensor_QPE_03H_Pass2",
    6:  "MRMS_MultiSensor_QPE_06H_Pass2",
    12: "MRMS_MultiSensor_QPE_12H_Pass2",
}

ACCUM_HOURS = [1, 3, 6, 12]

NWS_LEVELS = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00,
              2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00, 20.00]
NWS_COLORS = [
    "#c0e8ff", "#78c8ff", "#3090e0", "#1464b4",
    "#00c800", "#00a000", "#009600", "#006400",
    "#f0e800", "#e0c800", "#e09600", "#c06400",
    "#e00000", "#b40000", "#780000", "#e87dfc",
]

# ---------------------------------------------------------------------------
# HRRR
# ---------------------------------------------------------------------------
HRRR_BASE   = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_FXX    = [1, 3, 6, 12]   # forecast hours rendered by hrrr_asos module

# ---------------------------------------------------------------------------
# ASOS
# ---------------------------------------------------------------------------
ASOS_WINDOW_MINUTES = 20   # ±minutes around valid_time when querying ASOS
CONUS_STATES = [
    "AL","AR","AZ","CA","CO","CT","DE","FL","GA","IA","ID","IL","IN","KS",
    "KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH",
    "NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
    "VA","VT","WA","WI","WV","WY",
]
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CACHE_DIR  = "noaa_precip_cache"
FRAMES_DIR = "frames"
OUTPUT_DPI = 150
