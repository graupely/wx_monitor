"""
wx_monitor
==========
Operational weather data retrieval and visualisation package.

Products
--------
goes_mrms      GOES-19 visible + MRMS composite reflectivity overlay
mrms_hrrr      MRMS MultiSensor QPE vs HRRR accumulated precipitation (1/3/6/12 h)
hrrr_asos      HRRR precipitation-type forecast vs ASOS METAR verification

Shared valid time
-----------------
All products anchor to the same UTC hour:

    VALID_TIME = floor(utcnow, hour) − 1 hour

Import it with::

    from wx_monitor.config import VALID_TIME
"""

from wx_monitor.config import VALID_TIME  # noqa: F401 — re-exported for convenience

__version__ = "1.1.0"
__all__ = ["goes_mrms", "mrms_hrrr", "hrrr_asos", "config", "utils"]
