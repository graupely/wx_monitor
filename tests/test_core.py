"""
tests/test_core.py
==================
Unit tests for the wx_monitor package.

These tests cover logic that can be validated without live network access:
  - Shared valid-time computation
  - Filename-parsing utilities
  - ASOS METAR classification
  - NWS precipitation colormap construction
  - CLI argument parsing
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# config — valid time
# ---------------------------------------------------------------------------

class TestValidTime:
    def test_valid_time_is_utc(self):
        from wx_monitor.config import VALID_TIME
        assert VALID_TIME.tzinfo is not None
        assert VALID_TIME.tzinfo == timezone.utc

    def test_valid_time_is_on_the_hour(self):
        from wx_monitor.config import VALID_TIME
        assert VALID_TIME.minute == 0
        assert VALID_TIME.second == 0
        assert VALID_TIME.microsecond == 0

    def test_valid_time_is_one_hour_before_now(self):
        from wx_monitor.config import VALID_TIME
        now_floored = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        # Allow up to 1 second of clock drift between import and assertion
        assert abs((now_floored - VALID_TIME - timedelta(hours=1)).total_seconds()) < 2

    def test_all_modules_share_identical_valid_time(self):
        """Every module that imports VALID_TIME must see the same object."""
        from wx_monitor.config   import VALID_TIME as vt_cfg
        from wx_monitor          import VALID_TIME  as vt_pkg
        assert vt_cfg is vt_pkg


# ---------------------------------------------------------------------------
# utils — filename parsing
# ---------------------------------------------------------------------------

class TestParseGoesScanTime:
    def test_valid_filename(self):
        from wx_monitor.utils import parse_goes_scan_time
        fname = (
            "OR_ABI-L1b-RadC-M6C02_G19_s20260590001174_e20260590003547_"
            "c20260590003587.nc"
        )
        dt = parse_goes_scan_time(fname)
        assert dt is not None
        assert dt.year == 2026
        assert dt.tzinfo == timezone.utc

    def test_invalid_filename_returns_none(self):
        from wx_monitor.utils import parse_goes_scan_time
        assert parse_goes_scan_time("not_a_goes_file.nc") is None

    def test_s3_path_prefix_stripped(self):
        from wx_monitor.utils import parse_goes_scan_time
        full = (
            "noaa-goes19/ABI-L1b-RadC/2026/059/00/"
            "OR_ABI-L1b-RadC-M6C02_G19_s20260590001174_e20260590003547_"
            "c20260590003587.nc"
        )
        dt = parse_goes_scan_time(full)
        assert dt is not None


class TestParseMrmsTime:
    def test_valid_filename(self):
        from wx_monitor.utils import parse_mrms_time
        fname = "MRMS_MergedReflectivityQCComposite_00.50_20260228-190036.grib2.gz"
        dt = parse_mrms_time(fname)
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 28
        assert dt.hour == 19
        assert dt.minute == 0
        assert dt.second == 36

    def test_invalid_filename_returns_none(self):
        from wx_monitor.utils import parse_mrms_time
        assert parse_mrms_time("no_date_here.grib2.gz") is None


# ---------------------------------------------------------------------------
# utils — time helpers
# ---------------------------------------------------------------------------

class TestTimeHelpers:
    def test_floor_hour(self):
        from wx_monitor.utils import floor_hour
        dt = datetime(2026, 2, 28, 14, 37, 22, tzinfo=timezone.utc)
        result = floor_hour(dt)
        assert result == datetime(2026, 2, 28, 14, 0, 0, tzinfo=timezone.utc)

    def test_round_to_hour_rounds_up(self):
        from wx_monitor.utils import round_to_hour
        dt = datetime(2026, 2, 28, 14, 31, 0, tzinfo=timezone.utc)
        assert round_to_hour(dt) == datetime(2026, 2, 28, 15, 0, 0, tzinfo=timezone.utc)

    def test_round_to_hour_rounds_down(self):
        from wx_monitor.utils import round_to_hour
        dt = datetime(2026, 2, 28, 14, 29, 0, tzinfo=timezone.utc)
        assert round_to_hour(dt) == datetime(2026, 2, 28, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# hrrr_asos — METAR classification
# ---------------------------------------------------------------------------

class TestClassifyAsos:
    def test_rain(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("RA") == "Rain"
        assert classify_asos("-RA BR") == "Rain"

    def test_snow(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("SN") == "Snow"
        assert classify_asos("-SN OVC015") == "Snow"

    def test_drizzle_remapped_to_rain(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("DZ") == "Rain"

    def test_mixed_remapped_to_snow(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("RASN") == "Snow"

    def test_freezing_rain(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("FZRA") == "Freezing Rain"
        assert classify_asos("-FZRA") == "Freezing Rain"

    def test_ice_pellets(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("PL") == "Ice Pellets"

    def test_hail_remapped_to_none(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("GR") == "None"

    def test_empty_string(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("") == "None"

    def test_no_precip(self):
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("SKC") == "None"
        assert classify_asos("FG BR") == "None"

    def test_priority_freezing_over_rain(self):
        """FZRA should win over RA when both appear."""
        from wx_monitor.hrrr_asos import classify_asos
        assert classify_asos("RA FZRA") == "Freezing Rain"


# ---------------------------------------------------------------------------
# mrms_hrrr — MRMS QPE filename parsing
# ---------------------------------------------------------------------------

class TestParseMrmsQpeTime:
    """Tests for _parse_mrms_qpe_time() in mrms_hrrr."""

    def test_12h_filename(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        fname = "MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-230000.grib2.gz"
        dt = _parse_mrms_qpe_time(fname)
        assert dt is not None
        assert dt.year  == 2026
        assert dt.month == 2
        assert dt.day   == 28
        assert dt.hour  == 23
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo is not None

    def test_01h_filename(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        fname = "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260301-140500.grib2.gz"
        dt = _parse_mrms_qpe_time(fname)
        assert dt is not None
        assert dt.hour   == 14
        assert dt.minute == 5

    def test_06h_filename(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        fname = "MRMS_MultiSensor_QPE_06H_Pass2_00.00_20260228-060000.grib2.gz"
        dt = _parse_mrms_qpe_time(fname)
        assert dt is not None
        assert dt.hour == 6

    def test_no_timestamp_returns_none(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        assert _parse_mrms_qpe_time("no_date_here.grib2.gz") is None

    def test_empty_returns_none(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        assert _parse_mrms_qpe_time("") is None

    def test_wrong_extension_returns_none(self):
        from wx_monitor.mrms_hrrr import _parse_mrms_qpe_time
        # Must end in .grib2.gz
        assert _parse_mrms_qpe_time(
            "MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-230000.grib2"
        ) is None


class TestMrmsQpeUrlSelection:
    """_get_mrms_qpe_url selects the file closest to the target time."""

    def _mock_listing(self, filenames: list[str]) -> str:
        """Build a minimal HTML directory listing."""
        links = "\n".join(f'<a href="{f}">{f}</a>' for f in filenames)
        return f"<html><body>\n{links}\n</body></html>"

    def test_selects_closest_file(self, requests_mock):
        pytest.importorskip("requests_mock")
        from wx_monitor.mrms_hrrr import _get_mrms_qpe_url
        from wx_monitor.config import MRMS_QPE_BASE, MRMS_QPE_DIRS

        files = [
            "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260228-180000.grib2.gz",  # -60 min
            "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260228-190000.grib2.gz",  # exact match
            "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260228-200000.grib2.gz",  # +60 min
        ]
        target_url = f"{MRMS_QPE_BASE}{MRMS_QPE_DIRS[1]}/"
        requests_mock.get(target_url, text=self._mock_listing(files))

        target = datetime(2026, 2, 28, 19, 0, tzinfo=timezone.utc)
        url, dt = _get_mrms_qpe_url(1, target)

        assert "20260228-190000" in url
        assert dt.hour == 19

    def test_raises_when_no_files_found(self, requests_mock):
        pytest.importorskip("requests_mock")
        from wx_monitor.mrms_hrrr import _get_mrms_qpe_url
        from wx_monitor.config import MRMS_QPE_BASE, MRMS_QPE_DIRS

        target_url = f"{MRMS_QPE_BASE}{MRMS_QPE_DIRS[12]}/"
        requests_mock.get(target_url, text="<html><body>empty</body></html>")

        with pytest.raises(FileNotFoundError, match="No MRMS QPE files found"):
            _get_mrms_qpe_url(12, datetime(2026, 2, 28, 19, 0, tzinfo=timezone.utc))


class TestDownloadMrmsQpeValidation:
    """_download_mrms_qpe must reject non-gzip server responses."""

    def test_raises_on_html_response(self, tmp_path, monkeypatch, requests_mock):
        pytest.importorskip("requests_mock")
        from wx_monitor import mrms_hrrr

        html_body = b"<!DOCTYPE html><html><body>Not Found</body></html>"
        fake_url  = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_12H_Pass2/" \
                    "MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-190000.grib2.gz"
        requests_mock.get(fake_url, content=html_body, status_code=200)
        monkeypatch.setattr(mrms_hrrr, "CACHE_DIR", str(tmp_path))

        with pytest.raises(RuntimeError, match="did not return a gzip file"):
            mrms_hrrr._download_mrms_qpe(fake_url, 12)

    def test_accepts_valid_gzip(self, tmp_path, monkeypatch, requests_mock):
        pytest.importorskip("requests_mock")
        import gzip as gzip_mod
        from wx_monitor import mrms_hrrr

        # Minimal valid gzip payload (compresses the bytes b"GRIB...")
        gz_payload = gzip_mod.compress(b"GRIB" + b"\x00" * 64)
        fake_url   = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/" \
                     "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260228-190000.grib2.gz"
        requests_mock.get(fake_url, content=gz_payload, status_code=200)
        monkeypatch.setattr(mrms_hrrr, "CACHE_DIR", str(tmp_path))

        local = mrms_hrrr._download_mrms_qpe(fake_url, 1)
        assert Path(local).exists()
        # The cached file is the *decompressed* GRIB2
        assert Path(local).read_bytes()[:4] == b"GRIB"


class TestMrmsQpeCoordinates:
    """_load_mrms_qpe must return 2-D (ny, nx) arrays even when cfgrib
    provides 1-D coordinate vectors (regular lat/lon MRMS grid)."""

    def _make_fake_grib_ds(self, lats_1d, lons_1d):
        """Return a minimal xarray-style mock dataset."""
        import xarray as xr

        ny, nx = lats_1d.size, lons_1d.size
        data   = np.random.rand(ny, nx).astype(np.float32) * 25.4  # mm

        ds = xr.Dataset(
            {"unknown": xr.DataArray(data, dims=["latitude", "longitude"])},
            coords={
                "latitude":  xr.DataArray(lats_1d, dims=["latitude"]),
                "longitude": xr.DataArray(lons_1d, dims=["longitude"]),
            },
        )
        return ds

    def test_1d_coords_become_2d(self, tmp_path, monkeypatch):
        """Simulate cfgrib returning 1-D lat/lon vectors for a regular grid."""
        import cfgrib as cfgrib_mod
        from wx_monitor import mrms_hrrr

        lats_1d = np.linspace(20.0, 55.0,  35)   # (35,)
        lons_1d = np.linspace(-130.0, -60.0, 70)  # (70,)
        fake_ds = self._make_fake_grib_ds(lats_1d, lons_1d)

        monkeypatch.setattr(cfgrib_mod, "open_datasets", lambda *a, **kw: [fake_ds])

        lons, lats, data = mrms_hrrr._load_mrms_qpe("fake.grib2")

        assert lons.shape  == (35, 70), f"lons.shape={lons.shape}, expected (35, 70)"
        assert lats.shape  == (35, 70), f"lats.shape={lats.shape}, expected (35, 70)"
        assert data.shape  == (35, 70), f"data.shape={data.shape}, expected (35, 70)"
        assert lons.min()  >= -180.0
        assert lons.max()  <= 180.0
        # All values should be ≥ 0 (mm/25.4) or NaN
        assert np.all(np.isnan(data) | (data >= 0))

    def test_2d_coords_passthrough(self, tmp_path, monkeypatch):
        """When cfgrib returns 2-D coordinates (curvilinear), pass them through."""
        import cfgrib as cfgrib_mod
        import xarray as xr
        from wx_monitor import mrms_hrrr

        ny, nx  = 20, 30
        lats_2d = np.tile(np.linspace(25, 50, ny)[:, None], (1, nx))
        lons_2d = np.tile(np.linspace(-120, -70, nx)[None, :], (ny, 1))
        data_2d = np.ones((ny, nx), dtype=np.float32) * 10.0  # 10 mm

        ds = xr.Dataset(
            {"unknown": xr.DataArray(data_2d, dims=["y", "x"])},
            coords={
                "latitude":  xr.DataArray(lats_2d, dims=["y", "x"]),
                "longitude": xr.DataArray(lons_2d, dims=["y", "x"]),
            },
        )
        monkeypatch.setattr(cfgrib_mod, "open_datasets", lambda *a, **kw: [ds])

        lons, lats, data = mrms_hrrr._load_mrms_qpe("fake.grib2")

        assert lons.shape == (ny, nx)
        assert lats.shape == (ny, nx)


# ---------------------------------------------------------------------------
# mrms_hrrr — config constants integrity
# ---------------------------------------------------------------------------

class TestMrmsQpeConfig:
    def test_all_accum_hours_have_dirs(self):
        from wx_monitor.config import ACCUM_HOURS, MRMS_QPE_DIRS, MRMS_QPE_PREFIXES
        for h in ACCUM_HOURS:
            assert h in MRMS_QPE_DIRS,    f"MRMS_QPE_DIRS missing key {h}"
            assert h in MRMS_QPE_PREFIXES, f"MRMS_QPE_PREFIXES missing key {h}"

    def test_dir_names_include_hour_padding(self):
        from wx_monitor.config import MRMS_QPE_DIRS
        assert "01H" in MRMS_QPE_DIRS[1]
        assert "03H" in MRMS_QPE_DIRS[3]
        assert "06H" in MRMS_QPE_DIRS[6]
        assert "12H" in MRMS_QPE_DIRS[12]

    def test_base_url_ends_with_slash(self):
        from wx_monitor.config import MRMS_QPE_BASE
        assert MRMS_QPE_BASE.endswith("/")


# ---------------------------------------------------------------------------
# utils — NWS colormap
# ---------------------------------------------------------------------------

class TestNwsColormap:
    def test_colormap_construction(self):
        from wx_monitor.utils import make_nws_precip_cmap
        cmap, norm = make_nws_precip_cmap()
        assert cmap is not None
        assert norm is not None

    def test_colormap_level_count(self):
        from wx_monitor.utils import make_nws_precip_cmap
        from wx_monitor.config import NWS_LEVELS, NWS_COLORS
        cmap, _ = make_nws_precip_cmap()
        assert cmap.N == len(NWS_COLORS)
        assert len(NWS_LEVELS) == len(NWS_COLORS)


# ---------------------------------------------------------------------------
# CLI — argument parsing (no network calls)
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_valid_time_flag(self, capsys):
        from wx_monitor.cli import main
        rc = main(["--valid-time"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VALID_TIME" in out

    def test_unknown_product_exits(self):
        from wx_monitor.cli import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent_product"])

    def test_default_output_dir(self):
        from wx_monitor.cli import _build_parser
        args = _build_parser().parse_args([])
        assert args.output_dir == "output"
        assert args.n_hours == 12

    def test_product_choices(self):
        from wx_monitor.cli import _build_parser
        for product in ["goes_mrms", "mrms_hrrr", "hrrr_asos"]:
            args = _build_parser().parse_args([product])
            assert args.product == product
