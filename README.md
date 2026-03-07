# wx-monitor

Operational weather visualisation package that combines four NOAA data
streams into a unified, time-synchronised pipeline.

| Product | Data sources | Output |
|---------|-------------|--------|
| **GOES + MRMS** | GOES-19 ABI GeoColor (S3) + MRMS composite reflectivity | `goes19_conus_geocolor_mrms.png` |
| **MRMS QPE vs HRRR** | MRMS MultiSensor QPE Pass 2 + HRRR APCP (1 h, 3 h, 6 h, 12 h) | `mrms_qpe_vs_hrrr_*hr_*.png` × 4 |
| **HRRR + ASOS ptype** | HRRR precipitation type + ASOS METAR observations | `frames/frame_f??_??.png` + `pod_stats_*.csv` |
| **Reflectivity at −10 °C** | MRMS Reflectivity_-10C + HRRR REFD:263 K (F06) | `refl_m10_f06_*.png` |

---

## Shared valid-time anchor

All four products are anchored to the same UTC hour:

```
VALID_TIME = floor(utcnow, hour) − 1 hour
```

This value is computed **once** at package import and shared across every
module, so GOES scans, MRMS tiles, HRRR runs and ASOS observations all
refer to the same clock time regardless of how long each download takes.

```python
from wx_monitor.config import VALID_TIME
print(VALID_TIME)  # e.g. 2026-02-28 18:00:00+00:00
```

---

## Installation

### Prerequisites

The package requires several system-level libraries for GRIB and NetCDF
support, plus cartographic projection libraries.

**Debian / Ubuntu:**
```bash
sudo apt-get install \
    libgeos-dev libproj-dev proj-data proj-bin \
    libeccodes-dev libhdf5-dev libnetcdf-dev
```

**macOS (Homebrew):**
```bash
brew install eccodes hdf5 netcdf geos proj
```

### Python package

```bash
git clone https://github.com/<your-org>/wx-monitor.git
cd wx-monitor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Run everything

```bash
python -m wx_monitor
# or
wx-monitor
```

Output is written to `output/` by default.

### Run a single product

```bash
python -m wx_monitor goes_mrms   # GOES-19 GeoColor + MRMS composite reflectivity
python -m wx_monitor mrms_hrrr   # MRMS QPE vs HRRR accumulated precipitation
python -m wx_monitor hrrr_asos   # HRRR ptype forecast vs ASOS METAR verification
python -m wx_monitor refl_m10    # MRMS vs HRRR reflectivity at −10 °C
```

### Options

```
usage: python -m wx_monitor [-h] [--output-dir DIR] [--n-hours N] [--valid-time]
                             [{goes_mrms,mrms_hrrr,hrrr_asos,refl_m10}]

positional arguments:
  product               Product to run (omit to run all four)

options:
  --output-dir DIR      Directory for all output files (default: output/)
  --n-hours N           Back-hours for hrrr_asos frames (default: 12)
  --valid-time          Print the shared valid time and exit
```

### Python API

```python
from wx_monitor import goes_mrms, mrms_hrrr, hrrr_asos, refl_m10
from wx_monitor.config import VALID_TIME

goes_mrms.run(output_path="output/goes.png")
mrms_hrrr.run(output_dir="output/")
hrrr_asos.run(n_hours=12, output_dir="output/frames/")
refl_m10.run(output_dir="output/")
```

---

## Package layout

```
wx_monitor/
├── wx_monitor/
│   ├── __init__.py        # Re-exports VALID_TIME
│   ├── __main__.py        # python -m wx_monitor entry point
│   ├── cli.py             # Argument parsing & orchestration
│   ├── config.py          # ★ Shared VALID_TIME + all constants
│   ├── utils.py           # Shared helpers (HTTP, parsing, colormaps)
│   ├── goes_mrms.py       # GOES-19 GeoColor + MRMS composite reflectivity
│   ├── mrms_hrrr.py       # MRMS MultiSensor QPE vs HRRR accumulated precip
│   ├── hrrr_asos.py       # HRRR ptype forecast vs ASOS METAR verification
│   └── refl_m10.py        # MRMS vs HRRR reflectivity at −10 °C (263 K)
├── tests/
│   └── test_core.py       # Unit tests (no network required)
├── .github/
│   └── workflows/
│       └── update.yml     # Scheduled pipeline (4× daily) + GitHub Pages deploy
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Products

### GOES + MRMS (`goes_mrms`)

GOES-19 ABI GeoColor composite overlaid with MRMS composite reflectivity.

GeoColor follows the algorithm of Miller et al. (2020, *J. Atmos. Oceanic Technol.*, 37, 429–448):

- **Daytime** (µ₀ ≥ 0.3): True Color RGB — Red = C02, Green = hybrid simulated
  (`0.48·C02 + 0.46·C01 + 0.06·C03`), Blue = C01. Log₁₀ reflectance scaling
  over [−1.6, 0.176].
- **Nighttime** (µ₀ ≤ 0.1): Three-layer nested stack — high ice clouds (C13,
  white), low liquid clouds (C13−C07 BTD, pale blue), dark background.
- **Twilight** (0.1 < µ₀ < 0.3): Per-pixel cosine blend weighted by
  `N(µ₀)^1.5` per Miller et al. Eq. 5.

---

### MRMS QPE vs HRRR (`mrms_hrrr`)

Side-by-side comparison of observed and forecast accumulated precipitation
for four periods: 1 h, 3 h, 6 h, and 12 h.  Each figure is a 4-panel layout:

```
[MRMS QPE map]  [HRRR forecast map]  [spacer]  [KDE distribution]
```

The KDE panel shows normalised density of precipitation values > 0.01 inch on
a log-linear axis, with dashed vertical lines at the dataset medians.

---

### HRRR + ASOS ptype (`hrrr_asos`)

Animated PNG frames showing HRRR precipitation-type forecast fields
(CRAIN / CSNOW / CFRZR / CICEP) with ASOS METAR observations overlaid as
scatter markers.  Verification statistics (POD, FAR) are computed per category
per frame and written to a CSV.  Frames are rendered for forecast hours F01,
F03, F06, and F12.

#### Verification metrics

For each canonical precipitation type (Rain, Snow, Freezing Rain, Ice Pellets):

| Metric | Formula | Description |
|--------|---------|-------------|
| POD | TP / (TP + FN) | Fraction of observed events caught by the model |
| FAR | FP / (TP + FP) | Fraction of model forecasts that were incorrect |

Computed only at stations where at least one side (ASOS or HRRR) reports
precipitation, so trivially-correct "both quiet" pairs are excluded.

---

### Reflectivity at −10 °C (`refl_m10`)

Side-by-side comparison of MRMS observed and HRRR F06 forecast reflectivity
at the −10 °C isotherm (263 K), using a 4-panel layout matching `mrms_hrrr`:

```
[MRMS map]  [HRRR F06 map]  [spacer]  [KDE distribution]
```

The KDE panel shows normalised density of all reflectivity values ≥ 0 dBZ,
with dashed vertical lines at the dataset medians.

The −10 °C level is diagnostically important for winter precipitation: it lies
near the dendritic growth zone for snow crystals, so reflectivity at this level
is a useful proxy for snowfall intensity and storm organisation.

HRRR field: `REFD:263 K above mean sea level`, byte-range fetched from the
`wrfsfcf06` surface file.

---

## Data sources

| Source | URL / Bucket | Auth |
|--------|-------------|------|
| GOES-19 ABI L1b | `s3://noaa-goes19` | Anonymous (public) |
| MRMS Composite Reflectivity | `https://mrms.ncep.noaa.gov/2D/MergedReflectivityQCComposite/` | None |
| MRMS Reflectivity −10 °C | `https://mrms.ncep.noaa.gov/2D/Reflectivity_-10C/` | None |
| MRMS MultiSensor QPE 1 H | `https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/` | None |
| MRMS MultiSensor QPE 3 H | `https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_03H_Pass2/` | None |
| MRMS MultiSensor QPE 6 H | `https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_06H_Pass2/` | None |
| MRMS MultiSensor QPE 12 H | `https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_12H_Pass2/` | None |
| HRRR (APCP, REFD, ptype) | `https://noaa-hrrr-bdp-pds.s3.amazonaws.com` | None (byte-range) |
| ASOS METAR | Iowa Environmental Mesonet (`mesonet.agron.iastate.edu`) | None |

### MRMS file format

Files are gzip-compressed GRIB2 (`.grib2.gz`). The valid time is encoded in
the filename — no separate metadata request is needed:

```
MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-230000.grib2.gz
                                     ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                                     valid at 2026-02-28 23:00 UTC
```

The file whose timestamp is closest to `VALID_TIME` is selected automatically.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
