# D4sigma Beam Profiler

A Python desktop application for ISO 11146-compliant laser beam profiling from camera images. Computes D4σ beam widths, principal-axis moments, clip-level widths, and beam orientation from 2D intensity images captured by a beam profiling camera.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Features

- **ISO 11146-1 second-moment analysis** — centroid, σ_x, σ_y, D4σ_x, D4σ_y, and full tensor diagonalisation for principal axes (σ_maj, σ_min, θ, ellipticity)
- **Iterative integration-area masking** — ISO 11146-1 §7 rotated-rectangle mask converged to 3σ
- **ISO/TR 11146-3 background subtraction** — corner-seed → unilluminated-pixel refinement → n·σ zero threshold
- **TPA correction** — optional sign(x)·√|x| transform for two-photon-absorption cameras (ISO 11146-3 §3.1); toggleable per-session
- **Clip-level (D%pk) widths** — BeamGage-style 13.5% (1/e²) widths from marginal profiles, in both lab and principal axes
- **Robust auto-ROI** — blur + flat-region suppression for reliable beam finding even with damage clusters present, followed by ISO §7 iterative seed refinement
- **Damage masking** — draw rectangles over saturated damage clusters; pixels replaced by background mean
- **Saturation detection** — flags images where beam peak ≥ 80% of inferred ADC ceiling
- **Batch processing** — load a series of files, run all, export combined CSV and multi-page PDF report
- **Dark frame subtraction** — optional per-session dark frame
- **Marginal Gaussian fits** — display-only fits on profile plots (do not affect D4σ results)

---

## Screenshots

The application has a collapsible settings sidebar and a single canvas panel that switches between the raw image view and the four-panel analysis results (√ ROI image, BG-subtracted + D4σ ellipse, x-profile, y-profile).

---

## Installation

### Requirements

- Python 3.10 or newer
- PyQt6 (or PyQt5 as fallback)

### Install dependencies

```bash
pip install numpy pandas scipy matplotlib PyQt6
```

For PyQt5 instead:
```bash
pip install numpy pandas scipy matplotlib PyQt5
```

### Clone and run

```bash
git clone https://github.com/YOUR_USERNAME/d4sigma-beam-profiler.git
cd d4sigma-beam-profiler
python d4sigma_gui.py
```

No installation step is needed. The `beamprofiler/` package directory must be in the same folder as `d4sigma_gui.py`.

---

## Project structure

```
d4sigma-beam-profiler/
├── d4sigma_gui.py              # Main application (PyQt6/5 GUI)
├── beamprofiler/
│   ├── __init__.py
│   ├── models.py               # FileEntry dataclass
│   ├── analysis.py             # All physics / numerical functions
│   └── tests/
│       ├── __init__.py
│       └── test_analysis.py    # 38 unit tests (pytest)
└── README.md
```

The `beamprofiler` package has no Qt dependency and can be used independently from scripts or Jupyter notebooks:

```python
from beamprofiler.analysis import load_csv, apply_sqrt, run_analysis
from beamprofiler.models import FileEntry
import numpy as np

raw = load_csv("my_image.csv", has_header=False, has_index=False)
fe  = FileEntry(path="my_image.csv", raw=raw,
                clean_raw=raw.copy(), sqrt_img=apply_sqrt(raw))
fe  = run_analysis(fe, settings={"px": 4.65, "pad_sigma": 4.0,
                                  "fit_mode": "principal", "use_tpa": True})
print(fe.results["d4sigma_maj"], fe.results["theta_deg"])
```

---

## Workflow

1. **Load files** — Add one or more CSV files (rows × columns of pixel values, no header/index by default). Multi-file series supported.
2. **Mark damage** (if needed) — Draw rectangles over saturated damage clusters. Damaged pixels are replaced by the local background mean. Do this *before* running auto-ROI.
3. **Set ROI** — Use auto-ROI (recommended), draw manually, or apply one file's ROI to all others.
4. **Configure settings**
   - *Pixel size* — enter the camera's pixel pitch in µm. Results will be reported in µm; leave at 1.0 for pixel units.
   - *TPA correction* — enable if your camera uses two-photon absorption (signal ∝ I²); disable for linear cameras.
   - *Fit mode* — lab axes (σ_x, σ_y) or principal axes (σ_maj, σ_min, θ).
   - *Background parameters* — corner fraction (default 3.5%), n·σ threshold (default 3.0), mask factor (default 3.0 per ISO).
5. **Run** — Click **▶ Run (this file)** or **▶▶ Run all files**.
6. **Export** — Save results as CSV (one row per file) or PDF (one 4-panel figure page per file).

---

## Input file format

CSV files with one pixel value per cell, no header row, no index column (configurable). Values should be raw ADC counts. Tested with 8-bit cameras (values 0–255) but any bit depth works.

Example (3×4 image):
```
12,11,13,14
12,255,254,13
11,12,12,11
```

---

## Settings reference

| Setting | Default | Description |
|---|---|---|
| Pixel size | 1.0 µm | Camera pixel pitch. Set to 1.0 for pixel-unit output. |
| TPA correction | on | Apply sign(x)·√\|x\| before analysis (ISO 11146-3 §3.1). |
| Fit mode | Lab axes | Lab (σ_x, σ_y) or Principal axes (σ_maj, σ_min, θ). |
| vmax | 255 | Colour scale ceiling for the raw image viewer. |
| Corner fraction | 3.5% | Size of corner patches for background estimation. |
| n·σ threshold | 3.0 | Pixels below n·σ_bg are zeroed after BG subtraction. |
| Mask factor | 3.0 | Integration-area half-width in units of σ_maj (ISO: 3.0). |
| Pad σ | 4.0 | Auto-ROI seed half-width in 1/e² beam widths. |
| Saturation threshold | 80% | Fraction of ADC ceiling above which pixels are flagged. |

---

## Mathematics

All D4σ results come from direct numerical integration of the ISO 11146-1 §4 second-moment definitions — not from Gaussian curve fitting.

**Centroid:**
$$\bar{x} = \frac{\sum_{i,j} I_{ij}\, x_j}{\sum_{i,j} I_{ij}}, \quad \bar{y} = \frac{\sum_{i,j} I_{ij}\, y_i}{\sum_{i,j} I_{ij}}$$

**Second moments:**
$$\sigma_x^2 = \frac{\sum I_{ij}(x_j-\bar{x})^2}{\sum I_{ij}}, \quad \sigma_{xy} = \frac{\sum I_{ij}(x_j-\bar{x})(y_i-\bar{y})}{\sum I_{ij}}$$

**Principal axes (ISO 11146-1 Eqs. 25–27):**
$$\sigma_\text{maj}^2 = \frac{\sigma_x^2+\sigma_y^2}{2} + \sqrt{\left(\frac{\sigma_x^2-\sigma_y^2}{2}\right)^2+\sigma_{xy}^2}$$

**D4σ:** $D4\sigma_x = 4\sigma_x$, $D4\sigma_\text{maj} = 4\sigma_\text{maj}$

### Sources

- **ISO 11146-1:2021** — Test methods for laser beam widths, divergence angles and beam propagation ratios. Part 1: Stigmatic and simple astigmatic beams. §4 (second moments), §7 (iterative integration area).
- **ISO/TR 11146-3:2004** — Part 3: Intrinsic and geometrical laser beam classification. §3.1 (TPA correction), §3.2–3.4 (background subtraction procedure).
- **Scott Prahl, laserbeamsize** (https://laserbeamsize.readthedocs.io) — pragmatic background zeroing approach and iterative mask implementation.
- **Ophir/Spiricon BeamGage documentation** — clip-level (D%pk) width convention at 13.5% of marginal profile peak.

---

## Running the tests

```bash
cd d4sigma-beam-profiler
python -m pytest beamprofiler/tests/ -v
```

38 tests covering: `apply_sqrt`, `check_saturation`, `iso_background`, `rotated_rect_mask`, `_moments`, `beam_size_iso`, `auto_roi`, `clip_level_widths`, `marginal_gaussian_fit`, TPA round-trip. All tests use synthetic Gaussian beams with known analytical ground truth.

---

## Known limitations

- Input format is currently CSV only. TIFF and BeamGage `.bgData` support is planned.
- No uncertainty estimates on σ values (ISO/TR 11146-3 §5 noise propagation not yet implemented).
- No M² measurement workflow (single-file D4σ only; multi-z fitting not yet implemented).
- Session save/load (ROIs, damage masks, settings) not yet implemented.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. Please run the test suite before submitting a PR.
