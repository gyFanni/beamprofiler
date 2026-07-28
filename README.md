# beamprofiler

A Python desktop application for ISO 11146-compliant laser beam profiling from camera images. Computes D4σ beam widths, principal-axis moments, clip-level widths, and beam orientation from 2D intensity images captured by a beam profiling camera.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Features

- **ISO 11146-1 second-moment analysis** — centroid, σ_x, σ_y, D4σ_x, D4σ_y, and full tensor diagonalisation for principal axes (σ_maj, σ_min, θ, ellipticity)
- **Iterative integration-area masking** — ISO 11146-1 §7 rotated-rectangle mask converged to 3σ
- **Background subtraction** — corner-seed approximation method (ISO/TR 11146-3 §3.4.3); optional, can be disabled for cameras that output pre-subtracted data
- **TPA correction** — optional sign(x)·√|x| transform for two-photon-absorption cameras; toggleable per-session
- **Clip-level (D%pk) widths** — BeamGage-style 13.5% (1/e²) widths from marginal profiles, in both lab and principal axes
- **Robust auto-ROI** — Gaussian-blur + flat-region suppression for reliable beam finding even with saturated damage clusters present, followed by ISO §7 iterative seed refinement
- **Damage masking** — draw rectangles over saturated damage clusters; pixels replaced by background mean
- **Saturation detection** — flags images where beam peak ≥ 80% of inferred ADC ceiling
- **BeamGage .bgData support** — loads Ophir/Spiricon HDF5 files; pixel scale and ADC ceiling auto-populated from file metadata
- **Batch processing** — load a series of files, run all, export combined CSV and multi-page PDF report
- **Dark frame subtraction** — optional per-session dark frame (CSV or .bgData)
- **Marginal Gaussian fits** — display-only fits on profile plots (do not affect D4σ results)

---

## Screenshots

The application has a collapsible settings sidebar and a single canvas panel that switches between the raw image view and the four-panel analysis results (√ ROI image, BG-subtracted + D4σ ellipse, x-profile, y-profile).

---

## Installation

### Requirements

- Python 3.10 or newer
- PyQt6 (or PyQt5 as fallback)
- h5py (required only for .bgData files)

### Install dependencies

```bash
pip install numpy pandas scipy matplotlib PyQt6 h5py
```

For PyQt5 instead:
```bash
pip install numpy pandas scipy matplotlib PyQt5 h5py
```

### Clone and run

```bash
git clone https://github.com/YOUR_USERNAME/beamprofiler.git
cd beamprofiler
python d4sigma_gui.py
```

No installation step is needed. The `beamprofiler/` package directory must be in the same folder as `d4sigma_gui.py`.

---

## Project structure

```
beamprofiler/
├── d4sigma_gui.py              # Main application (PyQt6/5 GUI)
├── beamprofiler/
│   ├── __init__.py
│   ├── models.py               # FileEntry dataclass
│   ├── analysis.py             # All physics / numerical functions
│   └── tests/
│       ├── __init__.py
│       └── test_analysis.py    # 38 unit tests (pytest)
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

The `beamprofiler` package has no Qt dependency and can be used independently from scripts or Jupyter notebooks:

```python
from beamprofiler.analysis import load_csv, apply_sqrt, run_analysis
from beamprofiler.models import FileEntry

raw = load_csv("my_image.csv", has_header=False, has_index=False)
fe  = FileEntry(path="my_image.csv", raw=raw,
                clean_raw=raw.copy(), sqrt_img=apply_sqrt(raw))
fe  = run_analysis(fe, settings={"px": 4.65, "pad_sigma": 4.0,
                                  "fit_mode": "principal", "use_tpa": True})
print(fe.results["d4sigma_maj"], fe.results["theta_deg"])
```

For BeamGage files:

```python
from beamprofiler.analysis import load_bgdata, run_analysis
from beamprofiler.models import FileEntry

info = load_bgdata("measurement.bgData")
raw  = info["image"]
fe   = FileEntry(path="measurement.bgData", raw=raw,
                 clean_raw=raw.copy(), sqrt_img=raw.copy())  # TPA off for linear camera
fe   = run_analysis(fe, settings={"px": info["pixel_size_x"], "pad_sigma": 4.0,
                                   "fit_mode": "principal", "use_tpa": False})
```

---

## Workflow

1. **Load files** — Add CSV or .bgData files via the file dialog. For .bgData files, pixel scale and ADC ceiling are auto-populated from the file metadata and TPA correction is automatically disabled.
2. **Mark damage** (if needed) — Draw rectangles over saturated damage clusters. Damaged pixels are replaced by the local background mean. Do this *before* running auto-ROI.
3. **Set ROI** — Use auto-ROI (recommended), draw manually, or apply one file's ROI to all others with the "□→□ Apply this ROI to all files" button.
4. **Configure settings**
   - *Pixel size* — camera pixel pitch in µm. Leave at 1.0 for pixel-unit output.
   - *TPA correction* — enable for two-photon-absorption cameras (signal ∝ I²); disable for linear cameras. Auto-disabled for .bgData files.
   - *Background subtraction* — enable for cameras with background offset; disable if the camera already outputs background-free data.
   - *Fit mode* — lab axes (σ_x, σ_y) or principal axes (σ_maj, σ_min, θ).
   - *Background parameters* — corner fraction (default 3.5%), n·σ threshold (default 3.0), mask factor (default 3.0 per ISO).
5. **Run** — Click **▶ Run (this file)** or **▶▶ Run all files**.
6. **Export** — Save results as CSV (one row per file) or PDF (one 4-panel figure page per file).

---

## Input file formats

### CSV

Plain text, one pixel value per cell, no header row, no index column (both configurable). Values should be raw ADC counts.

```
12,11,13,14
12,255,254,13
11,12,12,11
```

Tested with 8-bit cameras (0–255) but any bit depth works.

### BeamGage .bgData

Standard HDF5 format produced by Ophir/Spiricon BeamGage software. Pixel values are stored as signed 32-bit fixed-point integers where the MSB of the camera's native N-bit data occupies bit position 30. The loader converts these to ADC counts automatically:

```
raw_counts = int32_value >> (30 - (N - 1))
```

The bit depth (N) is read from the `BITENCODING` dataset (e.g. `"S16"` for signed 16-bit). On loading, the GUI auto-populates:
- Pixel size (µm) from `PIXELSCALEXUM` / `PIXELSCALEYUM`
- vmax from the ADC ceiling (2^N − 1)
- TPA correction is automatically unchecked (BeamGage cameras are linear-response)

Multi-frame files are supported; frame 1 is loaded by default.

---

## Settings reference

| Setting | Default | Description |
|---|---|---|
| Pixel size | 1.0 µm | Camera pixel pitch. Auto-set from .bgData metadata. |
| TPA correction | on | Apply sign(x)·√\|x\| before analysis. Auto-off for .bgData. |
| BG subtraction | on | Estimate and subtract background (ISO/TR 11146-3 §3.4.3). |
| Fit mode | Lab axes | Lab (σ_x, σ_y) or Principal axes (σ_maj, σ_min, θ). |
| vmax | 255 | Colour scale ceiling for the raw image viewer. Auto-set from .bgData bit depth. |
| Corner fraction | 3.5% | Size of corner patches for background estimation (2–5% per ISO §3.4.3). |
| n·σ threshold | 3.0 | Pixels below n·σ_bg are zeroed after BG subtraction (2 < n < 4 per ISO §3.4.2). |
| Mask factor | 3.0 | Integration-area half-width in units of σ (ISO 11146-1 §7 specifies 3.0). |
| Pad σ | 4.0 | Auto-ROI seed half-width in 1/e² beam widths. |
| Saturation threshold | 80% | Fraction of ADC ceiling above which pixels are flagged. |

---

## Mathematics

All D4σ results come from direct numerical integration of the ISO 11146-1 §4 second-moment definitions — not from Gaussian curve fitting. The Gaussian fits visible in the profile plots are for display purposes only.

**Centroid (ISO 11146-3 Eqs. 9–10):**
$$\bar{x} = \frac{\sum_{i,j} I_{ij}\, x_j}{\sum_{i,j} I_{ij}}, \quad \bar{y} = \frac{\sum_{i,j} I_{ij}\, y_i}{\sum_{i,j} I_{ij}}$$

**Second moments (ISO 11146-3 Eqs. 11–13):**
$$\sigma_x^2 = \frac{\sum I_{ij}(x_j-\bar{x})^2}{\sum I_{ij}}, \quad \sigma_{xy} = \frac{\sum I_{ij}(x_j-\bar{x})(y_i-\bar{y})}{\sum I_{ij}}$$

**Principal axes (ISO 11146-3 Eqs. 19–20):**
$$\sigma_\text{maj}^2 = \frac{\sigma_x^2+\sigma_y^2}{2} + \sqrt{\left(\frac{\sigma_x^2-\sigma_y^2}{2}\right)^2+\sigma_{xy}^2}$$

**Orientation angle (ISO 11146-3 Eq. 24):**
$$\varphi = \frac{1}{2}\arctan\!\left(\frac{2\sigma_{xy}}{\sigma_x^2-\sigma_y^2}\right)$$

**D4σ:** $D4\sigma_x = 4\sigma_x$, $D4\sigma_\text{maj} = 4\sigma_\text{maj}$

### Known deviations from ISO/TR 11146-3

The background correction procedure deviates from the standard in two ways, both following the approach of the laserbeamsize library:

**1. Negative noise values (§3.1)**
The standard requires that negative values remaining after background subtraction are kept in the integral, so that positive and negative noise amplitudes cancel. The current implementation zeros pixels below `n·σ_bg`, which introduces a small positive bias in σ for low-SNR measurements. This is a known pragmatic deviation; it has negligible effect when beam width > 0.25× sensor size.

**2. Fine baseline correction (§3.4.2)**
The standard's statistical method uses a 2D convolution (averaging over n×m pixel sub-arrays, Eq. 60) before identifying non-illuminated pixels, in order to avoid digitisation errors. The current implementation uses a direct corner-patch estimate (§3.4.3 approximation method) without the convolution step. This is sufficient for most practical cases and is explicitly permitted by the standard as a first approximation.

### Sources

- **ISO 11146-1:2021** — *Lasers and laser-related equipment — Test methods for laser beam widths, divergence angles and beam propagation ratios — Part 1: Stigmatic and simple astigmatic beams.* §4 (second-moment definitions, Eqs. 24–27), §7 (iterative integration-area procedure).
- **ISO/TR 11146-3:2004** — *Part 3: Intrinsic and geometrical laser beam classification, propagation and details of test methods.* §2.3 (first- and second-order moments, Eqs. 9–13), §2.6 (principal axes, Eqs. 19–24), §3 (background and offset correction).
- **Scott Prahl, laserbeamsize** (https://laserbeamsize.readthedocs.io) — background zeroing approach and iterative mask implementation.
- **Ophir/Spiricon BeamGage** — clip-level (D%pk) width convention at 13.5% of marginal profile peak; .bgData HDF5 fixed-point encoding format.
- **Siegman, A.E. (1998)** — "How to (Maybe) Measure Laser Beam Quality", OSA Annual Meeting tutorial — sub-pixel centroid precision from second-moment integration.

---

## Running the tests

```bash
cd beamprofiler
python -m pytest beamprofiler/tests/ -v
```

38 tests covering `apply_sqrt`, `check_saturation`, `iso_background`, `rotated_rect_mask`, `_moments`, `beam_size_iso`, `auto_roi`, `clip_level_widths`, `marginal_gaussian_fit`, and a TPA round-trip integration test. All tests use synthetic Gaussian beams with known analytical ground truth and verify results to within 1–5% tolerance.

---

## Known limitations and planned improvements

- **Background correction** — current implementation deviates from ISO/TR 11146-3 §3.1 (see above). A fully standard-compliant statistical method (§3.4.2) is planned.
- **No uncertainty estimates** — ISO/TR 11146-3 §5 noise propagation to σ values is not yet implemented.
- **No M² measurement** — single-plane D4σ only; multi-z hyperbolic fit for M² is planned.
- **No session save/load** — ROIs, damage masks, and settings are not persisted between sessions.
- **No TIFF support** — only CSV and .bgData are currently supported; 16-bit TIFF via `tifffile` is planned.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. Please run the test suite before submitting a PR.