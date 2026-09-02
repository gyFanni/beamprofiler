# -*- coding: utf-8 -*-
"""
sim_beam_widths.py
==================
Simulates Gaussian beams with 2*sigma from 2 to 30 px, detected by a
linear and a TPA camera. Images are saved as CSVs, then loaded and
analysed with D4sigma corner background subtraction.

Beam widths:
  - 2.0 to 5.0 px in 0.5 px steps
  - 5.0 to 10.0 px in 1.0 px steps
  - 11.0 to 29.0 px in 3.0 px steps  (original coarse sweep)

Outputs
-------
  sim_beams/beam_lin_w2s*.csv     -- linear camera images
  sim_beams/beam_tpa_w2s*.csv     -- TPA camera images
  sim_beams/beam_metadata.csv     -- beam parameters
  sim_beams/analysis_results.csv  -- D4sigma fit results
  sim_beam_width_comparison.png   -- figure
"""
import sys, numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sys.path.insert(0, "..")
from beamprofiler.analysis import (iso_background, beam_size_iso, auto_roi)

# ── Parameters ────────────────────────────────────────────────────────────
ADC_MAX         = 255.0
TARGET_ADU      = 0.7 * ADC_MAX
READ_NOISE_ADU  = 0.5
SHOT_NOISE_FRAC = 0.05
N               = 256
N_HALF          = N // 2
BG_FRAC         = 0.1

# Build full width list without duplicates
widths_fine  = [round(x * 2) / 2 for x in np.arange(1.0, 6.5, 0.1)]  # 2.0..5.0
widths_mid   = list(range(7, 15, 1))                                   # 6..10
widths_coarse= list(range(16, 40, 6))                                  # 11,14,...,29
WIDTHS_2SIGMA = sorted(set(widths_fine + widths_mid + widths_coarse))
print(f"2*sigma values ({len(WIDTHS_2SIGMA)} total): {WIDTHS_2SIGMA}")

# ── Generate images ───────────────────────────────────────────────────────
os.makedirs("sim_beams", exist_ok=True)
rng = np.random.default_rng(42)
bg_adu    = BG_FRAC * TARGET_ADU
noise_std = SHOT_NOISE_FRAC * np.sqrt(bg_adu) + READ_NOISE_ADU

Y, X = np.mgrid[0:N, 0:N].astype(float)
cx = cy = N_HALF - 0.5

meta = []
print("\nGenerating images...")
for w2s in WIDTHS_2SIGMA:
    sigma_px = w2s / 2.0     # true sigma
    w0_px    = float(w2s)    # 1/e^2 radius = 2*sigma

    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    I = np.exp(-2 * r**2 / w0_px**2)   # normalised Gaussian, peak = 1

    # Linear camera: S = I + B + noise
    img_lin = np.clip(TARGET_ADU * I    + bg_adu + rng.normal(0, noise_std, (N, N)),
                      0, ADC_MAX)
    # TPA camera: S = I^2 + B + noise
    img_tpa = np.clip(TARGET_ADU * I**2 + bg_adu + rng.normal(0, noise_std, (N, N)),
                      0, ADC_MAX)

    # filename tag: replace "." with "p" for half-pixel values
    tag = f"{w2s:.1f}".replace(".", "p")
    fname_lin = f"sim_beams/beam_lin_w2s{tag}px.csv"
    fname_tpa = f"sim_beams/beam_tpa_w2s{tag}px.csv"

    pd.DataFrame(np.round(img_lin).astype(int)).to_csv(fname_lin,
                                                        header=False, index=False)
    pd.DataFrame(np.round(img_tpa).astype(int)).to_csv(fname_tpa,
                                                        header=False, index=False)
    meta.append(dict(w2sigma_px=w2s, sigma_px=sigma_px, w0_px=w0_px,
                     sigma_true_px=sigma_px,
                     file_linear=fname_lin, file_tpa=fname_tpa))
    print(f"  2*sigma={w2s:5.1f}px  sigma={sigma_px:5.2f}px  "
          f"peak_lin={img_lin.max():.0f}  peak_tpa={img_tpa.max():.0f}")

pd.DataFrame(meta).to_csv("sim_beams/beam_metadata.csv", index=False)
print(f"\nSaved {len(WIDTHS_2SIGMA) * 2} CSVs + beam_metadata.csv")

# ── Analyse: load CSVs and run D4sigma fits ───────────────────────────────
print("\nAnalysing images...")
results = []

for row in meta:
    w2s     = row["w2sigma_px"]
    sx_true = row["sigma_true_px"]

    for cam, fname, use_tpa, pad in [
        # pad=3 for linear (standard), pad=5 for TPA (compensates for
        # auto-ROI estimating sigma from I^2 which is narrower than I)
        ("linear",        row["file_linear"], False, 3.0),
        ("tpa_corrected", row["file_tpa"],    True,  5.0),
    ]:
        img = pd.read_csv(fname, header=None).values.astype(float)

        # Corner background from full image
        bm, bs = iso_background(img, corner_frac=0.035)

        try:
            x0, x1, y0, y1 = auto_roi(img, pad_sigma=pad)
            roi = img[y0:y1, x0:x1].astype(float)
            m, _, _ = beam_size_iso(roi, px=1.0, bg_mean=bm, bg_std=bs,
                                     n_sigma=3.0, mask_factor=3.0,
                                     bg_mode="corner", use_tpa=use_tpa)
            sx_est = m["sigma_x"]
            if not (0 < sx_est < N / 3):
                sx_est = np.nan
        except:
            sx_est = np.nan

        err = (sx_est - sx_true) / sx_true if np.isfinite(sx_est) else np.nan
        results.append(dict(
            w2sigma_px   = w2s,
            sigma_true_px= sx_true,
            d4sigma_true = 4 * sx_true,
            camera       = cam,
            sigma_est_px = sx_est,
            d4sigma_est  = 4 * sx_est if np.isfinite(sx_est) else np.nan,
            err_sigma    = err,
        ))
        print(f"  2*sigma={w2s:5.1f}px  {cam:15s}: "
              f"sigma_est={sx_est:.3f}px  err={err:+.3f}" if np.isfinite(sx_est)
              else f"  2*sigma={w2s:5.1f}px  {cam:15s}: FAILED")

df = pd.DataFrame(results)
df.to_csv("sim_beams/analysis_results.csv", index=False)
print(f"\nSaved analysis_results.csv  ({len(df)} rows)")

# ── Plot ───────────────────────────────────────────────────────────────────
OI = {"green":"#009E73", "blue":"#0072B2"}
cam_style = {
    "linear":        (OI["green"], "-",  "o", "Linear camera"),
    "tpa_corrected": (OI["blue"],  "--", "s", "TPA corrected (pad=5)"),
}

def style_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=10)
    ax.tick_params(which="minor", direction="in", length=3,
                   color="#aaa", top=True, right=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, ls="--", lw=0.5, color="#ddd", zorder=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("white")

# Panel 1: estimated vs true sigma
ax = axes[0]; style_ax(ax)
x_range = np.linspace(0, df["sigma_true_px"].max() * 1.05, 100)
ax.plot(x_range, x_range, "k--", lw=1.2, alpha=0.4, label="Identity (perfect)")
for cam, (col, ls, mk, lbl) in cam_style.items():
    sub = df[df.camera == cam].sort_values("sigma_true_px")
    ax.plot(sub["sigma_true_px"], sub["sigma_est_px"],
            color=col, ls=ls, marker=mk, ms=5, lw=1.8,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.8,
            label=lbl)
ax.set_xlabel("True $\\sigma$ (px)", fontsize=11)
ax.set_ylabel("Estimated $\\sigma$ (px)", fontsize=11)
ax.set_title("Estimated vs true $\\sigma$", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc")

# Panel 2: relative error vs 2*sigma
ax2 = axes[1]; style_ax(ax2)
ax2.axhline(0,     color="#888", lw=0.8, ls=":")
ax2.axhline( 0.05, color="#ccc", lw=0.6, ls="--")
ax2.axhline(-0.05, color="#ccc", lw=0.6, ls="--")
for cam, (col, ls, mk, lbl) in cam_style.items():
    sub = df[df.camera == cam].sort_values("w2sigma_px")
    ax2.plot(sub["w2sigma_px"], sub["err_sigma"],
             color=col, ls=ls, marker=mk, ms=5, lw=1.8,
             markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.8,
             label=lbl)
ax2.set_xlabel("True beam width $2\\sigma$ (px)", fontsize=11)
ax2.set_ylabel("Relative error  $(\\hat{\\sigma} - \\sigma)/\\sigma$", fontsize=11)
ax2.set_title("Relative $\\sigma$ error vs beam size", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc")

fig.suptitle(
    "D4$\\sigma$ fit from simulated camera images  |  corner BG  |  "
    f"bg = {BG_FRAC:.0%}  |  8-bit 256\u00d7256 px",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("sim_beam_width_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved sim_beam_width_comparison.png")
