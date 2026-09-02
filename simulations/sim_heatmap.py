# -*- coding: utf-8 -*-
"""
fig_width_heatmap.py
====================
Simulates single-image D4sigma_x width error at the beam waist for three
analysis configurations (Linear, TPA corrected, TPA x sqrt2) and three
background subtraction methods (BG off, Corner §3.4.3, ISO stat. §3.4.2),
across a matrix of background fractions and beam/sensor ratios.

Outputs
-------
  fig_width_heatmap.png
  fig_width_heatmap.pdf
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from beamprofiler.analysis import (iso_background, iso_background_statistical,
                                    beam_size_iso, auto_roi)
from itertools import product

# ── Parameters ────────────────────────────────────────────────────────────
ADC_MAX         = 255.0
TARGET_ADU      = 0.6 * ADC_MAX
READ_NOISE_ADU  = 0.5
SHOT_NOISE_FRAC = 0.05
N               = 256
N_HALF          = N // 2
BG_FRACTIONS    = [0.01, 0.02, 0.05, 0.10]
BEAM_SENSOR     = [0.10, 0.15, 0.20, 0.25]
BG_MODES        = ["off", "corner", "iso_statistical"]
N_REAL          = 50

# colorbar range
VMIN, VMAX = -0.20, 0.20

# font sizes
FS_SUP   = 15
FS_TITLE = 14
FS_LABEL = 14
FS_TICK  = 12
FS_CELL  = 12
FS_CBAR  = 12

# ── Camera models ─────────────────────────────────────────────────────────

def make_img(w0x, w0y, bg, noise, camera, rng):
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N_HALF - 0.5
    dx = (X - cx) * 0.005; dy = (Y - cy) * 0.005
    I = np.exp(-2 * (dx**2/w0x**2 + dy**2/w0y**2))
    sig = TARGET_ADU * I**2 if camera == "tpa" else TARGET_ADU * I
    return np.clip(sig + bg + rng.normal(0, noise, (N, N)), 0, ADC_MAX)

# ── Analysis ──────────────────────────────────────────────────────────────

def analyse(img, bgm, camera, use_tpa):
    """pad=5 for TPA corrected, pad=3 otherwise."""
    pad = 5.0 if use_tpa else 3.0
    if bgm == "corner":
        bm, bs = iso_background(img, corner_frac=0.035)
    elif bgm == "iso_statistical":
        bm, bs = iso_background_statistical(img, n_sigma=3.0)
    else:
        bm, bs = None, None
    try:
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=pad)
        roi = img[y0:y1, x0:x1].astype(float)
        m, _, _ = beam_size_iso(roi, px=0.005, bg_mean=bm, bg_std=bs,
                                 n_sigma=3.0, mask_factor=3.0,
                                 bg_mode=bgm, use_tpa=use_tpa)
        return m["sigma_x"]
    except:
        return np.nan

# ── Simulation ────────────────────────────────────────────────────────────

rng_master = np.random.default_rng(42)
records    = []

for bg_frac, bs in product(BG_FRACTIONS, BEAM_SENSOR):
    w0x = bs * N_HALF * 0.005; w0y = w0x * 0.8
    sx_true = w0x / 2
    bg      = bg_frac * TARGET_ADU
    noise   = SHOT_NOISE_FRAC * np.sqrt(bg) + READ_NOISE_ADU

    for _ in range(N_REAL):
        rng     = np.random.default_rng(rng_master.integers(0, 2**31))
        img_lin = make_img(w0x, w0y, bg, noise, "linear", rng)
        img_tpa = make_img(w0x, w0y, bg, noise, "tpa",    rng)

        for bgm in BG_MODES:
            sl = analyse(img_lin, bgm, "linear", False)
            records.append(dict(config="Linear", bg_mode=bgm,
                bg_frac=bg_frac, beam_sensor=bs,
                err=(sl - sx_true)/sx_true if np.isfinite(sl) else np.nan))

            sc = analyse(img_tpa, bgm, "tpa", True)
            records.append(dict(config="TPA corrected", bg_mode=bgm,
                bg_frac=bg_frac, beam_sensor=bs,
                err=(sc - sx_true)/sx_true if np.isfinite(sc) else np.nan))

            ss = analyse(img_tpa, bgm, "tpa", False)
            records.append(dict(config="TPA \u00d7\u221a2", bg_mode=bgm,
                bg_frac=bg_frac, beam_sensor=bs,
                err=(ss*np.sqrt(2) - sx_true)/sx_true if np.isfinite(ss) else np.nan))

df = pd.DataFrame(records)
print(f"Simulation done: {len(df)} rows")

# ── Plot ───────────────────────────────────────────────────────────────────

norm = plt.Normalize(vmin=VMIN, vmax=VMAX)

# white text thresholds on the jet scale
# norm(-0.10) = 0.25  -> switch to white below -10%
# norm(+0.15) = 0.875 -> switch to white above +15%
WHITE_LO = norm(-0.10)
WHITE_HI = norm( 0.15)

configs    = ["Linear", "TPA corrected", "TPA \u00d7\u221a2"]
config_lbl = {
    "Linear":        "Linear camera",
    "TPA corrected": "TPA corrected  (pad\u2009=\u20095)",
    "TPA \u00d7\u221a2": "TPA \u00d7\u221a2",
}
mode_order = ["off", "corner", "iso_statistical"]
mode_lbl   = {
    "off":              "BG off",
    "corner":           "Corner \u00a73.4.3",
    "iso_statistical":  "ISO stat. \u00a73.4.2",
}

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.patch.set_facecolor("white")

for ri, cfg in enumerate(configs):
    for ci, bgm in enumerate(mode_order):
        ax    = axes[ri, ci]
        pivot = df[(df.config == cfg) & (df.bg_mode == bgm)].groupby(
                    ["beam_sensor", "bg_frac"])["err"].mean().unstack()

        ax.imshow(pivot.values, norm=norm, cmap="jet",
                  aspect="auto", origin="lower")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns],
                           fontsize=FS_TICK)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0%}" for v in pivot.index],
                           fontsize=FS_TICK)
        ax.set_xlabel("Background fraction", fontsize=FS_LABEL)
        ax.set_ylabel("$w_0$ / Sensor half", fontsize=FS_LABEL)
        ax.set_title(f"{config_lbl[cfg]}\n{mode_lbl[bgm]}",
                     fontsize=FS_TITLE, fontweight="bold")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isfinite(v):
                    s = "\u2014"
                elif abs(v) > abs(VMIN):
                    s = f"{v:+.2f}*"
                else:
                    s = f"{v:+.3f}"
                nv    = norm(np.clip(v, VMIN, VMAX))
                white = nv < WHITE_LO or nv > WHITE_HI
                ax.text(j, i, s, ha="center", va="center",
                        fontsize=FS_CELL, fontweight="bold",
                        color="white" if white else "black")

cb   = fig.add_axes([0.93, 0.12, 0.015, 0.75])
sm   = plt.cm.ScalarMappable(cmap="jet", norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, cax=cb)
cbar.set_label(
    "Mean relative $\\sigma_x$ error  ($\\Delta\\sigma / \\sigma$)\n"
    "Blue = underestimate  |  Green \u2248 0  |  Red = overestimate\n"
    "* = outside \u00b120\u2009% range",
    fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_TICK)

'''fig.suptitle(
    "Single-image $\\sigma_x$ width error at beam waist\n"
    "Rows: analysis configuration  |  "
    "Columns: background subtraction method",
    fontsize=FS_SUP, fontweight="bold")'''

plt.tight_layout(rect=[0, 0, 0.92, 0.97])
for fmt in ["pdf", "png"]:
    fig.savefig(f"fig_width_heatmap.{fmt}", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"Saved fig_width_heatmap.{fmt}")