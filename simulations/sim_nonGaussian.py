# -*- coding: utf-8 -*-
"""
sim_nonGaussian.py
==================
Compares D4sigma width estimation accuracy for different beam profiles
using three camera/analysis combinations:

  1. Linear camera     : S = I + B + noise,   fit without TPA correction
  2. TPA uncorr. x√2  : S = I^2 + B + noise, fit without TPA, multiply by √2
  3. TPA corrected     : S = I^2 + B + noise, fit with TPA sqrt correction

Beam profiles tested:
  - Gaussian TEM00
  - LG01 donut
  - LG02
  - HG11 four-lobe
  - Bessel-Gauss

ISO statistical background subtraction used throughout.
bg_frac = 5%, w0 = 30px, N_REAL = 50 realisations.

Outputs
-------
  sim_nonGaussian_results.csv
  sim_nonGaussian_profiles.png
  sim_nonGaussian_beamprofiles.png
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from beamprofiler.analysis import (iso_background_statistical,
                                    beam_size_iso, auto_roi)
from scipy.special import j0

# ── Parameters ────────────────────────────────────────────────────────────
ADC_MAX         = 255.0
TARGET_ADU      = 0.6 * ADC_MAX
READ_NOISE_ADU  = 0.5
SHOT_NOISE_FRAC = 0.05
N               = 256
N_REAL          = 50
BG_FRAC         = 0.05
W0              = 30    # beam 1/e^2 radius in pixels

# ── Beam profiles ─────────────────────────────────────────────────────────

def coords(N):
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2 - 0.5
    dx = X - cx; dy = Y - cy
    return np.sqrt(dx**2 + dy**2), dx, dy

def beam_gauss(N, w):
    r, _, _ = coords(N)
    I = np.exp(-2 * r**2 / w**2)
    return I / I.max()

def beam_LG01(N, w):
    """LG_0^1 donut: I = (r/w)^2 * exp(-2r^2/w^2)."""
    r, _, _ = coords(N)
    I = (r / w)**2 * np.exp(-2 * r**2 / w**2)
    return I / I.max()

def beam_LG02(N, w):
    """LG_0^2: I = (r/w)^4 * exp(-2r^2/w^2)."""
    r, _, _ = coords(N)
    I = (r / w)**4 * np.exp(-2 * r**2 / w**2)
    return I / I.max()

def beam_HG11(N, w):
    """HG_11 four-lobe: I = (xy/w^2)^2 * exp(-2(x^2+y^2)/w^2)."""
    r, dx, dy = coords(N)
    I = (dx * dy / w**2)**2 * np.exp(-2 * (dx**2 + dy**2) / w**2)
    return I / I.max()

def beam_bessel(N, w):
    """Bessel-Gauss: J0^2 * Gaussian envelope."""
    r, _, _ = coords(N)
    kr = 2.4048 / w * 2.5
    I = j0(kr * r)**2 * np.exp(-2 * r**2 / (w * 3)**2)
    I = np.maximum(I, 0)
    return I / I.max()

beams = {
    "Gaussian TEM$_{00}$":   beam_gauss,
    "LG$_{0}^{1}$ (donut)":  beam_LG01,
    "LG$_{0}^{2}$":          beam_LG02,
    "HG$_{11}$ (4-lobe)":    beam_HG11,
    "Bessel-Gauss":          beam_bessel,
}

# ── True sigma (numerical on large noiseless grid) ────────────────────────

def true_sigma(fn, w, NL=1024):
    img = fn(NL, w) * TARGET_ADU
    Y, X = np.mgrid[0:NL, 0:NL].astype(float)
    cx = cy = NL / 2 - 0.5
    tot = img.sum()
    xb  = (img * (X - cx)).sum() / tot
    return np.sqrt((img * (X - cx - xb)**2).sum() / tot)

print("True sigma_x for each beam:")
true_sigmas = {}
for name, fn in beams.items():
    sx = true_sigma(fn, W0)
    true_sigmas[name] = sx
    print(f"  {name:35s}: sigma_x = {sx:.2f} px")

# ── Camera models ─────────────────────────────────────────────────────────

def make_linear(I_norm, bg_adu, noise_std, rng):
    """Linear camera: S = I + B + noise."""
    return np.clip(TARGET_ADU * I_norm + bg_adu
                   + rng.normal(0, noise_std, I_norm.shape), 0, ADC_MAX)

def make_tpa(I_norm, bg_adu, noise_std, rng):
    """TPA camera: S = I^2 + B + noise."""
    return np.clip(TARGET_ADU * I_norm**2 + bg_adu
                   + rng.normal(0, noise_std, I_norm.shape), 0, ADC_MAX)

# ── Analysis ──────────────────────────────────────────────────────────────

def analyse(img, use_tpa):
    """Return sigma_x using ISO statistical BG.
    pad=5.0 for TPA corrected (raw I^2 has narrower auto-ROI estimate),
    pad=3.0 for linear or TPA uncorrected."""
    pad = 5.0 if use_tpa else 3.0
    bm, bs = iso_background_statistical(img, n_sigma=3.0)
    try:
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=pad)
        roi = img[y0:y1, x0:x1].astype(float)
        m, _, _ = beam_size_iso(roi, px=1.0, bg_mean=bm, bg_std=bs,
                                 n_sigma=3.0, mask_factor=3.0,
                                 bg_mode="iso_statistical", use_tpa=use_tpa)
        return m["sigma_x"]
    except:
        return np.nan

# ── Simulation loop ───────────────────────────────────────────────────────

bg_adu     = BG_FRAC * TARGET_ADU
noise_std  = SHOT_NOISE_FRAC * np.sqrt(bg_adu) + READ_NOISE_ADU
rng_master = np.random.default_rng(42)
records    = []

# (label, camera_type, use_tpa_in_fit, scale_factor)
analysis_cases = [
    ("Linear camera",    "linear", False, 1.0),
    ("TPA uncorr. \u00d7\u221a2", "tpa",    False, np.sqrt(2)),
    ("TPA corrected",    "tpa",    True,  1.0),
]

print("\nRunning simulation...")
print(f"{'Beam':35s}  {'Method':25s}  {'err':>8}  {'std':>7}")
print("-"*80)

for beam_name, beam_fn in beams.items():
    I_norm  = beam_fn(N, W0)
    sx_true = true_sigmas[beam_name]

    for label, cam_type, use_tpa, scale in analysis_cases:
        exs = []
        for _ in range(N_REAL):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            img = (make_linear(I_norm, bg_adu, noise_std, rng)
                   if cam_type == "linear"
                   else make_tpa(I_norm, bg_adu, noise_std, rng))
            sx = analyse(img, use_tpa=use_tpa)
            if np.isfinite(sx):
                exs.append((sx * scale - sx_true) / sx_true)

        mn = np.nanmean(exs); sd = np.nanstd(exs)
        records.append(dict(beam=beam_name, method=label,
                            err_mean=mn, err_std=sd, n=len(exs)))
        print(f"{beam_name:35s}  {label:25s}  {mn:+8.3f}  {sd:7.4f}")

df = pd.DataFrame(records)
df.to_csv("sim_nonGaussian_results.csv", index=False)
print("Saved sim_nonGaussian_results.csv")

# ── Figure 1: error bar chart ─────────────────────────────────────────────
OI = {"green":"#009E73", "orange":"#E69F00", "blue":"#0072B2"}
methods = ["Linear camera", "TPA uncorr. \u00d7\u221a2", "TPA corrected"]
colors  = [OI["green"], OI["orange"], OI["blue"]]
beam_names = list(beams.keys())
x = np.arange(len(beam_names))
width = 0.25

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
ax.tick_params(direction="in", top=True, right=True, labelsize=10)
ax.grid(True, ls="--", lw=0.5, color="#ddd", axis="y", zorder=0)
ax.axhline(0,     color="#888", lw=0.8, ls=":")
ax.axhline( 0.05, color="#ccc", lw=0.6, ls="--")
ax.axhline(-0.05, color="#ccc", lw=0.6, ls="--")

for i, (method, col) in enumerate(zip(methods, colors)):
    sub = df[df.method == method].reset_index(drop=True)
    offsets = x + (i - 1) * width
    ax.bar(offsets, sub["err_mean"], width=width * 0.85,
           color=col, alpha=0.85, zorder=3, label=method)
    ax.errorbar(offsets, sub["err_mean"], yerr=sub["err_std"],
                fmt="none", color="#333", capsize=3, lw=1.2, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(beam_names, fontsize=10)
ax.set_ylabel(
    "Relative $\\sigma_x$ error  ($\\Delta\\sigma/\\sigma_\\mathrm{true}$)",
    fontsize=11)
ax.set_title(
    "D4$\\sigma_x$ error: linear vs TPA camera, different beam profiles\n"
    f"ISO stat. BG  |  bg = {BG_FRAC:.0%}  |  $w_0$ = {W0}px  |"
    "  bars: mean \u00b1 1\u03c3",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc")
plt.tight_layout()
fig.savefig(r"sim_nonGaussian_profiles.pdf", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved sim_nonGaussian_profiles.png")

# ── Figure 2: beam profile images ─────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4))
fig2.patch.set_facecolor("white")
for ax2, (name, fn) in zip(axes2, beams.items()):
    ax2.imshow(fn(N, W0), cmap="hot", origin="upper")
    ax2.set_title(name, fontsize=16, fontweight="bold")
    ax2.axis("off")
plt.tight_layout()
fig2.savefig(r"sim_nonGaussian_beamprofiles.pdf", dpi=150,
             bbox_inches="tight", facecolor="white")
print("Saved sim_nonGaussian_beamprofiles.png")