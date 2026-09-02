# -*- coding: utf-8 -*-
"""
sim_supergaussian.py
====================
Compares D4sigma width estimation accuracy for super-Gaussian beams of
increasing order n:

    I(r) = exp( -2 * (r/w0)^(2n) )

n=1 is the standard Gaussian; higher n approaches a flat-top / top-hat beam.

Three camera/analysis combinations are compared:
  1. Linear camera      : S = I + B + noise,   fit without TPA correction
  2. TPA uncorr. x√2   : S = I^2 + B + noise, fit without correction, result x√2
  3. TPA corrected      : S = I^2 + B + noise, fit with sqrt correction, pad=5

For each order n the true sigma (second moment of I) is computed numerically
on a large noiseless grid and used as the reference.

Outputs
-------
  sim_supergaussian_results.csv
  sim_supergaussian_profiles.png  -- error bar chart
  sim_supergaussian_beams.png     -- beam profile images
  sim_supergaussian_sigma.png     -- true sigma vs order (sigma_I and sigma_I^2)
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sys.path.insert(0, "..")
from beamprofiler.analysis import (iso_background_statistical,
                                    beam_size_iso, auto_roi)

# ── Parameters ────────────────────────────────────────────────────────────
ADC_MAX         = 255.0
TARGET_ADU      = 0.6 * ADC_MAX
READ_NOISE_ADU  = 0.5
SHOT_NOISE_FRAC = 0.05
N               = 256
N_HALF          = N // 2
N_REAL          = 50
BG_FRAC         = 0.05
W0              = 30    # 1/e^(2/n) radius in pixels for each order

# Super-Gaussian orders to test
SG_ORDERS = [1, 2, 3, 4, 5, 8, 12, 20]

# ── Beam profiles ─────────────────────────────────────────────────────────

def coords(N):
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2 - 0.5
    return np.sqrt((X - cx)**2 + (Y - cy)**2)

def beam_supergauss(N, w0, n):
    """Super-Gaussian order n: I = exp(-2*(r/w0)^(2n))."""
    r = coords(N)
    I = np.exp(-2.0 * (r / w0) ** (2 * n))
    return I / I.max()

# ── True sigma (second moment of I on large noiseless grid) ──────────────

def true_sigma(n, w0, NL=1024):
    """Numerically exact second-moment sigma of super-Gaussian order n."""
    r = coords(NL)
    I = np.exp(-2.0 * (r / w0) ** (2 * n))
    Y, X = np.mgrid[0:NL, 0:NL].astype(float)
    cx = cy = NL / 2 - 0.5
    tot = I.sum()
    xb  = (I * (X - cx)).sum() / tot
    sx  = np.sqrt((I * (X - cx - xb)**2).sum() / tot)
    # also compute sigma of I^2 (what TPA camera sees before correction)
    I2  = I**2; tot2 = I2.sum()
    xb2 = (I2 * (X - cx)).sum() / tot2
    sx2 = np.sqrt((I2 * (X - cx - xb2)**2).sum() / tot2)
    return sx, sx2

print(f"True sigma values for each super-Gaussian order (w0={W0}px):")
print(f"{'n':>4}  {'sigma_I':>9}  {'sigma_I^2':>11}  {'ratio I/I^2':>13}  "
      f"{'pad needed':>12}")
print("-"*60)
true_sigmas  = {}
true_sigmas2 = {}
for n in SG_ORDERS:
    sx, sx2 = true_sigma(n, W0)
    true_sigmas[n]  = sx
    true_sigmas2[n] = sx2
    ratio = sx / sx2
    pad_needed = 3.0 * ratio
    print(f"{n:>4}  {sx:>9.2f}px  {sx2:>11.2f}px  {ratio:>13.3f}  "
          f"{pad_needed:>12.1f}")

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
    """
    pad=5.0 for TPA corrected (I^2 profile is narrower, ROI would be too tight
    with pad=3), pad=3.0 for linear or TPA uncorrected.
    """
    pad = 5.0 if use_tpa else 3.0
    bm, bs = iso_background_statistical(img, n_sigma=3.0)
    try:
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=pad)
        roi = img[y0:y1, x0:x1].astype(float)
        m, _, _ = beam_size_iso(roi, px=1.0, bg_mean=bm, bg_std=bs,
                                 n_sigma=3.0, mask_factor=3.0,
                                 bg_mode="iso_statistical", use_tpa=use_tpa)
        sx = m["sigma_x"]
        if not (0 < sx < N / 3):
            return np.nan
        return sx
    except:
        return np.nan

# ── Simulation loop ───────────────────────────────────────────────────────

bg_adu     = BG_FRAC * TARGET_ADU
noise_std  = SHOT_NOISE_FRAC * np.sqrt(bg_adu) + READ_NOISE_ADU
rng_master = np.random.default_rng(42)
records    = []

analysis_cases = [
    ("Linear camera",         "linear", False, 1.0),
    ("TPA uncorr. \u00d7\u221a2", "tpa",    False, np.sqrt(2)),
    ("TPA corrected",         "tpa",    True,  1.0),
]

print(f"\nRunning simulation (N_REAL={N_REAL}, bg={BG_FRAC:.0%}, w0={W0}px)...")
print(f"{'n':>4}  {'method':25s}  {'err':>8}  {'std':>7}")
print("-"*50)

for n in SG_ORDERS:
    I_norm  = beam_supergauss(N, W0, n)
    sx_true = true_sigmas[n]

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
        records.append(dict(order=n, method=label,
                            sigma_true=sx_true,
                            sigma_I2_true=true_sigmas2[n],
                            ratio=true_sigmas[n]/true_sigmas2[n],
                            err_mean=mn, err_std=sd, n_valid=len(exs)))
        print(f"{n:>4}  {label:25s}  {mn:>+8.3f}  {sd:>7.4f}")

df = pd.DataFrame(records)
df.to_csv("sim_supergaussian_results.csv", index=False)
print("Saved sim_supergaussian_results.csv")

# ── Figure 1: error vs super-Gaussian order ───────────────────────────────
OI = {"green":"#009E73", "orange":"#E69F00", "blue":"#0072B2"}
methods = ["Linear camera", "TPA uncorr. \u00d7\u221a2", "TPA corrected"]
colors  = [OI["green"],    OI["orange"],                OI["blue"]]
markers = ["o",            "s",                         "^"]
lstyles = ["-",            "--",                        "-"]

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
    ax.axhline(0,     color="#888", lw=0.8, ls=":")
    ax.axhline( 0.05, color="#ccc", lw=0.6, ls="--")
    ax.axhline(-0.05, color="#ccc", lw=0.6, ls="--")

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("white"); style_ax(ax)

for method, col, mk, ls in zip(methods, colors, markers, lstyles):
    sub = df[df.method == method].sort_values("order")
    ax.plot(sub["order"], sub["err_mean"],
            color=col, ls=ls, marker=mk, ms=7, lw=1.8,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.8,
            label=method)
    ax.fill_between(sub["order"],
                    sub["err_mean"] - sub["err_std"],
                    sub["err_mean"] + sub["err_std"],
                    color=col, alpha=0.13)

ax.set_xlabel("Super-Gaussian order $n$", fontsize=11)
ax.set_ylabel("Relative $\\sigma_x$ error  ($\\Delta\\sigma/\\sigma_\\mathrm{true}$)",
              fontsize=10)
ax.set_title(
    "D4$\\sigma_x$ error vs super-Gaussian order\n"
    f"$I(r) = \\exp(-2(r/w_0)^{{2n}})$  |  "
    f"ISO stat. BG  |  bg={BG_FRAC:.0%}  |  $w_0$={W0}px  |  ±1σ shaded",
    fontsize=11, fontweight="bold")
ax.set_xticks(SG_ORDERS)
ax.legend(fontsize=10, framealpha=0.95, edgecolor="#cccccc")
plt.tight_layout()
fig.savefig(r"sim_supergaussian_profiles.pdf", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved sim_supergaussian_profiles.png")

# ── Figure 2: beam profile images ────────────────────────────────────────
n_show = [n for n in SG_ORDERS if n <= 12] + ([SG_ORDERS[-1]] if SG_ORDERS[-1] > 12 else [])
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
fig2.patch.set_facecolor("white")
for ax2, n in zip(axes2.ravel(), SG_ORDERS):
    I = beam_supergauss(N, W0, n)
    im = ax2.imshow(I, cmap="inferno", origin="upper", vmin=0, vmax=1,
                    extent=[-N_HALF, N_HALF, -N_HALF, N_HALF])
    ax2.set_title(f"$n={n}$  |  $\\sigma_I$={true_sigmas[n]:.1f}px",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("px", fontsize=8); ax2.set_ylabel("px", fontsize=8)
    ax2.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
plt.suptitle(f"Super-Gaussian beam profiles  |  $w_0$={W0}px",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig2.savefig("sim_supergaussian_beams.pdf", dpi=150,
             bbox_inches="tight", facecolor="white")
print("Saved sim_supergaussian_beams.png")

# ── Figure 3: sigma_I and sigma_I^2 vs order ─────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.patch.set_facecolor("white")

ax3a = axes3[0]; style_ax(ax3a)
orders = np.array(SG_ORDERS)
sig_I  = np.array([true_sigmas[n]  for n in SG_ORDERS])
sig_I2 = np.array([true_sigmas2[n] for n in SG_ORDERS])
ax3a.plot(orders, sig_I,  color=OI["green"], ls="-", marker="o", ms=6,
          lw=1.8, markerfacecolor=OI["green"],
          markeredgecolor="white", markeredgewidth=0.8, label="$\\sigma_I$ (true)")
ax3a.plot(orders, sig_I2, color=OI["blue"],  ls="--", marker="s", ms=6,
          lw=1.8, markerfacecolor=OI["blue"],
          markeredgecolor="white", markeredgewidth=0.8, label="$\\sigma_{I^2}$ (TPA signal)")
ax3a.set_xlabel("Super-Gaussian order $n$", fontsize=11)
ax3a.set_ylabel("$\\sigma$ (px)", fontsize=11)
ax3a.set_title("True $\\sigma$ of $I$ and $I^2$ vs order", fontsize=11,
               fontweight="bold")
ax3a.set_xticks(SG_ORDERS)
ax3a.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc")

ax3b = axes3[1]; style_ax(ax3b)
ratio = sig_I / sig_I2
ax3b.axhline(np.sqrt(2), color=OI["orange"], lw=1.5, ls="--",
             label=f"$\\sqrt{{2}}$ = {np.sqrt(2):.3f}  (Gaussian $n=1$)")
ax3b.plot(orders, ratio, color=OI["blue"], ls="-", marker="o", ms=6,
          lw=1.8, markerfacecolor=OI["blue"],
          markeredgecolor="white", markeredgewidth=0.8,
          label="$\\sigma_I / \\sigma_{{I^2}}$ (actual)")
ax3b.set_xlabel("Super-Gaussian order $n$", fontsize=11)
ax3b.set_ylabel("$\\sigma_I / \\sigma_{I^2}$", fontsize=11)
ax3b.set_title("Ratio determining ROI clipping severity\n"
               "(required pad $\\geq 3 \\times$ ratio)", fontsize=11,
               fontweight="bold")
ax3b.set_xticks(SG_ORDERS)
ax3b.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc")

plt.suptitle(
    "Super-Gaussian beam: $\\sigma$ scaling and ROI implications for TPA cameras",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig3.savefig("sim_supergaussian_sigma.pdf", dpi=150,
             bbox_inches="tight", facecolor="white")
print("Saved sim_supergaussian_sigma.png")