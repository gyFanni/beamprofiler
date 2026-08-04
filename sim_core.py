# -*- coding: utf-8 -*-
"""
sim_core.py
===========
Runs all three simulations and saves results as CSV files.
No plotting is done here.

Run this first:
    python sim_core.py

Then generate figures:
    python sim_plots.py

Outputs
-------
    sim_results.csv          -- M² fit error across full z-scan
    sim_images/              -- example camera image CSVs
    sim_width_results.csv    -- single-image width error at beam waist
    sim_zdep_results.csv     -- width error vs z-position
"""

from __future__ import annotations
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beamprofiler.analysis import (
    iso_background, iso_background_statistical, beam_size_iso, auto_roi)

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
LAM      = 1.750e-3   # mm  (1750 nm)
PX       = 0.005    # mm  (5.0 µm pixel pitch)
ADC_BITS = 8
ADC_MAX  = float(2**ADC_BITS - 1)   # 255
N        = 256                        # sensor size (pixels)
N_HALF   = N // 2                     # sensor half-width

# Discrete exposure model
# Camera offers exposure in doubling steps. At each z-position the exposure
# is chosen to bring the beam peak close to TARGET_ADU. Background (dark
# current) scales linearly with exposure. Noise = shot noise on background
# + fixed read noise.
EXPOSURE_STEPS = [1, 2, 4, 8, 16, 32, 64, 128]   # relative to base exposure
TARGET_ADU     = 0.6 * ADC_MAX                     # desired peak ADC level
BASE_BG_FRAC   = None   # set per condition (= bg_frac * TARGET_ADU at exp=1)
READ_NOISE_ADU = 0.5    # fixed read noise in ADC counts (independent of exposure)
SHOT_NOISE_FRAC = 0.05  # shot noise on background = this fraction * sqrt(bg_adu)

# True beam parameters
M2X_TRUE = 1.3        # M² along x
M2Y_TRUE = 1.6        # M² along y  (astigmatic)
W0Y_W0X  = 0.8        # w0y / w0x ratio

# Sweep
BG_FRACTIONS = [0.01, 0.02, 0.05, 0.10]
BEAM_SENSOR  = [0.10, 0.15, 0.20, 0.25]
BG_MODES     = ["off", "corner", "iso_statistical"]
CAMERA_TYPES = ["linear", "tpa"]
N_REAL       = 15     # noise realisations per condition

# ═══════════════════════════════════════════════════════════════════════════
# BEAM MODEL
# ═══════════════════════════════════════════════════════════════════════════

def rayleigh(w0, M2):
    """Rayleigh length [mm]."""
    return np.pi * w0**2 / (M2 * LAM)

def w_at_z(z, w0, M2, z0):
    """1/e^2 beam radius at position z [mm]."""
    return w0 * np.sqrt(1.0 + ((z - z0) / rayleigh(w0, M2))**2)

def exposure_for_z(z, w0x, w0y, M2x, M2y, z0x, z0y, camera,
                   base_bg_adu=None, n_sigma_sat=3.0):
    """
    Choose the discrete exposure step that brings the beam peak closest to
    TARGET_ADU without the brightest pixel saturating.

    The saturation check includes background and noise:
        peak(exp) + bg(exp) + n_sigma_sat * noise(exp) <= ADC_MAX

    where:
        peak(exp)  = peak_at_exp1 * exp
        bg(exp)    = base_bg_adu * exp
        noise(exp) = SHOT_NOISE_FRAC * sqrt(bg(exp)) + READ_NOISE_ADU

    base_bg_adu: background at exp=1. If None, assumed to be 0 (conservative
                 default keeps behaviour when bg is not known at call time;
                 callers should pass the actual base_bg for correctness).
    """
    wx = w_at_z(z, w0x, M2x, z0x)
    wy = w_at_z(z, w0y, M2y, z0y)
    geom = (w0x * w0y) / (wx * wy)

    if camera == "tpa":
        peak_at_exp1 = TARGET_ADU * geom**2
    else:
        peak_at_exp1 = TARGET_ADU * geom

    if peak_at_exp1 <= 0:
        return 1

    bg0 = base_bg_adu if base_bg_adu is not None else 0.0

    def worst_pixel(step):
        bg_z    = bg0 * step
        noise_z = SHOT_NOISE_FRAC * np.sqrt(max(bg_z, 0.)) + READ_NOISE_ADU
        return peak_at_exp1 * step + bg_z + n_sigma_sat * noise_z

    ideal = TARGET_ADU / peak_at_exp1

    valid = [s for s in EXPOSURE_STEPS if worst_pixel(s) <= ADC_MAX]
    if not valid:
        return 1
    return min(valid, key=lambda s: abs(s - ideal))


def make_z_array(z_mean, zR_min, n_near=3, n_far=3):
    """
    ISO 11146-1 recommended z sampling:
      n_near positions within 0.8 Rayleigh lengths of the waist,
      n_far  positions at 2.1-2.5 Rayleigh lengths on each side.
    """
    near   = np.linspace(z_mean - 0.8*zR_min, z_mean + 0.8*zR_min, n_near)
    far_lo = np.linspace(z_mean - 2.5*zR_min, z_mean - 2.1*zR_min, n_far)
    far_hi = np.linspace(z_mean + 2.1*zR_min, z_mean + 2.5*zR_min, n_far)
    return np.sort(np.concatenate([far_lo, near, far_hi]))

def make_image(z, w0x, w0y, M2x, M2y, z0x, z0y,
               peak_adu, bg_adu, noise_std, camera, rng):
    """
    Synthetic camera image at propagation distance z.

    camera="tpa":    S = peak_adu * I^2 + bg_adu + noise
    camera="linear": S = peak_adu * I   + bg_adu + noise

    peak_adu, bg_adu, noise_std are pre-computed by the caller using the
    discrete exposure model (exposure_for_z), so they vary per z-position.
    I is normalised so peak = 1 at beam centre; the exposure scaling is
    already absorbed into peak_adu.
    """
    Y, X   = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N_HALF - 0.5
    dx = (X - cx) * PX
    dy = (Y - cy) * PX
    wx = w_at_z(z, w0x, M2x, z0x)
    wy = w_at_z(z, w0y, M2y, z0y)
    I  = np.exp(-2.0 * (dx**2/wx**2 + dy**2/wy**2))
    sig = peak_adu * I**2 if camera == "tpa" else peak_adu * I
    return np.clip(sig + bg_adu + rng.normal(0, noise_std, (N, N)), 0, ADC_MAX)

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_image(img, bg_mode, use_tpa, bg_mean=None, bg_std=None):
    """
    Run beam_size_iso with auto ROI and return (sigma_x, sigma_y) in mm.
    Auto ROI crops to the relevant beam region before analysis.
    """
    try:
        # Get auto ROI bounds
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=3.0)
        roi_img = img[y0:y1, x0:x1].astype(float)

        m, _, _ = beam_size_iso(
            roi_img, px=PX,
            bg_mean=bg_mean, bg_std=bg_std,
            n_sigma=3.0, mask_factor=3.0,
            bg_mode=bg_mode, use_tpa=use_tpa)
        return m["sigma_x"], m["sigma_y"]
    except Exception:
        return np.nan, np.nan

def fit_m2(z, d4s):
    """
    Fit d4sigma(z)^2 = A + B*(z-z0)^2  ->  M^2, w0, z0.

    z and d4s must be in consistent units (both mm here).
    z is centred before fitting to prevent z0 going out of bounds
    when absolute z values are large (waist at 50*zR >> 500 mm).
    """
    mask = np.isfinite(d4s) & (d4s > 0)
    if mask.sum() < 5:
        return None
    zg, dg = z[mask], d4s[mask]
    z_off  = zg.mean()
    zg_c   = zg - z_off
    i0     = np.argmin(dg)
    dz_max = np.abs(zg_c).max()
    try:
        p, _ = curve_fit(
            lambda z, A, B, z0: A + B*(z - z0)**2,
            zg_c, dg**2,
            p0=[dg[i0]**2,
                max((dg.max()**2 - dg[i0]**2) /
                    max((zg_c.max() - zg_c[i0])**2, 1e-9), 1e-12),
                zg_c[i0]],
            maxfev=20000,
            bounds=([0, 0, -2*dz_max], [1e6, 1e6, 2*dz_max]))
        A, B, z0c = p
        if A <= 0 or B <= 0:
            return None
        M2 = (np.pi / (4*LAM)) * np.sqrt(A * B)
        return dict(M2=M2, w0=np.sqrt(A)/2, z0=z0c + z_off)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

print("Starting simulation...")
t_start = time.perf_counter()
records = []
rng_master = np.random.default_rng(42)
n_outer = len(BG_FRACTIONS) * len(BEAM_SENSOR) * len(CAMERA_TYPES)
done = 0

for bg_frac, bs, camera in product(BG_FRACTIONS, BEAM_SENSOR, CAMERA_TYPES):
    done += 1
    print(f"  [{done}/{n_outer}] bg={bg_frac:.0%}  bs={bs:.2f}  cam={camera}")

    w0x    = bs * N_HALF * PX
    w0y    = w0x * W0Y_W0X
    zR_min = min(rayleigh(w0x, M2X_TRUE), rayleigh(w0y, M2Y_TRUE))
    z0x    = 50 * zR_min              # waist far from z=0 to have room for scan
    z0y    = z0x + 2 * zR_min         # astigmatic separation = 2 Rayleigh lengths
    z_mean = (z0x + z0y) / 2
    z_arr  = make_z_array(z_mean, zR_min)

    # bg_frac is defined relative to TARGET_ADU at base exposure (exp=1)
    base_bg_adu = bg_frac * TARGET_ADU

    for real_i in range(N_REAL):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))

        # Generate all images for this realisation once, reuse across bg_modes.
        # Each z-position gets its own exposure step.
        images = []
        for z in z_arr:
            exp = exposure_for_z(z, w0x, w0y, M2X_TRUE, M2Y_TRUE,
                                 z0x, z0y, camera,
                                 base_bg_adu=base_bg_adu)
            wx = w_at_z(z, w0x, M2X_TRUE, z0x)
            wy = w_at_z(z, w0y, M2Y_TRUE, z0y)
            geom = (w0x * w0y) / (wx * wy)
            if camera == "tpa":
                peak_adu_z = TARGET_ADU * geom**2 * exp
            else:
                peak_adu_z = TARGET_ADU * geom * exp
            bg_adu_z   = base_bg_adu * exp
            # noise: shot noise on background + fixed read noise
            noise_std_z = SHOT_NOISE_FRAC * np.sqrt(bg_adu_z) + READ_NOISE_ADU
            images.append(make_image(z, w0x, w0y, M2X_TRUE, M2Y_TRUE, z0x, z0y,
                                     peak_adu_z, bg_adu_z, noise_std_z,
                                     camera, rng))

        for bg_mode in BG_MODES:
            d4sx = np.full(len(z_arr), np.nan)
            d4sy = np.full(len(z_arr), np.nan)

            for zi, img in enumerate(images):
                if bg_mode == "corner":
                    bm, bs2 = iso_background(img, corner_frac=0.035)
                elif bg_mode == "iso_statistical":
                    bm, bs2 = iso_background_statistical(img, n_sigma=3.0)
                else:
                    bm, bs2 = None, None

                sx, sy = analyse_image(img, bg_mode,
                                       use_tpa=(camera == "tpa"),
                                       bg_mean=bm, bg_std=bs2)
                d4sx[zi] = 4 * sx
                d4sy[zi] = 4 * sy

            for ax_i, (d4s, M2_true, w0_true) in enumerate([
                    (d4sx, M2X_TRUE, w0x),
                    (d4sy, M2Y_TRUE, w0y)]):
                fit   = fit_m2(z_arr, d4s)
                err_M2 = ((fit["M2"] - M2_true) / M2_true
                          if fit and np.isfinite(fit["M2"]) else np.nan)
                records.append(dict(
                    camera     = camera,
                    bg_mode    = bg_mode,
                    bg_frac    = bg_frac,
                    beam_sensor= bs,
                    axis       = ["x", "y"][ax_i],
                    realisation= real_i,
                    M2_true    = M2_true,
                    M2_est     = fit["M2"] if fit else np.nan,
                    w0_true_um = w0_true * 1e3,
                    w0_est_um  = fit["w0"] * 1e3 if fit else np.nan,
                    err_M2     = err_M2,
                ))

elapsed = time.perf_counter() - t_start
print(f"Simulation done in {elapsed:.0f} s")

df = pd.DataFrame(records)
df.to_csv("sim_results.csv", index=False)
print(f"Saved sim_results.csv  ({len(df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE EXAMPLE CAMERA IMAGES
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs("sim_images", exist_ok=True)
saved = 0
rng_img = np.random.default_rng(99)

for bg_frac, bs, camera in [(0.05, 0.20, "linear"), (0.05, 0.20, "tpa")]:
    w0x    = bs * N_HALF * PX
    w0y    = w0x * W0Y_W0X
    zR_min = min(rayleigh(w0x, M2X_TRUE), rayleigh(w0y, M2Y_TRUE))
    z0x    = 50 * zR_min
    z0y    = z0x + 3 * zR_min
    z_arr  = make_z_array((z0x+z0y)/2, zR_min)
    base_bg_adu = bg_frac * TARGET_ADU

    for zi, z in enumerate(z_arr):
        exp  = exposure_for_z(z, w0x, w0y, M2X_TRUE, M2Y_TRUE, z0x, z0y, camera,
                             base_bg_adu=base_bg_adu)
        wx   = w_at_z(z, w0x, M2X_TRUE, z0x)
        wy   = w_at_z(z, w0y, M2Y_TRUE, z0y)
        geom = (w0x * w0y) / (wx * wy)
        if camera == "tpa":
            peak_adu_z = TARGET_ADU * geom**2 * exp
        else:
            peak_adu_z = TARGET_ADU * geom * exp
        bg_adu_z   = base_bg_adu * exp
        noise_std_z = SHOT_NOISE_FRAC * np.sqrt(bg_adu_z) + READ_NOISE_ADU
        img = make_image(z, w0x, w0y, M2X_TRUE, M2Y_TRUE, z0x, z0y,
                         peak_adu_z, bg_adu_z, noise_std_z, camera, rng_img)
        fname = (f"sim_images/cam_{camera}_bg{int(bg_frac*100)}pct"
                 f"_bs{int(bs*100)}_z{zi:02d}.csv")
        pd.DataFrame(np.round(img).astype(int)).to_csv(
            fname, header=False, index=False)
        saved += 1

print(f"Saved {saved} image CSVs to sim_images/")



# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-Z WIDTH SIMULATION  (beam waist, fixed exposure)
# ═══════════════════════════════════════════════════════════════════════════
# At a fixed z-position (beam waist) with constant exposure, how do the
# fitted beam widths (D4sigma and clip-level) depend on:
#   - background fraction
#   - beam/sensor size ratio
#   - background subtraction method
#   - camera type (linear vs TPA)
# ═══════════════════════════════════════════════════════════════════════════

print("\nRunning single-z width simulation...")
t_width = time.perf_counter()

from beamprofiler.analysis import clip_level_widths

N_REAL_W = 30   # realisations for width simulation

def make_image_fixed(w0x, w0y, peak_adu, bg_adu, noise_std, camera, rng):
    """Single image at the beam waist (z=z0), no exposure scaling."""
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N_HALF - 0.5
    dx = (X - cx) * PX
    dy = (Y - cy) * PX
    I  = np.exp(-2.0 * (dx**2/w0x**2 + dy**2/w0y**2))
    sig = peak_adu * I**2 if camera == "tpa" else peak_adu * I
    return np.clip(sig + bg_adu + rng.normal(0, noise_std, (N, N)), 0, ADC_MAX)

# ROI strategies for corner method comparison:
#   "full_sensor" : no auto ROI; background and analysis on full image
#                  (pure ISO §3.4.3 as written)
#   "roi_fullbg"  : auto ROI for analysis; background from full-sensor corners
#                  (current default — protects corner estimate but crops moments)
#   "roi_cropbg"  : auto ROI for both analysis and background estimation
#                  (corners of the crop; least standard-compliant)
# For iso_statistical and off the ROI strategy is always roi_fullbg (canonical).

ROI_STRATEGIES = ["full_sensor", "roi_fullbg", "roi_cropbg"]

rng_w = np.random.default_rng(123)
width_records = []

for bg_frac, bs, camera in product(BG_FRACTIONS, BEAM_SENSOR, CAMERA_TYPES):
    w0x = bs * N_HALF * PX
    w0y = w0x * W0Y_W0X
    peak_adu  = TARGET_ADU
    bg_adu    = bg_frac * TARGET_ADU
    noise_std = SHOT_NOISE_FRAC * np.sqrt(bg_adu) + READ_NOISE_ADU

    sigma_x_true = w0x / 2
    sigma_y_true = w0y / 2

    for real in range(N_REAL_W):
        rng = np.random.default_rng(rng_w.integers(0, 2**31))
        img = make_image_fixed(w0x, w0y, peak_adu, bg_adu, noise_std, camera, rng)

        # compute ROI once per image (same ROI reused across strategies)
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=3.0)
        roi = img[y0:y1, x0:x1].astype(float)

        # background estimates
        bm_full, bs2_full = iso_background(img,  corner_frac=0.035)  # full sensor corners
        bm_crop, bs2_crop = iso_background(roi,  corner_frac=0.035)  # crop corners
        bm_iso,  bs2_iso  = iso_background_statistical(img, n_sigma=3.0)

        # define (bg_mode, roi_strategy, img_to_analyse, bm, bs2) tuples
        analysis_cases = [
            # corner method — three ROI strategies
            ("corner", "full_sensor", img.astype(float), bm_full, bs2_full),
            ("corner", "roi_fullbg",  roi,               bm_full, bs2_full),
            ("corner", "roi_cropbg",  roi,               bm_crop, bs2_crop),
            # iso_statistical — canonical (full bg, with ROI)
            ("iso_statistical", "roi_fullbg", roi, bm_iso, bs2_iso),
            # off — canonical (no bg, with ROI)
            ("off", "roi_fullbg", roi, None, None),
        ]

        for bg_mode, roi_strat, img_to_use, bm, bs2 in analysis_cases:
            try:
                m, bg_img, _ = beam_size_iso(
                    img_to_use, px=PX, bg_mean=bm, bg_std=bs2,
                    n_sigma=3.0, mask_factor=3.0,
                    bg_mode=bg_mode, use_tpa=(camera == "tpa"))
                cl = clip_level_widths(bg_img, px=PX, clip_frac=0.135,
                                       theta_rad=m["theta_rad"],
                                       xc_px=m["x_bar"]/PX,
                                       yc_px=m["y_bar"]/PX)
                width_records.append(dict(
                    camera=camera, bg_mode=bg_mode, roi_strategy=roi_strat,
                    bg_frac=bg_frac, beam_sensor=bs, realisation=real,
                    d4sx=m["sigma_x"]*4,      d4sy=m["sigma_y"]*4,
                    d4sx_true=sigma_x_true*4, d4sy_true=sigma_y_true*4,
                    err_d4sx=(m["sigma_x"]-sigma_x_true)/sigma_x_true,
                    err_d4sy=(m["sigma_y"]-sigma_y_true)/sigma_y_true,
                    clip_x=cl["clip_x"],      clip_y=cl["clip_y"],
                    clip_x_true=sigma_x_true*4,
                    clip_y_true=sigma_y_true*4,
                    err_clip_x=(cl["clip_x"]-sigma_x_true*4)/(sigma_x_true*4),
                    err_clip_y=(cl["clip_y"]-sigma_y_true*4)/(sigma_y_true*4),
                ))
            except Exception:
                width_records.append(dict(
                    camera=camera, bg_mode=bg_mode, roi_strategy=roi_strat,
                    bg_frac=bg_frac, beam_sensor=bs, realisation=real,
                    d4sx=np.nan, d4sy=np.nan,
                    d4sx_true=sigma_x_true*4, d4sy_true=sigma_y_true*4,
                    err_d4sx=np.nan, err_d4sy=np.nan,
                    clip_x=np.nan, clip_y=np.nan,
                    clip_x_true=sigma_x_true*4, clip_y_true=sigma_y_true*4,
                    err_clip_x=np.nan, err_clip_y=np.nan,
                ))

dfw = pd.DataFrame(width_records)
dfw.to_csv("sim_width_results.csv", index=False)
print(f"Single-z done in {time.perf_counter()-t_width:.0f}s  "
      f"|  {len(dfw)} rows  ->  sim_width_results.csv")

# ── style dicts needed by heatmaps below ──────────────────────────────────
OI = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00"}
cam_title  = {"linear":"Linear camera", "tpa":"TPA camera"}
mode_style = {
    "off":             (OI["blue"],  "-",  "o", "BG off"),
    "corner":          (OI["orange"],"--", "s", "Corner (\u00a73.4.3)"),
    "iso_statistical": (OI["green"], "-",  "^", "ISO stat. (\u00a73.4.2)"),
}

# style helpers also needed here
def style_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.tick_params(which="minor", direction="in", length=3, color="#aaaaaa",
                   top=True, right=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, ls="--", lw=0.5, color="#dddddd", zorder=0)
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.axhline( 0.1, color="#ccc", lw=0.6, ls="--")
    ax.axhline(-0.1, color="#ccc", lw=0.6, ls="--")

cam_col = {"linear": OI["blue"], "tpa": OI["orange"]}
cam_ls  = {"linear": "-",        "tpa": "--"}

# ── Heatmaps: one 2x3 figure per metric, plus combined ISO-stat figure ─────
metric_pairs = [
    ("err_d4sx",  "D4\u03c3\u2093  (axis x)"),
    ("err_d4sy",  "D4\u03c3\u1d67  (axis y)"),
    ("err_clip_x","Clip\u2093 13.5%"),
    ("err_clip_y","Clip\u1d67 13.5%"),
]
mode_order_w = ["off", "corner", "iso_statistical"]
mode_lbl_w   = {"off":"BG off","corner":"Corner (§3.4.3)",
                "iso_statistical":"ISO stat. (§3.4.2)"}
norm_w = plt.Normalize(vmin=-1, vmax=1)

def make_wpivot(cam, bgm, err_col):
    sub = dfw[(dfw.camera==cam) & (dfw.bg_mode==bgm)]
    return sub.groupby(["beam_sensor","bg_frac"])[err_col].mean().unstack()

def draw_heatmap_ax(ax, pivot):
    im = ax.imshow(pivot.values, norm=norm_w, cmap="jet",
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
    ax.set_xlabel("Background fraction", fontsize=9)
    ax.set_ylabel("Beam/sensor ratio", fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isfinite(v):
                s = "\u2014"
            else:
                s = f"{v:+.2f}" + ("*" if abs(v) > 1 else "")
            dark = np.isfinite(v) and (norm_w(v) < 0.25 or norm_w(v) > 0.75)
            ax.text(j, i, s, ha="center", va="center", fontsize=8,
                    color="white" if dark else "black")

hmap_suptitle = (
    "Single-image width error at beam waist  |  {metric}\n"
    "Blue = underestimate  |  Green = accurate  |  Red = overestimate  |"
    "  colorbar: \u22121 to +1  |  * = outside range")

width_figs = []

for err_col, metric_lbl in metric_pairs:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    for ri, cam in enumerate(["linear","tpa"]):
        for ci, bgm in enumerate(mode_order_w):
            ax = axes[ri, ci]
            pivot = make_wpivot(cam, bgm, err_col)
            draw_heatmap_ax(ax, pivot)
            ax.set_title(f"{cam_title[cam]}\n{mode_lbl_w[bgm]}",
                         fontsize=9, fontweight="bold")
    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm_w); sm.set_array([])
    fig.colorbar(sm, cax=cb_ax,
                 label="Mean relative width error  (\u0394w/w)\n"
                       "* = value outside \u00b11 range")
    fig.suptitle(hmap_suptitle.format(metric=metric_lbl),
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    tag = err_col.replace("err_", "")
    fname = f"sim_heatmap_width_{tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    width_figs.append(fname)
    print(f"Saved {fname}")

# Combined: all 4 metrics, ISO stat only
fig5, axes5 = plt.subplots(2, 4, figsize=(19, 9))
fig5.patch.set_facecolor("white")
for ri, cam in enumerate(["linear","tpa"]):
    for ci, (err_col, mlbl) in enumerate(metric_pairs):
        ax = axes5[ri, ci]
        pivot = make_wpivot(cam, "iso_statistical", err_col)
        draw_heatmap_ax(ax, pivot)
        ax.set_title(f"{cam_title[cam]}\n{mlbl}",
                     fontsize=8.5, fontweight="bold")
cb5 = fig5.add_axes([0.92, 0.15, 0.012, 0.7])
sm5 = plt.cm.ScalarMappable(cmap="jet", norm=norm_w); sm5.set_array([])
fig5.colorbar(sm5, cax=cb5, label="Mean relative width error  (\u0394w/w)")
fig5.suptitle(
    "Single-image width error at beam waist  |  ISO stat. BG (§3.4.2) only\n"
    "Blue = underestimate  |  Green = accurate  |  Red = overestimate  |"
    "  colorbar: \u22121 to +1",
    fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.91, 1])
fig5.savefig("sim_heatmap_width_combined.png", dpi=150,
             bbox_inches="tight", facecolor="white")
print("Saved sim_heatmap_width_combined.png")

print("\nAll outputs written (width simulation):")
for f in ["sim_width_results.csv"] + width_figs + ["sim_heatmap_width_combined.png"]:
    print(f"  {f}")



# ═══════════════════════════════════════════════════════════════════════════
# Z-DEPENDENT WIDTH ERROR SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
# Shows how width estimation error varies across the z-scan, not just at
# the waist. Uses discrete exposure model. This explains why M² fits are
# biased even when single-image width estimates look accurate at the waist:
# the far-field images have systematically different bias because the beam
# fills more of the sensor, corrupting the background estimate.
# ═══════════════════════════════════════════════════════════════════════════

print("\nRunning z-dependent width error simulation...")
t_zdep = time.perf_counter()

BEAM_SENSOR_Z = [0.15, 0.25]
BG_FRACS_Z    = [0.02, 0.10]
N_REAL_Z      = 20

zdep_records = []
rng_z = np.random.default_rng(77)

for bs, bg_frac, camera in product(BEAM_SENSOR_Z, BG_FRACS_Z, CAMERA_TYPES):
    w0x = bs * N_HALF * PX
    w0y = w0x * W0Y_W0X
    zRm = min(rayleigh(w0x, M2X_TRUE), rayleigh(w0y, M2Y_TRUE))
    z0x = 50 * zRm; z0y = z0x + 2 * zRm; zm = (z0x + z0y) / 2
    z_arr = make_z_array(zm, zRm)
    base_bg = bg_frac * TARGET_ADU

    for real in range(N_REAL_Z):
        rng = np.random.default_rng(rng_z.integers(0, 2**31))

        # generate all images with discrete exposure
        images, exposures = [], []
        for z in z_arr:
            exp = exposure_for_z(z, w0x, w0y, M2X_TRUE, M2Y_TRUE, z0x, z0y, camera,
                                 base_bg_adu=base_bg)
            wx = w_at_z(z, w0x, M2X_TRUE, z0x)
            wy = w_at_z(z, w0y, M2Y_TRUE, z0y)
            geom = (w0x * w0y) / (wx * wy)
            peak_z  = TARGET_ADU * (geom**2 if camera == "tpa" else geom) * exp
            bg_z    = base_bg * exp
            noise_z = SHOT_NOISE_FRAC * np.sqrt(bg_z) + READ_NOISE_ADU
            # reuse make_image_fixed — beam centred at sensor centre, sigma = wx/wy
            Y, X  = np.mgrid[0:N, 0:N].astype(float)
            dx = (X - (N_HALF - 0.5)) * PX
            dy = (Y - (N_HALF - 0.5)) * PX
            I  = np.exp(-2.0 * (dx**2/wx**2 + dy**2/wy**2))
            sig = peak_z * I**2 if camera == "tpa" else peak_z * I
            img = np.clip(sig + bg_z + rng.normal(0, noise_z, (N, N)), 0, ADC_MAX)
            images.append(img)
            exposures.append(exp)

        for bg_mode in BG_MODES:
            for zi, (z, img, exp) in enumerate(zip(z_arr, images, exposures)):
                wx_true = w_at_z(z, w0x, M2X_TRUE, z0x)
                wy_true = w_at_z(z, w0y, M2Y_TRUE, z0y)
                z_rel   = (z - zm) / zRm

                if bg_mode == "corner":
                    bm, bs2 = iso_background(img, corner_frac=0.035)
                elif bg_mode == "iso_statistical":
                    bm, bs2 = iso_background_statistical(img, n_sigma=3.0)
                else:
                    bm, bs2 = None, None

                try:
                    x0r, x1r, y0r, y1r = auto_roi(img, pad_sigma=3.0)
                    roi = img[y0r:y1r, x0r:x1r].astype(float)
                    m, _, _ = beam_size_iso(
                        roi, px=PX, bg_mean=bm, bg_std=bs2,
                        n_sigma=3.0, mask_factor=3.0,
                        bg_mode=bg_mode, use_tpa=(camera == "tpa"))
                    err_x = (m["sigma_x"] - wx_true/2) / (wx_true/2)
                    err_y = (m["sigma_y"] - wy_true/2) / (wy_true/2)
                except Exception:
                    err_x = err_y = np.nan

                zdep_records.append(dict(
                    camera=camera, bg_mode=bg_mode,
                    bg_frac=bg_frac, beam_sensor=bs,
                    realisation=real, zi=zi,
                    z_rel=z_rel, exposure=exp,
                    err_d4sx=err_x, err_d4sy=err_y,
                    beam_sensor_at_z=wx_true / (N_HALF * PX),
                ))

dfz = pd.DataFrame(zdep_records)
dfz.to_csv("sim_zdep_results.csv", index=False)
print(f"Z-dep done in {time.perf_counter()-t_zdep:.0f}s  "
      f"|  {len(dfz)} rows  ->  sim_zdep_results.csv")

# ── Plotting: z-dependent width error ─────────────────────────────────────
import matplotlib.lines as mlines

for bg_frac in [0.02, 0.10]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    fig.patch.set_facecolor("white")

    for ri, bs in enumerate([0.15, 0.25]):
        for ci, cam in enumerate(["linear", "tpa"]):
            ax = axes[ri, ci]
            style_ax(ax)
            ax.axvspan(-0.8,  0.8, alpha=0.05, color="gray")
            ax.axvspan(-2.5, -2.1, alpha=0.05, color="steelblue")
            ax.axvspan( 2.1,  2.5, alpha=0.05, color="steelblue")

            for bgm, (col, ls, mk, lbl) in mode_style.items():
                sub = dfz[(dfz.camera==cam)&(dfz.bg_mode==bgm)&
                          (dfz.bg_frac==bg_frac)&(dfz.beam_sensor==bs)]
                grp = sub.groupby("z_rel")["err_d4sx"]
                mn, sd = grp.mean(), grp.std()
                ax.plot(mn.index, mn.values, color=col, ls=ls, marker=mk,
                        ms=6, lw=1.8, markerfacecolor=col,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=lbl)
                ax.fill_between(mn.index, mn-sd, mn+sd, color=col, alpha=0.13)

            ax2 = ax.twinx()
            sub0 = dfz[(dfz.camera==cam)&(dfz.bg_mode=="corner")&
                       (dfz.bg_frac==bg_frac)&(dfz.beam_sensor==bs)]
            grp2 = sub0.groupby("z_rel")["beam_sensor_at_z"].mean()
            ax2.plot(grp2.index, grp2.values, color=OI["red"],
                     ls=":", lw=1.2, alpha=0.7)
            ax2.axhline(0.5, color=OI["red"], lw=0.7, ls="--", alpha=0.4)
            ax2.set_ylabel("$w_x(z)$ / sensor half", fontsize=8,
                           color=OI["red"])
            ax2.tick_params(axis="y", colors=OI["red"], labelsize=8)
            ax2.set_ylim(0, 1.4)
            ax2.spines["right"].set_edgecolor(OI["red"])

            ax.set_xlabel("$(z - z_0)\\ /\\ z_R$", fontsize=10)
            ax.set_ylabel("D4$\\sigma_x$ relative error", fontsize=9)
            ax.set_title(
                f"{cam_title[cam]}  |  beam/sensor(waist) = {bs:.2f}",
                fontsize=10, fontweight="bold")
            ax.set_xlim(-3, 3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], color=OI["red"], ls=":", lw=1.5))
    labels.append("$w_x(z)$ / sensor half")
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"D4$\\sigma_x$ error vs z-position  |  bg = {bg_frac:.0%}  |  \u00b11\u03c3 shaded\n"
        "Red dotted = beam filling fraction  |  Red dashed = ISO \u00a73.4.3 limit (0.5)\n"
        "Shaded zones: near-waist (grey) / far-field (blue)",
        fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fname = f"sim_zdep_bg{int(bg_frac*100)}pct.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {fname}")

# TPA vs linear, ISO stat
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
fig2.patch.set_facecolor("white")

for ri, bs in enumerate([0.15, 0.25]):
    for ci, bg_frac in enumerate([0.02, 0.10]):
        ax = axes2[ri, ci]
        style_ax(ax)
        ax.axvspan(-0.8,  0.8, alpha=0.05, color="gray")
        ax.axvspan(-2.5, -2.1, alpha=0.05, color="steelblue")
        ax.axvspan( 2.1,  2.5, alpha=0.05, color="steelblue")

        for cam in ["linear", "tpa"]:
            sub = dfz[(dfz.camera==cam)&(dfz.bg_mode=="iso_statistical")&
                      (dfz.bg_frac==bg_frac)&(dfz.beam_sensor==bs)]
            grp = sub.groupby("z_rel")["err_d4sx"]
            mn, sd = grp.mean(), grp.std()
            ax.plot(mn.index, mn.values, color=cam_col[cam],
                    ls=cam_ls[cam],
                    marker="o" if cam=="linear" else "s",
                    ms=6, lw=2, markerfacecolor=cam_col[cam],
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=cam_title[cam])
            ax.fill_between(mn.index, mn-sd, mn+sd,
                            color=cam_col[cam], alpha=0.15)

        ax2t = ax.twinx()
        sub_exp = dfz[(dfz.camera=="tpa")&(dfz.bg_mode=="iso_statistical")&
                      (dfz.bg_frac==bg_frac)&(dfz.beam_sensor==bs)]
        grp_exp = sub_exp.groupby("z_rel")["exposure"].mean()
        ax2t.step(grp_exp.index, grp_exp.values, color="#888",
                  lw=1.2, alpha=0.6, where="mid")
        ax2t.set_ylabel("TPA exposure (\u00d7)", fontsize=8, color="#888")
        ax2t.tick_params(axis="y", colors="#888", labelsize=8)
        ax2t.set_yscale("log", base=2)
        ax2t.set_ylim(0.8, 200)
        ax2t.spines["right"].set_edgecolor("#888")

        ax.set_xlabel("$(z - z_0)\\ /\\ z_R$", fontsize=10)
        ax.set_ylabel("D4$\\sigma_x$ relative error", fontsize=9)
        ax.set_title(
            f"beam/sensor(waist) = {bs:.2f}  |  bg = {bg_frac:.0%}",
            fontsize=10, fontweight="bold")
        ax.set_xlim(-3, 3)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=9, framealpha=0.95, edgecolor="#cccccc",
                      loc="lower left")

fig2.suptitle(
    "TPA vs Linear  |  ISO stat. BG  |  D4$\\sigma_x$ error vs z  |  \u00b11\u03c3 shaded\n"
    "Grey step = TPA exposure multiplier (log\u2082 scale, right axis)",
    fontsize=10, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("sim_zdep_tpa_vs_linear.png", dpi=150,
             bbox_inches="tight", facecolor="white")
print("Saved sim_zdep_tpa_vs_linear.png")

print("\nAll outputs written (z-dependent simulation):")
for f in ["sim_zdep_results.csv",
          "sim_zdep_bg5pct.png", "sim_zdep_bg20pct.png",
          "sim_zdep_tpa_vs_linear.png"]:
    print(f"  {f}")