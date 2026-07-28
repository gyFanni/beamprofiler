# -*- coding: utf-8 -*-
"""
beamprofiler.analysis
All numerical analysis functions. No Qt, no matplotlib.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .models import FileEntry

SAT_THRESH_DEFAULT = 0.8


def load_bgdata(path: str, frame: int = 1) -> dict:
    """
    Load a BeamGage .bgData file (HDF5 format, Ophir/Spiricon).

    Pixel data encoding
    -------------------
    BeamGage stores pixel values as signed 32-bit fixed-point integers where
    the MSB of the camera's native N-bit data occupies bit position 30 (just
    below the int32 sign bit). To recover ADC counts:

        raw_counts = int32_value >> (30 - (N - 1))

    BITENCODING is a string such as "8", "12", "S16", "U14" -- strip any
    leading S/U prefix to get N.

    Parameters
    ----------
    path  : path to the .bgData file
    frame : 1-based frame index (default 1; most files contain a single frame)

    Returns
    -------
    dict with keys:
        image        : np.ndarray (float64), shape (height, width), ADC counts
        bits         : int   -- native bit depth of the camera
        pixel_size_x : float -- pixel pitch in x, micrometres
        pixel_size_y : float -- pixel pitch in y, micrometres
        width        : int
        height       : int
        timestamp    : str
        saturated    : bool  -- BeamGage saturation flag
        binning_x    : int
        binning_y    : int
        n_frames     : int   -- total number of frames in the file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to load .bgData files.\n"
            "Install it with:  pip install h5py")

    with h5py.File(path, "r") as f:
        n_frames = len(f["BG_DATA"].keys())
        grp = f[f"BG_DATA/{frame}"]
        rf  = grp["RAWFRAME"]

        w    = int(rf["WIDTH"][0])
        h_px = int(rf["HEIGHT"][0])
        px_x = float(rf["PIXELSCALEXUM"][0])
        px_y = float(rf["PIXELSCALEYUM"][0])
        bx   = int(rf["BINNINGX"][0])
        by   = int(rf["BINNINGY"][0])
        sat  = bool(rf["METADATA/SATURATED"][0])

        ts_raw = rf["TIMESTAMP"][0]
        ts = (ts_raw.decode() if isinstance(ts_raw, (bytes, np.bytes_))
              else str(ts_raw))

        bits_raw = rf["BITENCODING"][0]
        bits_str = (bits_raw.decode() if isinstance(bits_raw, (bytes, np.bytes_))
                    else str(bits_raw)).strip()
        bits = int(bits_str.lstrip("SsUu"))

        data = grp["DATA"][:]

    # Fixed-point conversion: right-shift by (30 - (bits - 1))
    shift = 30 - (bits - 1)
    image = (data.astype(np.int64) >> shift).reshape(h_px, w).astype(float)

    return dict(
        image        = image,
        bits         = bits,
        pixel_size_x = px_x,
        pixel_size_y = px_y,
        width        = w,
        height       = h_px,
        timestamp    = ts,
        saturated    = sat,
        binning_x    = bx,
        binning_y    = by,
        n_frames     = n_frames,
    )


def check_saturation(img: np.ndarray, sat_thresh: float = SAT_THRESH_DEFAULT) -> dict:
    """
    Flag pixels at or above sat_thresh * ADC ceiling (default 80%).
    Returns: warning, sat_fraction, n_sat, adc_ceiling, threshold, message.
    """
    ceiling   = float(img.max())
    threshold = sat_thresh * ceiling
    n_sat     = int((img >= threshold).sum())
    sat_frac  = float(n_sat) / img.size
    warning   = n_sat > 0
    if warning:
        msg = (f"Saturation: {n_sat} px ({100*sat_frac:.2f}%) "
               f">= {sat_thresh*100:.0f}% of ADC ceiling ({ceiling:.0f}). "
               "D4sigma results may be unreliable.")
    else:
        msg = "No saturation detected."
    return dict(warning=warning, sat_fraction=sat_frac, n_sat=n_sat,
                adc_ceiling=ceiling, threshold=threshold, message=msg)


# ========================================================================
#  CORE ANALYSIS FUNCTIONS
# ========================================================================

def load_csv(path: str, has_header: bool, has_index: bool) -> np.ndarray:
    header    = 0 if has_header else None
    index_col = 0 if has_index  else None
    return pd.read_csv(path, header=header, index_col=index_col).values.astype(float)


def apply_sqrt(img: np.ndarray) -> np.ndarray:
    """TPA correction: sign(x)*sqrt(|x|)."""
    return np.sign(img) * np.sqrt(np.abs(img))


def iso_background(img: np.ndarray, corner_frac: float = 0.035, n_sigma: float = 3.0):
    """
    Estimate background mean and std using the laserbeamsize ISO method:
      1. Sample corners (corner_frac x min(ny,nx)) to get initial mean/std.
      2. Label all pixels below mean + n_sigma*std as 'unilluminated'.
      3. Recompute mean and std from those unilluminated pixels only.

    Returns (bg_mean, bg_std).
    """
    ny, nx = img.shape
    p = int(np.clip(corner_frac * min(ny, nx), 5, 50))
    corners = np.concatenate([img[:p,:p].ravel(), img[:p,-p:].ravel(),
                               img[-p:,:p].ravel(), img[-p:,-p:].ravel()])
    c_mean = float(np.mean(corners))
    c_std  = float(np.std(corners))

    # unilluminated mask: pixels below corner_mean + n_sigma*corner_std
    unlit = img[img < c_mean + n_sigma * c_std]
    if len(unlit) == 0:
        return c_mean, c_std
    return float(np.mean(unlit)), float(np.std(unlit))


def rotated_rect_mask(ny: int, nx: int,
                      xc: float, yc: float,
                      half_w: float, half_h: float,
                      theta_rad: float) -> np.ndarray:
    """
    Boolean mask: True inside a rectangle of half-widths (half_w, half_h)
    centred at (xc, yc), rotated by theta_rad (CCW from x-axis).

    half_w is along the major axis, half_h along the minor axis.
    """
    Y, X = np.mgrid[0:ny, 0:nx]
    dx = X - xc
    dy = Y - yc
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    # rotate into principal-axis frame
    u =  dx * cos_t + dy * sin_t   # along major axis
    v = -dx * sin_t + dy * cos_t   # along minor axis
    return (np.abs(u) <= half_w) & (np.abs(v) <= half_h)


def _moments(img: np.ndarray, px: float = 1.0):
    """
    Raw second-moment computation on img (pixels already zeroed outside mask).
    Returns dict with x_bar, y_bar, σ_x, σ_y, σ_xy,
    σ_maj, σ_min, theta_rad, θ_deg, ellipticity, d4* variants.
    """
    ny, nx = img.shape
    x = np.arange(nx, dtype=float) * px
    y = np.arange(ny, dtype=float) * px
    X, Y  = np.meshgrid(x, y)
    total = img.sum()
    if total <= 0:
        return {k: 0. for k in
                ("x_bar","y_bar","sigma_x","sigma_y","sigma_xy",
                 "d4sigma_x","d4sigma_y","sigma_maj","sigma_min",
                 "d4sigma_maj","d4sigma_min","theta_rad","theta_deg","ellipticity")}

    xb = float((img*X).sum() / total)
    yb = float((img*Y).sum() / total)
    var_x  = float((img*(X-xb)**2).sum() / total)
    var_y  = float((img*(Y-yb)**2).sum() / total)
    cov_xy = float((img*(X-xb)*(Y-yb)).sum() / total)

    sx = float(np.sqrt(max(var_x, 0.)))
    sy = float(np.sqrt(max(var_y, 0.)))

    # ISO 11146-1 Eq. (25)-(27): eigenvalues of [[var_x, cov],[cov, var_y]]
    mean_v = (var_x + var_y) / 2.
    disc   = float(np.sqrt(max(((var_x - var_y)/2.)**2 + cov_xy**2, 0.)))
    s_maj  = float(np.sqrt(max(mean_v + disc, 0.)))
    s_min  = float(np.sqrt(max(mean_v - disc, 0.)))

    theta_rad = 0.5 * np.arctan2(2.*cov_xy, var_x - var_y)
    θ_deg = float(np.degrees(theta_rad))
    ell       = float(s_min / s_maj) if s_maj > 0 else 1.

    return dict(
        x_bar=xb, y_bar=yb,
        sigma_x=sx, sigma_y=sy, sigma_xy=cov_xy,
        d4sigma_x=4*sx, d4sigma_y=4*sy,
        sigma_maj=s_maj, sigma_min=s_min,
        d4sigma_maj=4*s_maj, d4sigma_min=4*s_min,
        theta_rad=theta_rad, theta_deg=θ_deg,
        ellipticity=ell,
    )


def beam_size_iso(img: np.ndarray,
                  px: float = 1.0,
                  dark_frame: np.ndarray = None,
                  bg_mean: float = None,
                  bg_std:  float = None,
                  n_sigma: float = 3.0,
                  mask_factor: float = 3.0,
                  max_iter: int = 20,
                  bg_subtract: bool = True) -> tuple[dict, np.ndarray, dict]:
    """
    Integrated ISO 11146 beam size algorithm.

    bg_subtract : if True (default), estimate and subtract the background,
                  then zero pixels below n_sigma * bg_std.
                  If False, skip subtraction entirely (e.g. when the camera
                  already outputs background-free data). Only negative values
                  from dark-frame subtraction are clamped to zero.
    """
    img = img.astype(float).copy()

    # dark frame subtraction
    if dark_frame is not None:
        if dark_frame.shape != img.shape:
            raise ValueError(f"Dark frame shape {dark_frame.shape} != image {img.shape}.")
        img -= dark_frame.astype(float)

    if bg_subtract:
        # use pre-estimated background or fall back to own corners
        if bg_mean is None or bg_std is None:
            _bg_mean, _bg_std = iso_background(img)
        else:
            _bg_mean, _bg_std = float(bg_mean), float(bg_std)
        img_bg    = img - _bg_mean
        threshold = max(n_sigma * _bg_std, 0.0)
        img_bg[img_bg < threshold] = 0.
        bg_info = dict(bg_mean=_bg_mean, bg_std=_bg_std, threshold=threshold,
                       dark_used=dark_frame is not None, bg_subtract=True)
    else:
        # no background subtraction: clamp negatives only
        img_bg  = np.maximum(img, 0.)
        bg_info = dict(bg_mean=0., bg_std=0., threshold=0.,
                       dark_used=dark_frame is not None, bg_subtract=False)

    # initial moments on full zeroed ROI
    m = _moments(img_bg, px)
    if m["sigma_maj"] <= 0:
        return m, img_bg, bg_info

    # iterative rotated-rectangle masking
    ny, nx = img_bg.shape
    for _ in range(max_iter):
        half_maj = mask_factor * m["sigma_maj"] / px
        half_min = mask_factor * m["sigma_min"] / px

        # safety: if mask already covers the full image, further iteration is pointless
        if half_maj * 2 >= nx and half_min * 2 >= ny:
            break

        mask   = rotated_rect_mask(ny, nx,
                                   m["x_bar"] / px, m["y_bar"] / px,
                                   half_maj, half_min, m["theta_rad"])
        masked = np.where(mask, img_bg, 0.)
        m_new  = _moments(masked, px)

        if m_new["sigma_maj"] <= 0:
            break

        dx  = abs(m_new["x_bar"]    - m["x_bar"])    / px
        dy  = abs(m_new["y_bar"]    - m["y_bar"])    / px
        dsx = abs(m_new["sigma_maj"] - m["sigma_maj"]) / max(m["sigma_maj"], 1e-9)
        dsy = abs(m_new["sigma_min"] - m["sigma_min"]) / max(m["sigma_min"], 1e-9)
        m   = m_new
        if dx < 0.1 and dy < 0.1 and dsx < 0.01 and dsy < 0.01:
            break

    # final display image: masked with converged parameters
    half_maj = mask_factor * m["sigma_maj"] / px
    half_min = mask_factor * m["sigma_min"] / px
    if half_maj > 0 and half_min > 0:
        mask   = rotated_rect_mask(ny, nx,
                                   m["x_bar"] / px, m["y_bar"] / px,
                                   half_maj, half_min, m["theta_rad"])
        bg_img = np.where(mask, img_bg, 0.)
    else:
        bg_img = img_bg

    return m, bg_img, bg_info


def clip_level_widths(bg_img: np.ndarray,
                      px: float = 1.0,
                      clip_frac: float = 0.135,
                      theta_rad: float = 0.0,
                      xc_px: float = None,
                      yc_px: float = None) -> dict:
    """
    Clip-level (D%pk) beam widths from marginal profiles (BeamGage convention).

    Lab-axis widths (clip_x, clip_y)
    ---------------------------------
    Project bg_img onto the x and y axes (sum over y and x respectively),
    find the peak of each profile, threshold at clip_frac x peak (13.5% = 1/e^2),
    width = distance between outermost crossings.

    Principal-axis widths (clip_maj, clip_min)
    ------------------------------------------
    When theta_rad != 0, also rotate bg_img so the major axis aligns with
    the image x-axis, then take marginal profiles of the rotated image.
    Uses scipy.ndimage.rotate with reshape=False so the image stays the
    same size; pixels outside the original frame become zero and do not
    affect the marginal sums.

    Parameters
    ----------
    bg_img    : background-subtracted, masked image
    px        : pixel size in physical units
    clip_frac : clip level (default 0.135 = 13.5% = 1/e^2)
    theta_rad : angle of the major axis from x-axis in radians (from _moments).
                If 0, principal-axis widths are equal to lab widths.
    xc_px, yc_px : centroid in ROI-local pixel coords (used as rotation centre).
                   If None, image centre is used.

    Returns dict with clip_x, clip_y, clip_maj, clip_min, clip_frac.
    """
    from scipy.ndimage import rotate as _rotate

    ny, nx = bg_img.shape
    x_ax = np.arange(nx, dtype=float) * px
    y_ax = np.arange(ny, dtype=float) * px

    def _width(profile, coords):
        pk = float(profile.max())
        if pk <= 0: return 0.
        idx = np.where(profile >= clip_frac * pk)[0]
        return float(coords[idx[-1]] - coords[idx[0]]) if len(idx) else 0.

    clip_x = _width(bg_img.sum(axis=0), x_ax)
    clip_y = _width(bg_img.sum(axis=1), y_ax)

    # principal-axis widths: rotate image so major axis lies along x
    # theta_rad is angle of major axis from x-axis (CCW positive).
    # Rotating the image by +theta_deg (CCW) aligns the major axis with x,
    # so sum(axis=0) gives the projection along the major axis (clip_maj)
    # and sum(axis=1) gives the projection along the minor axis (clip_min).
    angle_deg = float(np.degrees(theta_rad))
    if abs(angle_deg) > 0.01:
        cx = xc_px if xc_px is not None else nx / 2.
        cy = yc_px if yc_px is not None else ny / 2.
        from scipy.ndimage import shift as _shift
        shifted   = _shift(bg_img, [-cy + ny/2., -cx + nx/2.], order=1, cval=0.)
        rotated   = _rotate(shifted, angle_deg, reshape=False, order=1, cval=0.)
        rotated   = np.maximum(rotated, 0.)
        clip_maj  = _width(rotated.sum(axis=0), x_ax)   # along major (x after rotation)
        clip_min  = _width(rotated.sum(axis=1), y_ax)   # along minor (y after rotation)
    else:
        # no rotation needed (theta ~ 0): major along x, minor along y
        clip_maj = _width(bg_img.sum(axis=0), x_ax)
        clip_min = _width(bg_img.sum(axis=1), y_ax)

    return dict(
        clip_x   = clip_x,
        clip_y   = clip_y,
        clip_maj = clip_maj,
        clip_min = clip_min,
        clip_frac = clip_frac,
    )


def marginal_gaussian_fit(profile, coords):
    """Fit Gaussian + constant baseline to a marginal profile."""
    def gauss_const(x, amp, mu, sig, offset):
        return amp * np.exp(-0.5*((x-mu)/sig)**2) + offset
    pk_idx = np.argmax(profile)
    p0 = [float(profile.max() - profile.min()),
          float(coords[pk_idx]),
          float((coords[-1]-coords[0])/6),
          float(profile.min())]
    try:
        popt, _ = curve_fit(gauss_const, coords, profile, p0=p0,
                            maxfev=10000,
                            bounds=([-np.inf, coords[0],  1e-9, -np.inf],
                                    [ np.inf, coords[-1], np.inf, np.inf]))
        return popt, gauss_const(coords, *popt)
    except Exception:
        return p0, gauss_const(coords, *p0)


# ========================================================================
#  AUTO-ROI  -  ISO 11146-1 S.7 iterative integration-area algorithm
# ========================================================================

def auto_roi(img, pad_sigma=3.0, max_iter=20):
    """
    Peak-seeded ISO 11146-1 S.7 iterative integration-area algorithm.

    Peak finding (robust against saturated damage clusters)
    -------------------------------------------------------
    A raw argmax fails when damage pixels are present even after marking,
    or when the damage was not marked. The fix uses two steps:
    1. Gaussian blur (sigma=5) so a smooth beam peak wins over a sharp
       saturated spike.
    2. Mask flat regions (local_std < 5% of global_std) before taking
       argmax, eliminating any remaining saturated plateaus.

    Half-width estimation (robust for small beams on large sensors)
    ---------------------------------------------------------------
    Walking the full-image marginal profile at the peak row/col gives
    unreliable 1/e^2 widths for small beams: background noise means the
    profile never cleanly drops below the threshold, returning a 1-pixel
    half-width and a tiny seed window. Instead, extract a local patch
    (+-60 px) around the peak and walk that patch's marginal profiles.

    Returns (x0, x1, y0, y1).
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    ny, nx = img.shape
    X, Y   = np.meshgrid(np.arange(nx, dtype=float),
                         np.arange(ny, dtype=float))

    # Use the ISO corner-seed background estimator rather than just the median.
    # This gives a better 1/e^2 threshold when background is non-uniform or
    # when the beam signal is weak relative to background noise.
    bg, _bg_std = iso_background(img, corner_frac=0.035, n_sigma=3.0)
    img_bg = img.astype(float) - bg

    # -- robust peak: blur then suppress flat regions ------------------
    blurred   = gaussian_filter(img_bg, sigma=5)
    lm2       = uniform_filter(img_bg ** 2, size=7)
    lm        = uniform_filter(img_bg,      size=7)
    local_std = np.sqrt(np.maximum(lm2 - lm ** 2, 0))
    pos       = img_bg[img_bg > 0]
    gstd      = float(np.std(pos)) if len(pos) else 1.
    bm        = blurred.copy()
    bm[local_std < 0.05 * gstd] = 0.
    pk_y, pk_x = np.unravel_index(np.argmax(bm), bm.shape)
    peak_sig   = float(img_bg[pk_y, pk_x])

    if peak_sig <= 0:
        return (0, nx, 0, ny)

    # -- local-patch 1/e^2 half-widths ---------------------------------
    # Use a +-60 px patch so background noise outside the beam does not
    # interfere with the half-width walk.
    patch_r = 60
    py0, py1 = max(0, pk_y - patch_r), min(ny, pk_y + patch_r)
    px0, px1 = max(0, pk_x - patch_r), min(nx, pk_x + patch_r)
    local    = img_bg[py0:py1, px0:px1]
    lrow     = local[pk_y - py0, :]
    lcol     = local[:, pk_x - px0]
    lpk_x    = pk_x - px0
    lpk_y    = pk_y - py0
    thr      = peak_sig / np.e ** 2

    def hw(profile, pidx):
        n = len(profile)
        r = pidx
        while r < n - 1 and profile[r] > thr: r += 1
        l = pidx
        while l > 0       and profile[l] > thr: l -= 1
        return max(r - pidx, pidx - l, 1)

    hw_x = hw(lrow, lpk_x)
    hw_y = hw(lcol, lpk_y)
    x0 = int(np.clip(pk_x - pad_sigma * hw_x, 0, nx - 1))
    x1 = int(np.clip(pk_x + pad_sigma * hw_x, 1, nx))
    y0 = int(np.clip(pk_y - pad_sigma * hw_y, 0, ny - 1))
    y1 = int(np.clip(pk_y + pad_sigma * hw_y, 1, ny))

    # -- ISO S.7 iteration from seed window ----------------------------
    for _ in range(max_iter):
        patch = img_bg[y0:y1, x0:x1]
        Xp    = X[y0:y1, x0:x1]
        Yp    = Y[y0:y1, x0:x1]
        total = patch.sum()
        if total <= 0: break
        xb = float((patch * Xp).sum() / total)
        yb = float((patch * Yp).sum() / total)
        sx = float(np.sqrt(max((patch * (Xp - xb)**2).sum() / total, 0.))); sx = max(sx, 1.)
        sy = float(np.sqrt(max((patch * (Yp - yb)**2).sum() / total, 0.))); sy = max(sy, 1.)
        nx0 = int(np.clip(xb - pad_sigma * sx, 0, nx - 1))
        nx1 = int(np.clip(xb + pad_sigma * sx, 1, nx))
        ny0 = int(np.clip(yb - pad_sigma * sy, 0, ny - 1))
        ny1 = int(np.clip(yb + pad_sigma * sy, 1, ny))
        if nx0 == x0 and nx1 == x1 and ny0 == y0 and ny1 == y1: break
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1

    return (x0, x1, y0, y1)




def run_analysis(fe: FileEntry, settings: dict) -> FileEntry:
    """
    Run the full pipeline on a FileEntry using the provided settings dict.
    Settings keys: dark_frame, corner_frac, n_sigma, mask_factor, px,
                   pad_sigma, use_auto_roi, fit_mode ('lab' | 'principal').
    """
    px          = settings["px"]
    fit_mode    = settings.get("fit_mode", "lab")
    use_tpa     = settings.get("use_tpa", True)
    dark_frame  = settings.get("dark_frame")
    sat_thresh  = settings.get("sat_thresh", SAT_THRESH_DEFAULT)
    bg_subtract = settings.get("bg_subtract", True)

    sat = check_saturation(fe.clean_raw, sat_thresh=sat_thresh)

    # use existing ROI or compute auto-ROI
    if fe.roi is None or settings.get("use_auto_roi", False):
        fe.roi = auto_roi(fe.sqrt_img, pad_sigma=settings["pad_sigma"])

    x0, x1, y0, y1 = fe.roi
    roi_img  = fe.sqrt_img[y0:y1, x0:x1]
    dark_roi = (dark_frame[y0:y1, x0:x1]
                if dark_frame is not None else None)

    # estimate background from the full image (not the small ROI crop)
    # so corner patches are guaranteed to be real background.
    # Skip entirely when bg_subtract is disabled.
    corner_frac = settings.get("corner_frac", 0.035)
    n_sigma     = settings.get("n_sigma", 3.0)
    if bg_subtract:
        bg_mean, bg_std = iso_background(fe.sqrt_img,
                                         corner_frac=corner_frac,
                                         n_sigma=n_sigma)
    else:
        bg_mean, bg_std = None, None

    m, bg_img, bg_info = beam_size_iso(
        roi_img,
        px           = px,
        dark_frame   = dark_roi,
        bg_mean      = bg_mean,
        bg_std       = bg_std,
        n_sigma      = n_sigma,
        mask_factor  = settings.get("mask_factor", 3.0),
        bg_subtract  = bg_subtract,
    )

    fe.roi_img = roi_img
    fe.bg_img  = bg_img
    fe.bg_info = bg_info

    unit = "µm" if px != 1.0 else "px"

    # clip-level widths: lab axes always; principal axes when theta != 0
    cl = clip_level_widths(
        bg_img, px=px, clip_frac=0.135,
        theta_rad = m["theta_rad"],
        xc_px     = m["x_bar"] / px if px else m["x_bar"],
        yc_px     = m["y_bar"] / px if px else m["y_bar"],
    )

    # shift centroid to global coordinates
    x_bar_g = m["x_bar"] + x0 * px
    y_bar_g = m["y_bar"] + y0 * px

    fe.results = dict(
        filename    = fe.fname,
        fit_mode    = fit_mode,
        use_tpa     = use_tpa,
        bg_subtract = bg_subtract,
        unit        = unit,
        pixel_size_um = px,
        bg_info     = bg_info,
        roi         = fe.roi,
        # centroid (global)
        x_bar       = x_bar_g,
        y_bar       = y_bar_g,
        # lab-axis moments
        σ_x     = m["sigma_x"],
        σ_y     = m["sigma_y"],
        σ_xy    = m["sigma_xy"],
        d4σ_x   = m["d4sigma_x"],
        d4σ_y   = m["d4sigma_y"],
        # principal-axis moments
        σ_maj   = m["sigma_maj"],
        σ_min   = m["sigma_min"],
        d4σ_maj = m["d4sigma_maj"],
        d4σ_min = m["d4sigma_min"],
        θ_deg   = m["theta_deg"],
        ellipticity = m["ellipticity"],
        # clip-level
        clip_x      = cl["clip_x"],
        clip_y      = cl["clip_y"],
        clip_maj    = cl["clip_maj"],
        clip_min    = cl["clip_min"],
        clip_frac   = cl["clip_frac"],
    )
    return fe


# ========================================================================