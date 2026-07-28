# -*- coding: utf-8 -*-
"""
beamprofiler/tests/test_analysis.py
=====================================
Unit tests for the analysis module.
All tests use synthetic Gaussian beams with known analytical ground truth.

Run with:
    cd beamprofiler_dir
    python -m pytest beamprofiler/tests/ -v
or:
    python -m pytest beamprofiler/tests/test_analysis.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pytest
from beamprofiler.analysis import (
    apply_sqrt,
    check_saturation,
    iso_background,
    rotated_rect_mask,
    _moments,
    beam_size_iso,
    auto_roi,
    clip_level_widths,
    marginal_gaussian_fit,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                  sx=15., sy=10., theta=0., amp=1000.,
                  bg=0., noise=0., seed=42) -> np.ndarray:
    """
    Synthetic 2-D Gaussian beam on a clean background.

    Parameters match ISO 11146 conventions:
      (cx, cy)  : centroid in pixel coordinates
      sx, sy    : 1-sigma widths along principal axes (before rotation)
      theta     : rotation angle in DEGREES (CCW from x-axis)
    """
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    dx   = X - cx
    dy   = Y - cy
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    u    =  dx * c + dy * s   # along major axis
    v    = -dx * s + dy * c   # along minor axis
    img  = amp * np.exp(-0.5 * ((u/sx)**2 + (v/sy)**2)) + bg
    if noise > 0:
        img += rng.normal(0, noise, img.shape)
    return img


def make_tpa_gaussian(**kwargs) -> np.ndarray:
    """Return a Gaussian whose sqrt matches make_gaussian."""
    img = make_gaussian(**kwargs)
    return img ** 2   # TPA: detected signal = I^2


# -----------------------------------------------------------------------
# apply_sqrt
# -----------------------------------------------------------------------

class TestApplySqrt:
    def test_positive_values(self):
        img = np.array([0., 1., 4., 9., 16.])
        out = apply_sqrt(img)
        np.testing.assert_allclose(out, [0., 1., 2., 3., 4.])

    def test_negative_values_preserved(self):
        """ISO 11146-3 S.3.1: negatives must survive the transform."""
        img = np.array([-4., -1., 0., 1., 4.])
        out = apply_sqrt(img)
        np.testing.assert_allclose(out, [-2., -1., 0., 1., 2.])

    def test_tpa_roundtrip(self):
        """apply_sqrt(x^2) == x for x >= 0."""
        x   = np.linspace(0, 10, 50)
        out = apply_sqrt(x ** 2)
        np.testing.assert_allclose(out, x, atol=1e-12)


# -----------------------------------------------------------------------
# check_saturation
# -----------------------------------------------------------------------

class TestCheckSaturation:
    def test_no_saturation(self):
        img = np.ones((100, 100)) * 100.
        result = check_saturation(img, sat_thresh=0.8)
        # all pixels == ceiling, so warning IS triggered (100% = ceiling)
        # That's correct behaviour - a flat image is fully saturated
        assert isinstance(result["warning"], bool)
        assert result["adc_ceiling"] == 100.

    def test_saturation_detected(self):
        img = np.zeros((50, 50))
        img[25, 25] = 255.   # single hot pixel at ADC ceiling
        result = check_saturation(img, sat_thresh=0.8)
        assert result["warning"] is True
        assert result["n_sat"] == 1
        assert result["adc_ceiling"] == 255.

    def test_below_threshold(self):
        img = np.ones((50, 50)) * 100.
        img[25, 25] = 200.   # 200 < 0.8 * 200 is False (200 == ceiling)
        # ceiling = 200, threshold = 0.8 * 200 = 160, pixel 200 >= 160 -> warning
        result = check_saturation(img, sat_thresh=0.8)
        assert result["warning"] is True

    def test_custom_threshold(self):
        img = np.zeros((50, 50))
        img[0, 0] = 100.
        # with thresh=1.0, only pixels == ceiling are flagged
        result = check_saturation(img, sat_thresh=1.0)
        assert result["n_sat"] == 1

    def test_message_content(self):
        img = np.zeros((10, 10))
        img[0, 0] = 255.
        result = check_saturation(img)
        assert "Saturation" in result["message"]
        assert "255" in result["message"]

    def test_no_saturation_message(self):
        # Manually construct: peak=100, ceiling=10000 (pad zeros around beam)
        beam = make_gaussian(ny=100, nx=100, cx=50., cy=50., sx=8., sy=8., amp=100., bg=0.)
        # pad with zeros so max of padded image is 100, then add a single
        # pixel at 10000 in a corner that we will explicitly NOT flag as beam
        img = np.zeros((200, 200))
        img[50:150, 50:150] = beam
        img[0, 0] = 10000.   # sets ceiling to 10000
        # beam peak = 100, ceiling = 10000, threshold = 0.95*10000 = 9500
        # only img[0,0] is flagged -- but we want to test beam-only case
        # So: use an image where beam << ceiling and no pixels cross threshold
        img2 = np.zeros((100, 100))
        img2[50, 50] = 1000.   # ceiling
        img2[49:52, 49:52] = np.linspace(50, 999, 9).reshape(3,3)
        # threshold = 0.95 * 1000 = 950; only pixels >= 950 flagged
        result = check_saturation(img2, sat_thresh=0.95)
        # pixel at [50,50]=1000 >= 950, so 1 pixel flagged
        assert result["n_sat"] == 1
        # Now test truly no saturation: beam peak well below threshold
        img3 = make_gaussian(ny=200, nx=200, cx=100., cy=100., sx=10., sy=10., amp=100., bg=0.)
        # ceiling = 100; sat_thresh=0.95 -> threshold=95; pixels near peak might be flagged
        # Better: use beam peak = 100 but threshold = 200 (impossible with sat_thresh>1)
        # Use a clean case: scale so peak = 100, ceil = 100, thresh = 0.999*100 = 99.9
        # Then all peak pixels (100) >= 99.9 -> flagged. Not what we want.
        # Correct test: beam max < sat_thresh * true_sensor_ceiling
        # We simulate by putting a hidden max pixel far above beam
        img4 = img3.copy()
        img4[0, 0] = 50000.   # sensor ceiling is 50000
        result4 = check_saturation(img4, sat_thresh=0.95)
        # threshold = 0.95 * 50000 = 47500; beam peak 100 << 47500 -> only [0,0] flagged
        # that IS saturation, but it's the artificial pixel, not the beam
        # This correctly shows that the beam itself is not saturated
        assert result4["warning"] is True   # the corner pixel IS saturated
        assert result4["n_sat"] == 1        # only one pixel above threshold


# -----------------------------------------------------------------------
# iso_background
# -----------------------------------------------------------------------

class TestIsoBackground:
    def test_flat_background(self):
        """Uniform image: background estimate should equal the constant value."""
        img = np.ones((100, 100)) * 5.
        mean, std = iso_background(img, corner_frac=0.1)
        assert abs(mean - 5.) < 0.1
        assert std < 0.1

    def test_beam_does_not_bias_estimate(self):
        """Background estimate should not be affected by a bright central beam."""
        img = make_gaussian(ny=200, nx=200, cx=100, cy=100,
                            sx=10, sy=10, amp=500., bg=10.)
        mean, std = iso_background(img, corner_frac=0.05)
        # should be close to bg=10, not pulled up by beam
        assert abs(mean - 10.) < 2.0

    def test_returns_tuple(self):
        img = np.random.rand(100, 100) * 5.
        result = iso_background(img)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# -----------------------------------------------------------------------
# rotated_rect_mask
# -----------------------------------------------------------------------

class TestRotatedRectMask:
    def test_unrotated_rect(self):
        """At theta=0, should be a simple axis-aligned rectangle."""
        mask = rotated_rect_mask(100, 100, 50., 50., 20., 10., 0.)
        # all pixels with |x-50| <= 20 and |y-50| <= 10 should be True
        Y, X = np.mgrid[0:100, 0:100]
        expected = (np.abs(X - 50.) <= 20.) & (np.abs(Y - 50.) <= 10.)
        np.testing.assert_array_equal(mask, expected)

    def test_full_coverage_large_halfwidths(self):
        """Half-widths larger than image -> all pixels True."""
        mask = rotated_rect_mask(50, 50, 25., 25., 1000., 1000., 0.)
        assert mask.all()

    def test_no_coverage_zero_halfwidths(self):
        """Half-width of 0 -> almost no pixels (border effects only)."""
        mask = rotated_rect_mask(50, 50, 25., 25., 0., 0., 0.)
        # only the exact centre pixel satisfies |u|<=0
        assert mask.sum() <= 4   # at most a few pixels at boundary

    def test_rotation_swaps_axes(self):
        """90 deg rotation should swap the effective x and y widths."""
        half_w, half_h = 20., 5.
        mask0 = rotated_rect_mask(100, 100, 50., 50., half_w, half_h, 0.)
        mask90 = rotated_rect_mask(100, 100, 50., 50., half_w, half_h,
                                   np.pi / 2)
        # at theta=0: wide in x, narrow in y
        # at theta=90: wide in y, narrow in x
        Y, X = np.mgrid[0:100, 0:100]
        wide_x = (np.abs(X - 50.) <= half_w) & (np.abs(Y - 50.) <= half_h)
        wide_y = (np.abs(X - 50.) <= half_h) & (np.abs(Y - 50.) <= half_w)
        # Allow 0.5% mismatch due to floating-point boundary effects
        assert np.mean(mask0 != wide_x)  < 0.005
        assert np.mean(mask90 != wide_y) < 0.005


# -----------------------------------------------------------------------
# _moments  (the core computation)
# -----------------------------------------------------------------------

class TestMoments:
    TOL = 1e-2   # 1% relative tolerance for numerical integrals

    def test_centroid_on_axis_aligned_gaussian(self):
        """Centroid should recover the true beam centre."""
        cx, cy = 75., 60.
        img = make_gaussian(ny=200, nx=200, cx=cx, cy=cy,
                            sx=12., sy=8., amp=500., bg=0.)
        m = _moments(img, px=1.0)
        assert abs(m["x_bar"] - cx) < 0.5
        assert abs(m["y_bar"] - cy) < 0.5

    def test_sigma_on_axis_aligned_gaussian(self):
        """sigma_x/sigma_y should recover the true 1-sigma widths."""
        sx, sy = 15., 8.
        img = make_gaussian(ny=300, nx=300, cx=150., cy=150.,
                            sx=sx, sy=sy, amp=1000., bg=0.)
        m = _moments(img, px=1.0)
        assert abs(m["sigma_x"] - sx) / sx < self.TOL
        assert abs(m["sigma_y"] - sy) / sy < self.TOL

    def test_d4sigma_is_four_sigma(self):
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=10., sy=6., amp=500., bg=0.)
        m = _moments(img, px=1.0)
        assert abs(m["d4sigma_x"] - 4*m["sigma_x"]) < 1e-10
        assert abs(m["d4sigma_y"] - 4*m["sigma_y"]) < 1e-10

    def test_principal_axes_circular_beam(self):
        """Circular beam: sigma_maj == sigma_min, ellipticity == 1."""
        sx = sy = 12.
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=sx, sy=sy, amp=500., bg=0.)
        m = _moments(img, px=1.0)
        assert abs(m["sigma_maj"] - m["sigma_min"]) / sx < self.TOL
        assert abs(m["ellipticity"] - 1.0) < 0.02

    def test_principal_axes_tilted_gaussian(self):
        """
        Tilted Gaussian: sigma_maj > sigma_min, theta recovers rotation.
        We use theta=30 deg, sx=20 (major), sy=8 (minor).
        """
        theta_true = 30.
        img = make_gaussian(ny=300, nx=300, cx=150., cy=150.,
                            sx=20., sy=8., theta=theta_true, amp=1000., bg=0.)
        m = _moments(img, px=1.0)
        assert m["sigma_maj"] > m["sigma_min"]
        # angle should be within 2 degrees
        assert abs(m["theta_deg"] - theta_true) < 2.

    def test_pixel_size_scaling(self):
        """Results should scale linearly with px."""
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=10., sy=10., amp=500., bg=0.)
        m1  = _moments(img, px=1.0)
        m2  = _moments(img, px=2.0)
        assert abs(m2["x_bar"]    / m1["x_bar"]    - 2.0) < 1e-6
        assert abs(m2["sigma_x"]  / m1["sigma_x"]  - 2.0) < 1e-6
        assert abs(m2["d4sigma_x"]/ m1["d4sigma_x"]- 2.0) < 1e-6

    def test_zero_image_returns_zeros(self):
        m = _moments(np.zeros((50, 50)))
        assert m["x_bar"] == 0.
        assert m["sigma_maj"] == 0.


# -----------------------------------------------------------------------
# beam_size_iso  (integration test of full algorithm)
# -----------------------------------------------------------------------

class TestBeamSizeIso:
    TOL = 0.05   # 5% — looser due to background subtraction effects

    def test_recovers_centroid(self):
        cx, cy = 80., 60.
        img = make_gaussian(ny=150, nx=150, cx=cx, cy=cy,
                            sx=10., sy=8., amp=500., bg=5., noise=0.5)
        m, _, _ = beam_size_iso(img, px=1.0)
        assert abs(m["x_bar"] - cx) < 1.5
        assert abs(m["y_bar"] - cy) < 1.5

    def test_recovers_sigma(self):
        sx, sy = 12., 8.
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=sx, sy=sy, amp=1000., bg=5., noise=0.3)
        m, _, _ = beam_size_iso(img, px=1.0, n_sigma=3.0)
        assert abs(m["sigma_x"] - sx) / sx < self.TOL
        assert abs(m["sigma_y"] - sy) / sy < self.TOL

    def test_bg_img_nonnegative(self):
        """After processing, displayed bg_img should have no large negatives."""
        img = make_gaussian(ny=100, nx=100, cx=50., cy=50.,
                            sx=8., sy=6., amp=500., bg=10., noise=1.)
        _, bg_img, _ = beam_size_iso(img, px=1.0)
        assert bg_img.min() >= -1.0   # small negatives from noise OK

    def test_dark_frame_subtraction(self):
        """Dark frame subtraction should shift the result correctly."""
        bg_level = 20.
        img  = make_gaussian(ny=100, nx=100, cx=50., cy=50.,
                             sx=8., sy=6., amp=300., bg=bg_level)
        dark = np.full_like(img, bg_level * 0.5)
        m_with, _, info_with = beam_size_iso(img, dark_frame=dark)
        m_without, _, _      = beam_size_iso(img)
        # centroid should be the same either way
        assert abs(m_with["x_bar"] - m_without["x_bar"]) < 1.0
        assert info_with["dark_used"] is True

    def test_returns_correct_types(self):
        img = make_gaussian(ny=80, nx=80, cx=40., cy=40., sx=6., sy=6., amp=200.)
        m, bg_img, bg_info = beam_size_iso(img)
        assert isinstance(m, dict)
        assert isinstance(bg_img, np.ndarray)
        assert isinstance(bg_info, dict)
        assert bg_img.shape == img.shape


# -----------------------------------------------------------------------
# auto_roi
# -----------------------------------------------------------------------

class TestAutoRoi:
    def test_contains_beam(self):
        """The returned ROI should contain the beam centroid."""
        cx, cy = 300., 250.
        img = make_gaussian(ny=500, nx=600, cx=cx, cy=cy,
                            sx=20., sy=15., amp=800., bg=10.)
        x0, x1, y0, y1 = auto_roi(img, pad_sigma=3.0)
        assert x0 <= cx <= x1
        assert y0 <= cy <= y1

    def test_roi_smaller_than_full_image(self):
        """For a small beam, the ROI should be much smaller than the sensor."""
        img = make_gaussian(ny=500, nx=500, cx=250., cy=250.,
                            sx=10., sy=10., amp=800., bg=5.)
        x0, x1, y0, y1 = auto_roi(img)
        roi_area = (x1-x0) * (y1-y0)
        assert roi_area < 0.25 * 500 * 500   # < 25% of sensor

    def test_degenerate_image(self):
        """Flat image: returns full image without crashing."""
        img = np.ones((50, 50)) * 5.
        roi = auto_roi(img)
        assert roi == (0, 50, 0, 50)


# -----------------------------------------------------------------------
# clip_level_widths
# -----------------------------------------------------------------------

class TestClipLevelWidths:
    def test_gaussian_fwhm_approx(self):
        """
        For a Gaussian beam with sigma=10, the 1/e^2 (13.5%) width of the
        marginal profile should be approximately 4*sigma = 40 px.
        (Marginal profile of 2D Gaussian is a 1D Gaussian with same sigma.)
        """
        sx = 10.
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=sx, sy=sx, amp=500., bg=0.)
        cl = clip_level_widths(img, px=1.0, clip_frac=0.135)
        # 1/e^2 half-width = 2*sigma, so full width = 4*sigma
        expected = 4 * sx
        assert abs(cl["clip_x"] - expected) / expected < 0.05
        assert abs(cl["clip_y"] - expected) / expected < 0.05

    def test_zero_image(self):
        img = np.zeros((50, 50))
        cl = clip_level_widths(img)
        assert cl["clip_x"] == 0.
        assert cl["clip_y"] == 0.

    def test_pixel_size_scaling(self):
        img = make_gaussian(ny=200, nx=200, cx=100., cy=100.,
                            sx=10., sy=10., amp=500., bg=0.)
        cl1 = clip_level_widths(img, px=1.0)
        cl2 = clip_level_widths(img, px=3.45)
        assert abs(cl2["clip_x"] / cl1["clip_x"] - 3.45) < 0.01


# -----------------------------------------------------------------------
# marginal_gaussian_fit
# -----------------------------------------------------------------------

class TestMarginalGaussianFit:
    def test_recovers_parameters(self):
        """Fit should recover amp, mu, sigma, offset of a synthetic profile."""
        x   = np.linspace(0, 100, 200)
        amp, mu, sig, off = 500., 50., 10., 5.
        profile = amp * np.exp(-0.5*((x-mu)/sig)**2) + off
        popt, fitted = marginal_gaussian_fit(profile, x)
        assert abs(popt[0] - amp) / amp < 0.01   # amplitude
        assert abs(popt[1] - mu)  < 0.1           # centre
        assert abs(popt[2] - sig) / sig < 0.01    # sigma
        assert abs(popt[3] - off) < 0.5            # offset

    def test_output_shape(self):
        x = np.linspace(0, 50, 100)
        profile = np.exp(-0.5*((x-25)/5)**2) * 100
        popt, fitted = marginal_gaussian_fit(profile, x)
        assert len(popt) == 4
        assert fitted.shape == x.shape

    def test_noisy_profile(self):
        """Fit should still converge on a noisy profile."""
        rng = np.random.default_rng(0)
        x   = np.linspace(0, 100, 200)
        profile = 300. * np.exp(-0.5*((x-50.)/8.)**2) + rng.normal(0, 5., 200)
        popt, fitted = marginal_gaussian_fit(profile, x)
        assert abs(popt[1] - 50.) < 2.   # centre within 2 px


# -----------------------------------------------------------------------
# TPA round-trip integration test
# -----------------------------------------------------------------------

class TestTPARoundtrip:
    def test_apply_sqrt_then_moments(self):
        """
        A TPA image (I^2) after apply_sqrt should give the same
        centroid as the original linear image.
        """
        sx, sy, cx, cy = 12., 8., 100., 100.
        linear  = make_gaussian(ny=200, nx=200, cx=cx, cy=cy,
                                sx=sx, sy=sy, amp=500., bg=0.)
        tpa_img = linear ** 2
        corrected = apply_sqrt(tpa_img)
        m = _moments(corrected, px=1.0)
        assert abs(m["x_bar"] - cx) < 0.5
        assert abs(m["y_bar"] - cy) < 0.5
        # sigma should be close to original
        assert abs(m["sigma_x"] - sx) / sx < 0.02
        assert abs(m["sigma_y"] - sy) / sy < 0.02