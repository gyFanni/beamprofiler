# -*- coding: utf-8 -*-
# NOTE: This file uses UTF-8 encoding (Greek letters, symbols).
# On Windows, run with:  set PYTHONUTF8=1 && python d4sigma_gui.py
# Or add PYTHONUTF8=1 to your IDE's environment variables.
import sys as _sys, os as _os
if _sys.platform == "win32" and _os.environ.get("PYTHONUTF8", "0") != "1":
    _os.environ["PYTHONUTF8"] = "1"
    import subprocess
    _sys.exit(subprocess.call([_sys.executable, *_sys.argv]))
"""
D4σ Beam Profile GUI  —  2-Photon Absorption
=============================================
Interactive PyQt application for D4sigma beam analysis.
Supports batch processing of multiple files with a shared settings panel.

Dependencies
------------
    pip install numpy pandas matplotlib scipy

Qt binding (one of):
    pip install PyQt6          # preferred
    pip install PyQt5          # fallback

Run
---
    python d4sigma_gui.py
"""

# -- Qt compatibility shim ----------------------------------------------------
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QGroupBox, QLabel, QDoubleSpinBox, QPushButton,
        QCheckBox, QComboBox, QFileDialog, QMessageBox, QTextEdit,
        QSizePolicy, QStatusBar, QFrame, QListWidget, QListWidgetItem,
        QAbstractItemView, QProgressDialog, QStackedWidget,
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui  import QFont, QColor
    PYQT = 6
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QGroupBox, QLabel, QDoubleSpinBox, QPushButton,
        QCheckBox, QComboBox, QFileDialog, QMessageBox, QTextEdit,
        QSizePolicy, QStatusBar, QFrame, QListWidget, QListWidgetItem,
        QAbstractItemView, QProgressDialog, QStackedWidget,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui  import QFont, QColor
    PYQT = 5

import sys, os, copy
import numpy as np
import pandas as pd
from typing import Optional
from scipy.ndimage import label as _nd_label
# Pre-import scipy.ndimage submodules and h5py at startup.
# Both have a ~100-500 ms first-import cost (scipy.ndimage parses all
# function docstrings; h5py loads HDF5 C libraries). Importing here
# moves that cost to launch time rather than the first file open.
import scipy.ndimage as _scipy_ndimage_preload   # noqa: F401
try:
    import h5py as _h5py_preload                 # noqa: F401
except ImportError:
    pass   # h5py is optional; missing it is caught gracefully in load_bgdata
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.ticker as ticker


# ========================================================================
#  DATA MODEL + ANALYSIS  (imported from beamprofiler package)
# ========================================================================
import os as _os_pkg
import sys as _sys_pkg
# Ensure the directory containing this file is on sys.path so the
# beamprofiler package can always be found regardless of cwd.
_gui_dir = _os_pkg.path.dirname(_os_pkg.path.abspath(__file__))
if _gui_dir not in _sys_pkg.path:
    _sys_pkg.path.insert(0, _gui_dir)

from beamprofiler.models import FileEntry
from beamprofiler.analysis import (
    load_csv,
    apply_sqrt,
    check_saturation,
    iso_background,
    iso_background_statistical,
    rotated_rect_mask,
    beam_size_iso,
    auto_roi,
    clip_level_widths,
    marginal_gaussian_fit,
    run_analysis,
    load_bgdata,
)

# ========================================================================
#  MATPLOTLIB CANVASES
# ========================================================================

class ImageCanvas(FigureCanvas):
    roi_selected = pyqtSignal(int, int, int, int)
    dmg_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._selector = None
        self._selector_mode = "roi"
        self._toolbar = None   # set by MainWindow after toolbar creation
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _safe_draw(self):
        """Deferred redraw - avoids synchronous render crashes on Windows."""
        QtCore.QTimer.singleShot(0, self.draw)

    def _deactivate_toolbar(self):
        """Deactivate any active pan/zoom mode on the navigation toolbar.
        Matplotlib toolbar modes capture mouse events and prevent the
        RectangleSelector from receiving them."""
        if self._toolbar is None:
            return
        # NavigationToolbar2QT stores the active mode in .mode (a string)
        # Calling pan() or zoom() while already active toggles them off.
        try:
            mode = self._toolbar.mode
            if mode == "pan/zoom":
                self._toolbar.pan()     # toggles off
            elif mode == "zoom rect":
                self._toolbar.zoom()    # toggles off
        except Exception:
            pass

    def show_image(self, img, title="", cmap="jet", roi=None,
                   unit="px", dmask=None, vmax=None):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        arr = np.asarray(img, dtype=float)
        if arr.size == 0:
            self._safe_draw()
            return
        _vmax = float(vmax) if vmax is not None else float(np.percentile(arr, 99.5)) or 1.0
        vmin  = min(0.0, float(arr.min()))
        im = self.ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=_vmax,
                            interpolation="nearest", aspect="equal")
        self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        if dmask is not None and dmask.any():
            overlay = np.zeros((*dmask.shape, 4), dtype=float)
            overlay[dmask] = [1., 0., 0., 0.45]
            self.ax.imshow(overlay, origin="upper", interpolation="nearest",
                           aspect="equal")
        if roi:
            x0, x1, y0, y1 = roi
            self.ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--"))
        self.ax.set_title(title, fontsize=9)
        self.ax.set_xlabel(f"x [{unit}]")
        self.ax.set_ylabel(f"y [{unit}]")
        self._safe_draw()

    def enable_roi_selector(self, mode="roi"):
        self._deactivate_toolbar()   # release pan/zoom before drawing ROI
        if self._selector:
            self._selector.set_active(False)
        self._selector_mode = mode
        def on_select(eclick, erelease):
            x0 = int(min(eclick.xdata, erelease.xdata))
            x1 = int(max(eclick.xdata, erelease.xdata))
            y0 = int(min(eclick.ydata, erelease.ydata))
            y1 = int(max(eclick.ydata, erelease.ydata))
            if self._selector_mode == "damage":
                self.dmg_selected.emit(x0, x1, y0, y1)
            else:
                self.roi_selected.emit(x0, x1, y0, y1)
        self._selector = RectangleSelector(self.ax, on_select, useblit=True,
            button=[1], minspanx=4, minspany=4, spancoords="pixels", interactive=True)
        self._safe_draw()

    def disable_roi_selector(self):
        if self._selector:
            self._selector.set_active(False)
            self._selector = None


class ResultsCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _safe_draw(self):
        """Deferred redraw - avoids synchronous render crashes on Windows."""
        QtCore.QTimer.singleShot(0, self.draw)

    def plot(self, roi_img, bg_img, r, roi, px, unit, cmap="jet",
             vmax_raw=None, use_tpa=True):
        """
        vmax_raw : ADC ceiling (e.g. 255) from the raw image spinbox.
        use_tpa  : if True, analysis panels use sqrt(vmax_raw) as their
                   colour ceiling since the image is sqrt-transformed.
        """

        fit_mode = r.get("fit_mode", "lab")
        self.fig.clear()
        gs  = self.fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax4 = self.fig.add_subplot(gs[1, 1])

        x0_off = (roi[0] if roi else 0) * px
        y0_off = (roi[2] if roi else 0) * px
        ny, nx = bg_img.shape
        x_ax = np.arange(nx) * px + x0_off
        y_ax = np.arange(ny) * px + y0_off

        # Panel 1 shows raw ADC values (roi_img = clean_raw crop, no sqrt).
        # Panel 2 shows bg_img = background-subtracted + TPA-corrected.
        # Their colour ceilings are therefore different:
        #   panel 1: always vmax_raw (ADC ceiling, e.g. 255)
        #   panel 2: sqrt(vmax_raw) if TPA on, else vmax_raw
        _vmax_p1 = float(vmax_raw) if vmax_raw is not None else (
            float(np.percentile(roi_img, 99.5)) or 1.0)
        _vmax_p2 = (float(np.sqrt(vmax_raw)) if (use_tpa and vmax_raw is not None)
                    else _vmax_p1)

        # -- panel 1: raw ROI image (no BG sub, no sqrt) ---------------
        im1 = ax1.imshow(roi_img, origin="upper", cmap=cmap,
                         vmin=0, vmax=_vmax_p1, interpolation="nearest", aspect="equal")
        ax1.set_title("Raw ROI image (before BG sub)", fontsize=8)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v*px + x0_off:.0f}"))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v*px + y0_off:.0f}"))
        ax1.set_xlabel(f"x [{unit}]"); ax1.set_ylabel(f"y [{unit}]")
        self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # -- panel 2: BG-subtracted + TPA-corrected + ellipse ----------
        im2 = ax2.imshow(bg_img, origin="upper", cmap=cmap,
                         vmin=0, vmax=_vmax_p2, interpolation="nearest", aspect="equal")

        # centroid in ROI-local pixel coords
        cx_px = (r["x_bar"] - x0_off) / px if px else (r["x_bar"] - x0_off)
        cy_px = (r["y_bar"] - y0_off) / px if px else (r["y_bar"] - y0_off)
        ax2.plot(cx_px, cy_px, "+", color="white", ms=12, mew=2, label="centroid")

        if fit_mode == "lab":
            d4x_px = r["d4\u03c3_x"] / px if px else r["d4\u03c3_x"]
            d4y_px = r["d4\u03c3_y"] / px if px else r["d4\u03c3_y"]
            ax2.add_patch(Ellipse((cx_px, cy_px), width=d4x_px, height=d4y_px,
                                   angle=0, linewidth=1.8, edgecolor="white",
                                   facecolor="none", label="D4\u03c3 (lab)"))
            title2 = "BG-subtracted + D4\u03c3 (lab axes)"
        else:
            d4maj_px = r["d4\u03c3_maj"] / px if px else r["d4\u03c3_maj"]
            d4min_px = r["d4\u03c3_min"] / px if px else r["d4\u03c3_min"]
            ax2.add_patch(Ellipse((cx_px, cy_px),
                                   width=d4maj_px, height=d4min_px,
                                   angle=r["\u03b8_deg"],
                                   linewidth=1.8, edgecolor="white",
                                   facecolor="none", label="D4\u03c3 (principal)"))
            title2 = f"BG-subtracted + D4\u03c3  \u03b8={r['\u03b8_deg']:.1f}\u00b0"

        ax2.legend(fontsize=7, loc="upper right")
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v*px + x0_off:.0f}"))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v*px + y0_off:.0f}"))
        ax2.set_title(title2, fontsize=8)
        ax2.set_xlabel(f"x [{unit}]"); ax2.set_ylabel(f"y [{unit}]")
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # -- panels 3 & 4: marginal profiles ----------------------
        prof_x = bg_img.sum(axis=0)
        prof_y = bg_img.sum(axis=1)
        (_, mu_x, sig_x, _off_x), fit_x = marginal_gaussian_fit(prof_x, x_ax)
        (_, mu_y, sig_y, _off_y), fit_y = marginal_gaussian_fit(prof_y, y_ax)

        # clip-level markers
        pk_x  = float(prof_x.max()); thr_x = 0.135 * pk_x
        pk_y  = float(prof_y.max()); thr_y = 0.135 * pk_y

        if fit_mode == "lab":
            sx_lbl = f"\u03c3\u2093={r['\u03c3_x']:.2f} {unit}"
            sy_lbl = f"\u03c3\u1d67={r['\u03c3_y']:.2f} {unit}"
            ctr_x  = r["x_bar"]; hw_x = 2*r["\u03c3_x"]
            ctr_y  = r["y_bar"]; hw_y = 2*r["\u03c3_y"]
        else:
            sx_lbl = (f"\u03c3_maj={r['\u03c3_maj']:.2f} {unit}"
                      f"  \u03b5={r['ellipticity']:.3f}")
            sy_lbl = (f"\u03c3_min={r['\u03c3_min']:.2f} {unit}"
                      f"  \u03b8={r['\u03b8_deg']:.1f}\u00b0")
            ctr_x  = r["x_bar"]; hw_x = 2*r["\u03c3_maj"]
            ctr_y  = r["y_bar"]; hw_y = 2*r["\u03c3_min"]

        # x profile
        ax3.plot(x_ax, prof_x, color="steelblue", lw=1.5,
                 label="\u222b dy \u00b7 I(x,y)")
        ax3.plot(x_ax, fit_x,  color="tomato",    lw=1.5, ls="--",
                 label=f"Gauss fit  {sx_lbl}")
        ax3.axvline(ctr_x, color="gray", ls=":", lw=1)
        ax3.axvspan(ctr_x - hw_x, ctr_x + hw_x,
                    alpha=0.12, color="tomato", label="\u00b12\u03c3")
        if thr_x > 0:
            _clip_x_lbl = (r.get('clip_maj', r['clip_x']) if fit_mode == "principal"
                           else r['clip_x'])
            _clip_x_key = "maj" if fit_mode == "principal" else "x"
            ax3.axhline(thr_x, color="orange", ls="--", lw=1,
                        label=f"13.5%  \u2192  {_clip_x_lbl:.2f} {unit} (clip_{_clip_x_key})")
        ax3.set_title("Marginal x-profile" if fit_mode == "lab" else
                      "Marginal profile (major axis)", fontsize=8)
        ax3.set_xlabel(f"x [{unit}]"); ax3.set_ylabel("Intensity [a.u.]")
        ax3.legend(fontsize=7)

        # y profile
        ax4.plot(y_ax, prof_y, color="steelblue", lw=1.5,
                 label="\u222b dx \u00b7 I(x,y)")
        ax4.plot(y_ax, fit_y,  color="tomato",    lw=1.5, ls="--",
                 label=f"Gauss fit  {sy_lbl}")
        ax4.axvline(ctr_y, color="gray", ls=":", lw=1)
        ax4.axvspan(ctr_y - hw_y, ctr_y + hw_y,
                    alpha=0.12, color="tomato", label="\u00b12\u03c3")
        if thr_y > 0:
            _clip_y_lbl = (r.get('clip_min', r['clip_y']) if fit_mode == "principal"
                           else r['clip_y'])
            _clip_y_key = "min" if fit_mode == "principal" else "y"
            ax4.axhline(thr_y, color="orange", ls="--", lw=1,
                        label=f"13.5%  \u2192  {_clip_y_lbl:.2f} {unit} (clip_{_clip_y_key})")
        ax4.set_title("Marginal y-profile" if fit_mode == "lab" else
                      "Marginal profile (minor axis)", fontsize=8)
        ax4.set_xlabel(f"y [{unit}]"); ax4.set_ylabel("Intensity [a.u.]")
        ax4.legend(fontsize=7)

        self._safe_draw()



# ========================================================================
#  COLLAPSIBLE SECTION WIDGET
# ========================================================================

class CollapsibleSection(QWidget):
    """
    A titled section that can be collapsed/expanded by clicking its header.
    Replaces QGroupBox for a more modern sidebar feel.
    """
    def __init__(self, title: str, parent=None, expanded: bool = True):
        super().__init__(parent)
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 2)
        outer.setSpacing(0)

        # Header button
        self._toggle = QPushButton(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px 8px;
                font-weight: bold;
                font-size: 9pt;
                border: none;
                border-radius: 5px;
                background-color: #313145;
                color: #4a9eff;
            }
            QPushButton:checked   { background-color: #2a2a3e; }
            QPushButton:hover     { background-color: #44446a; }
        """)
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        # Content container
        self._container = QWidget()
        self._container.setStyleSheet(
            "QWidget { background: #2a2a3e; border-radius: 4px; }")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(6, 4, 6, 6)
        self._container_layout.setSpacing(4)
        self._container.setVisible(expanded)
        outer.addWidget(self._container)

    def _on_toggle(self, checked: bool):
        self._expanded = checked
        arrow = "▼" if checked else "▶"
        # update arrow prefix without changing rest of title
        txt = self._toggle.text()
        # strip existing arrow if present
        for a in ("▼ ", "▶ "):
            if txt.startswith(a):
                txt = txt[len(a):]
        self._toggle.setText(arrow + " " + txt)
        self._container.setVisible(checked)

    def layout(self):            # noqa: A003
        return self._container_layout

    def addWidget(self, w):
        self._container_layout.addWidget(w)

    def addLayout(self, lay):
        self._container_layout.addLayout(lay)


# ========================================================================
#  MAIN WINDOW
# ========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("D4σ Beam Profiler  —  2-Photon Absorption")
        self.resize(1500, 860)

        self._files: list[FileEntry] = []
        self._current_idx: int = -1
        self._dark_frame = None   # shared dark frame

        self._build_ui()
        self._connect_signals()
        self._status("Ready - add CSV files to start.")

    # -- properties for the currently active FileEntry -----------------
    @property
    def _fe(self) -> Optional[FileEntry]:
        if 0 <= self._current_idx < len(self._files):
            return self._files[self._current_idx]
        return None

    # -- UI ------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        # ===========================================================
        # LEFT PANEL  -  two-column scrollable sidebar
        # Column A (left):  Files * Pixel size * ROI * Damage
        # Column B (right): Background * Display * Analysis * Save
        # Both columns sit inside a QScrollArea so nothing is clipped
        # on small screens.
        # ===========================================================

        # QScrollArea is already imported at the top of the file
        # via the Qt compatibility shim - use it directly.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(530)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        two_col = QHBoxLayout(scroll_content)
        two_col.setContentsMargins(2, 2, 2, 2)
        two_col.setSpacing(4)

        col_a = QVBoxLayout(); col_a.setSpacing(6)
        col_b = QVBoxLayout(); col_b.setSpacing(6)
        two_col.addLayout(col_a)
        sep_cols = QFrame(); sep_cols.setFrameShape(QFrame.Shape.VLine)
        two_col.addWidget(sep_cols)
        two_col.addLayout(col_b)
        scroll.setWidget(scroll_content)

        # -- COLUMN A ----------------------------------------------

        # 1  —  Files
        grp_file = CollapsibleSection("1  —  Files")
        fl = grp_file.layout()
        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("📂 Add…")
        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.setEnabled(False)
        btn_row.addWidget(self.btn_add); btn_row.addWidget(self.btn_remove)
        fl.addLayout(btn_row)
        self.lst_files = QListWidget()
        self.lst_files.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lst_files.setFixedHeight(100)
        fl.addWidget(self.lst_files)
        hdr_row = QHBoxLayout()
        self.chk_header = QCheckBox("Header row")
        self.chk_index  = QCheckBox("Index col")
        hdr_row.addWidget(self.chk_header); hdr_row.addWidget(self.chk_index)
        fl.addLayout(hdr_row)
        col_a.addWidget(grp_file)

        # 2  -  Pixel size
        grp_px = CollapsibleSection("2  \u2014  Pixel size")
        px_lay = QHBoxLayout()
        self.spin_px = QDoubleSpinBox()
        self.spin_px.setRange(0.01, 1e6); self.spin_px.setValue(1.0)
        self.spin_px.setSuffix(" \u00b5m"); self.spin_px.setDecimals(3)
        px_lay.addWidget(QLabel("Size:")); px_lay.addWidget(self.spin_px)
        grp_px.addLayout(px_lay)
        col_a.addWidget(grp_px)

        # 3 * ROI
        grp_roi = CollapsibleSection("3  —  Region of Interest")
        rl = grp_roi.layout()
        self.btn_roi_auto     = QPushButton("\U0001f50d Auto ROI (this file)")
        self.btn_roi_auto_all = QPushButton("\U0001f50d\u2715 Auto ROI all files")
        self.btn_roi_auto_all.setEnabled(False)
        self.btn_roi_auto_all.setToolTip(
            "Run auto-ROI on every loaded file using the current Pad (\u03c3) setting.")
        self.btn_roi_apply_all = QPushButton("\u25a1\u2192\u25a1 Apply this ROI to all files")
        self.btn_roi_apply_all.setEnabled(False)
        self.btn_roi_apply_all.setToolTip(
            "Copy the current file's ROI to every other loaded file.\n"
            "Useful when the beam is at the same position in all images.")
        self.btn_roi_draw  = QPushButton("\u270f Draw ROI manually")
        self.btn_roi_clear = QPushButton("\u2715 Clear ROI")
        self.btn_roi_clear.setEnabled(False)
        self.lbl_roi = QLabel("ROI: full image")
        self.lbl_roi.setStyleSheet("font-size: 9px; color: #7f849c;")
        self.lbl_roi.setWordWrap(True)
        sep_auto = QFrame(); sep_auto.setFrameShape(QFrame.Shape.HLine)
        pad_row  = QHBoxLayout()
        self.spin_pad_sigma = QDoubleSpinBox()
        self.spin_pad_sigma.setRange(1.0, 10.0); self.spin_pad_sigma.setValue(4.0)
        self.spin_pad_sigma.setSingleStep(0.5)
        pad_row.addWidget(QLabel("Pad (\u03c3):")); pad_row.addWidget(self.spin_pad_sigma)
        self.chk_auto_roi_all = QCheckBox("Re-run auto ROI when running all")
        self.chk_auto_roi_all.setToolTip(
            "When using 'Run all files', recompute auto-ROI for every file.\n"
            "Uncheck to keep manually set ROIs.")
        self.chk_show_dmask = QCheckBox("Show damage mask (red)")
        self.chk_show_dmask.setChecked(True)
        rl.addWidget(self.btn_roi_auto)
        rl.addWidget(self.btn_roi_auto_all)
        rl.addWidget(self.btn_roi_apply_all)
        rl.addWidget(self.btn_roi_draw)
        rl.addWidget(self.btn_roi_clear)
        rl.addWidget(self.lbl_roi)
        rl.addWidget(sep_auto)
        rl.addLayout(pad_row)
        rl.addWidget(self.chk_auto_roi_all)
        rl.addWidget(self.chk_show_dmask)
        col_a.addWidget(grp_roi)

        # 3b —  Damage masking
        grp_dmg = CollapsibleSection("3b —  Damage masking")
        dl = grp_dmg.layout()
        self.btn_dmg_draw     = QPushButton("⚠ Mark damage (this file)")
        self.btn_dmg_draw.setEnabled(False)
        self.btn_dmg_draw.setToolTip(
            "Draw a rectangle over damaged pixels.\n"
            "Pixels are replaced by the mean of pixels below the threshold.\n"
            "Multiple draws are additive.")
        self.btn_dmg_apply_all = QPushButton("⚠⚠ Apply damage to all files")
        self.btn_dmg_apply_all.setEnabled(False)
        self.btn_dmg_apply_all.setToolTip(
            "Copy the current file's damage mask to every other loaded file.\n"
            "Each file's raw data is reprocessed using the same replacement\n"
            "regions and the current BG threshold.")
        self.btn_dmg_clear    = QPushButton("✕ Clear damage (this file)")
        self.btn_dmg_clear.setEnabled(False)
        thr_row = QHBoxLayout()
        self.spin_dmg_thr = QDoubleSpinBox()
        self.spin_dmg_thr.setRange(0, 1e6); self.spin_dmg_thr.setValue(50)
        self.spin_dmg_thr.setDecimals(1)
        self.spin_dmg_thr.setToolTip("Pixels < threshold are treated as background.")
        thr_row.addWidget(QLabel("BG thr:")); thr_row.addWidget(self.spin_dmg_thr)
        self.lbl_dmg = QLabel("No damage marked.")
        self.lbl_dmg.setStyleSheet("font-size: 9px; color: #7f849c;")
        self.lbl_dmg.setWordWrap(True)
        dl.addWidget(self.btn_dmg_draw)
        dl.addWidget(self.btn_dmg_apply_all)
        dl.addLayout(thr_row)
        dl.addWidget(self.btn_dmg_clear)
        dl.addWidget(self.lbl_dmg)
        col_a.addWidget(grp_dmg)
        col_a.addStretch()

        # -- COLUMN B ----------------------------------------------

        # 4 - Background
        grp_bg = CollapsibleSection("4  \u2014  Background (ISO 11146)")
        bl = grp_bg.layout()

        bg_mode_row = QHBoxLayout()
        bg_mode_row.addWidget(QLabel("Method:"))
        self.combo_bg_mode = QComboBox()
        self.combo_bg_mode.addItems([
            "Off (no subtraction)",
            "Corner estimate  (\u00a7\u200b3.4.3)",
            "ISO statistical  (\u00a7\u200b3.4.2)",
        ])
        self.combo_bg_mode.setCurrentIndex(1)   # default: corner
        self.combo_bg_mode.setToolTip(
            "Off: skip background subtraction entirely.\n"
            "\n"
            "Corner estimate (ISO/TR 11146-3 \u00a73.4.3):\n"
            "  Fast approximation. Corner patches seed the estimate;\n"
            "  unilluminated pixels refine it. Pixels below n\u00b7\u03c3\n"
            "  are zeroed after subtraction.\n"
            "  Sufficient for beams < 0.5\u00d7 sensor size.\n"
            "\n"
            "ISO statistical (\u00a73.4.2, fully standard-compliant):\n"
            "  2D local-mean convolution identifies illuminated pixels.\n"
            "  Negatives are KEPT after subtraction as required by \u00a73.1\n"
            "  so that positive and negative noise cancel in the integral.\n"
            "  Recommended for small beams or high-accuracy measurements.")
        bg_mode_row.addWidget(self.combo_bg_mode)
        bl.addLayout(bg_mode_row)

        sep_bg0 = QFrame(); sep_bg0.setFrameShape(QFrame.Shape.HLine); bl.addWidget(sep_bg0)

        # dark frame (optional)
        self.btn_dark = QPushButton("\U0001f4c2 Load dark frame\u2026")
        self.lbl_dark = QLabel("No dark frame (optional).")
        self.lbl_dark.setStyleSheet("font-size: 9px; color:#7f849c;")
        self.lbl_dark.setWordWrap(True)
        bl.addWidget(self.btn_dark); bl.addWidget(self.lbl_dark)

        sep_bg1 = QFrame(); sep_bg1.setFrameShape(QFrame.Shape.HLine); bl.addWidget(sep_bg1)

        corner_row = QHBoxLayout()
        self.spin_corner_pct = QDoubleSpinBox()
        self.spin_corner_pct.setRange(1.0, 20.0); self.spin_corner_pct.setValue(3.5)
        self.spin_corner_pct.setSuffix(" %")
        self.spin_corner_pct.setToolTip(
            "Corner patch size as % of shorter image dimension.\n"
            "Used by the corner-estimate method (\u00a73.4.3).\n"
            "ISO recommends 2\u20135% of image size.")
        corner_row.addWidget(QLabel("Corner:")); corner_row.addWidget(self.spin_corner_pct)
        bl.addLayout(corner_row)

        kernel_row = QHBoxLayout()
        self.spin_kernel_pct = QDoubleSpinBox()
        self.spin_kernel_pct.setRange(1.0, 10.0); self.spin_kernel_pct.setValue(3.0)
        self.spin_kernel_pct.setSuffix(" %")
        self.spin_kernel_pct.setToolTip(
            "Convolution kernel size as % of shorter image dimension.\n"
            "Used by the ISO statistical method (\u00a73.4.2).\n"
            "ISO recommends 2\u20135% of image size.")
        kernel_row.addWidget(QLabel("Kernel:")); kernel_row.addWidget(self.spin_kernel_pct)
        bl.addLayout(kernel_row)

        nsig_row = QHBoxLayout()
        self.spin_nsig = QDoubleSpinBox()
        self.spin_nsig.setRange(1.0, 10.0); self.spin_nsig.setValue(3.0)
        self.spin_nsig.setSingleStep(0.5)
        self.spin_nsig.setToolTip(
            "Threshold multiplier n\u1d57 (ISO \u00a73.4.2 specifies 2 < n\u1d57 < 4).\n"
            "Corner method: pixels below n\u00b7\u03c3_bg are zeroed after subtraction.\n"
            "ISO statistical method: used to identify illuminated pixels.")
        nsig_row.addWidget(QLabel("n\u00b7\u03c3:")); nsig_row.addWidget(self.spin_nsig)
        bl.addLayout(nsig_row)

        mask_row = QHBoxLayout()
        self.spin_mask_factor = QDoubleSpinBox()
        self.spin_mask_factor.setRange(1.0, 10.0); self.spin_mask_factor.setValue(3.0)
        self.spin_mask_factor.setSingleStep(0.5)
        self.spin_mask_factor.setToolTip(
            "Rotated rectangle mask size = factor \u00d7 D4\u03c3.\n"
            "ISO 11146-1 \u00a77 specifies 3.0.")
        mask_row.addWidget(QLabel("Mask (\u00d7D4\u03c3):")); mask_row.addWidget(self.spin_mask_factor)
        bl.addLayout(mask_row)

        col_b.addWidget(grp_bg)

        # 5  —  Display
        grp_cm = CollapsibleSection("5  \u2014  Display")
        cm_lay = grp_cm.layout()
        cmap_row = QHBoxLayout()
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["jet","inferno","hot","viridis","plasma","gray"])
        cmap_row.addWidget(QLabel("Cmap:")); cmap_row.addWidget(self.combo_cmap)
        cm_lay.addLayout(cmap_row)
        vmax_row = QHBoxLayout()
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(1, 1e9)
        self.spin_vmax.setValue(255)
        self.spin_vmax.setDecimals(0)
        self.spin_vmax.setToolTip(
            "Colour scale maximum for the image viewer and results plots.\n"
            "Default 255 = 8-bit ADC ceiling.\n"
            "Set to 65535 for 16-bit cameras.")
        vmax_row.addWidget(QLabel("vmax:")); vmax_row.addWidget(self.spin_vmax)
        cm_lay.addLayout(vmax_row)
        mode_row = QHBoxLayout()
        self.combo_fit_mode = QComboBox()
        self.combo_fit_mode.addItems(["Lab axes (x/y)", "Principal axes (maj/min)"])
        self.combo_fit_mode.setToolTip(
            "Lab axes: D4sigma in x and y directions.\n"
            "Principal axes: ISO 11146-1 S.4 tensor diagonalisation -\n"
            "  gives \u03c3_maj, \u03c3_min, orientation angle theta, ellipticity.")
        mode_row.addWidget(QLabel("Fit:")); mode_row.addWidget(self.combo_fit_mode)
        cm_lay.addLayout(mode_row)
        self.chk_tpa = QCheckBox("TPA correction (\u221a transform)")
        self.chk_tpa.setChecked(True)
        self.chk_tpa.setToolTip(
            "Apply sign(x)\u00b7\u221a|x| before analysis.\n"
            "Enable for two-photon absorption cameras (signal \u221d I\u00b2).\n"
            "Disable for linear-response cameras.")
        cm_lay.addWidget(self.chk_tpa)
        col_b.addWidget(grp_cm)

        # 6  —  Analysis
        grp_run = CollapsibleSection("6  \u2014  Analysis")
        run_lay = grp_run.layout()
        self.btn_run = QPushButton("\u25b6  Run (this file)")
        self.btn_run.setEnabled(False)
        self.btn_run.setObjectName("btn_run")
        self.btn_run_all = QPushButton("\u25b6\u25b6  Run all files")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setObjectName("btn_run_all")
        run_lay.addWidget(self.btn_run); run_lay.addWidget(self.btn_run_all)
        col_b.addWidget(grp_run)

        # Results text
        grp_res = CollapsibleSection("Results")
        res_lay = grp_res.layout()
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setFont(QFont("Courier", 8))
        self.txt_results.setFixedHeight(190)
        res_lay.addWidget(self.txt_results)
        col_b.addWidget(grp_res)

        # Save buttons
        self.btn_save        = QPushButton("💾  Save selected…")
        self.btn_save.setEnabled(False)
        self.btn_save_series = QPushButton("📊  Save series report…")
        self.btn_save_series.setEnabled(False)
        col_b.addWidget(self.btn_save)
        col_b.addWidget(self.btn_save_series)
        col_b.addStretch()

        # -- RIGHT PANEL  —  single canvas with view selector ----------
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Header bar: view selector + file name label
        hdr_widget = QWidget()
        hdr_widget.setStyleSheet("background: #2a2a3e; border-bottom: 1px solid #44446a;")
        hdr_lay = QHBoxLayout(hdr_widget)
        hdr_lay.setContentsMargins(8, 4, 8, 4)

        self.combo_view = QComboBox()
        self.combo_view.addItems(["Raw", "Analysis results"])
        self.combo_view.setFixedWidth(200)
        hdr_lay.addWidget(self.combo_view)
        hdr_lay.addStretch()
        right_layout.addWidget(hdr_widget)

        # Stacked widget: page 0 = ImageCanvas, page 1 = ResultsCanvas
        self._stack = QStackedWidget()

        # Page 0 — image viewer
        img_page = QWidget()
        img_lay  = QVBoxLayout(img_page)
        img_lay.setContentsMargins(0, 0, 0, 0)
        self.img_canvas = ImageCanvas()
        self.img_nav    = NavigationToolbar(self.img_canvas, self)
        self.img_canvas._toolbar = self.img_nav   # so canvas can deactivate toolbar
        img_lay.addWidget(self.img_nav)
        img_lay.addWidget(self.img_canvas)
        self._stack.addWidget(img_page)

        # Page 1 — analysis results
        res_page = QWidget()
        res_lay  = QVBoxLayout(res_page)
        res_lay.setContentsMargins(0, 0, 0, 0)
        self.res_canvas = ResultsCanvas()
        self.res_nav    = NavigationToolbar(self.res_canvas, self)
        res_lay.addWidget(self.res_nav)
        res_lay.addWidget(self.res_canvas)
        self._stack.addWidget(res_page)

        right_layout.addWidget(self._stack, stretch=1)

        # Sidebar toggle button — sits between the scroll panel and the plots
        self._sidebar = scroll          # keep reference for toggling
        self._sidebar_toggle = QPushButton("\u25c4")   # ◄
        self._sidebar_toggle.setFixedWidth(16)
        self._sidebar_toggle.setToolTip("Hide / show settings panel")
        self._sidebar_toggle.setStyleSheet("""
            QPushButton {
                background: #313145;
                color: #4a9eff;
                border: none;
                border-radius: 0px;
                font-size: 10pt;
                padding: 0px;
            }
            QPushButton:hover { background: #44446a; }
        """)
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)

        root.addWidget(scroll)
        root.addWidget(self._sidebar_toggle)
        root.addWidget(right, stretch=1)
        self.setStatusBar(QStatusBar())

    # -- signals -------------------------------------------------------

    def _connect_signals(self):
        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_remove.clicked.connect(self._on_remove_file)
        self.lst_files.currentRowChanged.connect(self._on_file_selected)
        self.btn_dark.clicked.connect(self._on_load_dark)
        self.btn_roi_auto.clicked.connect(self._on_auto_roi)
        self.btn_roi_auto_all.clicked.connect(self._on_auto_roi_all)
        self.btn_roi_apply_all.clicked.connect(self._on_roi_apply_all)
        self.btn_roi_draw.clicked.connect(self._on_roi_draw)
        self.btn_roi_clear.clicked.connect(self._on_roi_clear)
        self.btn_dmg_draw.clicked.connect(self._on_dmg_draw)
        self.btn_dmg_apply_all.clicked.connect(self._on_dmg_apply_all)
        self.btn_dmg_clear.clicked.connect(self._on_dmg_clear)
        self.img_canvas.roi_selected.connect(self._on_roi_received)
        self.img_canvas.dmg_selected.connect(self._on_dmg_received)

        self.combo_view.currentIndexChanged.connect(self._on_view_changed)
        self.combo_cmap.currentTextChanged.connect(self._refresh_image_view)
        self.spin_vmax.valueChanged.connect(self._refresh_image_view)
        self.combo_fit_mode.currentTextChanged.connect(self._on_fit_mode_changed)
        self.chk_tpa.stateChanged.connect(self._on_tpa_changed)
        self.chk_show_dmask.stateChanged.connect(lambda _: self._refresh_image_view())
        self.btn_run.clicked.connect(self._on_run)
        self.btn_run_all.clicked.connect(self._on_run_all)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_series.clicked.connect(self._on_save_series)

    def _on_tpa_changed(self):
        """TPA toggle changed: re-run analysis to apply the corrected pipeline."""
        self._refresh_image_view()
        self._status("TPA correction " +
                     ("enabled" if self.chk_tpa.isChecked() else "disabled") +
                     " \u2014 re-run analysis to update results.")

    def _toggle_sidebar(self):
        visible = self._sidebar.isVisible()
        self._sidebar.setVisible(not visible)
        self._sidebar_toggle.setText("\u25ba" if visible else "\u25c4")  # ► / ◄

    # -- file list management ------------------------------------------

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add image files", "",
            "Image files (*.csv *.txt *.dat *.bgData *.bgdata);;"
            "CSV files (*.csv *.txt *.dat);;"
            "BeamGage files (*.bgData *.bgdata);;"
            "All files (*)")
        if not paths:
            return
        added = 0
        for path in paths:
            if any(f.path == path for f in self._files):
                continue
            try:
                raw, meta = self._load_image_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"{os.path.basename(path)}: {e}")
                continue

            fe = FileEntry(path=path, raw=raw, clean_raw=raw.copy())
            self._files.append(fe)
            item = QListWidgetItem(fe.fname)
            item.setToolTip(path)
            self.lst_files.addItem(item)
            added += 1

            # auto-populate pixel size and vmax from .bgData metadata
            if meta:
                if meta.get("pixel_size_x") and meta["pixel_size_x"] > 0:
                    self.spin_px.setValue(meta["pixel_size_x"])
                if meta.get("bits"):
                    adc_ceiling = float(2 ** meta["bits"] - 1)
                    self.spin_vmax.setValue(adc_ceiling)
                if meta.get("is_bgdata"):
                    self.chk_tpa.setChecked(False)

        if added:
            self.btn_run_all.setEnabled(True)
            self.btn_roi_auto_all.setEnabled(True)
            if self._current_idx < 0:
                self.lst_files.setCurrentRow(0)
            self._status(f"Added {added} file(s). Total: {len(self._files)}")

    def _load_image_file(self, path: str) -> tuple:
        """
        Load any supported image file. Returns (raw_array, metadata_dict).
        metadata_dict is empty for CSV; contains pixel_size_x/bits/is_bgdata for .bgData.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in (".bgdata",):
            info = load_bgdata(path, frame=1)
            info["is_bgdata"] = True
            return info["image"], info
        else:
            raw = load_csv(path, self.chk_header.isChecked(),
                           self.chk_index.isChecked())
            return raw, {}

    def _on_remove_file(self):
        row = self.lst_files.currentRow()
        if row < 0: return
        self.lst_files.takeItem(row)
        self._files.pop(row)
        if not self._files:
            self._current_idx = -1
            self.btn_run.setEnabled(False)
            self.btn_run_all.setEnabled(False)
            self.btn_remove.setEnabled(False)
            self.btn_save.setEnabled(False)
            self._clear_display()
        else:
            new_row = min(row, len(self._files)-1)
            self.lst_files.setCurrentRow(new_row)

    def _on_file_selected(self, row):
        if row < 0 or row >= len(self._files):
            self._current_idx = -1
            return
        self._current_idx = row
        fe = self._fe
        self.btn_remove.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_dmg_draw.setEnabled(True)

        # sync ROI / damage labels
        if fe.roi:
            x0,x1,y0,y1 = fe.roi
            self.lbl_roi.setText(f"ROI: x=[{x0},{x1}), y=[{y0},{y1})")
            self.btn_roi_clear.setEnabled(True)
            self.btn_roi_apply_all.setEnabled(True)
        else:
            self.lbl_roi.setText("ROI: full image")
            self.btn_roi_clear.setEnabled(False)
            self.btn_roi_apply_all.setEnabled(False)

        if fe.dmask is not None:
            self.lbl_dmg.setText(f"{fe.dmask.sum()} px masked")
            self.btn_dmg_clear.setEnabled(True)
        else:
            self.lbl_dmg.setText("No damage marked.")
            self.btn_dmg_clear.setEnabled(False)

        if fe.results:
            self._show_results_text(fe.results)
            self.btn_save.setEnabled(True)
        else:
            self.txt_results.clear()
            self.btn_save.setEnabled(False)

        self._refresh_image_view()
        if fe.results and fe.roi_img is not None and fe.bg_img is not None:
            r = fe.results
            self.res_canvas.plot(fe.roi_img, fe.bg_img, r,
                fe.roi, r["pixel_size_um"], r["unit"], self.combo_cmap.currentText(),
                vmax_raw=self.spin_vmax.value(), use_tpa=self.chk_tpa.isChecked())
            # show results page if analysis is done, otherwise image page
            self.combo_view.setCurrentIndex(1)
        else:
            self.combo_view.setCurrentIndex(0)

    def _update_list_item(self, idx):
        """Colour list item: green=done, orange=done+saturated, grey=pending."""
        item = self.lst_files.item(idx)
        if item is None: return
        fe = self._files[idx]
        if fe.done:
            if fe.results.get("saturation_warning"):
                item.setForeground(QColor("#fab387"))   # orange = saturated
                item.setText(f"⚠ {fe.fname}")
            else:
                item.setForeground(QColor("#a6e3a1"))   # green = clean
                item.setText(f"✓ {fe.fname}")
        else:
            item.setForeground(QColor("#7f849c"))
            item.setText(fe.fname)

    # -- ROI -----------------------------------------------------------

    def _on_auto_roi(self):
        fe = self._fe
        if fe is None: return
        self._status("Running auto-ROI ...")
        QApplication.processEvents()
        try:
            roi = auto_roi(fe.clean_raw, pad_sigma=self.spin_pad_sigma.value())
        except Exception as e:
            QMessageBox.critical(self, "Auto-ROI error", str(e)); return
        fe.roi = roi
        x0,x1,y0,y1 = roi
        self.lbl_roi.setText(f"Auto ROI: x=[{x0},{x1}), y=[{y0},{y1})")
        self.btn_roi_clear.setEnabled(True)
        self.btn_roi_apply_all.setEnabled(True)
        self._refresh_image_view()
        self._status(f"Auto ROI: x=[{x0},{x1}), y=[{y0},{y1})")

    def _on_roi_apply_all(self):
        """Copy the current file's ROI to every other loaded file."""
        fe = self._fe
        if fe is None or fe.roi is None:
            QMessageBox.information(self, "No ROI",
                "Set a ROI on the current file first."); return
        roi = fe.roi
        n = 0
        for other in self._files:
            if other is fe: continue
            other.roi = roi
            n += 1
        self._status(f"ROI {roi} applied to {n} other file(s).")

    def _on_roi_draw(self):
        if self._fe is None: return
        self._status("Draw a rectangle on the image.")
        self.img_canvas.enable_roi_selector()

    def _on_roi_received(self, x0, x1, y0, y1):
        self.img_canvas.disable_roi_selector()
        fe = self._fe
        if fe is None: return
        ny, nx = fe.clean_raw.shape
        x0=max(0,x0); x1=min(nx,x1); y0=max(0,y0); y1=min(ny,y1)
        if x1-x0 < 2 or y1-y0 < 2: return
        fe.roi = (x0,x1,y0,y1)
        self.lbl_roi.setText(f"ROI: x=[{x0},{x1}), y=[{y0},{y1})")
        self.btn_roi_clear.setEnabled(True)
        self.btn_roi_apply_all.setEnabled(True)
        self._refresh_image_view()

    def _on_roi_clear(self):
        fe = self._fe
        if fe is None: return
        fe.roi = None
        self.lbl_roi.setText("ROI: full image")
        self.btn_roi_clear.setEnabled(False)
        self._refresh_image_view()

    # -- damage masking ------------------------------------------------

    def _on_dmg_draw(self):
        if self._fe is None: return
        self._status("Draw a rectangle over the damage region.")
        self.img_canvas.enable_roi_selector(mode="damage")

    def _on_dmg_received(self, x0, x1, y0, y1):
        self.img_canvas.disable_roi_selector()
        fe = self._fe
        if fe is None: return
        ny, nx = fe.raw.shape
        x0=max(0,x0); x1=min(nx,x1); y0=max(0,y0); y1=min(ny,y1)
        if x1-x0 < 1 or y1-y0 < 1: return
        thr     = self.spin_dmg_thr.value()
        bg_mean = float(np.mean(fe.raw[fe.raw < thr])) if (fe.raw < thr).any() else 0.0
        fe.clean_raw[y0:y1, x0:x1] = bg_mean
        if fe.dmask is None:
            fe.dmask = np.zeros((ny, nx), dtype=bool)
        fe.dmask[y0:y1, x0:x1] = True
        n_px = int(fe.dmask.sum())
        self.lbl_dmg.setText(f"{n_px} px masked  (bg={bg_mean:.1f}, thr<{thr:.0f})")
        self.btn_dmg_clear.setEnabled(True)
        self.btn_dmg_apply_all.setEnabled(True)
        self._refresh_image_view()
        self._status(f"Damage replaced with bg mean {bg_mean:.2f}.")

    def _on_auto_roi_all(self):
        """Run auto-ROI on every loaded file."""
        if not self._files: return
        pad = self.spin_pad_sigma.value()
        prog = QProgressDialog("Auto-ROI...", "Cancel", 0, len(self._files), self)
        prog.setWindowTitle("Auto ROI all files"); prog.setMinimumDuration(0)
        prog.setValue(0)
        for i, fe in enumerate(self._files):
            if prog.wasCanceled(): break
            prog.setLabelText(f"{fe.fname}  ({i+1}/{len(self._files)})")
            QApplication.processEvents()
            try:
                fe.roi = auto_roi(fe.clean_raw, pad_sigma=pad)
            except Exception as e:
                self._status(f"Auto-ROI failed for {fe.fname}: {e}")
            prog.setValue(i+1)
        prog.close()
        # refresh display for current file
        self._on_file_selected(self._current_idx)
        self._status(f"Auto-ROI done for {len(self._files)} file(s).")

    def _on_dmg_apply_all(self):
        """Copy the current file's damage regions to every other file."""
        fe = self._fe
        if fe is None or fe.dmask is None:
            QMessageBox.information(self, "No damage mask",
                "Mark a damage region on this file first."); return
        thr = self.spin_dmg_thr.value()
        # Collect the bounding rectangles from the mask so we can replay them
        # on images that may have different per-file background levels.
        labeled, n = _nd_label(fe.dmask)
        rects = []
        for k in range(1, n+1):
            ys, xs = np.where(labeled == k)
            rects.append((xs.min(), xs.max()+1, ys.min(), ys.max()+1))

        n_applied = 0
        for other in self._files:
            if other is fe: continue
            ny, nx = other.raw.shape
            # reset clean_raw and dmask for this file, then replay rectangles
            other.clean_raw = other.raw.copy()
            other.dmask     = np.zeros((ny, nx), dtype=bool)
            for x0, x1, y0, y1 in rects:
                x0c = max(0, x0); x1c = min(nx, x1)
                y0c = max(0, y0); y1c = min(ny, y1)
                if x1c <= x0c or y1c <= y0c: continue
                bg_mean = float(np.mean(other.raw[other.raw < thr])) \
                          if (other.raw < thr).any() else 0.0
                other.clean_raw[y0c:y1c, x0c:x1c] = bg_mean
                other.dmask[y0c:y1c, x0c:x1c]     = True

            n_applied += 1

        # refresh display
        self._on_file_selected(self._current_idx)
        self._status(f"Damage mask applied to {n_applied} other file(s).")

    def _on_dmg_clear(self):
        fe = self._fe
        if fe is None: return
        fe.clean_raw = fe.raw.copy()
        fe.dmask     = None
        self.lbl_dmg.setText("No damage marked.")
        self.btn_dmg_clear.setEnabled(False)
        self._refresh_image_view()

    # -- dark frame ----------------------------------------------------

    def _on_load_dark(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open dark frame", "",
            "Image files (*.csv *.txt *.dat *.bgData *.bgdata);;All files (*)")
        if not path: return
        try:
            dark, _ = self._load_image_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Dark frame load error", str(e)); return
        self._dark_frame = dark.astype(float)
        self.lbl_dark.setText(f"Dark: {os.path.basename(path)}")
        self._status(f"Dark frame loaded: {os.path.basename(path)}")

    def _on_fit_mode_changed(self, _=None):
        """Re-plot the current file with the newly selected fit mode,
        updating both the result panel and the results text."""
        fe = self._fe
        if fe is None or not fe.results: return
        # stamp the chosen mode into the stored results so text + plot agree
        fe.results["fit_mode"] = (
            "lab" if self.combo_fit_mode.currentIndex() == 0 else "principal")
        self._show_results_text(fe.results)
        r = fe.results
        self.res_canvas.plot(fe.roi_img, fe.bg_img, r,
            fe.roi, r["pixel_size_um"], r["unit"], self.combo_cmap.currentText(),
                vmax_raw=self.spin_vmax.value(), use_tpa=self.chk_tpa.isChecked())

    # -- analysis ------------------------------------------------------

    def _collect_settings(self) -> dict:
        fit_mode = ("lab" if self.combo_fit_mode.currentIndex() == 0
                    else "principal")
        _bg_idx = self.combo_bg_mode.currentIndex()
        bg_mode = ["off", "corner", "iso_statistical"][_bg_idx]
        return dict(
            px           = self.spin_px.value(),
            dark_frame   = self._dark_frame,
            corner_frac  = self.spin_corner_pct.value() / 100.0,
            kernel_frac  = self.spin_kernel_pct.value() / 100.0,
            n_sigma      = self.spin_nsig.value(),
            mask_factor  = self.spin_mask_factor.value(),
            pad_sigma    = self.spin_pad_sigma.value(),
            use_auto_roi = self.chk_auto_roi_all.isChecked(),
            fit_mode     = fit_mode,
            use_tpa      = self.chk_tpa.isChecked(),
            bg_subtract  = bg_mode != "off",
            bg_mode      = bg_mode,
            adc_ceiling  = self.spin_vmax.value(),
        )

    def _on_run(self):
        fe = self._fe
        if fe is None: return
        settings = self._collect_settings()
        if False:  # dark frame is now optional, handled inside beam_size_iso
            QMessageBox.warning(self, "No dark frame", "Load a dark frame CSV first."); return
        try:
            run_analysis(fe, settings)
        except Exception as e:
            QMessageBox.critical(self, "Analysis error", str(e)); return
        self._update_list_item(self._current_idx)
        self._show_results_text(fe.results)
        r = fe.results
        self.res_canvas.plot(fe.roi_img, fe.bg_img, r,
            fe.roi, r["pixel_size_um"], r["unit"], self.combo_cmap.currentText(),
                vmax_raw=self.spin_vmax.value(), use_tpa=self.chk_tpa.isChecked())
        self.btn_save.setEnabled(True)
        self.btn_save_series.setEnabled(any(f.done for f in self._files))
        # switch view to analysis results automatically
        self.combo_view.setCurrentIndex(1)
        self._status(f"Done: {fe.fname}")

    def _on_run_all(self):
        if not self._files: return
        settings = self._collect_settings()
        if False:  # dark frame is now optional, handled inside beam_size_iso
            QMessageBox.warning(self, "No dark frame", "Load a dark frame CSV first."); return

        prog = QProgressDialog("Running analysis...", "Cancel", 0, len(self._files), self)
        prog.setWindowTitle("Batch analysis"); prog.setMinimumDuration(0)
        prog.setValue(0)

        for i, fe in enumerate(self._files):
            if prog.wasCanceled(): break
            prog.setLabelText(f"Processing {fe.fname}  ({i+1}/{len(self._files)})")
            QApplication.processEvents()
            try:
                run_analysis(fe, settings)
            except Exception as e:
                self._status(f"Error on {fe.fname}: {e}")
            self._update_list_item(i)
            prog.setValue(i+1)

        prog.close()
        # refresh display for currently selected file
        self._on_file_selected(self._current_idx)
        self.btn_save_series.setEnabled(any(f.done for f in self._files))
        n_done = sum(1 for f in self._files if f.done)
        self._status(f"Batch complete: {n_done}/{len(self._files)} files analysed.")

    # -- results text --------------------------------------------------

    def _show_results_text(self, r):
        fi       = r.get("bg_info", {})
        unit     = r.get("unit", "px")
        fit_mode = r.get("fit_mode", "lab")
        lines = [
            "="*44,
            "  D4σ BEAM PROFILE RESULTS  (ISO 11146)",
            "="*44,
            f"  File            : {r.get('filename','')}",
            f"  Fit mode        : {fit_mode}",
            f"  TPA correction  : {'yes (\u221a transform)' if r.get('use_tpa', True) else 'no (linear)'}",
            ("  BG subtraction  : " + {
                "off":            "disabled",
                "corner":         "corner estimate (\u00a73.4.3)",
                "iso_statistical":"ISO statistical (\u00a73.4.2)",
            }.get(r.get("bg_mode","corner"), r.get("bg_mode","corner"))),
            "-"*44,
            f"  Centroid x_bar     = {r['x_bar']:10.3f}  {unit}",
            f"  Centroid y_bar     = {r['y_bar']:10.3f}  {unit}",
        ]
        if fit_mode == "lab":
            lines += [
                f"  σ_x            = {r['σ_x']:10.3f}  {unit}",
                f"  σ_y            = {r['σ_y']:10.3f}  {unit}",
                f"  D4σ_x          = {r['d4σ_x']:10.3f}  {unit}",
                f"  D4σ_y          = {r['d4σ_y']:10.3f}  {unit}",
            ]
        else:
            lines += [
                f"  σ_maj          = {r['σ_maj']:10.3f}  {unit}",
                f"  σ_min          = {r['σ_min']:10.3f}  {unit}",
                f"  D4σ_maj        = {r['d4σ_maj']:10.3f}  {unit}",
                f"  D4σ_min        = {r['d4σ_min']:10.3f}  {unit}",
                f"  theta (x-axis)     = {r['θ_deg']:10.2f}   deg",
                f"  Ellipticity    = {r['ellipticity']:10.4f}",
                f"  σ_xy           = {r['σ_xy']:10.4f}  {unit}^2",
            ]
        if fit_mode == "lab":
            lines += [
                "-"*44,
                f"  D%pk  clip_x    = {r['clip_x']:10.3f}  {unit}  (13.5%)",
                f"  D%pk  clip_y    = {r['clip_y']:10.3f}  {unit}  (13.5%)",
            ]
        else:
            lines += [
                "-"*44,
                f"  D%pk  clip_maj  = {r.get('clip_maj', r['clip_x']):10.3f}  {unit}  (13.5%)",
                f"  D%pk  clip_min  = {r.get('clip_min', r['clip_y']):10.3f}  {unit}  (13.5%)",
            ]
        lines += [
            "-"*44,
            f"  BG mean         : {fi.get('bg_mean',0):.4g}",
            f"  BG std          : {fi.get('bg_std',0):.4g}",
            f"  Zero threshold  : {fi.get('threshold',0):.4g}",
            f"  Dark frame used : {'yes' if fi.get('dark_used') else 'no'}",
            f"  ROI             : {r.get('roi','full image')}",
            "="*44,
        ]
        self.txt_results.setPlainText("\n".join(lines))

    # -- save ----------------------------------------------------------

    def _on_save(self):
        fe = self._fe
        if fe is None or not fe.results: return
        path, _ = QFileDialog.getSaveFileName(self, "Save result",
            os.path.splitext(fe.path)[0]+"_result",
            "PNG image (*.png);;PDF (*.pdf);;CSV summary (*.csv)")
        if not path: return
        base, ext = os.path.splitext(path)
        if ext.lower() == ".csv":
            self._write_csv(path, [fe])
            self.res_canvas.fig.savefig(base+"_plot.png", dpi=150, bbox_inches="tight")
        else:
            self.res_canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
            with open(base+"_summary.txt","w") as f:
                f.write(self.txt_results.toPlainText())
        QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        self._status(f"Saved: {path}")

    def _on_save_series(self):
        done = [fe for fe in self._files if fe.done]
        if not done: return
        path, _ = QFileDialog.getSaveFileName(self, "Save series report",
            "series_report", "CSV results (*.csv);;All files (*)")
        if not path: return
        base, _ = os.path.splitext(path)
        if not path.lower().endswith(".csv"):
            path = base + ".csv"

        # write combined CSV
        self._write_csv(path, done)

        # write all plots into one multi-page PDF
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = base + "_plots.pdf"
        cmap = self.combo_cmap.currentText()

        with PdfPages(pdf_path) as pdf:
            for fe in done:
                r        = fe.results
                fit_mode = r.get("fit_mode", "lab")
                px       = r["pixel_size_um"]; unit = r["unit"]
                x0_off   = (fe.roi[0] if fe.roi else 0) * px
                y0_off   = (fe.roi[2] if fe.roi else 0) * px
                ny, nx   = fe.bg_img.shape
                x_ax = np.arange(nx) * px + x0_off
                y_ax = np.arange(ny) * px + y0_off

                fig = plt.figure(figsize=(14, 10.5))
                fig.suptitle(fe.fname, fontsize=11, fontweight="bold", y=0.98)
                gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 0])
                ax4 = fig.add_subplot(gs[1, 1])

                # panel 1: raw ROI (clean_raw crop, no sqrt, no BG sub)
                _vmax_raw = self.spin_vmax.value()
                _use_tpa  = r.get("use_tpa", True)
                _vmax_p1  = float(_vmax_raw)
                _vmax_p2  = float(np.sqrt(_vmax_raw)) if _use_tpa else float(_vmax_raw)
                im1 = ax1.imshow(fe.roi_img, origin="upper", cmap=cmap,
                                 vmin=0, vmax=_vmax_p1, interpolation="nearest", aspect="equal")
                ax1.set_title("Raw ROI image (before BG sub)", fontsize=8)
                ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda v, _, _x0=x0_off, _px=px: f"{v*_px + _x0:.0f}"))
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda v, _, _y0=y0_off, _px=px: f"{v*_px + _y0:.0f}"))
                ax1.set_xlabel(f"x [{unit}]"); ax1.set_ylabel(f"y [{unit}]")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # panel 2: BG-subtracted + TPA-corrected
                im2 = ax2.imshow(fe.bg_img, origin="upper", cmap=cmap,
                                 vmin=0, vmax=_vmax_p2, interpolation="nearest", aspect="equal")
                cx_px = (r["x_bar"] - x0_off) / px if px else (r["x_bar"] - x0_off)
                cy_px = (r["y_bar"] - y0_off) / px if px else (r["y_bar"] - y0_off)
                ax2.plot(cx_px, cy_px, "+", color="white", ms=12, mew=2, label="centroid")
                if fit_mode == "lab":
                    d4w = r["d4σ_x"] / px if px else r["d4σ_x"]
                    d4h = r["d4σ_y"] / px if px else r["d4σ_y"]
                    ax2.add_patch(Ellipse((cx_px, cy_px), width=d4w, height=d4h, angle=0,
                                     linewidth=1.8, edgecolor="white", facecolor="none",
                                     label="D4sigma (lab)"))
                    title2 = "BG-subtracted + D4sigma (lab)"
                else:
                    d4w = r["d4\u03c3_maj"] / px if px else r["d4\u03c3_maj"]
                    d4h = r["d4\u03c3_min"] / px if px else r["d4\u03c3_min"]
                    ax2.add_patch(Ellipse((cx_px, cy_px), width=d4w, height=d4h,
                                     angle=r["\u03b8_deg"],
                                     linewidth=1.8, edgecolor="white", facecolor="none",
                                     label="D4\u03c3 (principal)"))
                    title2 = f"BG-subtracted + D4\u03c3  \u03b8={r['\u03b8_deg']:.1f}\u00b0"
                ax2.legend(fontsize=7, loc="upper right")
                _xoff, _yoff, _p = x0_off, y0_off, px
                ax2.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda v, _, xo=_xoff, p=_p: f"{v*p+xo:.0f}"))
                ax2.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda v, _, yo=_yoff, p=_p: f"{v*p+yo:.0f}"))
                ax2.set_title(title2, fontsize=8)
                ax2.set_xlabel(f"x [{unit}]"); ax2.set_ylabel(f"y [{unit}]")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                # panel 3
                prof_x = fe.bg_img.sum(axis=0)
                (_, _, sig_x, _), fit_x = marginal_gaussian_fit(prof_x, x_ax)
                pk_x = float(prof_x.max()); thr_x = 0.135 * pk_x
                if fit_mode == "lab":
                    sx_lbl = f"σ_x={r['σ_x']:.2f} {unit}"
                    ctr_x = r["x_bar"]; hw_x = 2*r["σ_x"]
                else:
                    sx_lbl = f"σ_maj={r['σ_maj']:.2f} {unit}  eps={r['ellipticity']:.3f}"
                    ctr_x = r["x_bar"]; hw_x = 2*r["σ_maj"]
                ax3.plot(x_ax, prof_x, color="steelblue", lw=1.5, label="int_dy I(x,y)")
                ax3.plot(x_ax, fit_x, color="tomato", lw=1.5, ls="--", label=f"Gauss {sx_lbl}")
                ax3.axvline(ctr_x, color="gray", ls=":", lw=1)
                ax3.axvspan(ctr_x-hw_x, ctr_x+hw_x, alpha=0.12, color="tomato", label="+-2sigma")
                if thr_x > 0:
                    _clx = (r.get('clip_maj', r['clip_x']) if fit_mode == "principal"
                            else r['clip_x'])
                    _lx  = "maj" if fit_mode == "principal" else "x"
                    ax3.axhline(thr_x, color="orange", ls="--", lw=1,
                                label=f"13.5% -> {_clx:.2f} {unit} (clip_{_lx})")
                ax3.set_title("Marginal x-profile" if fit_mode == "lab"
                              else "Marginal profile (major axis)", fontsize=8)
                ax3.set_xlabel(f"x [{unit}]"); ax3.set_ylabel("Intensity [a.u.]")
                ax3.legend(fontsize=7)

                # panel 4
                prof_y = fe.bg_img.sum(axis=1)
                (_, _, sig_y, _), fit_y = marginal_gaussian_fit(prof_y, y_ax)
                pk_y = float(prof_y.max()); thr_y = 0.135 * pk_y
                if fit_mode == "lab":
                    sy_lbl = f"σ_y={r['σ_y']:.2f} {unit}"
                    ctr_y = r["y_bar"]; hw_y = 2*r["σ_y"]
                else:
                    sy_lbl = f"σ_min={r['σ_min']:.2f} {unit}  theta={r['θ_deg']:.1f} deg"
                    ctr_y = r["y_bar"]; hw_y = 2*r["σ_min"]
                ax4.plot(y_ax, prof_y, color="steelblue", lw=1.5, label="int_dx I(x,y)")
                ax4.plot(y_ax, fit_y, color="tomato", lw=1.5, ls="--", label=f"Gauss {sy_lbl}")
                ax4.axvline(ctr_y, color="gray", ls=":", lw=1)
                ax4.axvspan(ctr_y-hw_y, ctr_y+hw_y, alpha=0.12, color="tomato", label="+-2sigma")
                if thr_y > 0:
                    _cly = (r.get('clip_min', r['clip_y']) if fit_mode == "principal"
                            else r['clip_y'])
                    _ly  = "min" if fit_mode == "principal" else "y"
                    ax4.axhline(thr_y, color="orange", ls="--", lw=1,
                                label=f"13.5% -> {_cly:.2f} {unit} (clip_{_ly})")
                ax4.set_title("Marginal y-profile" if fit_mode == "lab"
                              else "Marginal profile (minor axis)", fontsize=8)
                ax4.set_xlabel(f"y [{unit}]"); ax4.set_ylabel("Intensity [a.u.]")
                ax4.legend(fontsize=7)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        QMessageBox.information(self, "Series saved",
            f"Results CSV:\n  {path}\n\nPlots PDF ({len(done)} pages):\n  {pdf_path}")
        self._status(f"Series report saved: {len(done)} files -> {pdf_path}")

    @staticmethod
    def _write_csv(path, entries):
        rows = []
        for fe in entries:
            r  = fe.results
            bi = r.get("bg_info", {})
            rows.append({
                "filename":          r.get("filename", fe.fname),
                "fit_mode":          r.get("fit_mode", "lab"),
                "tpa_correction":    r.get("use_tpa", True),
                "bg_mode":           r.get("bg_mode", "corner"),
                "x_bar":             r["x_bar"],
                "y_bar":             r["y_bar"],
                "σ_x":           r["σ_x"],
                "σ_y":           r["σ_y"],
                "σ_xy":          r.get("σ_xy", ""),
                "d4σ_x":         r["d4σ_x"],
                "d4σ_y":         r["d4σ_y"],
                "σ_maj":         r.get("σ_maj", ""),
                "σ_min":         r.get("σ_min", ""),
                "d4σ_maj":       r.get("d4σ_maj", ""),
                "d4σ_min":       r.get("d4σ_min", ""),
                "θ_deg":         r.get("θ_deg", ""),
                "ellipticity":       r.get("ellipticity", ""),
                "clip_x_13.5pct":    r.get("clip_x", ""),
                "clip_y_13.5pct":    r.get("clip_y", ""),
                "clip_maj_13.5pct":  r.get("clip_maj", ""),
                "clip_min_13.5pct":  r.get("clip_min", ""),
                "unit":              r["unit"],
                "pixel_size_um":     r["pixel_size_um"],
                "bg_mean":           bi.get("bg_mean", ""),
                "bg_std":            bi.get("bg_std", ""),
                "bg_threshold":      bi.get("threshold", ""),
                "dark_frame_used":   bi.get("dark_used", False),
                "roi":               str(r.get("roi", "")),
                "iso_compliant":     "yes",
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    # -- display -------------------------------------------------------

    def _on_view_changed(self, index: int):
        """Switch the stacked widget page and refresh the relevant canvas."""
        self._stack.setCurrentIndex(index)   # 0=image, 1=analysis
        if index == 0:
            self._refresh_image_view()
        else:
            # re-plot results if available
            fe = self._fe
            if fe and fe.results and fe.roi_img is not None:
                r = fe.results
                self.res_canvas.plot(
                    fe.roi_img, fe.bg_img, r,
                    fe.roi, r["pixel_size_um"], r["unit"],
                    self.combo_cmap.currentText(),
                    vmax_raw=self.spin_vmax.value(), use_tpa=self.chk_tpa.isChecked())

    def _refresh_image_view(self):
        fe = self._fe
        if fe is None: return
        if self._stack.currentIndex() != 0:
            return
        cmap  = self.combo_cmap.currentText()
        unit  = "\u00b5m" if self.spin_px.value() != 1.0 else "px"
        img   = fe.clean_raw
        title = (f"{fe.fname}  \u2014  Raw"
                 + (" (damage-corrected)" if fe.dmask is not None else ""))
        dmask = fe.dmask if self.chk_show_dmask.isChecked() else None
        self.img_canvas.show_image(img, title=title, cmap=cmap,
                                   roi=fe.roi, unit=unit, dmask=dmask,
                                   vmax=self.spin_vmax.value())

    def _clear_display(self):
        self.img_canvas.fig.clear()
        self.img_canvas._safe_draw()
        self.res_canvas.fig.clear()
        self.res_canvas._safe_draw()
        self.txt_results.clear()

    def _status(self, msg):
        self.statusBar().showMessage(msg)


# ========================================================================
#  ENTRY POINT
# ========================================================================

def _apply_stylesheet(app: QApplication) -> None:
    """Modern dark theme with accent colours."""
    app.setStyle("Fusion")

    ACCENT   = "#4a9eff"
    ACCENT2  = "#2d7dd2"
    BG       = "#1e1e2e"
    BG2      = "#2a2a3e"
    BG3      = "#313145"
    BORDER   = "#44446a"
    TEXT     = "#cdd6f4"
    TEXT_DIM = "#7f849c"
    GREEN    = "#a6e3a1"
    RED      = "#f38ba8"
    ORANGE   = "#fab387"

    qss = f"""
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
        font-size: 10pt;
    }}
    QGroupBox {{
        background-color: {BG2};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 8px;
        padding: 6px 4px 4px 4px;
        font-weight: bold;
        font-size: 9pt;
        color: {ACCENT};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        left: 8px;
    }}
    QPushButton {{
        background-color: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 5px;
        padding: 5px 10px;
        font-size: 9pt;
    }}
    QPushButton:hover {{
        background-color: {BORDER};
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT2};
        color: white;
    }}
    QPushButton:disabled {{
        background-color: {BG2};
        color: {TEXT_DIM};
        border-color: {BG3};
    }}
    QPushButton#btn_run {{
        background-color: {ACCENT2};
        color: white;
        font-weight: bold;
        border: 1px solid {ACCENT2};
        padding: 6px 10px;
    }}
    QPushButton#btn_run:hover {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}
    QPushButton#btn_run:pressed {{
        background-color: #1a5fa0;
        border-color: #1a5fa0;
    }}
    QPushButton#btn_run:disabled {{ background-color: {BG3}; color: {TEXT_DIM}; border-color: {BG3}; }}
    QPushButton#btn_run_all {{
        background-color: #1a5fa0;
        color: white;
        font-weight: bold;
        border: 1px solid #1a5fa0;
        padding: 6px 10px;
    }}
    QPushButton#btn_run_all:hover {{
        background-color: {ACCENT2};
        border-color: {ACCENT2};
    }}
    QPushButton#btn_run_all:pressed {{
        background-color: #114070;
        border-color: #114070;
    }}
    QPushButton#btn_run_all:disabled {{ background-color: {BG3}; color: {TEXT_DIM}; border-color: {BG3}; }}
    QLabel {{
        color: {TEXT};
        background: transparent;
    }}
    QLabel#lbl_dim {{
        color: {TEXT_DIM};
        font-size: 8pt;
    }}
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit, QListWidget {{
        background-color: {BG3};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 6px;
        selection-background-color: {ACCENT2};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {BORDER};
        border-radius: 2px;
        width: 14px;
    }}
    QComboBox::drop-down {{
        border: none;
        background: {BORDER};
        border-radius: 3px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG3};
        color: {TEXT};
        selection-background-color: {ACCENT2};
        border: 1px solid {BORDER};
    }}
    QListWidget {{
        background-color: {BG3};
        alternate-background-color: {BG2};
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT2};
        color: white;
    }}
    QListWidget::item:hover {{
        background-color: {BORDER};
    }}
    QCheckBox {{
        color: {TEXT};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {BORDER};
        border-radius: 3px;
        background: {BG3};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT2};
        border-color: {ACCENT};
    }}
    QScrollArea, QScrollBar {{
        background-color: {BG};
        border: none;
    }}
    QScrollBar:vertical {{
        width: 8px;
        background: {BG2};
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        height: 8px;
        background: {BG2};
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 4px;
        min-width: 20px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QSplitter::handle {{
        background: {BORDER};
        width: 2px; height: 2px;
    }}
    QFrame[frameShape="4"], QFrame[frameShape="5"] {{
        color: {BORDER};
        background: {BORDER};
    }}
    QStatusBar {{
        background: {BG2};
        color: {TEXT_DIM};
        font-size: 8pt;
        border-top: 1px solid {BORDER};
    }}
    QProgressDialog {{
        background: {BG2};
        color: {TEXT};
    }}
    QMessageBox {{
        background: {BG2};
        color: {TEXT};
    }}
    QFileDialog {{
        background: {BG2};
        color: {TEXT};
    }}
    NavigationToolbar2QT {{
        background: {BG2};
        border-bottom: 1px solid {BORDER};
    }}
    """
    app.setStyleSheet(qss)


def main():
    app = QApplication(sys.argv)
    _apply_stylesheet(app)

    # Use a clean modern font if available
    from PyQt6.QtGui import QFont as _QFont
    try:
        f = _QFont("Segoe UI", 10)
        app.setFont(f)
    except Exception:
        pass

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()