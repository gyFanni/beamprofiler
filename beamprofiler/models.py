# -*- coding: utf-8 -*-
"""
beamprofiler.models
===================
Data container for a single loaded image file and its analysis state.
No Qt, no matplotlib — pure data.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FileEntry:
    """All state associated with one loaded image file."""
    path:       str
    raw:        np.ndarray           # original pixel data as loaded
    clean_raw:  np.ndarray           # raw with damage regions replaced by bg mean
    roi:        Optional[tuple]      = None   # (x0, x1, y0, y1) pixel indices
    dmask:      Optional[np.ndarray] = None   # bool mask, True = damaged pixel
    roi_img:    Optional[np.ndarray] = None   # clean_raw cropped to ROI (for display)
    bg_img:     Optional[np.ndarray] = None   # background-subtracted & TPA-corrected
    bg_info:    dict = field(default_factory=dict)
    results:    dict = field(default_factory=dict)

    @property
    def fname(self) -> str:
        return os.path.basename(self.path)

    @property
    def done(self) -> bool:
        """True if analysis has been run and produced results."""
        return bool(self.results)

    @property
    def saturated(self) -> bool:
        """True if the last analysis flagged saturation."""
        return self.results.get("saturation_warning", False)