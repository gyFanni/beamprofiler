# -*- coding: utf-8 -*-
"""
sim_plots.py
============
Generates all figures from the simulation CSVs produced by sim_core.py.

Run after sim_core.py:
    python sim_plots.py

Inputs (must exist)
-------------------
    sim_results.csv
    sim_width_results.csv
    sim_zdep_results.csv

Outputs
-------
    sim_heatmap.png
    sim_M2_vs_bgfrac.png
    sim_M2_vs_beamsensor.png
    sim_tpa_vs_linear.png
    sim_heatmap_width_d4sx.png
    sim_heatmap_width_d4sy.png
    sim_heatmap_width_clip_x.png
    sim_heatmap_width_clip_y.png
    sim_heatmap_width_combined.png
    sim_zdep_bg5pct.png
    sim_zdep_bg20pct.png
    sim_zdep_tpa_vs_linear.png
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

# load CSVs
df  = pd.read_csv("sim_results.csv")
dfw = pd.read_csv("sim_width_results.csv")
dfz = pd.read_csv("sim_zdep_results.csv")

# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

OI = {"blue":"#0072B2", "orange":"#E69F00", "green":"#009E73", "red":"#D55E00"}

mode_style = {
    "off":              (OI["blue"],   "-",  "o", "BG off"),
    "corner":           (OI["orange"], "--", "s", "Corner (§3.4.3)"),
    "iso_statistical":  (OI["green"],  "-",  "^", "ISO stat. (§3.4.2)"),
}
cam_title = {"linear":"Linear camera", "tpa":"TPA camera"}
cam_col   = {"linear":OI["blue"], "tpa":OI["orange"]}
cam_ls    = {"linear":"-",        "tpa":"--"}
cam_mk    = {"linear":"o",        "tpa":"s"}

def style_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True, labelsize=10)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=3, color="#aaaaaa")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, ls="--", lw=0.5, color="#dddddd", zorder=0)

def tol_lines(ax):
    ax.axhline(0,     color="#888888", lw=0.8, ls=":")
    ax.axhline( 0.10, color="#cccccc", lw=0.6, ls="--")
    ax.axhline(-0.10, color="#cccccc", lw=0.6, ls="--")

# ── Figure 1: M² error vs background fraction ─────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
fig1.patch.set_facecolor("white")

for ri, bs in enumerate([0.15, 0.25]):
    for ci, cam in enumerate(["linear", "tpa"]):
        ax = axes1[ri, ci]
        style_ax(ax); tol_lines(ax)
        for bgm, (col, ls, mk, lbl) in mode_style.items():
            sub = df[(df.camera==cam)&(df.bg_mode==bgm)&
                     (df.beam_sensor==bs)&(df.axis=="x")]
            grp = sub.groupby("bg_frac")["err_M2"]
            mn, sd = grp.mean(), grp.std()
            ax.plot(mn.index*100, mn.values, color=col, ls=ls, marker=mk,
                    ms=6, lw=1.8, markerfacecolor=col,
                    markeredgecolor="white", markeredgewidth=0.8, label=lbl)
            ax.fill_between(mn.index*100, mn-sd, mn+sd, color=col, alpha=0.13)
        ax.set_xlabel("Background fraction (%)", fontsize=10.5)
        ax.set_ylabel("Relative M² error  (ΔM²/M²)", fontsize=10.5)
        ax.set_title(f"{cam_title[cam]}\nbeam/sensor = {bs:.2f}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.legend(fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")
        ax.text(0.5, 0.11, "±10% tolerance",
                transform=ax.get_yaxis_transform(), fontsize=7.5, color="#888888")

fig1.suptitle("M² relative error vs background fraction  |  axis x  |  shaded = ±1σ",
              fontsize=12, fontweight="bold")
plt.tight_layout()
fig1.savefig("sim_M2_vs_bgfrac.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved sim_M2_vs_bgfrac.png")

# ── Figure 2: M² error vs beam/sensor ratio ────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
fig2.patch.set_facecolor("white")

for ri, bg_frac in enumerate([0.02, 0.10]):
    for ci, cam in enumerate(["linear", "tpa"]):
        ax = axes2[ri, ci]
        style_ax(ax); tol_lines(ax)
        for bgm, (col, ls, mk, lbl) in mode_style.items():
            sub = df[(df.camera==cam)&(df.bg_mode==bgm)&
                     (df.bg_frac==bg_frac)&(df.axis=="x")]
            grp = sub.groupby("beam_sensor")["err_M2"]
            mn, sd = grp.mean(), grp.std()
            ax.plot(mn.index, mn.values, color=col, ls=ls, marker=mk,
                    ms=6, lw=1.8, markerfacecolor=col,
                    markeredgecolor="white", markeredgewidth=0.8, label=lbl)
            ax.fill_between(mn.index, mn-sd, mn+sd, color=col, alpha=0.13)
        ax.set_xlabel("Beam σ / sensor half-width", fontsize=10.5)
        ax.set_ylabel("Relative M² error  (ΔM²/M²)", fontsize=10.5)
        ax.set_title(f"{cam_title[cam]}\nbg fraction = {bg_frac:.0%}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.legend(fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

fig2.suptitle("M² relative error vs beam/sensor size ratio  |  axis x  |  shaded = ±1σ",
              fontsize=12, fontweight="bold")
plt.tight_layout()
fig2.savefig("sim_M2_vs_beamsensor.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved sim_M2_vs_beamsensor.png")

# ── Figure 3: TPA vs linear, ISO stat mode ────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig3.patch.set_facecolor("white")

for ci, bg_frac in enumerate([0.02, 0.10]):
    ax = axes3[ci]
    style_ax(ax); tol_lines(ax)
    for cam in ["linear", "tpa"]:
        sub = df[(df.camera==cam)&(df.bg_mode=="iso_statistical")&
                 (df.bg_frac==bg_frac)&(df.axis=="x")]
        grp = sub.groupby("beam_sensor")["err_M2"]
        mn, sd = grp.mean(), grp.std()
        ax.plot(mn.index, mn.values, color=cam_col[cam], ls=cam_ls[cam],
                marker=cam_mk[cam], ms=7, lw=2,
                markerfacecolor=cam_col[cam], markeredgecolor="white",
                markeredgewidth=0.8, label=cam_title[cam])
        ax.fill_between(mn.index, mn-sd, mn+sd, color=cam_col[cam], alpha=0.15)
    ax.set_xlabel("Beam σ / sensor half-width", fontsize=11)
    ax.set_ylabel("Relative M² error  (ΔM²/M²)", fontsize=11)
    ax.set_title(f"ISO stat. BG  |  bg = {bg_frac:.0%}",
                 fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

fig3.suptitle(
    "TPA vs Linear camera  |  ISO §3.4.2 background  |  axis x  |  ±1σ shaded",
    fontsize=12, fontweight="bold")
plt.tight_layout()
fig3.savefig("sim_tpa_vs_linear.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved sim_tpa_vs_linear.png")

# ── Figure 4: M² bias heatmap (corrected colorbar) ────────────────────────
mode_order = ["off", "corner", "iso_statistical"]
mode_lbl   = {"off":"BG off","corner":"Corner (§3.4.3)",
               "iso_statistical":"ISO stat. (§3.4.2)"}

def make_pivot(cam, bgm, axis="x"):
    sub = df[(df.camera==cam)&(df.bg_mode==bgm)&(df.axis==axis)]
    return sub.groupby(["beam_sensor","bg_frac"])["err_M2"].mean().unstack()

vmin, vmax = -1.0, 1.0
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))
fig4.patch.set_facecolor("white")

for ri, cam in enumerate(["linear","tpa"]):
    for ci, bgm in enumerate(mode_order):
        ax = axes4[ri, ci]
        pivot = make_pivot(cam, bgm)
        im = ax.imshow(pivot.values, norm=norm, cmap="jet",
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("Background fraction", fontsize=9)
        ax.set_ylabel("Beam/sensor ratio", fontsize=9)
        ax.set_title(f"{cam_title[cam]}\n{mode_lbl[bgm]}",
                     fontsize=9, fontweight="bold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="gray")
                else:
                    clipped = "*" if (v > vmax or v < vmin) else ""
                    s = f"{v:+.2f}{clipped}"
                    ax.text(j, i, s, ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(v) > 0.6 else "black")

cb_ax = fig4.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
sm.set_array([])
cb = fig4.colorbar(sm, cax=cb_ax)
cb.set_label("Mean relative M² error  (ΔM²/M²)\n* value outside colorbar range",
             fontsize=9)

fig4.suptitle(
    "M² error heatmap  |  axis x  |  colorbar: \u22121 to +1\n"
    "Blue = underestimate  |  Green = accurate  |  Red = overestimate\n"
    "* = cell value outside \u00b11 range",
    fontsize=10, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.91, 1])
fig4.savefig("sim_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved sim_heatmap.png")

print("\nAll outputs written (M² simulation):")
for f in ["sim_results.csv", "sim_M2_vs_bgfrac.png",
          "sim_M2_vs_beamsensor.png", "sim_tpa_vs_linear.png",
          "sim_heatmap.png"]:
    print(f"  {f}")
print(f"  sim_images/  (see sim_core.py output)")

# load width results

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

def make_wpivot(cam, bgm, err_col, roi_strategy="roi_fullbg"):
    """
    roi_strategy filters the corner method variants:
      'roi_fullbg'  : auto ROI + full-sensor background (canonical, default)
      'full_sensor' : no ROI, full-sensor background (pure §3.4.3)
      'roi_cropbg'  : auto ROI + crop background
    For non-corner modes the filter is applied but only one strategy exists.
    """
    sub = dfw[(dfw.camera==cam) & (dfw.bg_mode==bgm)]
    if "roi_strategy" in dfw.columns:
        sub = sub[sub.roi_strategy==roi_strategy]
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
            pivot = make_wpivot(cam, bgm, err_col)   # default roi_fullbg
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

# ── Corner ROI strategy comparison figure ─────────────────────────────────
# Compare the three corner method variants for D4sigma_x:
#   full_sensor : no ROI, full-sensor corners (pure §3.4.3)
#   roi_fullbg  : auto ROI + full-sensor corners (canonical app behaviour)
#   roi_cropbg  : auto ROI + crop corners (least standard-compliant)
if "roi_strategy" in dfw.columns:
    strat_style = {
        "full_sensor": (OI["blue"],   "-",  "o", "Full sensor (§3.4.3 pure)"),
        "roi_fullbg":  (OI["orange"], "--", "s", "Auto ROI + full-sensor BG"),
        "roi_cropbg":  (OI["green"],  "-",  "^", "Auto ROI + crop BG"),
    }

    fig6, axes6 = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    fig6.patch.set_facecolor("white")

    for ri, cam in enumerate(["linear", "tpa"]):
        for ci, bgf in enumerate(sorted(dfw["bg_frac"].unique())):
            ax = axes6[ri, ci]
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
            ax.tick_params(direction="in", top=True, right=True, labelsize=9)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.tick_params(which="minor", direction="in", length=3,
                           color="#aaaaaa", top=True, right=True)
            ax.grid(True, ls="--", lw=0.5, color="#dddddd", zorder=0)
            ax.axhline(0,     color="#888", lw=0.8, ls=":")
            ax.axhline( 0.05, color="#ccc", lw=0.6, ls="--")
            ax.axhline(-0.05, color="#ccc", lw=0.6, ls="--")

            for strat, (col, ls, mk, lbl) in strat_style.items():
                sub = dfw[(dfw.camera==cam) & (dfw.bg_mode=="corner") &
                          (dfw.bg_frac==bgf) & (dfw.roi_strategy==strat)]
                grp = sub.groupby("beam_sensor")["err_d4sx"]
                mn, sd = grp.mean(), grp.std()
                ax.plot(mn.index, mn.values, color=col, ls=ls, marker=mk,
                        ms=6, lw=1.8, markerfacecolor=col,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=lbl)
                ax.fill_between(mn.index, mn-sd, mn+sd, color=col, alpha=0.13)

            # also show iso_statistical for reference
            sub_iso = dfw[(dfw.camera==cam) & (dfw.bg_mode=="iso_statistical") &
                          (dfw.bg_frac==bgf)]
            grp_iso = sub_iso.groupby("beam_sensor")["err_d4sx"]
            mn_iso = grp_iso.mean()
            ax.plot(mn_iso.index, mn_iso.values, color=OI["red"], ls=":",
                    lw=1.2, alpha=0.7, label="ISO stat. (ref)")

            ax.set_xlabel("Beam $w_0$ / sensor half-width", fontsize=9)
            if ci == 0:
                ax.set_ylabel("D4$\\sigma_x$ relative error", fontsize=9)
            ax.set_title(f"{cam_title[cam]}\nbg = {bgf:.0%}",
                         fontsize=9, fontweight="bold")
            bsl = sorted(dfw["beam_sensor"].unique())
            ax.set_xticks(bsl)
            ax.set_xticklabels([f"{v:.0%}" for v in bsl], fontsize=8)

    handles, labels = axes6[0, 0].get_legend_handles_labels()
    fig6.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
                framealpha=0.95, edgecolor="#cccccc",
                bbox_to_anchor=(0.5, -0.01))
    fig6.suptitle(
        "Corner method ROI strategy comparison  |  D4$\\sigma_x$ error\n"
        "Full sensor = no auto ROI, pure ISO §3.4.3  |  "
        "Auto ROI variants  |  ISO stat. shown for reference  |  ±1σ shaded",
        fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig6.savefig("sim_corner_roi_comparison.png", dpi=150,
                 bbox_inches="tight", facecolor="white")
    print("Saved sim_corner_roi_comparison.png")
else:
    print("roi_strategy column not found — re-run sim_core.py to generate updated CSV")

print("\nAll outputs written (width simulation):")
for f in (["sim_width_results.csv"] + width_figs +
          ["sim_heatmap_width_combined.png", "sim_corner_roi_comparison.png"]):
    print(f"  {f}")

# load zdep results

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