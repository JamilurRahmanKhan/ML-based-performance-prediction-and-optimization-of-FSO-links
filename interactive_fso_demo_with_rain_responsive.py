"""
Interactive FSO Analysis Tool
Machine-Learning-Based Performance Prediction and Visualization

Update: Added a Rain Attenuation toggle that lets you switch between:
  - Clear-air (your original plots)
  - Rain-affected plots (SNR reduced with rain attenuation; BER recomputed from rain-adjusted SNR)

How rain is applied:
  1) We read your rain CSV (daily rain mm).
  2) Convert to an average rain rate (mm/hr) = rain_mm / 24.
  3) Convert rain rate -> specific attenuation (dB/km) using a simple power-law model.
  4) Apply extra loss A(dB) = gamma(dB/km) * distance(km)
  5) Reduce received power (and therefore SNR) by 10^(-A/10)

NOTE:
- Your existing ML model predicts BER from (distance, Pt, Div) only.
  If we kept using it, BER would not change with rain. To visualize rain impact,
  we recompute BER from the (rain-adjusted) SNR using a standard OOK/BPSK-like
  approximation: BER = 0.5 * erfc(sqrt(SNR/2)).
- You can change this behavior in compute_ber_array().
"""

import matplotlib
matplotlib.use("TkAgg")  # interactive desktop backend

import os
import re
import csv
import warnings
from math import erfc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.patches as mpatches
import joblib
import mplcursors

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------
MODEL_FILE = "fsomodel_rf.joblib"

# Fixed physical parameters
PT_FIXED = 10.0       # mW
DIV_FIXED = 2.0       # degrees (full angle)
RX_APERTURE_M = 0.05  # receiver diameter (m)
NOISE_POWER_W = 1e-9  # noise power (W)
BER_THRESHOLD = 1e-3  # reliability cutoff

# Distance range
DIST_MIN = 10.0
DIST_MAX_DEFAULT = 2000.0
DIST_MAX_SLIDER = 3000.0

# Calibrate SNR to match your desired anchor at 10 m
SNR_TARGET_10M = 8e4  # anchor at 10 m

# Rain data
RAIN_CSV = "CCS_20140101_20240101 (1).csv"   # put this CSV next to the script (or change path)
RAIN_STAT = "p95"                            # one of: mean, median, p90, p95, max
RAIN_RATE_FALLBACK_MMHR = 0.0                # used if CSV not found or parsing fails

# Simple rain attenuation model (power law): gamma(dB/km) = A * R^B
# (This is a generic model; tune A,B for your wavelength/area if you have a better reference.)
RAIN_ATTEN_A = 1.076
RAIN_ATTEN_B = 0.67

# ---------------------------------------------------
# RAIN HELPERS
# ---------------------------------------------------
def _find_best_column(cols, keyword):
    """Return the first column that contains `keyword` (case-insensitive)."""
    keyword = keyword.lower()
    for c in cols:
        if keyword in str(c).lower():
            return c
    return None

def load_rain_rate_from_csv(path: str):
    """
    Reads the rain CSV and returns:
      rain_rate_mmhr_used, stats_dict
    Assumptions:
      - Your CSV has a 'Time' column and a rain column like 'Rain(mm)' (daily totals).
    """
    if not os.path.exists(path):
        return RAIN_RATE_FALLBACK_MMHR, {"note": f"Rain CSV not found: {path}"}

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = _find_best_column(df.columns, "time")
    rain_col = _find_best_column(df.columns, "rain")

    if rain_col is None:
        return RAIN_RATE_FALLBACK_MMHR, {"note": "Could not find a rain column."}

    # Parse rain column
    rain = pd.to_numeric(df[rain_col], errors="coerce")
    rain = rain.dropna()

    if len(rain) == 0:
        return RAIN_RATE_FALLBACK_MMHR, {"note": "Rain column has no numeric values."}

    # If these are daily totals in mm, approximate mean hourly rate
    rain_rate_mmhr = rain / 24.0

    stats = {
        "mean": float(rain_rate_mmhr.mean()),
        "median": float(rain_rate_mmhr.median()),
        "p90": float(rain_rate_mmhr.quantile(0.90)),
        "p95": float(rain_rate_mmhr.quantile(0.95)),
        "max": float(rain_rate_mmhr.max()),
        "rain_col": rain_col,
        "time_col": time_col,
        "rows_used": int(len(rain_rate_mmhr)),
        "note": "Converted daily rain(mm) to avg mm/hr by /24."
    }

    val = stats.get(RAIN_STAT, stats["p95"])
    return float(val), stats

def rain_specific_attenuation_db_per_km(r_mmhr: float) -> float:
    """gamma(dB/km) = A * R^B"""
    r = max(float(r_mmhr), 0.0)
    if r == 0.0:
        return 0.0
    return RAIN_ATTEN_A * (r ** RAIN_ATTEN_B)

# ---------------------------------------------------
# MODEL LOADING / FALLBACK
# ---------------------------------------------------
if not os.path.exists(MODEL_FILE):
    print("Model file not found. Creating a simple fallback BER model...")
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    n = 2500
    distances = np.random.uniform(10, 2000, n)
    pt_samples = np.full_like(distances, PT_FIXED)
    divs = np.full_like(distances, DIV_FIXED)

    def _snr_for_model(d):
        theta_rad = np.deg2rad(DIV_FIXED)
        beam_radius = d * np.tan(theta_rad / 2.0)
        beam_area = np.pi * beam_radius**2 + 1e-12
        rx_area = np.pi * (RX_APERTURE_M / 2.0) ** 2
        pr_mw = PT_FIXED * (rx_area / beam_area)
        pr_w = pr_mw * 1e-3
        snr = max(pr_w / NOISE_POWER_W, 1e-12)
        return snr

    snrs = np.array([_snr_for_model(d) for d in distances])
    bers = np.array([min(max(0.5 * erfc(np.sqrt(s / 2.0)), 1e-12), 0.5) for s in snrs])

    X = np.column_stack((distances, pt_samples, divs))
    y = bers

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X, y)
    joblib.dump({"model": rf, "features": ["distance_m", "Pt_mW", "div_deg"]}, MODEL_FILE)
    print(f"Created fallback model and saved to {MODEL_FILE}")

data = joblib.load(MODEL_FILE)
model = data["model"]

# ---------------------------------------------------
# PHYSICS HELPERS (with calibration + optional rain)
# ---------------------------------------------------
theta_rad_10 = np.deg2rad(DIV_FIXED)
beam_radius_10 = 10.0 * np.tan(theta_rad_10 / 2.0)
beam_area_10 = np.pi * beam_radius_10**2 + 1e-12
rx_area_global = np.pi * (RX_APERTURE_M / 2.0) ** 2
Pr10_mW = PT_FIXED * (rx_area_global / beam_area_10)
Pr10_W = Pr10_mW * 1e-3
SNR_phys_10 = max(Pr10_W / NOISE_POWER_W, 1e-12)
SNR_SCALE = SNR_TARGET_10M / SNR_phys_10

def compute_snr_array(distances_m: np.ndarray, rain_on: bool, rain_rate_mmhr: float) -> np.ndarray:
    """
    Physics-based, calibrated SNR:
      SNR_raw(d) = Pt * (Arx / (π*(d*tan(θ/2))^2)) / N0  ~ 1/d^2
      SNR_clear(d) = SNR_SCALE * SNR_raw(d)
    If rain_on:
      Apply extra loss A(dB) = gamma(dB/km) * distance(km)
      => SNR_rain = SNR_clear * 10^(-A/10)
    """
    theta_rad = np.deg2rad(DIV_FIXED)
    beam_radius = distances_m * np.tan(theta_rad / 2.0)
    beam_area = np.pi * beam_radius**2 + 1e-12

    pr_mw = PT_FIXED * (rx_area_global / beam_area)
    pr_w = pr_mw * 1e-3
    snr_raw = np.maximum(pr_w / NOISE_POWER_W, 1e-12)
    snr_clear = np.maximum(SNR_SCALE * snr_raw, 1e-12)

    if not rain_on:
        return snr_clear

    gamma = rain_specific_attenuation_db_per_km(rain_rate_mmhr)
    dist_km = distances_m / 1000.0
    A_db = gamma * dist_km
    atten_linear = 10.0 ** (-A_db / 10.0)
    snr_rain = np.maximum(snr_clear * atten_linear, 1e-12)
    return snr_rain

def compute_ber_array(distances_m: np.ndarray, snr_used: np.ndarray, rain_on: bool) -> np.ndarray:
    """
    Clear-air:
      ML-based BER prediction from distance + fixed Pt, Div.
    Rain-on:
      BER recomputed from *rain-adjusted* SNR so you can see the impact.
    """
    if not rain_on:
        X = np.column_stack((
            distances_m,
            np.full_like(distances_m, PT_FIXED),
            np.full_like(distances_m, DIV_FIXED),
        ))
        bers = model.predict(X)
        return np.clip(bers, 1e-12, 0.5)

    # Vectorized erfc for numpy arrays (pure python fallback)
    v_erfc = np.vectorize(erfc)
    bers = 0.5 * v_erfc(np.sqrt(np.maximum(snr_used, 1e-12) / 2.0))
    return np.clip(bers, 1e-12, 0.5)

# ---------------------------------------------------
# STATE
# ---------------------------------------------------
class AppState:
    def __init__(self):
        self.current_mode = "dist_snr_ber"
        self.three_d_fig = None
        self.ax_ber_twin = None
        self.cursor_handles = []
        self.colorbar = None
        self.original_position = None
        self.ax_cbar_dedicated = None
        self.distances = None

        self.rain_enabled = False
        self.rain_rate_mmhr = 0.0
        self.rain_stats = {}

state = AppState()

# Load rain stats once at startup
state.rain_rate_mmhr, state.rain_stats = load_rain_rate_from_csv(RAIN_CSV)

# ---------------------------------------------------
# FIGURE LAYOUT (responsive)
# ---------------------------------------------------
plt.style.use("default")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#f0f0f0")

# We keep axes stable and only move/resize them on window resize so widgets remain usable.
ax_main = fig.add_axes([0.13, 0.58, 0.75, 0.30])   # placeholder (will be resized)
ax_cbar = fig.add_axes([0.90, 0.58, 0.03, 0.30])   # placeholder (will be resized)
ax_dist_slider = fig.add_axes([0.13, 0.48, 0.80, 0.04])

# Controls (placeholders)
ax_radio = fig.add_axes([0.13, 0.15, 0.24, 0.25])
ax_rain = fig.add_axes([0.13, 0.41, 0.24, 0.06])

ax_btn_export  = fig.add_axes([0.50, 0.26, 0.16, 0.05])
ax_btn_save    = fig.add_axes([0.70, 0.26, 0.16, 0.05])
ax_btn_animate = fig.add_axes([0.50, 0.19, 0.16, 0.05])
ax_btn_reset   = fig.add_axes([0.70, 0.19, 0.16, 0.05])

def _apply_responsive_layout():
    """Reposition UI elements based on current window aspect ratio."""
    # size in pixels (fallback to inches if backend doesn't expose)
    try:
        w_px, h_px = fig.canvas.get_width_height()
        aspect = (w_px / max(h_px, 1))
    except Exception:
        w_in, h_in = fig.get_size_inches()
        aspect = (w_in / max(h_in, 1e-6))

    left = 0.13
    right = 0.92
    width = right - left
    cbar_w = 0.03
    gap = 0.02

    # Wide layout: plot on top, controls bottom-left, buttons bottom-right
    if aspect >= 1.20:
        ax_main.set_position([left, 0.56, width - cbar_w - gap, 0.33])
        ax_cbar.set_position([right - cbar_w, 0.56, cbar_w, 0.33])

        ax_dist_slider.set_position([left, 0.44, width, 0.04])

        # Move radio + rain below slider so title never overlaps slider
        ax_rain.set_position([left, 0.34, 0.28, 0.07])
        ax_radio.set_position([left, 0.12, 0.28, 0.20])

        # Buttons to the right
        bx = left + 0.40
        bw = 0.18
        bh = 0.055
        ax_btn_export.set_position([bx,        0.22, bw, bh])
        ax_btn_save.set_position([bx + 0.22,  0.22, bw, bh])
        ax_btn_animate.set_position([bx,       0.14, bw, bh])
        ax_btn_reset.set_position([bx + 0.22,  0.14, bw, bh])

    # Narrow layout: stack controls vertically for small windows
    else:
        ax_main.set_position([left, 0.58, width - cbar_w - gap, 0.31])
        ax_cbar.set_position([right - cbar_w, 0.58, cbar_w, 0.31])

        ax_dist_slider.set_position([left, 0.48, width, 0.04])

        ax_rain.set_position([left, 0.40, width, 0.06])
        ax_radio.set_position([left, 0.18, width, 0.20])

        bw = 0.40
        bh = 0.055
        ax_btn_export.set_position([left,            0.10, bw, bh])
        ax_btn_save.set_position([left + bw + 0.04, 0.10, bw, bh])
        ax_btn_animate.set_position([left,          0.03, bw, bh])
        ax_btn_reset.set_position([left + bw + 0.04,0.03, bw, bh])

    # Keep these synced for clean_axes() reset logic
    state.original_position = ax_main.get_position()
    state.ax_cbar_dedicated = ax_cbar

    fig.canvas.draw_idle()

# Apply initial responsive layout and re-apply whenever the window is resized
_apply_responsive_layout()
fig.canvas.mpl_connect("resize_event", lambda evt: _apply_responsive_layout())

# ---------------------------------------------------
# SLIDER: MAX DISTANCE

# ---------------------------------------------------
slider_dist = Slider(
    ax_dist_slider,
    "Max Distance (m)",
    100.0,
    DIST_MAX_SLIDER,
    valinit=DIST_MAX_DEFAULT,
    valstep=10.0,
    color="#4ECDC4",
    track_color="#E0F7F6",
)

slider_dist.label.set_fontsize(12)
slider_dist.label.set_weight("bold")
slider_dist.label.set_color("#000")
slider_dist.label.set_horizontalalignment("left")
slider_dist.label.set_position((0.0, 0.5))  # move label slightly right inside the slider axis
slider_dist.valtext.set_fontsize(11)
slider_dist.valtext.set_weight("bold")
slider_dist.valtext.set_color("#000")

# If you want to move the slider label a bit to the right:
# slider_dist.label.set_position((0.02, 0.5))  # (x,y) in axes coords

# ---------------------------------------------------
# RADIO BUTTONS (VIEW MODES)
# ---------------------------------------------------
radio = RadioButtons(
    ax_radio,
    (
        "Dist-SNR-BER",
        "Dist vs SNR",
        "Dist vs BER",
        "SNR vs BER",
        "SNR vs Dist",
        "BER vs Dist",
        "BER vs SNR",
        "3D View",
    ),
    active=0,
)

# Make the list easier to click/read: increase font and vertical spacing via axes height (GridSpec)
for t in radio.labels:
    t.set_fontsize(14)
    t.set_fontweight("bold")

ax_radio.set_title("Visualization Mode", fontsize=14, weight="bold", pad=12, color="#000")
for spine in ax_radio.spines.values():
    spine.set_visible(False)
ax_radio.set_facecolor("#ffffff")

# ---------------------------------------------------
# RAIN TOGGLE (CheckButtons)
# ---------------------------------------------------
ax_rain.set_title("Environment", fontsize=12, weight="bold", pad=10)
for spine in ax_rain.spines.values():
    spine.set_visible(False)
ax_rain.set_facecolor("#ffffff")

rain_label = f"Rain ({RAIN_STAT}={state.rain_rate_mmhr:.3f} mm/hr)"
chk = CheckButtons(ax_rain, [rain_label], [state.rain_enabled])

for t in chk.labels:
    t.set_fontsize(12)
    t.set_fontweight("bold")

# ---------------------------------------------------
# BUTTONS
# ---------------------------------------------------
btn_export = Button(ax_btn_export, "Export CSV", color="#95E1D3", hovercolor="#7FD1C3")
btn_save = Button(ax_btn_save, "Save Figure", color="#F38181", hovercolor="#E37171")
btn_animate = Button(ax_btn_animate, "Animate Distance", color="#EAFFD0", hovercolor="#DAEFC0")
btn_reset = Button(ax_btn_reset, "Reset", color="#FFEAA7", hovercolor="#EEDA97")

for btn in [btn_export, btn_save, btn_animate, btn_reset]:
    btn.label.set_fontsize(11)
    btn.label.set_weight("bold")
    btn.label.set_color("#000")

# ---------------------------------------------------
# CURSOR HANDLING
# ---------------------------------------------------
def clear_cursors():
    for c in state.cursor_handles:
        try:
            c.disconnect()
        except Exception:
            pass
    state.cursor_handles = []

def attach_cursor(scatter_obj, d_sparse, snr_sparse, ber_sparse):
    clear_cursors()
    try:
        cursor = mplcursors.cursor(scatter_obj, hover=True)

        def fmt(sel):
            i = sel.index
            return (
                f"Distance: {d_sparse[i]:.1f} m\n"
                f"SNR: {snr_sparse[i]:.2e}\n"
                f"BER: {ber_sparse[i]:.2e}"
            )

        cursor.connect("add", lambda sel: sel.annotation.set_text(fmt(sel)))
        cursor.connect(
            "add",
            lambda sel: sel.annotation.get_bbox_patch().set(
                facecolor="#FFFACD",
                alpha=0.95,
                edgecolor="#333",
                linewidth=2,
            ),
        )
        state.cursor_handles.append(cursor)
    except Exception:
        # if mplcursors is not working, just skip tooltips
        pass

# ---------------------------------------------------
# CLEAN AXES
# ---------------------------------------------------
def clean_axes():
    original_pos = state.original_position
    ax_main.clear()

    if state.ax_ber_twin is not None:
        try:
            state.ax_ber_twin.clear()
            state.ax_ber_twin.remove()
        except Exception:
            pass
        state.ax_ber_twin = None

    state.ax_cbar_dedicated.clear()
    state.ax_cbar_dedicated.set_visible(False)
    state.colorbar = None

    ax_main.set_position(original_pos)
    fig.canvas.draw_idle()
    clear_cursors()

# ---------------------------------------------------
# REDRAW
# ---------------------------------------------------
def redraw():
    max_d = slider_dist.val
    distances = np.linspace(DIST_MIN, max_d, 200)
    state.distances = distances

    snrs = compute_snr_array(distances, state.rain_enabled, state.rain_rate_mmhr)
    bers = compute_ber_array(distances, snrs, state.rain_enabled)

    clean_axes()

    env_txt = "Rain ON" if state.rain_enabled else "Rain OFF"
    if state.rain_enabled:
        gamma = rain_specific_attenuation_db_per_km(state.rain_rate_mmhr)
        env_txt += f" | R={state.rain_rate_mmhr:.3f} mm/hr | γ={gamma:.2f} dB/km"

    param_text = (
        f"Pt={PT_FIXED:.1f} mW | Div={DIV_FIXED:.1f}° | "
        f"Range: {DIST_MIN:.0f}–{max_d:.0f} m | {env_txt}"
    )

    mode = state.current_mode

    # sample subset for tooltips
    idx = np.linspace(0, len(distances) - 1, 30, dtype=int)
    d_s, snr_s, ber_s = distances[idx], snrs[idx], bers[idx]

    # --------- Mode 1: Dist-SNR-BER ----------
    if mode == "dist_snr_ber":
        pos = state.original_position
        state.ax_ber_twin = ax_main.twinx()
        ax_main.set_position(pos)
        state.ax_ber_twin.set_position(pos)

        ax_main.set_yscale("log")
        state.ax_ber_twin.set_yscale("log")

        ax_main.plot(distances, snrs, color="#1976D2", linewidth=2.5, alpha=0.8)
        state.ax_ber_twin.plot(distances, bers, "r-", linewidth=2.5, alpha=0.8)

        scatter_ber = state.ax_ber_twin.scatter(
            d_s, ber_s,
            c="darkred",
            s=40,
            alpha=0.9,
            edgecolors="white",
            linewidth=1.5,
            zorder=5,
        )

        ax_main.set_ylim(min(snrs[snrs > 0]) * 0.1, max(snrs) * 10)
        state.ax_ber_twin.set_ylim(1e-12, 1e-1)
        state.ax_ber_twin.axhline(
            y=BER_THRESHOLD,
            color="#FF1744",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            zorder=0,
        )

        ax_main.set_xlabel("Distance (m)", fontsize=12, weight="bold")
        ax_main.set_ylabel("SNR (log scale)", fontsize=12, weight="bold", color="#1976D2")
        state.ax_ber_twin.set_ylabel("BER (log scale)", fontsize=12, weight="bold", color="#D32F2F")

        ax_main.tick_params(axis="y", labelcolor="#1976D2", labelsize=10)
        ax_main.tick_params(axis="x", labelsize=10)
        state.ax_ber_twin.tick_params(axis="y", labelcolor="#D32F2F", labelsize=10)

        ax_main.set_title(f"Distance vs SNR vs BER\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")

        snr_patch = mpatches.Patch(color="#1976D2", alpha=0.8, label="SNR")
        ber_patch = mpatches.Patch(color="red", alpha=0.8, label="BER")
        thresh_patch = mpatches.Patch(color="#FF1744", label="BER Threshold (1e-3)")
        ax_main.legend(handles=[snr_patch, ber_patch, thresh_patch], loc="upper right", fontsize=11, framealpha=0.95)

        attach_cursor(scatter_ber, d_s, snr_s, ber_s)

    # --------- Mode 2: Dist vs SNR ----------
    elif mode == "dist_snr":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.set_yscale("log")
        ax_main.plot(distances, snrs, "navy", alpha=0.9, linewidth=2)

        scatter = ax_main.scatter(
            d_s, snr_s,
            c=snr_s,
            cmap="plasma",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_ylim(min(snrs[snrs > 0]) * 0.1, max(snrs) * 10)
        ax_main.set_xlabel("Distance (m)", fontsize=12, weight="bold")
        ax_main.set_ylabel("SNR (log scale)", fontsize=12, weight="bold")
        ax_main.set_title(f"Distance vs SNR\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("SNR Level", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 3: Dist vs BER ----------
    elif mode == "dist_ber":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.set_yscale("log")
        ax_main.plot(distances, bers, "red", alpha=0.9, linewidth=2)

        scatter = ax_main.scatter(
            d_s, ber_s,
            c=ber_s,
            cmap="viridis",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_ylim(1e-12, 1e-1)
        ax_main.axhline(
            y=BER_THRESHOLD,
            color="#FF1744",
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
        )

        ax_main.set_xlabel("Distance (m)", fontsize=12, weight="bold")
        ax_main.set_ylabel("BER (log scale)", fontsize=12, weight="bold")
        ax_main.set_title(f"Distance vs BER\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("BER Level", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 4: SNR vs BER ----------
    elif mode == "snr_ber":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.plot(snrs, bers, "purple", alpha=0.8, linewidth=2)
        scatter = ax_main.scatter(
            snr_s, ber_s,
            c=d_s,
            cmap="coolwarm",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlim(min(snrs[snrs > 0]) * 0.1, max(snrs) * 10)
        ax_main.set_ylim(1e-12, 1e-1)
        ax_main.axhline(
            y=BER_THRESHOLD,
            color="#FF1744",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        ax_main.set_xlabel("SNR (log scale)", fontsize=12, weight="bold")
        ax_main.set_ylabel("BER (log scale)", fontsize=12, weight="bold")
        ax_main.set_title(f"SNR vs BER\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("Distance (m)", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 5: SNR vs Dist ----------
    elif mode == "snr_dist":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.plot(snrs, distances, "teal", alpha=0.85, linewidth=2)
        scatter = ax_main.scatter(
            snr_s, d_s,
            c=ber_s,
            cmap="magma",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_xscale("log")
        ax_main.set_xlim(min(snrs[snrs > 0]) * 0.1, max(snrs) * 10)

        ax_main.set_xlabel("SNR (log scale)", fontsize=12, weight="bold")
        ax_main.set_ylabel("Distance (m)", fontsize=12, weight="bold")
        ax_main.set_title(f"SNR vs Distance\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("BER Level", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 6: BER vs Dist ----------
    elif mode == "ber_dist":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.plot(bers, distances, "darkred", alpha=0.85, linewidth=2)
        scatter = ax_main.scatter(
            ber_s, d_s,
            c=snr_s,
            cmap="cividis",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_xscale("log")
        ax_main.set_xlim(1e-12, 1e-1)

        ax_main.set_xlabel("BER (log scale)", fontsize=12, weight="bold")
        ax_main.set_ylabel("Distance (m)", fontsize=12, weight="bold")
        ax_main.set_title(f"BER vs Distance\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("SNR Level", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 7: BER vs SNR ----------
    elif mode == "ber_snr":
        pos = state.original_position
        ax_main.set_position(pos)

        ax_main.plot(bers, snrs, "forestgreen", alpha=0.85, linewidth=2)
        scatter = ax_main.scatter(
            ber_s, snr_s,
            c=d_s,
            cmap="plasma",
            s=40,
            alpha=0.9,
            edgecolors="#333",
            linewidth=0.6,
        )

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlim(1e-12, 1e-1)
        ax_main.set_ylim(min(snrs[snrs > 0]) * 0.1, max(snrs) * 10)

        ax_main.set_xlabel("BER (log scale)", fontsize=12, weight="bold")
        ax_main.set_ylabel("SNR (log scale)", fontsize=12, weight="bold")
        ax_main.set_title(f"BER vs SNR\n{param_text}", fontsize=14, weight="bold", pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=":")
        ax_main.set_facecolor("#FAFAFA")
        ax_main.tick_params(axis="both", labelsize=10)

        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label("Distance (m)", fontsize=11, weight="bold")

        attach_cursor(scatter, d_s, snr_s, ber_s)

    # --------- Mode 8: 3D View ----------
    elif mode == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if state.three_d_fig is not None:
            try:
                plt.close(state.three_d_fig)
            except Exception:
                pass

        state.three_d_fig = plt.figure(figsize=(8, 6))
        state.three_d_fig.patch.set_facecolor("#f0f0f0")
        ax3 = state.three_d_fig.add_subplot(111, projection="3d")

        log_snrs = np.log10(snrs)
        log_bers = np.log10(bers)

        colors = cm.turbo(np.linspace(0, 1, len(distances)))
        ax3.scatter(distances, log_snrs, log_bers, c=colors, s=25, alpha=0.8)
        ax3.plot(distances, log_snrs, log_bers, "k-", alpha=0.3, linewidth=1)

        ax3.set_xlabel("\nDistance (m)", fontsize=12, weight="bold", labelpad=15)
        ax3.set_ylabel("\nlog₁₀(SNR)", fontsize=12, weight="bold", labelpad=15)
        ax3.set_zlabel("\nlog₁₀(BER)", fontsize=12, weight="bold", labelpad=15)
        ax3.set_title(f"3D: Distance vs log(SNR) vs log(BER)\n{param_text}", fontsize=14, weight="bold", pad=25)
        ax3.grid(True, alpha=0.3)
        ax3.view_init(elev=25, azim=45)

        plt.tight_layout()
        state.three_d_fig.show()

    fig.canvas.draw_idle()

# ---------------------------------------------------
# EVENT HANDLERS
# ---------------------------------------------------
def on_radio_clicked(label):
    mode_map = {
        "Dist-SNR-BER": "dist_snr_ber",
        "Dist vs SNR": "dist_snr",
        "Dist vs BER": "dist_ber",
        "SNR vs BER": "snr_ber",
        "SNR vs Dist": "snr_dist",
        "BER vs Dist": "ber_dist",
        "BER vs SNR": "ber_snr",
        "3D View": "3d",
    }
    state.current_mode = mode_map[label]
    redraw()

def on_rain_toggle(_label):
    # Only one checkbox
    state.rain_enabled = not state.rain_enabled
    redraw()

def export_csv(event):
    if state.distances is None:
        return
    d = state.distances
    snrs = compute_snr_array(d, state.rain_enabled, state.rain_rate_mmhr)
    bers = compute_ber_array(d, snrs, state.rain_enabled)

    filename = f"fso_dist_{int(d[-1])}m_{'rain' if state.rain_enabled else 'clear'}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Distance_m", "SNR_linear", "BER", "RainEnabled", "RainRate_mm_hr", "RAIN_STAT"])
        for di, s, b in zip(d, snrs, bers):
            writer.writerow([f"{di:.3f}", f"{s:.6e}", f"{b:.6e}", int(state.rain_enabled), f"{state.rain_rate_mmhr:.6f}", RAIN_STAT])
    print(f"✓ Exported: {filename}")

animating = False
def animate(event):
    global animating
    animating = True
    original = slider_dist.val
    for md in np.linspace(500, DIST_MAX_SLIDER, 20):
        if not animating:
            break
        slider_dist.set_val(md)
        plt.pause(0.15)
    slider_dist.set_val(original)
    animating = False

def save_figure(event):
    filename = f"fso_{state.current_mode}_{'rain' if state.rain_enabled else 'clear'}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {filename}")

def reset_params(event):
    slider_dist.set_val(DIST_MAX_DEFAULT)
    if state.rain_enabled:
        # turn it off
        state.rain_enabled = False
        try:
            chk.set_active(0)  # toggles visually
        except Exception:
            pass
    print("✓ Reset")

# ---------------------------------------------------
# CONNECT EVENTS & START
# ---------------------------------------------------
slider_dist.on_changed(lambda _val: redraw())
radio.on_clicked(on_radio_clicked)
chk.on_clicked(on_rain_toggle)

btn_export.on_clicked(export_csv)
btn_save.on_clicked(save_figure)
btn_animate.on_clicked(animate)
btn_reset.on_clicked(reset_params)

print("Initializing FSO Analysis Tool...")
print("Rain stats:", state.rain_stats)
redraw()

fig.text(
    0.5, 0.04,
    "Interactive FSO Analysis Tool  |  Hover over data points for details  |  © 2025",
    ha="center",
    fontsize=10,
    style="italic",
    color="#333",
    weight="bold",
)

print("✓ Tool ready! Showing window...")
plt.show()