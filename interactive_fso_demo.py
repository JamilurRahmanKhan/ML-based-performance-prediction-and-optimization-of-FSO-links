# """
# Interactive FSO Demo (High-Performance Version)
# - Sliders: Pt (mW), Divergence (deg)
# - Buttons: Dist vs SNR+BER, BER vs SNR, SNR vs BER, 3D Plot
# - Export CSV, Save Figure
# - Hover tooltips
# - Color-mapped curves
# - Threshold line at BER=1e-3
# - Animation button
# """

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
# import joblib
# import csv
# import mplcursors
# from matplotlib import cm
# import time

# # ---------------- CONFIG ----------------
# MODEL_FILE = "fsomodel_rf.joblib"
# if not os.path.exists(MODEL_FILE):
#     raise FileNotFoundError(f"{MODEL_FILE} not found. Run build_and_train_fsober_model.py first.")

# data = joblib.load(MODEL_FILE)
# model = data['model']

# PT_DEFAULT = 10.0
# DIV_DEFAULT = 2.0
# DISTANCES = np.linspace(10, 2000, 300)
# RX_APERTURE_M = 0.05
# NOISE_POWER_W = 1e-9
# BER_THRESHOLD = 1e-3  # reliability cutoff

# # ---------------- FUNCTIONS ----------------
# def compute_snr_array(distances_m, Pt_mW, div_deg):
#     theta_rad = np.deg2rad(div_deg)
#     beam_radius = distances_m * np.tan(theta_rad / 2.0)
#     beam_area = np.pi * beam_radius**2 + 1e-12
#     rx_area = np.pi * (RX_APERTURE_M / 2.0)**2
#     Pr_mW = Pt_mW * (rx_area / beam_area)
#     Pr_W = Pr_mW * 1e-3
#     snr_linear = np.maximum(Pr_W / NOISE_POWER_W, 1e-12)
#     return snr_linear

# def compute_ber_array(distances_m, Pt_mW, div_deg):
#     X = np.column_stack((distances_m, np.full_like(distances_m, Pt_mW), np.full_like(distances_m, div_deg)))
#     return model.predict(X)

# # ---------------- STATE ----------------
# current_mode = "dist"
# three_d_fig = None
# animating = False

# # ---------------- MAIN FIGURE ----------------
# plt.rcParams.update({'font.size': 11})
# fig, ax_main = plt.subplots(figsize=(11, 7))
# plt.subplots_adjust(left=0.1, bottom=0.3, right=0.75)

# ax_snr = ax_main
# ax_ber = ax_main.twinx()
# ax_ber.set_yscale('log')

# snrs0 = compute_snr_array(DISTANCES, PT_DEFAULT, DIV_DEFAULT)
# bers0 = compute_ber_array(DISTANCES, PT_DEFAULT, DIV_DEFAULT)

# # color maps
# colors_snr = cm.Greens(np.linspace(0.3, 1, len(DISTANCES)))
# colors_ber = cm.Reds(np.linspace(0.3, 1, len(DISTANCES)))

# line_snr = ax_snr.scatter(DISTANCES, snrs0, c=colors_snr, label='SNR (linear)')
# line_ber = ax_ber.scatter(DISTANCES, bers0, c=colors_ber, label='BER (log)')

# ax_snr.set_xlabel("Distance (m)")
# ax_snr.set_ylabel("SNR (linear)", color='tab:blue')
# ax_ber.set_ylabel("BER (log scale)", color='tab:orange')
# ax_main.set_title("Distance vs SNR and BER")
# ax_snr.grid(True, linestyle='--', alpha=0.4)

# # threshold line
# ax_ber.axhline(y=BER_THRESHOLD, color='red', linestyle='--', lw=1.2)
# ax_ber.text(DISTANCES[-1], BER_THRESHOLD*1.2, "BER=1e-3 (unreliable)", color="red",
#             ha="right", va="bottom", fontsize=9)

# # ---------------- SLIDERS ----------------
# axcolor = 'lightgoldenrodyellow'
# ax_pt = plt.axes([0.1, 0.20, 0.6, 0.04], facecolor=axcolor)
# ax_div = plt.axes([0.1, 0.14, 0.6, 0.04], facecolor=axcolor)

# slider_pt = Slider(ax_pt, 'Power Pt (mW)', 1.0, 20.0, valinit=PT_DEFAULT, valstep=0.1)
# slider_div = Slider(ax_div, 'Divergence (deg)', 0.5, 5.0, valinit=DIV_DEFAULT, valstep=0.1)

# # ---------------- REDRAW ----------------
# cursor_handles = []
# def clear_cursors():
#     global cursor_handles
#     for c in cursor_handles:
#         try: c.disconnect()
#         except: pass
#     cursor_handles = []

# def attach_cursor(ax, distances, snrs, bers):
#     clear_cursors()
#     sc = ax.scatter(distances, snrs, s=1, alpha=0)
#     cursor = mplcursors.cursor(sc, hover=True)
#     def fmt(sel):
#         i = sel.index
#         return f"Dist: {distances[i]:.1f} m\nSNR: {snrs[i]:.2e}\nBER: {bers[i]:.2e}"
#     cursor.connect("add", lambda sel: sel.annotation.set_text(fmt(sel)))
#     cursor_handles.append(cursor)

# def redraw():
#     global current_mode, three_d_fig
#     Pt, div = slider_pt.val, slider_div.val

#     snrs = compute_snr_array(DISTANCES, Pt, div)
#     bers = compute_ber_array(DISTANCES, Pt, div)

#     ax_main.clear()
#     if current_mode == "dist":
#         ax_snr = ax_main
#         ax_ber = ax_main.twinx()
#         ax_ber.set_yscale('log')
#         ax_snr.scatter(DISTANCES, snrs, c=colors_snr, label="SNR (linear)")
#         ax_ber.scatter(DISTANCES, bers, c=colors_ber, label="BER (log)")
#         ax_snr.set_xlabel("Distance (m)")
#         ax_snr.set_ylabel("SNR (linear)")
#         ax_ber.set_ylabel("BER (log scale)")
#         ax_main.set_title(f"Distance vs SNR & BER | Pt={Pt:.1f} mW, Div={div:.1f}°")
#         ax_snr.grid(True, linestyle='--', alpha=0.4)
#         ax_ber.axhline(y=BER_THRESHOLD, color='red', linestyle='--', lw=1.2)
#         ax_ber.text(DISTANCES[-1], BER_THRESHOLD*1.2, "BER=1e-3 (unreliable)", color="red",
#                     ha="right", va="bottom", fontsize=9)
#         attach_cursor(ax_main, DISTANCES, snrs, bers)

#     elif current_mode == "ber_snr":
#         ax_main.semilogy(snrs, bers, 'r-o', ms=3)
#         ax_main.set_xlabel("SNR (linear)")
#         ax_main.set_ylabel("BER (log scale)")
#         ax_main.set_title(f"BER vs SNR | Pt={Pt:.1f} mW, Div={div:.1f}°")
#         ax_main.grid(True, linestyle='--', alpha=0.4)

#     elif current_mode == "snr_ber":
#         ax_main.plot(bers, snrs, 'g-o', ms=3)
#         ax_main.set_xscale('log')
#         ax_main.set_xlabel("BER (log scale)")
#         ax_main.set_ylabel("SNR (linear)")
#         ax_main.set_title(f"SNR vs BER | Pt={Pt:.1f} mW, Div={div:.1f}°")
#         ax_main.grid(True, linestyle='--', alpha=0.4)

#     elif current_mode == "3d":
#         from mpl_toolkits.mplot3d import Axes3D
#         if three_d_fig is not None: plt.close(three_d_fig)
#         three_d_fig = plt.figure(figsize=(9,6))
#         ax3 = three_d_fig.add_subplot(111, projection="3d")
#         ax3.plot(DISTANCES, snrs, np.log10(bers), 'p-', ms=3, color="purple")
#         ax3.set_xlabel("Distance (m)")
#         ax3.set_ylabel("SNR (linear)")
#         ax3.set_zlabel("log10(BER)")
#         ax3.set_title(f"3D Plot | Pt={Pt:.1f} mW, Div={div:.1f}°")
#         plt.show(block=False)

#     fig.canvas.draw_idle()

# slider_pt.on_changed(lambda val: redraw())
# slider_div.on_changed(lambda val: redraw())

# # ---------------- BUTTONS ----------------
# btn_positions = {
#     'Dist vs SNR': [0.78, 0.23, 0.18, 0.05],
#     'BER vs SNR':  [0.78, 0.16, 0.18, 0.05],
#     'SNR vs BER':  [0.78, 0.09, 0.18, 0.05],
#     '3D Plot':     [0.78, 0.02, 0.18, 0.05],
#     'Export CSV':  [0.58, 0.02, 0.18, 0.05],
#     'Save Figure': [0.38, 0.02, 0.18, 0.05],
#     'Animate':     [0.18, 0.02, 0.18, 0.05]
# }
# buttons = {name: Button(plt.axes(pos), name) for name, pos in btn_positions.items()}

# def set_mode(mode): 
#     global current_mode; current_mode = mode; redraw()
# buttons['Dist vs SNR'].on_clicked(lambda e: set_mode("dist"))
# buttons['BER vs SNR'].on_clicked(lambda e: set_mode("ber_snr"))
# buttons['SNR vs BER'].on_clicked(lambda e: set_mode("snr_ber"))
# buttons['3D Plot'].on_clicked(lambda e: set_mode("3d"))

# def export_csv(event):
#     Pt, div = slider_pt.val, slider_div.val
#     snrs = compute_snr_array(DISTANCES, Pt, div)
#     bers = compute_ber_array(DISTANCES, Pt, div)
#     with open("exported_data.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Distance_m","SNR_linear","BER"])
#         for d,s,b in zip(DISTANCES, snrs, bers):
#             writer.writerow([f"{d:.3f}", f"{s:.3e}", f"{b:.3e}"])
#     print("Exported data to exported_data.csv")

# def save_figure(event):
#     fig.savefig("current_plot.png", dpi=200, bbox_inches="tight")
#     print("Saved current plot as current_plot.png")

# def animate(event):
#     global animating
#     animating = True
#     for Pt in np.linspace(5, 20, 20):
#         if not animating:
#             break
#         slider_pt.set_val(Pt)
#         redraw()
#         plt.pause(0.3)
#     animating = False

# buttons['Export CSV'].on_clicked(export_csv)
# buttons['Save Figure'].on_clicked(save_figure)
# buttons['Animate'].on_clicked(animate)

# # ---------------- START ----------------
# redraw()
# plt.show()















"""
Professional FSO Analysis Tool - Production Ready Version
All UI/UX issues fixed: proper spacing, visible controls, professional layout
Physics and ML calculations preserved and working correctly
"""
import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import joblib
import csv
import mplcursors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
MODEL_FILE = "fsomodel_rf.joblib"

# --- Model Check ---
if not os.path.exists(MODEL_FILE):
    print("Model file not found. Creating a simple model...")
    from sklearn.ensemble import RandomForestRegressor
    from math import erfc, sqrt
    
    np.random.seed(42)
    n = 2000
    distances = np.random.uniform(10, 2000, n)
    pt_samples = np.random.uniform(5, 20, n)
    divs = np.random.uniform(0.5, 5, n)
    
    rows = []
    for d, p, div in zip(distances, pt_samples, divs):
        theta_rad = np.deg2rad(div)
        beam_radius = d * np.tan(theta_rad / 2.0)
        beam_area = np.pi * beam_radius**2 + 1e-12
        rx_area = np.pi * (0.05 / 2.0)**2
        Pr = p * (rx_area / beam_area) * 1e-3
        snr = max(Pr / 1e-9, 1e-12)
        ber = min(max(0.5 * erfc(np.sqrt(snr / 2.0)), 1e-12), 0.5)
        rows.append((d, p, div, ber))
    
    X = np.array([[r[0], r[1], r[2]] for r in rows])
    y = np.array([r[3] for r in rows])
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    joblib.dump({'model': model, 'features': ['distance_m','Pt_mW','div_deg']}, MODEL_FILE)
    print(f"Created and saved model to {MODEL_FILE}")

data = joblib.load(MODEL_FILE)
model = data['model']

PT_DEFAULT = 10.0
DIV_DEFAULT = 2.0
DISTANCES = np.linspace(10, 2000, 200) 
RX_APERTURE_M = 0.05
NOISE_POWER_W = 1e-9
BER_THRESHOLD = 1e-3

# ---------------- PHYSICS FUNCTIONS ----------------
def compute_snr_array(distances_m, Pt_mW, div_deg):
    theta_rad = np.deg2rad(div_deg)
    beam_radius = distances_m * np.tan(theta_rad / 2.0)
    beam_area = np.pi * beam_radius**2 + 1e-12
    rx_area = np.pi * (RX_APERTURE_M / 2.0)**2
    Pr_mW = Pt_mW * (rx_area / beam_area)
    Pr_W = Pr_mW * 1e-3
    snr_linear = np.maximum(Pr_W / NOISE_POWER_W, 1e-12)
    return snr_linear

def compute_ber_array(distances_m, Pt_mW, div_deg):
    X = np.column_stack((distances_m, 
                         np.full_like(distances_m, Pt_mW), 
                         np.full_like(distances_m, div_deg)))
    return model.predict(X)

# ---------------- STATE ----------------
class AppState:
    def __init__(self):
        self.current_mode = "dist_snr_ber"
        self.three_d_fig = None
        self.ax_ber_twin = None
        self.cursor_handles = []
        self.colorbar = None
        self.original_position = None
        self.ax_cbar_dedicated = None

state = AppState()

# ---------------- FIGURE SETUP (PRODUCTION LAYOUT) ----------------
plt.style.use('default')
fig = plt.figure(figsize=(18, 10)) 
fig.patch.set_facecolor('#f0f0f0')

# FIXED: Optimal spacing for all UI elements
gs = GridSpec(8, 9, figure=fig, 
              left=0.13, right=0.92, bottom=0.18, top=0.88,
              hspace=0.5, wspace=0.4,
              height_ratios=[6, 0.4, 0.4, 0.3, 0.7, 0.7, 0.7, 0.7])

# Main plot area (spans columns 0-7)
ax_main = fig.add_subplot(gs[0, 0:8]) 
# Dedicated Colorbar area (column 8)
ax_cbar = fig.add_subplot(gs[0, 8])

# Sliders - with more space between plot and sliders
ax_pt_slider = fig.add_subplot(gs[3, :])
ax_div_slider = fig.add_subplot(gs[4, :])

# Controls - pushed down further
ax_radio = fig.add_subplot(gs[6:8, 0:3]) 
ax_btn_export = fig.add_subplot(gs[6, 4:6])
ax_btn_save = fig.add_subplot(gs[6, 6:8])
ax_btn_animate = fig.add_subplot(gs[7, 4:6])
ax_btn_reset = fig.add_subplot(gs[7, 6:8])

# Store original plot position and dedicated colorbar axis
state.original_position = ax_main.get_position()
state.ax_cbar_dedicated = ax_cbar

# ---------------- SLIDERS (Professional styling) ----------------
slider_pt = Slider(ax_pt_slider, 'Power (mW)', 1.0, 20.0, valinit=PT_DEFAULT, valstep=0.1, color='#FF6B6B', track_color='#FFE5E5')
slider_div = Slider(ax_div_slider, 'Divergence (°)', 0.5, 5.0, valinit=DIV_DEFAULT, valstep=0.1, color='#4ECDC4', track_color='#E0F7F6')
for slider in [slider_pt, slider_div]:
    slider.label.set_fontsize(12)
    slider.label.set_weight('bold')
    slider.label.set_color('#000')
    slider.valtext.set_fontsize(11)
    slider.valtext.set_weight('bold')
    slider.valtext.set_color('#000')

# ---------------- RADIO BUTTONS ----------------
radio = RadioButtons(ax_radio, ('Dist-SNR-BER', 'Dist vs SNR', 'SNR vs BER', '3D View'), active=0)
for label in radio.labels:
    label.set_fontsize(11)
    label.set_weight('bold')
ax_radio.set_title('Visualization Mode', fontsize=12, weight='bold', pad=10, color='#000')
for spine in ax_radio.spines.values():
    spine.set_visible(False)
ax_radio.set_facecolor('#ffffff')

# ---------------- BUTTONS ----------------
btn_export = Button(ax_btn_export, 'Export CSV', color='#95E1D3', hovercolor='#7FD1C3')
btn_save = Button(ax_btn_save, 'Save Figure', color='#F38181', hovercolor='#E37171')
btn_animate = Button(ax_btn_animate, 'Animate Power', color='#EAFFD0', hovercolor='#DAEFC0')
btn_reset = Button(ax_btn_reset, 'Reset Parameters', color='#FFEAA7', hovercolor='#EEDA97')
for btn in [btn_export, btn_save, btn_animate, btn_reset]:
    btn.label.set_fontsize(11)
    btn.label.set_weight('bold')
    btn.label.set_color('#000')

# ---------------- CURSOR HANDLING ----------------
def clear_cursors():
    for c in state.cursor_handles:
        try: 
            c.disconnect()
        except: 
            pass
    state.cursor_handles = []

def attach_cursor(scatter_obj, x_data_sparse, y_data_sparse):
    clear_cursors()
    try:
        cursor = mplcursors.cursor(scatter_obj, hover=True)
        
        def fmt(sel):
            i_sparse = sel.index
            i_full = i_sparse * 15
            
            d_val = DISTANCES[i_full]
            snr_val = compute_snr_array(np.array([d_val]), slider_pt.val, slider_div.val)[0]
            ber_val = compute_ber_array(np.array([d_val]), slider_pt.val, slider_div.val)[0]

            return (f"Distance: {d_val:.1f} m\n"
                    f"SNR: {snr_val:.2e}\n"
                    f"BER: {ber_val:.2e}")
        
        cursor.connect("add", lambda sel: sel.annotation.set_text(fmt(sel)))
        cursor.connect("add", lambda sel: sel.annotation.get_bbox_patch().set(
            facecolor='#FFFACD', alpha=0.95, edgecolor='#333', linewidth=2))
        state.cursor_handles.append(cursor)
    except Exception:
        pass

# ---------------- CLEAN AXES ----------------
def clean_axes():
    """Clean axes while preserving position and managing the dedicated colorbar axis."""
    original_pos = state.original_position
    
    ax_main.clear()
    
    if state.ax_ber_twin is not None:
        try:
            state.ax_ber_twin.clear()
            state.ax_ber_twin.remove()
        except:
            pass
        state.ax_ber_twin = None
    
    state.ax_cbar_dedicated.clear()
    state.ax_cbar_dedicated.set_visible(False)
    state.colorbar = None
    
    ax_main.set_position(original_pos)
    
    fig.canvas.draw_idle()
    clear_cursors()

# ---------------- REDRAW FUNCTION ----------------
def redraw():
    Pt = slider_pt.val
    div = slider_div.val
    
    # Recompute data based on CURRENT slider values
    snrs = compute_snr_array(DISTANCES, Pt, div)
    bers = compute_ber_array(DISTANCES, Pt, div)
    
    clean_axes()
    
    param_text = f"Pt={Pt:.1f} mW | Div={div:.1f}°"
    
    if state.current_mode == "dist_snr_ber":
        pos = state.original_position
        state.ax_ber_twin = ax_main.twinx()
        ax_main.set_position(pos)
        state.ax_ber_twin.set_position(pos)
        
        ax_main.set_yscale('log') 
        ax_main.plot(DISTANCES, snrs, color='#1976D2', linewidth=2.5, alpha=0.8)
        
        state.ax_ber_twin.plot(DISTANCES, bers, 'r-', linewidth=2.5, alpha=0.8)
        scatter_ber = state.ax_ber_twin.scatter(DISTANCES[::15], bers[::15], 
                                               c='darkred', s=40, alpha=0.9, 
                                               edgecolors='white', linewidth=1.5, zorder=5)
        
        ax_main.set_ylim(min(snrs[snrs > 0])*0.1, max(snrs)*10)
        state.ax_ber_twin.set_yscale('log')
        state.ax_ber_twin.set_ylim(1e-12, 1e-1) 
        state.ax_ber_twin.axhline(y=BER_THRESHOLD, color='#FF1744', 
                                  linestyle='--', linewidth=2.5, alpha=0.8, zorder=0)
        
        ax_main.set_xlabel('Distance (m)', fontsize=12, weight='bold')
        ax_main.set_ylabel('SNR (log scale)', fontsize=12, weight='bold', color='#1976D2')
        state.ax_ber_twin.set_ylabel('BER (log scale)', fontsize=12, weight='bold', color='#D32F2F')
        
        ax_main.tick_params(axis='y', labelcolor='#1976D2', labelsize=10)
        ax_main.tick_params(axis='x', labelsize=10)
        state.ax_ber_twin.tick_params(axis='y', labelcolor='#D32F2F', labelsize=10)
        
        ax_main.set_title(f'Distance vs SNR vs BER\n{param_text}', 
                         fontsize=14, weight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=':')
        ax_main.set_facecolor('#FAFAFA')
        
        snr_patch = mpatches.Patch(color='#1976D2', alpha=0.8, label='SNR')
        ber_patch = mpatches.Patch(color='red', alpha=0.8, label='BER')
        thresh_patch = mpatches.Patch(color='#FF1744', label='BER Threshold')
        ax_main.legend(handles=[snr_patch, ber_patch, thresh_patch], 
                      loc='upper right', fontsize=11, framealpha=0.95)
        
        attach_cursor(scatter_ber, DISTANCES[::15], bers[::15])
        
    elif state.current_mode == "dist_snr":
        pos = state.original_position
        ax_main.set_position(pos)
        
        ax_main.set_yscale('log')
        ax_main.plot(DISTANCES, snrs, 'navy', alpha=0.8, linewidth=2)
        
        scatter = ax_main.scatter(DISTANCES[::15], snrs[::15], c=snrs[::15], 
                                 cmap='plasma', s=40, alpha=0.9, 
                                 edgecolors='#333', linewidth=0.5)
        
        ax_main.set_ylim(min(snrs[snrs > 0])*0.1, max(snrs)*10)
        
        ax_main.set_xlabel('Distance (m)', fontsize=12, weight='bold')
        ax_main.set_ylabel('SNR (log scale)', fontsize=12, weight='bold')
        ax_main.set_title(f'Distance vs SNR\n{param_text}', 
                         fontsize=14, weight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=':')
        ax_main.set_facecolor('#FAFAFA')
        ax_main.tick_params(axis='both', labelsize=10)
        
        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label('SNR Level', fontsize=11, weight='bold')
        
        attach_cursor(scatter, DISTANCES[::15], snrs[::15])
        
    elif state.current_mode == "snr_ber":
        pos = state.original_position
        ax_main.set_position(pos)
        
        ax_main.plot(snrs, bers, 'purple', alpha=0.8, linewidth=2)
        scatter = ax_main.scatter(snrs[::15], bers[::15], c=DISTANCES[::15], 
                                 cmap='coolwarm', s=40, alpha=0.9, 
                                 edgecolors='#333', linewidth=0.5)
        
        ax_main.set_xscale('log') 
        ax_main.set_yscale('log')
        ax_main.set_xlim(min(snrs[snrs > 0])*0.1, max(snrs)*10)
        ax_main.set_ylim(1e-12, 1e-1) 
        ax_main.axhline(y=BER_THRESHOLD, color='#FF1744', 
                        linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax_main.set_xlabel('SNR (log scale)', fontsize=12, weight='bold')
        ax_main.set_ylabel('BER (log scale)', fontsize=12, weight='bold')
        ax_main.set_title(f'SNR vs BER\n{param_text}', 
                         fontsize=14, weight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle=':')
        ax_main.set_facecolor('#FAFAFA')
        ax_main.tick_params(axis='both', labelsize=10)
        
        state.ax_cbar_dedicated.set_visible(True)
        state.colorbar = fig.colorbar(scatter, cax=state.ax_cbar_dedicated)
        state.colorbar.set_label('Distance (m)', fontsize=11, weight='bold')
        
        attach_cursor(scatter, snrs[::15], bers[::15])
        
    elif state.current_mode == "3d":
        from mpl_toolkits.mplot3d import Axes3D
        
        if state.three_d_fig is not None:
            try:
                plt.close(state.three_d_fig)
            except:
                pass
        
        state.three_d_fig = plt.figure(figsize=(8, 6))
        state.three_d_fig.patch.set_facecolor('#f0f0f0')
        ax3 = state.three_d_fig.add_subplot(111, projection='3d')
        
        log_snrs = np.log10(snrs)
        log_bers = np.log10(bers)

        colors = cm.turbo(np.linspace(0, 1, len(DISTANCES)))
        ax3.scatter(DISTANCES, log_snrs, log_bers, c=colors, s=25, alpha=0.7)
        ax3.plot(DISTANCES, log_snrs, log_bers, 'k-', alpha=0.3, linewidth=1)
        
        ax3.set_xlabel('\nDistance (m)', fontsize=12, weight='bold', labelpad=15)
        ax3.set_ylabel('\nlog₁₀(SNR)', fontsize=12, weight='bold', labelpad=15)
        ax3.set_zlabel('\nlog₁₀(BER)', fontsize=12, weight='bold', labelpad=15)
        ax3.set_title(f'3D: Distance vs log(SNR) vs log(BER)\n{param_text}', 
                     fontsize=14, weight='bold', pad=25)
        ax3.grid(True, alpha=0.3)
        ax3.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        state.three_d_fig.show()
    
    fig.canvas.draw_idle()

# ---------------- EVENT HANDLERS ----------------
def on_radio_clicked(label):
    mode_map = {
        'Dist-SNR-BER': 'dist_snr_ber',
        'Dist vs SNR': 'dist_snr',
        'SNR vs BER': 'snr_ber',
        '3D View': '3d'
    }
    state.current_mode = mode_map[label]
    redraw()

def export_csv(event):
    Pt, div = slider_pt.val, slider_div.val
    snrs = compute_snr_array(DISTANCES, Pt, div)
    bers = compute_ber_array(DISTANCES, Pt, div)
    
    filename = f"fso_Pt{Pt:.1f}_Div{div:.1f}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance_m', 'SNR_linear', 'BER'])
        for d, s, b in zip(DISTANCES, snrs, bers):
            writer.writerow([f'{d:.3f}', f'{s:.6e}', f'{b:.6e}'])
    print(f"✓ Exported: {filename}")

animating = False
def animate(event):
    global animating
    animating = True
    original = slider_pt.val
    for Pt in np.linspace(2, 20, 20):
        if not animating:
            break
        slider_pt.set_val(Pt)
        plt.pause(0.15)
    slider_pt.set_val(original)
    animating = False

def save_figure(event):
    filename = f"fso_{state.current_mode}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")

def reset_params(event):
    slider_pt.set_val(PT_DEFAULT)
    slider_div.set_val(DIV_DEFAULT)
    print("✓ Reset")

# Connect events
slider_pt.on_changed(lambda val: redraw())
slider_div.on_changed(lambda val: redraw())
radio.on_clicked(on_radio_clicked)
btn_export.on_clicked(export_csv)
btn_save.on_clicked(save_figure)
btn_animate.on_clicked(animate)
btn_reset.on_clicked(reset_params)

# Initial render
print("Initializing FSO Analysis Tool...")
redraw()

# Footer - positioned better to avoid overlap
fig.text(0.5, 0.04, "Interactive FSO Analysis Tool  |  Hover over data points for details  |  © 2025",
         ha='center', fontsize=10, style='italic', color='#333', weight='bold')

print("✓ Tool ready! Showing window...")
plt.show()



