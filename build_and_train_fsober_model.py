# file: build_and_train_fsober_model.py
# Purpose: generate synthetic distance->BER dataset (no-weather), train model, save it.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
from math import erfc, sqrt

# ---------- Physical / simulation constants (tweakable) ----------
Pt_mW = 10.0           # Transmit optical power (mW) -- example
tx_aperture_m = 0.05   # transmitter aperture diameter (m)
rx_aperture_m = 0.05   # receiver aperture diameter (m)
beam_divergence_deg = 2.0   # full-angle beam divergence (degrees)
responsivity_A_per_W = 0.8  # photodiode responsivity (A/W) - example
noise_power_W = 1e-9    # equivalent noise power (W) - tunable
# (we will map received power to an SNR via SNR = Pr / noise_power_W)

# Helper: convert divergence to radius at distance L (r = distance * tan(theta/2))
def received_power_mW(Pt_mW, distance_m, beam_divergence_deg, tx_ap_m, rx_ap_m):
    # geometric spreading (approx): fraction = receiver_area / beam_area_at_rx
    theta_rad = np.deg2rad(beam_divergence_deg)  # full-angle
    beam_radius = distance_m * np.tan(theta_rad / 2.0)  # meters
    beam_area = np.pi * (beam_radius ** 2) + 1e-12
    rx_area = np.pi * (rx_ap_m / 2.0) ** 2
    fraction = rx_area / beam_area
    Pr_mW = Pt_mW * fraction
    return Pr_mW

# SNR -> BER mapping (approx for binary signaling)
def ber_from_snr_linear(snr_linear):
    # use BER ≈ 0.5 * erfc(sqrt(SNR/2))
    val = 0.5 * erfc(np.sqrt(snr_linear / 2.0))
    return float(val)

# Generate dataset
np.random.seed(42)
n_samples = 3000
# Distance range: 10 m to 2000 m (tune as needed)
distances = np.random.uniform(10, 2000, size=n_samples)
# Optional: vary Pt or divergence slightly to add diversity
pt_samples = Pt_mW * np.random.uniform(0.9, 1.1, size=n_samples)
divs = np.random.uniform(1.5, 2.5, size=n_samples)

rows = []
for d, p, div in zip(distances, pt_samples, divs):
    Pr_mW = received_power_mW(p, d, div, tx_aperture_m, rx_aperture_m)
    Pr_W = Pr_mW * 1e-3
    # Convert to SNR (simple): SNR = received_power / noise_power
    snr_linear = max(Pr_W / noise_power_W, 1e-12)
    ber = ber_from_snr_linear(snr_linear)
    # Clip BER to avoid exact zeros (for ML stability)
    ber = min(max(ber, 1e-12), 0.5)
    rows.append((d, p, div, Pr_mW, snr_linear, ber))

df = pd.DataFrame(rows, columns=['distance_m', 'Pt_mW', 'div_deg', 'Pr_mW', 'snr_linear', 'ber'])
# Keep only needed features (client asked only distance as input; but we keep Pt/div too)
X = df[['distance_m', 'Pt_mW', 'div_deg']].values
y = df['ber'].values

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)

# Save model and a small scaler-like meta (we don't scale here, but save columns)
joblib.dump({'model': model, 'features': ['distance_m','Pt_mW','div_deg']}, 'fsomodel_rf.joblib')
print("Saved model to fsomodel_rf.joblib")

# Optional: plot true vs predicted (for a sample)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, s=6, alpha=0.6)
plt.xlabel("True BER")
plt.ylabel("Predicted BER")
plt.title("RandomForest: True vs Predicted BER")
plt.grid(True)
plt.tight_layout()
plt.savefig("ber_true_vs_pred.png", dpi=150)
print("Saved ber_true_vs_pred.png")
