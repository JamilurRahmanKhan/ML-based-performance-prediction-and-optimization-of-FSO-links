# file: predict_ber.py
# Purpose: Load trained FSO BER model and predict BER for multiple distances with plot

import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load saved model
model_data = joblib.load("fsomodel_rf.joblib")
model = model_data['model']
features = model_data['features']

print("FSO BER Prediction Demo (No Weather Effects)")
print("============================================")

# Ask user for distances (comma-separated)
distances_input = input("Enter distance(s) in meters (comma-separated for multiple): ")
distances_list = [float(d.strip()) for d in distances_input.split(",")]

# Fixed power and divergence
Pt_mW = 10.0
div_deg = 2.0

# Prepare input and predict
X_new = np.array([[d, Pt_mW, div_deg] for d in distances_list])
ber_preds = model.predict(X_new)

# Show table of results
print("\nDistance (m)   Predicted BER")
print("-----------------------------")
for d, ber in zip(distances_list, ber_preds):
    print(f"{d:10.2f}   {ber:.6e}")

# Plot BER vs Distance
plt.figure(figsize=(7,4))
plt.plot(distances_list, ber_preds, marker='o', linestyle='-', color='b')
plt.xlabel("Distance (m)")
plt.ylabel("Predicted BER")
plt.title("FSO BER vs Distance (No Weather)")
plt.grid(True)
plt.tight_layout()
plt.show()
