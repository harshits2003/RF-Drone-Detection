#----SESSION 1: RF ACTIVITY SIMULATION-------
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
# -----------------------------
# Simulation parameters
# -----------------------------
fs = 1000            # Sampling frequency (Hz)
duration = 1.0       # seconds
t = np.linspace(0, duration, int(fs * duration))

# -----------------------------
# RF activity models
# -----------------------------

# No-drone scenario (background RF noise)
noise_only = np.random.normal(0, 0.4, len(t))

# Drone-present scenario (noise + bursty activity)
burst_mask = (np.random.rand(len(t)) > 0.7).astype(int)
bursts = burst_mask * np.sin(2 * np.pi * 40 * t)
drone_signal = noise_only + bursts

# --- Visualizing the simulated RF activity ---
plt.figure(figsize=(10, 4))
plt.plot(t, noise_only, label="No Drone", alpha=0.8)
plt.plot(t, drone_signal, label="Drone Present", alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("RF Activity Level")
plt.title("Simulated RF Activity")
plt.legend()
plt.tight_layout()
# Save plot
plt.savefig("drone_rf_detection/results/plots/simulated_rf_activity.png", dpi=300)
plt.show()

# -----------------------------
# Feature extraction
# -----------------------------

def extract_features(signal):
    mean_energy = np.mean(signal ** 2)
    variance = np.var(signal)
    return mean_energy, variance

no_drone_features = extract_features(noise_only)
drone_features = extract_features(drone_signal)

print("No Drone Features:")
print("  Mean Energy :", no_drone_features[0])
print("  Variance    :", no_drone_features[1])

print("\nDrone Features:")
print("  Mean Energy :", drone_features[0])
print("  Variance    :", drone_features[1])

# -----------------------------
# Simple detection logic
# -----------------------------

ENERGY_THRESHOLD = no_drone_features[0] * 1.3 # The threshold is set relative to background RF energy, not as a fixed constant.

print("\nDetection Result:")
if (drone_features[0] > ENERGY_THRESHOLD and
    drone_features[1] > no_drone_features[1] * 1.2):
    print("🚨 DRONE DETECTED")
else:
    print("✅ AREA CLEAR")

#----- SESSION 2: FEATURE-SPACE VISUALIZATION -------

# -----------------------------
# Multiple-run simulation
# -----------------------------

NUM_SAMPLES = 50

no_drone_data = []
drone_data = []

for _ in range(NUM_SAMPLES):

    # No-drone signal
    noise_only = np.random.normal(0, 0.4, len(t))
    
    # Drone signal
    burst_mask = (np.random.rand(len(t)) > 0.9).astype(int)
    bursts = 2.5 * burst_mask * np.sin(2 * np.pi * 40 * t)
    drone_signal = noise_only + bursts

    # Extract features
    no_drone_data.append(extract_features(noise_only))
    drone_data.append(extract_features(drone_signal))

# Convert to numpy arrays
no_drone_data = np.array(no_drone_data)
drone_data = np.array(drone_data)

# Feature-Space scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    no_drone_data[:, 0],
    no_drone_data[:, 1],
    label="No Drone",
    alpha=0.7
)

plt.scatter(
    drone_data[:, 0],
    drone_data[:, 1],
    label="Drone Present",
    alpha=0.7
)

plt.xlabel("Mean Energy")
plt.ylabel("Variance")
plt.title("Feature Space: Drone vs No Drone")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("drone_rf_detection/results/plots/feature_space_separation.png", dpi=300)
plt.show()
print("Feature-space plot saved to results/plots/feature_space_separation.png")
# This plot shows Detectibility; Although raw RF activity looks noisy, the extracted features form separable clusters, which enables reliable detection using simple classifiers.

#----SESSION 3: THRESHOLD CALIBRATION & DECISION BOUNDARY DESIGN-------

# -----------------------------
# Threshold based on background RF
# -----------------------------

mean_energy_no_drone = no_drone_data[:, 0]
mean_energy_drone = drone_data[:, 0]

# Threshold: midway between background mean and drone mean
threshold_energy = 0.5 * (
    np.mean(mean_energy_no_drone) + np.mean(mean_energy_drone)
)

print("\n--- Threshold Calibration ---")
print("Mean energy (No Drone):", np.mean(mean_energy_no_drone))
print("Mean energy (Drone)   :", np.mean(mean_energy_drone))
print("Chosen energy threshold:", threshold_energy)

# -----------------------------
# Detection performance
# -----------------------------

false_alarms = np.sum(mean_energy_no_drone > threshold_energy)
detections = np.sum(mean_energy_drone > threshold_energy)

print("\n--- Detection Performance ---")
print("Total No-Drone samples :", len(mean_energy_no_drone))
print("False alarms           :", false_alarms)

print("Total Drone samples    :", len(mean_energy_drone))
print("Correct detections     :", detections)

# -----------------------------
# Decision boundary visualization
# -----------------------------

plt.figure(figsize=(6, 6))

plt.scatter(
    no_drone_data[:, 0],
    no_drone_data[:, 1],
    label="No Drone",
    alpha=0.7
)

plt.scatter(
    drone_data[:, 0],
    drone_data[:, 1],
    label="Drone Present",
    alpha=0.7
)

# Decision boundary (vertical line)
plt.axvline(
    threshold_energy,
    color="red",
    linestyle="--",
    label="Decision Threshold"
)

plt.xlabel("Mean Energy")
plt.ylabel("Variance")
plt.title("Decision Boundary for Drone Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("drone_rf_detection/results/plots/decision_boundary.png", dpi=300)
plt.show()

print("Saved decision boundary plot.")
# Calibrated the detection threshold using background RF statistics and evaluate false alarms versus detections. This allows us to tune sensitivity without decoding any protocol.

