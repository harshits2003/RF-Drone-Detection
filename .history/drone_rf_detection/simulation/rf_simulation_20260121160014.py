import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
