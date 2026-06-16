import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load files
lin_acc = pd.read_csv("BOULDERING_DATA/Batch1/Batch1_unzipped/L6 D Y kar2026-06-05 4-09-47 PM/Accelerometer.csv")
gyro = pd.read_csv("BOULDERING_DATA/Batch1/Batch1_unzipped/L6 D Y kar2026-06-05 4-09-47 PM/Gyroscope.csv")
orientation = pd.read_csv("BOULDERING_DATA/Batch1/Batch1_unzipped/L6 D Y kar2026-06-05 4-09-47 PM/Orientation.csv")

# Create magnitudes
lin_acc["lin_acc_magnitude"] = np.sqrt(
    lin_acc["X (m/s^2)"]**2 +
    lin_acc["Y (m/s^2)"]**2 +
    lin_acc["Z (m/s^2)"]**2
)

gyro["gyro_magnitude"] = np.sqrt(
    gyro["X (rad/s)"]**2 +
    gyro["Y (rad/s)"]**2 +
    gyro["Z (rad/s)"]**2
)

# Smooth signals
lin_acc["lin_acc_smooth"] = lin_acc["lin_acc_magnitude"].rolling(window=50, center=True).mean()
gyro["gyro_smooth"] = gyro["gyro_magnitude"].rolling(window=50, center=True).mean()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(lin_acc["Time (s)"], lin_acc["lin_acc_magnitude"], alpha=0.35, label="Raw")
axes[0].plot(lin_acc["Time (s)"], lin_acc["lin_acc_smooth"], linewidth=2, label="Smoothed")
axes[0].set_ylabel("Linear acc. magnitude\n(m/s²)")
axes[0].set_title("Hard Difficulty Bouldering Instance")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(gyro["Time (s)"], gyro["gyro_magnitude"], alpha=0.35, label="Raw")
axes[1].plot(gyro["Time (s)"], gyro["gyro_smooth"], linewidth=2, label="Smoothed")
axes[1].set_ylabel("Gyro magnitude\n(rad/s)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(orientation["Time (s)"], orientation["Pitch (°)"], label="Pitch")
axes[2].plot(orientation["Time (s)"], orientation["Roll (°)"], label="Roll")
axes[2].plot(orientation["Time (s)"], orientation["Yaw (°)"], label="Yaw")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Orientation (°)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Easy 9 
# Medium 18
# Hard 8

# Yes 26
# No 9 

# Normal 19
# Slab 5 
# Overhand 8 
# Dynamic 2