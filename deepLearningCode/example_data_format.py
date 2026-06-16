"""
Example script showing the exact format of data at every stage of the TCN preprocessing pipeline.

Run this after merge_raw_data.py has created:
  - tcn_raw_concatenated.csv
  - attempt_id_mapping.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tcn_inputdata import (
    add_angle_sin_cos_features,
    build_tcn_dataset,
    standardize_train_test_sequences,
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ============================================================================
# STAGE 1: Load the raw concatenated dataframe
# ============================================================================
print_section("STAGE 1: Raw Concatenated Data (tcn_raw_concatenated.csv)")

raw_df = pd.read_csv("tcn_raw_concatenated.csv")

print(f"\nDataFrame shape: {raw_df.shape}")
print(f"  Rows (sensor samples across all attempts): {raw_df.shape[0]:,}")
print(f"  Columns (metadata + sensor channels): {raw_df.shape[1]}")

print(f"\nColumn names and dtypes:")
for col in raw_df.columns:
    print(f"  {col:25s} {str(raw_df[col].dtype):15s}")

print(f"\nFirst 3 rows of data:")
print(raw_df.head(3).to_string())

print(f"\nAttempt ID summary:")
print(f"  Unique attempt IDs: {raw_df['attempt_id'].nunique()}")
print(f"  Attempt ID range: {raw_df['attempt_id'].min()} to {raw_df['attempt_id'].max()}")
print(f"  Samples per attempt (min/max/mean):")
samples_per_attempt = raw_df.groupby('attempt_id').size()
print(f"    Min:  {samples_per_attempt.min():6d} samples")
print(f"    Max:  {samples_per_attempt.max():6d} samples")
print(f"    Mean: {samples_per_attempt.mean():6.1f} samples")

print(f"\nDifficulty label summary:")
print(f"  Unique labels: {raw_df['difficulty_label'].nunique()}")
print(raw_df['difficulty_label'].value_counts().sort_index())

print(f"\nDuration summary (seconds per attempt):")
print(f"  Min:  {raw_df['duration_seconds'].min():6.2f} seconds")
print(f"  Max:  {raw_df['duration_seconds'].max():6.2f} seconds")
print(f"  Mean: {raw_df['duration_seconds'].mean():6.2f} seconds")
print(f"  Std:  {raw_df['duration_seconds'].std():6.2f} seconds")


# ============================================================================
# STAGE 2: Attempt ID mapping
# ============================================================================
print_section("STAGE 2: Attempt ID to Folder Mapping (attempt_id_mapping.csv)")

id_mapping = pd.read_csv("attempt_id_mapping.csv")

print(f"\nMapping dataframe shape: {id_mapping.shape}")
print(f"  Rows (one per unique attempt): {id_mapping.shape[0]}")
print(f"  Columns: {list(id_mapping.columns)}")

print(f"\nFirst 5 rows:")
print(id_mapping.head(5).to_string())

print(f"\nLast 5 rows:")
print(id_mapping.tail(5).to_string())


# ============================================================================
# STAGE 3: Angle encoding (sin/cos conversion)
# ============================================================================
print_section("STAGE 3: Add Angle Sin/Cos Features")

df_with_angles = raw_df.copy()
df_with_angles, angle_cols = add_angle_sin_cos_features(
    df_with_angles,
    angle_cols=("yaw", "pitch", "roll"),
)

print(f"\nOriginal angle columns: yaw, pitch, roll")
print(f"New angle columns created: {angle_cols}")

print(f"\nDataFrame shape after angle encoding: {df_with_angles.shape}")
print(f"  Original yaw/pitch/roll columns removed (circular encoding applied)")
print(f"  New sin/cos columns added: {len(angle_cols)} columns")

sensor_cols = [
    "acc_x", "acc_y", "acc_z",
    "lin_acc_x", "lin_acc_y", "lin_acc_z",
    "gyro_x", "gyro_y", "gyro_z",
] + angle_cols

print(f"\nAll sensor columns used for TCN:")
for i, col in enumerate(sensor_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSample angle conversions (one row):")
sample_row = raw_df.iloc[0]
yaw_rad = np.deg2rad(sample_row['yaw'])
pitch_rad = np.deg2rad(sample_row['pitch'])
roll_rad = np.deg2rad(sample_row['roll'])
print(f"  Original yaw (degrees):  {sample_row['yaw']:8.3f}°")
print(f"  sin(yaw):                {np.sin(yaw_rad):8.3f}")
print(f"  cos(yaw):                {np.cos(yaw_rad):8.3f}")


# ============================================================================
# STAGE 4: Resample to fixed 500 time steps
# ============================================================================
print_section("STAGE 4: Resample Each Attempt to 500 Fixed Time Steps")

X, y, durations = build_tcn_dataset(
    full_df=df_with_angles,
    attempt_id_col="attempt_id",
    time_col="time_seconds",
    sensor_cols=sensor_cols,
    label_col="difficulty_label",
    n_steps=500,
)

print(f"\nOutput arrays:")
print(f"  X shape:          {X.shape}")
print(f"    (attempts, time_steps, channels) = ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
print(f"  y shape:          {y.shape}")
print(f"  durations shape:  {durations.shape}")

print(f"\nX array properties:")
print(f"  dtype:          {X.dtype}")
print(f"  Min value:      {X.min():12.6f}")
print(f"  Max value:      {X.max():12.6f}")
print(f"  Mean value:     {X.mean():12.6f}")
print(f"  Std value:      {X.std():12.6f}")

print(f"\ny array (labels):")
print(f"  dtype:          {y.dtype}")
print(f"  Unique values:  {np.unique(y)}")
print(f"  Class counts:")
for label in np.unique(y):
    count = (y == label).sum()
    pct = 100 * count / len(y)
    print(f"    Label {label}: {count:3d} attempts ({pct:5.1f}%)")

print(f"\ndurations array (seconds per attempt):")
print(f"  dtype:          {durations.dtype}")
print(f"  Min:            {durations.min():7.2f} seconds")
print(f"  Max:            {durations.max():7.2f} seconds")
print(f"  Mean:           {durations.mean():7.2f} seconds")
print(f"  Std:            {durations.std():7.2f} seconds")

print(f"\nSample sequence (first 3 time steps of first attempt):")
print(f"  Attempt 0, Time steps 0-2:")
sample_seq = X[0, :3, :]
for t in range(3):
    print(f"    t={t}: {sample_seq[t, :6]}")
    print(f"          ... (showing first 6 of {X.shape[2]} channels)")


# ============================================================================
# STAGE 5: Train/test split
# ============================================================================
print_section("STAGE 5: Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain/test split (80/20):")
print(f"  X_train shape:  {X_train.shape}")
print(f"  X_test shape:   {X_test.shape}")
print(f"  y_train shape:  {y_train.shape}")
print(f"  y_test shape:   {y_test.shape}")

print(f"\nTraining set class balance:")
for label in np.unique(y_train):
    count = (y_train == label).sum()
    pct = 100 * count / len(y_train)
    print(f"  Label {label}: {count:3d} attempts ({pct:5.1f}%)")

print(f"\nTest set class balance:")
for label in np.unique(y_test):
    count = (y_test == label).sum()
    pct = 100 * count / len(y_test)
    print(f"  Label {label}: {count:3d} attempts ({pct:5.1f}%)")


# ============================================================================
# STAGE 6: Standardization (fit on training data only)
# ============================================================================
print_section("STAGE 6: Standardization (StandardScaler fitted on training data only)")

X_train_scaled, X_test_scaled, scaler = standardize_train_test_sequences(X_train, X_test)

print(f"\nStandardized array shapes:")
print(f"  X_train_scaled shape:  {X_train_scaled.shape}")
print(f"  X_test_scaled shape:   {X_test_scaled.shape}")

print(f"\nStandardization parameters (fitted on training data only):")
print(f"  Mean (learned from training):  {scaler.mean_[:6]}")
print(f"                                ... (showing first 6 of {len(scaler.mean_)} channels)")
print(f"  Scale (std from training):     {scaler.scale_[:6]}")
print(f"                                ... (showing first 6 of {len(scaler.scale_)} channels)")

print(f"\nScaled data properties (training set):")
print(f"  Min value:      {X_train_scaled.min():12.6f}")
print(f"  Max value:      {X_train_scaled.max():12.6f}")
print(f"  Mean value:     {X_train_scaled.mean():12.6f} (approx 0)")
print(f"  Std value:      {X_train_scaled.std():12.6f} (approx 1)")

print(f"\nScaled data properties (test set):")
print(f"  Min value:      {X_test_scaled.min():12.6f}")
print(f"  Max value:      {X_test_scaled.max():12.6f}")
print(f"  Mean value:     {X_test_scaled.mean():12.6f}")
print(f"  Std value:      {X_test_scaled.std():12.6f}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_section("FINAL SUMMARY: Ready for TCN Model")

print(f"\nData pipeline complete. You now have:")
print(f"  1. X_train_scaled : shape {X_train_scaled.shape}")
print(f"     - {X_train_scaled.shape[0]} training attempts")
print(f"     - {X_train_scaled.shape[1]} time steps per attempt (normalized full duration)")
print(f"     - {X_train_scaled.shape[2]} sensor channels (9 + 6 angle encodings)")
print(f"     - Standardized (mean≈0, std≈1)")
print(f"")
print(f"  2. X_test_scaled  : shape {X_test_scaled.shape}")
print(f"     - {X_test_scaled.shape[0]} test attempts")
print(f"     - Same time steps and channels as training")
print(f"     - Scaled with training parameters (no leakage)")
print(f"")
print(f"  3. y_train        : shape {y_train.shape}")
print(f"     - {len(y_train)} labels (one per training attempt)")
print(f"     - Classes: {np.unique(y_train)}")
print(f"")
print(f"  4. y_test         : shape {y_test.shape}")
print(f"     - {len(y_test)} labels (one per test attempt)")
print(f"     - Classes: {np.unique(y_test)}")
print(f"")
print(f"Ready to feed into TCN model!")
print(f"  Model input: X_train_scaled with shape {X_train_scaled.shape}")
print(f"  Model output: y_train with shape {y_train.shape}")


print("\n" + "=" * 80)
print("  End of Example")
print("=" * 80 + "\n")
