import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

"""
4. Engineer features (cf. Chapter 4), again describe your choices for setting, and analyse
the usefulness of the resulting set of features.
"""

# TODO highlight the final best model when printing results

BOULDERING_ROOT = "BOULDERING_DATA/"

# idea 2 # of final rows per attempt
WINDOW_SIZES = [1, 5, 10, 20, 50, 100]  # Number of rows per attempt (e.g., 1 row = whole attempt, 2 rows = split attempt in half, etc.)

def magnitude(df, x_col, y_col, z_col):
    """
    Compute vector magnitude: sqrt(x^2 + y^2 + z^2)
    """
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)


def summarize_series(series, prefix):
    """
    Return mean, standard deviation, min, and max for one signal.
    """
    return {
        f"{prefix}_mean": series.mean(),
        f"{prefix}_std": series.std(),
        f"{prefix}_min": series.min(),
        f"{prefix}_max": series.max()
    }


def get_metadata(folder_name):
    """
    Regex to extract metadata from folder name: level, style, topped, participant, date
    Example folder name: "L1 N Y teo2026 2024-05-01 14-30
    Numerical mappings: difficulty (1-3), style (1-3), topped (0-1)
    """
    parts = folder_name.split()
    features = {}

    diff = parts[0] if len(parts) > 0 else ""
    if diff in {"L1", "L2"}:
        diff = 1
    elif diff in {"L3", "L4"}:
        diff = 2
    elif diff in {"L5", "L6"}:
        diff = 3
    else:
        diff = 0
    features["difficulty"] = diff

    style = parts[1].upper() if len(parts) > 1 else ""
    style_map = {"N": 1, "O": 2, "S": 3, "D": 4}
    features["style"] = style_map.get(style, 0)

    topped_value = parts[2] if len(parts) > 2 else ""
    features["topped"] = 1 if topped_value.lower() in {"y", "yes"} else 0

    return features


def get_window_ranges(length, window_size):
    """
    Get start and end indices for each window based on total length and desired number of windows.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    indices = np.array_split(np.arange(length), window_size)
    ranges = []
    for chunk in indices:
        if len(chunk) == 0:
            ranges.append((0, 0))
        else:
            ranges.append((int(chunk[0]), int(chunk[-1]) + 1))

    return ranges


def prefix_window_features(window_features, window_index):
    """Prefix each feature name with the window index so columns stay unique."""
    return {f"window_{window_index + 1}_{key}": value for key, value in window_features.items()}


def add_angle_sin_cos_features(df):
    """Convert circular orientation direction features to sin/cos pairs.

    Mean/min/max are direction-like angle summaries, so they need circular
    encoding. Standard deviation and range stay as numeric variation measures.
    """
    angle_suffixes = (
        "_yaw_mean", "_pitch_mean", "_roll_mean",
        "_yaw_min", "_pitch_min", "_roll_min",
        "_yaw_max", "_pitch_max", "_roll_max",
    )

    angle_columns = [
        column
        for column in df.columns
        if column.endswith(angle_suffixes)
    ]

    if not angle_columns:
        return df

    df = df.copy()

    for column in angle_columns:
        radians = np.deg2rad(df[column])
        df[f"{column}_sin"] = np.sin(radians)
        df[f"{column}_cos"] = np.cos(radians)

    # Remove direction-like angle summaries after circular encoding.
    df = df.drop(columns=angle_columns)

    return df

def summarise_window(acc_window, lin_acc_window, gyro_window, gravity_window, orientation_window):
    features = {}

    if acc_window.empty:
        return {
            "acc_mag_mean": np.nan,
            "acc_mag_std": np.nan,
            "acc_mag_min": np.nan,
            "acc_mag_max": np.nan,
            "lin_acc_mag_mean": np.nan,
            "lin_acc_mag_std": np.nan,
            "lin_acc_mag_min": np.nan,
            "lin_acc_mag_max": np.nan,
            "gyro_mag_mean": np.nan,
            "gyro_mag_std": np.nan,
            "gyro_mag_min": np.nan,
            "gyro_mag_max": np.nan,
            # "gravity_mag_mean": np.nan,
            # "gravity_mag_std": np.nan,
            # "gravity_mag_min": np.nan,
            # "gravity_mag_max": np.nan,
            "yaw_mean": np.nan,
            "yaw_std": np.nan,
            "yaw_min": np.nan,
            "yaw_max": np.nan,
            "pitch_mean": np.nan,
            "pitch_std": np.nan,
            "pitch_min": np.nan,
            "pitch_max": np.nan,
            "roll_mean": np.nan,
            "roll_std": np.nan,
            "roll_min": np.nan,
            "roll_max": np.nan,
            "yaw_range": np.nan,
            "pitch_range": np.nan,
            "roll_range": np.nan,
        }
    
    # Accelerometer magnitude
    acc_mag = magnitude(
        acc_window,
        "X (m/s^2)",
        "Y (m/s^2)",
        "Z (m/s^2)"
    )
    features.update(summarize_series(acc_mag, "acc_mag"))

    # Linear acceleration magnitude
    lin_acc_mag = magnitude(
        lin_acc_window,
        "X (m/s^2)",
        "Y (m/s^2)",
        "Z (m/s^2)"
    )
    features.update(summarize_series(lin_acc_mag, "lin_acc_mag"))

    # Gyroscope magnitude
    gyro_mag = magnitude(
        gyro_window,
        "X (rad/s)",
        "Y (rad/s)",
        "Z (rad/s)"
    )
    features.update(summarize_series(gyro_mag, "gyro_mag"))

    # # Gravity magnitude
    # gravity_mag = magnitude(
    #     gravity_window,
    #     "Gravity X (m/s^2)",
    #     "Gravity Y (m/s^2)",
    #     "Gravity Z (m/s^2)"
    # )
    # features.update(summarize_series(gravity_mag, "gravity_mag"))

    # Orientation summaries: mean, std, min, max
    for col in ["Yaw (°)", "Pitch (°)", "Roll (°)"]:
        clean_name = col.replace(" (°)", "").lower()
        features.update(summarize_series(orientation_window[col], clean_name))

    # Orientation ranges
    features["yaw_range"] = orientation_window["Yaw (°)"].max() - orientation_window["Yaw (°)"].min()
    features["pitch_range"] = orientation_window["Pitch (°)"].max() - orientation_window["Pitch (°)"].min()
    features["roll_range"] = orientation_window["Roll (°)"].max() - orientation_window["Roll (°)"].min()

    return features


def summarize_attempt(attempt_folder, window_size=10):
    """
    Summarise one bouldering attempt into one machine-learning row.

    Summaries the whole attempt into fixed num of rows based on window_size, e.g. if window_size=10, then split the attempt into 10 equal time windows and compute features for each window.
    """

    attempt_folder = Path(attempt_folder)

    # Load files
    acc = pd.read_csv(attempt_folder / "Accelerometer.csv")
    lin_acc = pd.read_csv(attempt_folder / "Linear Accelerometer.csv")
    gyro = pd.read_csv(attempt_folder / "Gyroscope.csv")
    gravity = pd.read_csv(attempt_folder / "Gravity.csv")
    orientation = pd.read_csv(attempt_folder / "Orientation.csv")

    features = get_metadata(attempt_folder.name)
    attempt_id = attempt_folder.name
    features["window_count"] = window_size
    features["duration_seconds"] = acc["Time (s)"].max() - acc["Time (s)"].min()

    # metadata by reading folder name (L1 N Y teo2026... --> difficulty=L1, style=N, topped=Y, participant=teo)
    length = min(len(acc), len(lin_acc), len(gyro), len(gravity), len(orientation))
    window_ranges = get_window_ranges(length, window_size) # output = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)] if length=103 and window_size=10

    # loop in windows of the attempt based on window_size and compute features for each window
    for window_index, window_range in enumerate(window_ranges):
        # grab the time window for this window index
        acc_window = acc.iloc[window_range[0]:window_range[1]]
        lin_acc_window = lin_acc.iloc[window_range[0]:window_range[1]]
        gyro_window = gyro.iloc[window_range[0]:window_range[1]]
        gravity_window = gravity.iloc[window_range[0]:window_range[1]]
        orientation_window = orientation.iloc[window_range[0]:window_range[1]]

        # compute features for this window and add to overall features with a suffix indicating the window index
        window_features = summarise_window(acc_window, lin_acc_window, gyro_window, gravity_window, orientation_window)
        features.update(prefix_window_features(window_features, window_index))

    return pd.DataFrame([features]), attempt_id


def summarize_batch(batch_root, drop_metadata=False, window_size=None):
    """
    Summarise every attempt folder inside a batch directory.

    This walks only the immediate subfolders under batch_root and keeps
    folders that contain the expected sensor CSV files.
    """

    batch_root = Path(batch_root)
    summaries = []

    # Loop through attempt folders and summarize each attempt 
    for attempt_folder in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        required_files = [
            attempt_folder / "Accelerometer.csv",
            attempt_folder / "Linear Accelerometer.csv",
            attempt_folder / "Gyroscope.csv",
            # attempt_folder / "Gravity.csv",
            attempt_folder / "Orientation.csv",
        ]

        # Only process folders that contain all required files
        if not all(path.exists() for path in required_files):
            print(f"Skipping {attempt_folder} - missing required files")
            continue

        summary_df, attempt_id = summarize_attempt(attempt_folder, window_size=window_size)
        summary_df["_attempt_id"] = attempt_id
        summaries.append(summary_df)

    if not summaries:
        return pd.DataFrame()

    batch_features = pd.concat(summaries, ignore_index=True)

    if drop_metadata:
        batch_features = batch_features.drop(
            columns=["attempt_id", "difficulty", "topped", "style"],
            errors="ignore",
        )

    return batch_features


def summarize_dataset(batch_root, window_size):
    """
    Summarise the whole batch directory into one row per attempt.

    Each row represents one attempt, and each window's features are flattened
    into uniquely named columns.
    """

    return summarize_batch(batch_root, drop_metadata=False, window_size=window_size)


def summarize_all_data(window_size):
    """Summarise the entire bouldering dataset.
    Returns combined dataframe and mapping of index to attempt_id.
    """
    dataset_features = summarize_dataset(BOULDERING_ROOT, window_size=window_size)
    
    if dataset_features.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract attempt_id mapping before removing the column
    attempt_mapping = dataset_features[["_attempt_id"]].copy()
    attempt_mapping.reset_index(drop=True, inplace=True)
    attempt_mapping.rename(columns={"_attempt_id": "attempt_id"}, inplace=True)
    
    # Remove temporary column from the main dataframe
    dataset_features = dataset_features.drop(columns=["_attempt_id"])

    # Encode circular direction features once at the full-dataset level.
    dataset_features = add_angle_sin_cos_features(dataset_features)
    
    return dataset_features, attempt_mapping

if __name__ == "__main__":
    os.makedirs("FEATURES", exist_ok=True)

    # loop through each window size and save attempt table
    for window_size in WINDOW_SIZES:
        attempt_features, attempt_mapping = summarize_all_data(window_size=window_size)

        print(f"Summary for window size {window_size}:")
        print(attempt_features.head())

        output_name = f"FEATURES/bouldering_summary_{window_size}.csv"
        attempt_features.to_csv(output_name, index=False)
        
        # Save the mapping file with index -> attempt_id mapping
        mapping_name = f"FEATURES/bouldering_summary_{window_size}_mapping.csv"
        attempt_mapping.to_csv(mapping_name, index=True)
        print(f"Saved mapping to {mapping_name}")
