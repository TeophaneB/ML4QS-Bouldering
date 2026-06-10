import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# idea 2 # of final rows per attempt
WINDOW_SIZES = [10, 20, 40, 60]  # Number of rows per attempt (e.g., 1 row = whole attempt, 2 rows = split attempt in half, etc.)


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
    """
    parts = folder_name.split()
    features = {}
    
    features["difficulty"] = parts[0]  # e.g., L1
    
    features["style"] = parts[1]  # e.g., N
    
    for el in parts[2]:
        if el == "No" or el == "N":
            features["topped"] = "N"
        else:
            features["topped"] = "Y"
    match = re.match(r"([a-zA-Z]+)(\d{4}-\d{2}-\d{2})", parts[3])
    if match:
        features["participant"] = match.group(1)   # e.g. teo
        date_str = match.group(2)                  # e.g. 2026-06-05
    else:
        features["participant"] = parts[3]
        date_str = ""

    time_str  = parts[4].replace("-", ":")         # 2:20:25
    ampm      = parts[5] if len(parts) > 5 else ""
    features["datetime"] = f"{date_str} {time_str} {ampm}".strip()

    return features


def get_window_ranges(length, window_size):
    """
    Get start and end indices for each window based on total length and desired number of windows.
    """
    window_length = length // window_size # if len=103 and window_size=10, then window_length=10, and we get 10 windows of length 10, and the last window will have the remaining 3 rows
    ranges = [(i * window_length, (i + 1) * window_length) for i in range(window_size)]

    # add final skipped rows to last window
    if ranges[-1][1] < length:
        ranges[-1] = (ranges[-1][0], length)

    return ranges

def summarise_window(acc_window, lin_acc_window, gyro_window, gravity_window, orientation_window):
    features = {}
    
    # Duration, based on accelerometer recording
    features["duration_seconds"] = acc_window["Time (s)"].max() - acc_window["Time (s)"].min()

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

    # Gravity magnitude
    gravity_mag = magnitude(
        gravity_window,
        "Gravity X (m/s^2)",
        "Gravity Y (m/s^2)",
        "Gravity Z (m/s^2)"
    )
    features.update(summarize_series(gravity_mag, "gravity_mag"))

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

    features = {}

    # metadata by reading folder name (L1 N Y teo2026... --> difficulty=L1, style=N, topped=Y, participant=teo)
    metadata = get_metadata(attempt_folder.name)
    length = min(len(acc), len(lin_acc), len(gyro), len(gravity), len(orientation))
    window_ranges = get_window_ranges(length, window_size) # output = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)] if length=103 and window_size=10

    # loop in windows of the attempt based on window_size and compute features for each window
    for window_range in window_ranges:
        # grab the time window for this window index
        acc_window = acc.iloc[window_range[0]:window_range[1]]
        lin_acc_window = lin_acc.iloc[window_range[0]:window_range[1]]
        gyro_window = gyro.iloc[window_range[0]:window_range[1]]
        gravity_window = gravity.iloc[window_range[0]:window_range[1]]
        orientation_window = orientation.iloc[window_range[0]:window_range[1]]

        # compute features for this window and add to overall features with a suffix indicating the window index
        window_features = summarise_window(acc_window, lin_acc_window, gyro_window, gravity_window, orientation_window)
        features.update(window_features)

    return pd.DataFrame([features])


def summarize_batch(batch_root, drop_metadata=True, window_size=None):
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
            attempt_folder / "Gravity.csv",
            attempt_folder / "Orientation.csv",
        ]

        # Only process folders that contain all required files
        if not all(path.exists() for path in required_files):
            print(f"Skipping {attempt_folder} - missing required files")
            continue

        summary = summarize_attempt(attempt_folder, window_size=window_size)
        summaries.append(summary)

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
    Summarise the whole batch directory into one global feature table.

    This keeps every valid attempt folder, including folders that are
    effectively duplicates, then computes mean, std, min, and max across
    the per-attempt feature columns.
    """


    batch_features = summarize_batch(batch_root, drop_metadata=True, window_size=window_size)

    if batch_features.empty:
        return pd.DataFrame()

    numeric_features = batch_features.select_dtypes(include=[np.number])
    global_summary = numeric_features.agg(["mean", "std", "min", "max"]).T
    global_summary.columns = [f"dataset_{stat}" for stat in global_summary.columns]

    return global_summary.reset_index().rename(columns={"index": "feature"})

if __name__ == "__main__":

    # loop through window sizes and save summaries for each window size to /FEATURES folder
    for window_size in WINDOW_SIZES:
        dataset_summary = summarize_dataset(
            "BOULDERING_DATA/batch1/batch1_unzipped",
            window_size=window_size
        )

        print(f"Dataset summary for window size {window_size}:")
        print(dataset_summary.head())

        # save to /FEATURES folder with filename indicating window size
        os.makedirs("FEATURES", exist_ok=True)
        dataset_summary.to_csv(f"FEATURES/bouldering_summary_{window_size}s.csv", index=False)