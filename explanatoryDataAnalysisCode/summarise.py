import pandas as pd
import numpy as np
from pathlib import Path


def parse_difficulty_label(folder_name):
    """Map folder difficulty token (L1-L6) to 1-3 buckets."""
    first_token = Path(folder_name).name.split()[0].upper() if Path(folder_name).name.split() else ""

    if first_token in {"L1", "L2"}:
        return 1
    if first_token in {"L3", "L4"}:
        return 2
    if first_token in {"L5", "L6"}:
        return 3

    return None


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


def summarize_attempt(attempt_folder, attempt_id=None, difficulty=None, topped=None, style=None):
    """
    Summarise one bouldering attempt into one machine-learning row.

    Expected files in the attempt folder:
    - Accelerometer.csv
    - Linear Accelerometer.csv
    - Gyroscope.csv
    - Gravity.csv
    - Orientation.csv
    """

    attempt_folder = Path(attempt_folder)

    # Load files
    acc = pd.read_csv(attempt_folder / "Accelerometer.csv")
    lin_acc = pd.read_csv(attempt_folder / "Linear Accelerometer.csv")
    gyro = pd.read_csv(attempt_folder / "Gyroscope.csv")
    # gravity = pd.read_csv(attempt_folder / "Gravity.csv")
    orientation = pd.read_csv(attempt_folder / "Orientation.csv")

    features = {}

    # Optional metadata
    features["attempt_id"] = attempt_id
    features["difficulty"] = difficulty
    features["topped"] = topped
    features["style"] = style

    # Duration, based on accelerometer recording
    features["duration_seconds"] = acc["Time (s)"].max() - acc["Time (s)"].min()

    # Accelerometer magnitude
    acc_mag = magnitude(
        acc,
        "X (m/s^2)",
        "Y (m/s^2)",
        "Z (m/s^2)"
    )
    features.update(summarize_series(acc_mag, "acc_mag"))

    # Linear acceleration magnitude
    lin_acc_mag = magnitude(
        lin_acc,
        "X (m/s^2)",
        "Y (m/s^2)",
        "Z (m/s^2)"
    )
    features.update(summarize_series(lin_acc_mag, "lin_acc_mag"))

    # Gyroscope magnitude
    gyro_mag = magnitude(
        gyro,
        "X (rad/s)",
        "Y (rad/s)",
        "Z (rad/s)"
    )
    features.update(summarize_series(gyro_mag, "gyro_mag"))

    # Gravity magnitude
    # gravity_mag = magnitude(
    #     gravity,
    #     "Gravity X (m/s^2)",
    #     "Gravity Y (m/s^2)",
    #     "Gravity Z (m/s^2)"
    # )
    # features.update(summarize_series(gravity_mag, "gravity_mag"))

    # Orientation summaries: mean, std, min, max
    for col in ["Yaw (°)", "Pitch (°)", "Roll (°)"]:
        clean_name = col.replace(" (°)", "").lower()
        features.update(summarize_series(orientation[col], clean_name))

    # Orientation ranges
    features["yaw_range"] = orientation["Yaw (°)"].max() - orientation["Yaw (°)"].min()
    features["pitch_range"] = orientation["Pitch (°)"].max() - orientation["Pitch (°)"].min()
    features["roll_range"] = orientation["Roll (°)"].max() - orientation["Roll (°)"].min()

    return pd.DataFrame([features])


def summarize_batch(batch_root, drop_metadata=False):
    """
    Summarise every attempt folder inside a batch directory.

    This walks only the immediate subfolders under batch_root and keeps
    folders that contain the expected sensor CSV files.
    """

    batch_root = Path(batch_root)
    summaries = []

    for attempt_folder in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        required_files = [
            attempt_folder / "Accelerometer.csv",
            attempt_folder / "Linear Accelerometer.csv",
            # attempt_folder / "Gyroscope.csv",
            attempt_folder / "Orientation.csv",
        ]

        if not all(path.exists() for path in required_files):
            continue

        summary = summarize_attempt(
            attempt_folder,
            difficulty=parse_difficulty_label(attempt_folder.name),
        )
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


def summarize_dataset(batch_root, split_by_difficulty=True):
    """
    Summarise the whole batch directory into one global feature table.

    This keeps every valid attempt folder, including folders that are
    effectively duplicates, then computes mean, std, min, and max across
    the per-attempt feature columns.
    """
    batch_features = summarize_batch(batch_root, drop_metadata=False)

    if batch_features.empty:
        return pd.DataFrame()

    if not split_by_difficulty:
        numeric_features = batch_features.select_dtypes(include=[np.number])
        global_summary = numeric_features.agg(["mean", "std", "min", "max"]).T
        global_summary.columns = [f"dataset_{stat}" for stat in global_summary.columns]
        return global_summary.reset_index().rename(columns={"index": "feature"})

    per_difficulty_summaries = []
    for difficulty, group_df in batch_features.groupby("difficulty", dropna=True):
        numeric_features = group_df.select_dtypes(include=[np.number]).drop(columns=["difficulty"], errors="ignore")
        difficulty_summary = numeric_features.agg(["mean", "std", "min", "max"]).T
        difficulty_summary.columns = [f"dataset_{stat}" for stat in difficulty_summary.columns]
        difficulty_summary = difficulty_summary.reset_index().rename(columns={"index": "feature"})
        difficulty_summary.insert(0, "difficulty", int(difficulty))
        per_difficulty_summaries.append(difficulty_summary)

    if not per_difficulty_summaries:
        return pd.DataFrame(columns=["difficulty", "feature", "dataset_mean", "dataset_std", "dataset_min", "dataset_max"])

    return pd.concat(per_difficulty_summaries, ignore_index=True)

if __name__ == "__main__":
    SPLIT_BY_DIFFICULTY = True

    dataset_summary = summarize_dataset(
        "/Users/karm1616/Desktop/Univeristy/Masters/Machine Learning for the Quanitfied Self/ML4QS-Bouldering/BOULDERING_DATA",
        split_by_difficulty=SPLIT_BY_DIFFICULTY,
    )

    print(dataset_summary.head())
    output_name = "bouldering_dataset_summary_by_difficulty.csv" if SPLIT_BY_DIFFICULTY else "bouldering_dataset_summary.csv"
    dataset_summary.to_csv(output_name, index=False)