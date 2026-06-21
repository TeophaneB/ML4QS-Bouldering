import pandas as pd
import numpy as np
from pathlib import Path


def parse_difficulty_label(folder_name):
    """Map folder difficulty token (L1-L6) to 1-3 buckets."""
    tokens = Path(folder_name).name.split()
    first_token = tokens[0].upper() if tokens else ""

    if first_token in {"L1", "L2"}:
        return 1
    if first_token in {"L3", "L4"}:
        return 2
    if first_token in {"L5", "L6"}:
        return 3

    return None


def magnitude(df, x_col, y_col, z_col):
    """Compute vector magnitude: sqrt(x^2 + y^2 + z^2)."""
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)


def summarize_series(series, prefix):
    """Return mean, standard deviation, min, and max for one signal within one attempt."""
    return {
        f"{prefix}_mean": series.mean(),
        f"{prefix}_std": series.std(),
        f"{prefix}_min": series.min(),
        f"{prefix}_max": series.max(),
    }


def summarize_attempt(attempt_folder, attempt_id=None, difficulty=None, topped=None, style=None):
    """
    Summarise one bouldering attempt into one row.

    Expected files:
    - Accelerometer.csv
    - Linear Accelerometer.csv
    - Gyroscope.csv
    - Orientation.csv
    """

    attempt_folder = Path(attempt_folder)

    acc = pd.read_csv(attempt_folder / "Accelerometer.csv")
    lin_acc = pd.read_csv(attempt_folder / "Linear Accelerometer.csv")
    gyro = pd.read_csv(attempt_folder / "Gyroscope.csv")
    orientation = pd.read_csv(attempt_folder / "Orientation.csv")

    features = {}

    features["attempt_id"] = attempt_id
    features["difficulty"] = difficulty
    features["topped"] = topped
    features["style"] = style

    features["duration_seconds"] = acc["Time (s)"].max() - acc["Time (s)"].min()

    acc_mag = magnitude(acc, "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)")
    features.update(summarize_series(acc_mag, "acc_mag"))

    lin_acc_mag = magnitude(lin_acc, "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)")
    features.update(summarize_series(lin_acc_mag, "lin_acc_mag"))

    gyro_mag = magnitude(gyro, "X (rad/s)", "Y (rad/s)", "Z (rad/s)")
    features.update(summarize_series(gyro_mag, "gyro_mag"))

    for col in ["Yaw (°)", "Pitch (°)", "Roll (°)"]:
        clean_name = col.replace(" (°)", "").lower()
        features.update(summarize_series(orientation[col], clean_name))

    features["yaw_range"] = orientation["Yaw (°)"].max() - orientation["Yaw (°)"].min()
    features["pitch_range"] = orientation["Pitch (°)"].max() - orientation["Pitch (°)"].min()
    features["roll_range"] = orientation["Roll (°)"].max() - orientation["Roll (°)"].min()

    return pd.DataFrame([features])


def summarize_batch(batch_root, drop_metadata=False):
    """
    Summarise every valid attempt folder inside a batch directory.
    """

    batch_root = Path(batch_root)
    summaries = []

    for attempt_folder in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        required_files = [
            attempt_folder / "Accelerometer.csv",
            attempt_folder / "Linear Accelerometer.csv",
            attempt_folder / "Gyroscope.csv",
            attempt_folder / "Orientation.csv",
        ]

        if not all(path.exists() for path in required_files):
            continue

        summary = summarize_attempt(
            attempt_folder,
            attempt_id=attempt_folder.name,
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


def summarize_report_by_difficulty(batch_root):
    """
    Create the exact summary needed for the report table.

    This avoids confusing second-level summaries such as:
    mean of std, std of std, min of std, max of std.

    Instead, it computes meaningful values directly per difficulty class.
    """

    batch_features = summarize_batch(batch_root, drop_metadata=False)

    if batch_features.empty:
        return pd.DataFrame()

    difficulty_names = {
        1: "Easy",
        2: "Medium",
        3: "Hard",
    }

    rows = []

    for difficulty, group_df in batch_features.groupby("difficulty", dropna=True):
        difficulty = int(difficulty)

        row = {
            "Difficulty": difficulty_names.get(difficulty, str(difficulty)),
            "Mean Duration": group_df["duration_seconds"].mean(),
            "Max Duration": group_df["duration_seconds"].max(),

            "Mean Acceleration Mag.": group_df["acc_mag_mean"].mean(),
            "Max Acceleration Mag.": group_df["acc_mag_max"].max(),

            "Mean Gyroscope Mag.": group_df["gyro_mag_mean"].mean(),
            "Max Gyroscope Mag.": group_df["gyro_mag_max"].max(),

            "Mean Linear Accel. Mag.": group_df["lin_acc_mag_mean"].mean(),
            "Max Linear Accel. Mag.": group_df["lin_acc_mag_max"].max(),
        }

        rows.append(row)

    report_summary = pd.DataFrame(rows)

    difficulty_order = ["Easy", "Medium", "Hard"]
    report_summary["Difficulty"] = pd.Categorical(
        report_summary["Difficulty"],
        categories=difficulty_order,
        ordered=True,
    )

    report_summary = report_summary.sort_values("Difficulty").reset_index(drop=True)

    return report_summary


def make_latex_difficulty_table(report_summary):
    """
    Convert the difficulty summary into the LaTeX table format:
    rows = sensor features
    columns = Easy, Medium, Hard
    """

    table_df = report_summary.set_index("Difficulty").T
    table_df = table_df[["Easy", "Medium", "Hard"]]

    latex = table_df.to_latex(
        float_format="%.1f",
        column_format="lrrr",
        escape=False,
    )

    return latex


if __name__ == "__main__":
    batch_root = "/Users/karm1616/Desktop/Univeristy/Masters/Machine Learning for the Quanitfied Self/ML4QS-Bouldering/BOULDERING_DATA"

    report_summary = summarize_report_by_difficulty(batch_root)

    print("\nReport summary:")
    print(report_summary)

    report_summary.to_csv("bouldering_report_summary_by_difficulty.csv", index=False)

    latex_table = make_latex_difficulty_table(report_summary)

    print("\nLaTeX table body:")
    print(latex_table)

    with open("bouldering_report_summary_by_difficulty.tex", "w") as f:
        f.write(latex_table)