from pathlib import Path
import re

import pandas as pd


ATTEMPT_LABEL_PATTERN = re.compile(
	r"^L\d+\s+(?P<style>N|O|S|D)\s+(?P<topped>Y|No)\b",
	re.IGNORECASE,
)


def parse_difficulty_label(folder_name):
	"""Convert the attempt folder name into a simple difficulty label."""
	first_token = folder_name.split()[0].upper() if folder_name.split() else ""

	if first_token in {"L1", "L2"}:
		return 1
	if first_token in {"L3", "L4"}:
		return 2
	if first_token in {"L5", "L6"}:
		return 3

	return None

def _parse_attempt_labels(folder_name):
	"""Extract the style and topped tokens from an attempt folder name."""
	match = ATTEMPT_LABEL_PATTERN.match(Path(folder_name).name)
	if not match:
		return None, None

	style_token = match.group("style").upper()
	topped_token = match.group("topped").upper()

	style_map = {"N": 0, "O": 1, "S": 2, "D": 3}
	topped_map = {"Y": 1, "NO": 0}

	return style_map.get(style_token), topped_map.get(topped_token)


def parse_style_label(folder_name):
	"""Convert the attempt folder name into a simple style label."""
	style_label, _ = _parse_attempt_labels(folder_name)
	return style_label


def parse_topped_label(folder_name):
	"""Convert the attempt folder name into a simple topped label."""
	_, topped_label = _parse_attempt_labels(folder_name)
	return topped_label


def _read_and_rename(csv_path, rename_map):
	df = pd.read_csv(csv_path)
	df = df.rename(columns=rename_map)
	df = df[[column for column in rename_map.values() if column in df.columns]]
	return df


def merge_attempt_folder(attempt_folder, attempt_id=None, difficulty_label=None, style_label=None, topped_label=None, tolerance_seconds=0.02):
	"""Merge all sensor CSV files from one attempt folder into one dataframe.

	The accelerometer timestamps are used as the base time axis. The other sensor
	streams are aligned to that axis with a nearest-time merge.
	"""
	attempt_folder = Path(attempt_folder)
	if attempt_id is None:
		attempt_id = attempt_folder.name
	if difficulty_label is None:
		difficulty_label = parse_difficulty_label(attempt_folder.name)
	if style_label is None:
		style_label = parse_style_label(attempt_folder.name)
	if topped_label is None:
		topped_label = parse_topped_label(attempt_folder.name)
	required_files = {
		"Accelerometer.csv",
		"Linear Accelerometer.csv",
		"Gyroscope.csv",
		"Orientation.csv",
	}
	missing_files = [name for name in required_files if not (attempt_folder / name).exists()]
	if missing_files:
		raise FileNotFoundError(f"Missing files in {attempt_folder}: {missing_files}")

	acc = _read_and_rename(
		attempt_folder / "Accelerometer.csv",
		{
			"Time (s)": "time_seconds",
			"X (m/s^2)": "acc_x",
			"Y (m/s^2)": "acc_y",
			"Z (m/s^2)": "acc_z",
		},
	)
	lin_acc = _read_and_rename(
		attempt_folder / "Linear Accelerometer.csv",
		{
			"Time (s)": "time_seconds",
			"X (m/s^2)": "lin_acc_x",
			"Y (m/s^2)": "lin_acc_y",
			"Z (m/s^2)": "lin_acc_z",
		},
	)
	gyro = _read_and_rename(
		attempt_folder / "Gyroscope.csv",
		{
			"Time (s)": "time_seconds",
			"X (rad/s)": "gyro_x",
			"Y (rad/s)": "gyro_y",
			"Z (rad/s)": "gyro_z",
		},
	)
	orientation = _read_and_rename(
		attempt_folder / "Orientation.csv",
		{
			"Time (s)": "time_seconds",
			"Yaw (°)": "yaw",
			"Pitch (°)": "pitch",
			"Roll (°)": "roll",
		},
	)

	# Sort each stream by time so merge_asof can align them.
	acc = acc.sort_values("time_seconds")
	lin_acc = lin_acc.sort_values("time_seconds")
	gyro = gyro.sort_values("time_seconds")
	orientation = orientation.sort_values("time_seconds")

	merged = pd.merge_asof(
		acc,
		lin_acc,
		on="time_seconds",
		direction="nearest",
		tolerance=tolerance_seconds,
	)
	merged = pd.merge_asof(
		merged,
		gyro,
		on="time_seconds",
		direction="nearest",
		tolerance=tolerance_seconds,
	)
	merged = pd.merge_asof(
		merged,
		orientation,
		on="time_seconds",
		direction="nearest",
		tolerance=tolerance_seconds,
	)

	merged["attempt_id"] = attempt_id
	merged["difficulty_label"] = difficulty_label
	merged["style_label"] = style_label
	merged["topped_label"] = topped_label
	merged["duration_seconds"] = merged["time_seconds"].max() - merged["time_seconds"].min()

	return merged


def get_valid_attempt_folders(data_root):
	"""Return the attempt folders that contain the required sensor files."""
	data_root = Path(data_root)
	valid_folders = []

	for attempt_folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
		required_files = {
			"Accelerometer.csv",
			"Linear Accelerometer.csv",
			"Gyroscope.csv",
			"Orientation.csv",
		}
		if all((attempt_folder / name).exists() for name in required_files):
			valid_folders.append(attempt_folder)

	return valid_folders


def build_concatenated_dataframe(data_root, tolerance_seconds=0.02):
	"""Load every valid attempt folder and concatenate them into one dataframe.

	Returns:
		full_df, id_mapping_df
	"""
	data_root = Path(data_root)
	all_attempts = []
	mapping_rows = []

	for numeric_attempt_id, attempt_folder in enumerate(get_valid_attempt_folders(data_root), start=0):
		try:
			attempt_df = merge_attempt_folder(
				attempt_folder,
				attempt_id=numeric_attempt_id,
				tolerance_seconds=tolerance_seconds,
			)
		except FileNotFoundError:
			continue

		all_attempts.append(attempt_df)
		mapping_rows.append(
			{
				"attempt_id": numeric_attempt_id,
				"folder_name": attempt_folder.name,
			}
		)

	if not all_attempts:
		return pd.DataFrame(), pd.DataFrame(columns=["attempt_id", "folder_name"])

	full_df = pd.concat(all_attempts, ignore_index=True)
	id_mapping_df = pd.DataFrame(mapping_rows)

	return full_df, id_mapping_df


if __name__ == "__main__":
	# This script creates one long dataframe where every row is one timestamp sample.
	# That is the format expected by tcn_inputdata.py.
	data_root = Path("../BOULDERING_DATA")
	output_path = Path("tcn_raw_concatenated.csv")
	mapping_path = Path("TCNattempt_id_mapping.csv")

	full_df, id_mapping_df = build_concatenated_dataframe(data_root)
	print("Combined dataframe shape:", full_df.shape)
	full_df.to_csv(output_path, index=False)
	print(f"Saved concatenated raw data to {output_path}")
	id_mapping_df.to_csv(mapping_path, index=False)
	print(f"Saved attempt ID mapping to {mapping_path}")