"""Build TCN-ready bouldering attempt data in memory and train a TCN classifier.

This script keeps the full-attempt preprocessing and the model training in one
place so the generated X and y arrays can be used immediately by the model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tcn import TCN
from sklearn.preprocessing import StandardScaler
N_STEPS = 500  # Number of time steps to resample each attempt to.

def _ensure_columns(df, required_columns, context="dataframe"):
	missing_columns = [column for column in required_columns if column not in df.columns]
	if missing_columns:
		raise KeyError(f"Missing columns in {context}: {missing_columns}")


def add_angle_sin_cos_features(df, angle_cols=("yaw", "pitch", "roll")):
	"""Create sine/cosine versions of angle columns and return the updated dataframe."""
	df = df.copy()
	angle_feature_cols = []

	for angle_col in angle_cols:
		if angle_col not in df.columns:
			raise KeyError(f"Angle column not found: {angle_col}")

		radians = np.deg2rad(df[angle_col].astype(float))
		sin_col = f"sin_{angle_col}"
		cos_col = f"cos_{angle_col}"

		df[sin_col] = np.sin(radians)
		df[cos_col] = np.cos(radians)
		angle_feature_cols.extend([sin_col, cos_col])

	return df, angle_feature_cols


def _fill_missing_sensor_values(df_attempt, sensor_cols):
	"""Fill missing sensor values inside one attempt using linear interpolation."""
	df_attempt = df_attempt.copy()

	if df_attempt[sensor_cols].isna().any().any():
		df_attempt[sensor_cols] = df_attempt[sensor_cols].interpolate(
			method="linear",
			limit_direction="both",
		)
		df_attempt[sensor_cols] = df_attempt[sensor_cols].ffill().bfill()

	return df_attempt


def standardize_sequence_data(X_train, X_val, X_test):
	"""Standardize 3D sequence data using statistics from the training split only."""
	scaler = StandardScaler()
	n_samples, n_steps, n_channels = X_train.shape

	X_train_reshaped = X_train.reshape(-1, n_channels)
	X_val_reshaped = X_val.reshape(-1, n_channels)
	X_test_reshaped = X_test.reshape(-1, n_channels)

	scaler.fit(X_train_reshaped)

	X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_steps, n_channels)
	X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape[0], n_steps, n_channels)
	X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], n_steps, n_channels)

	return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def standardize_tabular_data(X_train, X_val, X_test):
	"""Standardize 2D metadata features using training-set statistics only."""
	scaler = StandardScaler()

	scaler.fit(X_train)

	X_train_scaled = scaler.transform(X_train)
	X_val_scaled = scaler.transform(X_val)
	X_test_scaled = scaler.transform(X_test)

	return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def resample_attempt_to_fixed_steps(df_attempt, time_col, sensor_cols, n_steps=N_STEPS):
	"""Resample one complete attempt to exactly n_steps evenly spaced time points."""
	_ensure_columns(df_attempt, [time_col] + list(sensor_cols), context="attempt dataframe")

	if len(df_attempt) < 2:
		raise ValueError("Attempt has fewer than 2 rows")

	attempt = df_attempt.sort_values(time_col).copy()
	attempt = attempt.dropna(subset=[time_col])

	if len(attempt) < 2:
		raise ValueError("Attempt has fewer than 2 valid time rows")

	attempt = attempt[[time_col] + list(sensor_cols)].copy()
	attempt = _fill_missing_sensor_values(attempt, sensor_cols)

	if attempt[sensor_cols].isna().any().any():
		raise ValueError("Attempt still contains missing sensor values after interpolation")

	# Make time start at 0 for every attempt so all climbs are aligned from start to end.
	attempt[time_col] = attempt[time_col] - attempt[time_col].iloc[0]
	attempt = attempt.drop_duplicates(subset=time_col, keep="last")
	attempt = attempt.sort_values(time_col)

	if len(attempt) < 2:
		raise ValueError("Attempt does not have enough unique time points after cleanup")

	time_values = attempt[time_col].to_numpy(dtype=float)
	sensor_values = attempt[sensor_cols].to_numpy(dtype=float)
	duration_seconds = float(time_values[-1])

	# Recreate the full attempt on a fixed, normalized time grid.
	new_time = np.linspace(0.0, duration_seconds, n_steps)

	if duration_seconds == 0.0:
		# If the attempt has zero duration after cleanup, repeat the first sample.
		return np.repeat(sensor_values[:1], n_steps, axis=0)

	resampled = np.empty((n_steps, len(sensor_cols)), dtype=float)
	for channel_index, _ in enumerate(sensor_cols):
		resampled[:, channel_index] = np.interp(
			new_time,
			time_values,
			sensor_values[:, channel_index],
		)

	return resampled


def build_tcn_dataset(
	full_df,
	attempt_id_col,
	time_col,
	sensor_cols,
	label_col,
	metadata_cols=None,
	n_steps=N_STEPS,
):
	"""Build fixed-length TCN input arrays from a dataframe with all attempts."""
	_ensure_columns(
		full_df,
		[attempt_id_col, time_col, label_col] + list(sensor_cols),
		context="full dataframe",
	)
	metadata_cols = list(metadata_cols or [])

	X = []
	metadata = []
	y = []
	durations = []

	for attempt_id, df_attempt in full_df.groupby(attempt_id_col, sort=False):
		df_attempt = df_attempt.sort_values(time_col).copy()

		if len(df_attempt) < 2:
			print(f"Skipping attempt {attempt_id}: fewer than 2 rows")
			continue

		labels = df_attempt[label_col].dropna().unique()
		if len(labels) == 0:
			print(f"Skipping attempt {attempt_id}: missing label")
			continue
		if len(labels) > 1:
			print(f"Skipping attempt {attempt_id}: multiple labels found {labels}")
			continue

		try:
			resampled_attempt = resample_attempt_to_fixed_steps(
				df_attempt=df_attempt,
				time_col=time_col,
				sensor_cols=sensor_cols,
				n_steps=n_steps,
			)
		except ValueError as exc:
			print(f"Skipping attempt {attempt_id}: {exc}")
			continue

		metadata_row = []
		for metadata_col in metadata_cols:
			if metadata_col not in df_attempt.columns:
				raise KeyError(f"Missing metadata column in attempt {attempt_id}: {metadata_col}")

			metadata_values = df_attempt[metadata_col].dropna().unique()
			if len(metadata_values) == 0:
				print(f"Skipping attempt {attempt_id}: missing metadata value for {metadata_col}")
				metadata_row = None
				break

			metadata_row.append(metadata_values[0])

		if metadata_row is None:
			continue

		duration_seconds = float(df_attempt[time_col].max() - df_attempt[time_col].min())

		X.append(resampled_attempt)
		metadata.append(metadata_row)
		y.append(labels[0])
		durations.append(duration_seconds)

	if not X:
		return (
			np.empty((0, n_steps, len(sensor_cols)), dtype=float),
			np.empty((0, len(metadata_cols)), dtype=float),
			np.asarray(y),
			np.asarray(durations, dtype=float),
		)

	X = np.asarray(X, dtype=float)
	metadata = np.asarray(metadata, dtype=float) if metadata_cols else np.empty((len(X), 0), dtype=float)
	y = np.asarray(y)
	durations = np.asarray(durations, dtype=float)

	return X, metadata, y, durations

#function to standardize X_train and X_test
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    n_samples, n_steps, n_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_channels)
    X_test_reshaped = X_test.reshape(-1, n_channels)

    scaler.fit(X_train_reshaped)

    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_steps, n_channels)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], n_steps, n_channels)

    return X_train_scaled, X_test_scaled

class TCNClassifier(nn.Module):
	def __init__(self, num_inputs, metadata_dim, num_classes):
		super().__init__()
		self.tcn = TCN(
			num_inputs=num_inputs,
			num_channels=[16,32], ## ORIGINAL IS 16,32
			kernel_size=3,
			dropout=0.2,
			causal=False,
			use_norm="weight_norm",
			activation="relu",
			kernel_initializer="xavier_uniform",
			input_shape="NLC",

		)
		self.metadata_encoder = None
		pooled_dim = 64
		if metadata_dim > 0:
			self.metadata_encoder = nn.Sequential(
				nn.Linear(metadata_dim, 16),
				nn.ReLU(),
				nn.Dropout(0.2),
		)
			pooled_dim += 16
		self.classifier = nn.Sequential(
			nn.Linear(pooled_dim, 32),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(32, num_classes),
		)

	def forward(self, x, metadata=None):
		# causal=False is appropriate here because this is full-attempt classification,
		# so the model is allowed to use the entire sequence.
		out = self.tcn(x)

		# Global average and max pooling summarize the whole attempt while keeping the
		# classifier small and making it less sensitive to exact temporal alignment.
		avg_pool = out.mean(dim=1)
		max_pool = out.max(dim=1).values
		pooled = torch.cat([avg_pool, max_pool], dim=1)

		if self.metadata_encoder is not None:
			if metadata is None:
				raise ValueError("Metadata tensor is required when metadata features are enabled")
			metadata_features = self.metadata_encoder(metadata)
			pooled = torch.cat([pooled, metadata_features], dim=1)

		# Return raw logits. CrossEntropyLoss expects integer labels, not one-hot labels,
		# and applies the softmax internally.
		return self.classifier(pooled)


def train_tcn_classifier(X, metadata, y):
	"""Split the data, train the TCN, and evaluate on the held-out test set."""
	X = np.asarray(X)
	metadata = np.asarray(metadata)
	y = np.asarray(y)
	if metadata.ndim == 1:
		metadata = metadata.reshape(-1, 1)

	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)
	class_names = label_encoder.classes_.astype(str)
	num_classes = len(class_names)

	# The test set is held out completely until the end so it is not used for
	# early stopping or model selection.
	X_train_val, X_test, metadata_train_val, metadata_test, y_train_val, y_test = train_test_split(
		X,
		metadata,
		y_encoded,
		test_size=0.2,
		stratify=y_encoded,
		random_state=42,
	)

	X_train, X_val, metadata_train, metadata_val, y_train, y_val = train_test_split(
		X_train_val,
		metadata_train_val,
		y_train_val,
		test_size=0.2,
		stratify=y_train_val,
		random_state=42,
	)

	X_train, X_val, X_test, _ = standardize_sequence_data(X_train, X_val, X_test)
	if metadata_train.shape[1] > 0:
		metadata_train, metadata_val, metadata_test, _ = standardize_tabular_data(
			metadata_train,
			metadata_val,
			metadata_test,
		)
	else:
		metadata_train = np.empty((X_train.shape[0], 0), dtype=float)
		metadata_val = np.empty((X_val.shape[0], 0), dtype=float)
		metadata_test = np.empty((X_test.shape[0], 0), dtype=float)

	# Keep X in NLC format (batch, time, channels) because the TCN is configured
	# with input_shape="NLC", so no manual transpose is needed.
	X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
	metadata_train_tensor = torch.tensor(metadata_train, dtype=torch.float32)
	metadata_val_tensor = torch.tensor(metadata_val, dtype=torch.float32)
	metadata_test_tensor = torch.tensor(metadata_test, dtype=torch.float32)

	y_train_tensor = torch.tensor(y_train, dtype=torch.long)
	y_val_tensor = torch.tensor(y_val, dtype=torch.long)
	y_test_tensor = torch.tensor(y_test, dtype=torch.long)

	train_dataset = TensorDataset(X_train_tensor, metadata_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, metadata_val_tensor, y_val_tensor)
	test_dataset = TensorDataset(X_test_tensor, metadata_test_tensor, y_test_tensor)

	train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = TCNClassifier(num_inputs=X.shape[2], metadata_dim=metadata.shape[1], num_classes=num_classes).to(device)
	criterion = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 1.0], device=device))  # Adjust weights if needed
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	max_epochs = 100
	patience = 10

	def evaluate(loader):
		model.eval()
		total_loss = 0.0
		all_preds = []
		all_targets = []

		with torch.no_grad():
			for xb, mb, yb in loader:
				xb = xb.to(device)
				mb = mb.to(device)
				yb = yb.to(device)
				logits = model(xb, mb)
				loss = criterion(logits, yb)
				total_loss += loss.item() * xb.size(0)
				all_preds.append(logits.argmax(dim=1).cpu())
				all_targets.append(yb.cpu())

		all_preds = torch.cat(all_preds).numpy()
		all_targets = torch.cat(all_targets).numpy()
		avg_loss = total_loss / len(loader.dataset)
		macro_f1 = f1_score(all_targets, all_preds, average="macro")
		accuracy = accuracy_score(all_targets, all_preds)
		return avg_loss, macro_f1, accuracy, all_targets, all_preds

	train_losses = []
	val_losses = []
	val_macro_f1s = []

	best_val_macro_f1 = -1.0
	best_model_state = None
	epochs_without_improvement = 0

	for epoch in range(max_epochs):
		model.train()
		running_loss = 0.0

		for xb, mb, yb in train_loader:
			xb = xb.to(device)
			mb = mb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad()
			logits = model(xb, mb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * xb.size(0)

		train_loss = running_loss / len(train_loader.dataset)
		val_loss, val_macro_f1, _, _, _ = evaluate(val_loader)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		val_macro_f1s.append(val_macro_f1)

		print(
			f"Epoch {epoch + 1:03d} | "
			f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macro_f1={val_macro_f1:.4f}"
		)

		if val_macro_f1 > best_val_macro_f1:
			best_val_macro_f1 = val_macro_f1
			best_model_state = {
				key: value.detach().cpu().clone()
				for key, value in model.state_dict().items()
			}
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				print(f"Early stopping after {epoch + 1} epochs.")
				break

	model.load_state_dict(best_model_state)
	model.to(device)
	test_loss, test_macro_f1, test_accuracy, y_test_true, y_test_pred = evaluate(test_loader)

	print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
	print(f"Test accuracy: {test_accuracy:.4f}")
	print(f"Test macro F1: {test_macro_f1:.4f}")
	print("Classification report:")
	print(classification_report(y_test_true, y_test_pred, target_names=class_names, zero_division=0))

	return {
		"model": model,
		"label_encoder": label_encoder,
		"best_val_macro_f1": best_val_macro_f1,
		"test_accuracy": test_accuracy,
		"test_macro_f1": test_macro_f1,
		"classification_report": classification_report(
			y_test_true,
			y_test_pred,
			target_names=class_names,
			zero_division=0,
		),
		"train_losses": train_losses,
		"val_losses": val_losses,
		"val_macro_f1s": val_macro_f1s,
	}


if __name__ == "__main__":
	# Replace this with your dataframe if it is already in memory.
	# The important part is that X and y are created here, kept in memory, and then
	# passed straight into the TCN model training function.
	full_df = pd.read_csv("tcn_raw_concatenated.csv")

	# Angle features are converted to sin/cos before resampling so circular values
	# are represented in a model-friendly way.
	full_df, angle_sensor_cols = add_angle_sin_cos_features(
		full_df,
		angle_cols=("yaw", "pitch", "roll"),
	)

	sensor_cols = [
		"acc_x", "acc_y", "acc_z",
		"lin_acc_x", "lin_acc_y", "lin_acc_z",
		"gyro_x", "gyro_y", "gyro_z",
	] + angle_sensor_cols
	metadata_cols = [
		column for column in ["duration_seconds", "style_label", "topped_label"]
		if column in full_df.columns
	]

	X, metadata, y, durations = build_tcn_dataset(
		full_df=full_df,
		attempt_id_col="attempt_id",
		time_col="time_seconds",
		sensor_cols=sensor_cols,
		label_col="difficulty_label",
		metadata_cols=metadata_cols,
		n_steps=N_STEPS,
	)

	print("X shape:", X.shape)
	print("y shape:", y.shape)
	print("durations shape:", durations.shape)

	train_tcn_classifier(X, metadata, y)