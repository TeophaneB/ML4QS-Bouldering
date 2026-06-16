# """PyTorch TCN classifier with cross-validation.

# This module provides a compact Temporal Convolutional Network implementation
# and a helper that performs stratified cross-validation in the same spirit as
# ``classify.py``.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Sequence, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import StratifiedKFold
# from torch.utils.data import DataLoader, TensorDataset


# class Chomp1d(nn.Module):
# 	def __init__(self, chomp_size: int):
# 		super().__init__()
# 		self.chomp_size = chomp_size

# 	def forward(self, x: torch.Tensor) -> torch.Tensor:
# 		return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


# class TemporalBlock(nn.Module):
# 	def __init__(
# 		self,
# 		in_channels: int,
# 		out_channels: int,
# 		kernel_size: int,
# 		stride: int,
# 		dilation: int,
# 		padding: int,
# 		dropout: float,
# 	):
# 		super().__init__()
# 		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
# 							   stride=stride, padding=padding, dilation=dilation)
# 		self.chomp1 = Chomp1d(padding)
# 		self.relu1 = nn.ReLU()
# 		self.drop1 = nn.Dropout(dropout)

# 		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
# 							   stride=stride, padding=padding, dilation=dilation)
# 		self.chomp2 = Chomp1d(padding)
# 		self.relu2 = nn.ReLU()
# 		self.drop2 = nn.Dropout(dropout)

# 		self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
# 		self.relu = nn.ReLU()

# 		self.init_weights()

# 	def init_weights(self) -> None:
# 		for m in (self.conv1, self.conv2):
# 			nn.init.kaiming_normal_(m.weight)
# 			if m.bias is not None:
# 				nn.init.zeros_(m.bias)
# 		if self.downsample is not None:
# 			nn.init.kaiming_normal_(self.downsample.weight)
# 			if self.downsample.bias is not None:
# 				nn.init.zeros_(self.downsample.bias)

# 	def forward(self, x: torch.Tensor) -> torch.Tensor:
# 		out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
# 		out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
# 		res = x if self.downsample is None else self.downsample(x)
# 		return self.relu(out + res)


# class TCN(nn.Module):
# 	def __init__(
# 		self,
# 		input_size: int,
# 		num_classes: int,
# 		num_channels: Sequence[int] = (32, 32, 32),
# 		kernel_size: int = 3,
# 		dropout: float = 0.2,
# 	):
# 		super().__init__()
# 		layers = []
# 		for i, out_channels in enumerate(num_channels):
# 			in_channels = input_size if i == 0 else num_channels[i - 1]
# 			dilation = 2 ** i
# 			padding = (kernel_size - 1) * dilation
# 			layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, padding, dropout))
# 		self.network = nn.Sequential(*layers)
# 		self.classifier = nn.Linear(num_channels[-1], num_classes)

# 	def forward(self, x: torch.Tensor) -> torch.Tensor:
# 		x = x.transpose(1, 2)
# 		y = self.network(x)
# 		y = y[:, :, -1]
# 		return self.classifier(y)


# @dataclass
# class CVResult:
# 	fold_accuracies: list[float]
# 	fold_f1_macro: list[float]

# 	@property
# 	def accuracy_mean(self) -> float:
# 		return float(np.mean(self.fold_accuracies)) if self.fold_accuracies else 0.0

# 	@property
# 	def f1_macro_mean(self) -> float:
# 		return float(np.mean(self.fold_f1_macro)) if self.fold_f1_macro else 0.0


# def _to_sequence_array(X: np.ndarray) -> np.ndarray:
# 	X = np.asarray(X, dtype=np.float32)
# 	if X.ndim == 2:
# 		return X[:, :, None]
# 	if X.ndim != 3:
# 		raise ValueError("X must be a 2D or 3D array")
# 	return X


# def _train_one_fold(
# 	model: nn.Module,
# 	train_loader: DataLoader,
# 	val_X: torch.Tensor,
# 	val_y: torch.Tensor,
# 	device: torch.device,
# 	epochs: int,
# 	lr: float,
# ) -> Tuple[float, float]:
# 	criterion = nn.CrossEntropyLoss()
# 	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 	model.to(device)

# 	for _ in range(epochs):
# 		model.train()
# 		for xb, yb in train_loader:
# 			xb = xb.to(device)
# 			yb = yb.to(device)
# 			optimizer.zero_grad()
# 			loss = criterion(model(xb), yb)
# 			loss.backward()
# 			optimizer.step()

# 	model.eval()
# 	with torch.no_grad():
# 		preds = torch.argmax(model(val_X.to(device)), dim=1).cpu().numpy()
# 	y_true = val_y.cpu().numpy()
# 	return accuracy_score(y_true, preds), f1_score(y_true, preds, average="macro")


# def run_tcn_cv(
# 	X: np.ndarray,
# 	y: np.ndarray,
# 	n_splits: int = 5,
# 	epochs: int = 20,
# 	batch_size: int = 64,
# 	lr: float = 1e-3,
# 	num_channels: Sequence[int] = (32, 32, 32),
# 	kernel_size: int = 3,
# 	dropout: float = 0.2,
# 	random_state: int = 42,
# ) -> CVResult:
# 	X = _to_sequence_array(X)
# 	y = np.asarray(y, dtype=np.int64)
# 	n_classes = int(np.unique(y).size)
# 	input_size = int(X.shape[-1])

# 	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	fold_accuracies: list[float] = []
# 	fold_f1_macro: list[float] = []

# 	for train_idx, val_idx in skf.split(X, y):
# 		X_train, X_val = X[train_idx], X[val_idx]
# 		y_train, y_val = y[train_idx], y[val_idx]

# 		train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
# 		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# 		val_X = torch.from_numpy(X_val)
# 		val_y = torch.from_numpy(y_val)

# 		model = TCN(input_size=input_size, num_classes=n_classes, num_channels=num_channels,
# 					kernel_size=kernel_size, dropout=dropout)
# 		acc, f1m = _train_one_fold(model, train_loader, val_X, val_y, device, epochs, lr)
# 		fold_accuracies.append(acc)
# 		fold_f1_macro.append(f1m)

# 	return CVResult(fold_accuracies=fold_accuracies, fold_f1_macro=fold_f1_macro)


# if __name__ == "__main__":
# 	X = np.random.randn(200, 50, 8).astype(np.float32)
# 	y = np.random.randint(0, 3, size=200)
# 	result = run_tcn_cv(X, y)
# 	print({"accuracy_mean": result.accuracy_mean, "f1_macro_mean": result.f1_macro_mean})
