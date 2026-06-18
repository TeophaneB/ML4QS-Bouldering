from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "FEATURES"
RESULTS_DIR = BASE_DIR / "results"


CLASSIFIERS = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

SELECTORS = {
    "Variance Threshold": VarianceThreshold(threshold=0.0),
    "ANOVA": SelectKBest(score_func=f_classif),
    "PCA": PCA(),
}


def _load_best_row() -> pd.Series:
    candidate_files = sorted(RESULTS_DIR.glob("best_model_found_window*.csv"))
    if not candidate_files:
        candidate_files = [RESULTS_DIR / "best_model_per_window.csv"]

    rows = []
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path)
        if not df.empty:
            rows.append(df)

    if not rows:
        raise FileNotFoundError("No best-model results were found in the results folder")

    combined = pd.concat(rows, ignore_index=True)
    if "mean_f1_macro" in combined.columns:
        best_index = combined["mean_f1_macro"].astype(float).idxmax()
        return combined.loc[best_index]

    return combined.iloc[0]


def _build_pipeline(best_row: pd.Series) -> Pipeline:
    classifier_name = str(best_row["classifier"])
    selector_name = str(best_row["selector"])

    if classifier_name not in CLASSIFIERS:
        raise KeyError(f"Unsupported classifier: {classifier_name}")
    if selector_name not in SELECTORS:
        raise KeyError(f"Unsupported selector: {selector_name}")

    classifier = CLASSIFIERS[classifier_name]
    selector = SELECTORS[selector_name]
    best_parameters = ast.literal_eval(str(best_row["best_parameters"]))

    pipeline = Pipeline(
        [
            ("base_cleaning_vt", VarianceThreshold(threshold=0.0)),
            ("feature_selector", selector),
            ("standardscaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    pipeline.set_params(**best_parameters)
    return pipeline


def save_confusion_matrix(best_row: pd.Series) -> Path:
    window_size = int(best_row["window_size"])
    source_file = Path(str(best_row["source_file"]))
    feature_file = FEATURES_DIR / f"bouldering_summary_{window_size}.csv"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df = pd.read_csv(feature_file)
    if "difficulty" not in df.columns:
        raise KeyError("The feature file does not contain a difficulty column")

    X = df.drop(columns=["difficulty"])
    y = df["difficulty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = _build_pipeline(best_row)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    labels = sorted(pd.unique(y))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"confusion_matrix_window_{window_size}_"
        f"{str(best_row['classifier']).replace(' ', '_')}_"
        f"{str(best_row['selector']).replace(' ', '_')}.png"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(label) for label in labels],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(
        f"Confusion Matrix - Window {window_size}\n{best_row['classifier']} + {best_row['selector']}"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix to {output_path}")
    print(f"Best model source file: {source_file}")
    print(f"Final macro precision: {precision_macro:.3f}")
    print(f"Final macro recall: {recall_macro:.3f}")
    return output_path


def main() -> None:
    best_row = _load_best_row()
    save_confusion_matrix(best_row)


if __name__ == "__main__":
    main()