"""
5. Define an appropriate train/test setup and apply classical machine learning techniques
(cf. Chapter 7) on the resulting dataset from (4). Describe your rationale, the results,
and how you optimized the hyperparameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# --- Classifiers ---
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=3)
classifier_grid_knn = (
    'classifier__n_neighbors', [1, 3, 5, 10, 20, 50],
    'classifier__weights', ['uniform', 'distance']
)

from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(max_iter=5000)
classifier_grid_logistic = ('classifier__C', [0.001, 0.01, 0.1, 0.5, 1, 10])

from sklearn.svm import SVC
classifier_svm = SVC()
classifier_grid_svm = (
    'classifier__C', [0.001, 0.01, 0.1, 0.5, 1, 10], 
    'classifier__kernel', ['linear', 'rbf']
)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier()
classifier_grid_rf = (
    'classifier__n_estimators', [1, 3, 5, 10, 20, 50],
    'classifier__max_depth', [None, 5, 10, 20]
)

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_grid_dt = ('classifier__max_depth', [None, 5, 10, 20])

Classifiers = [
 (classifier_knn, classifier_grid_knn, "KNN"),
 (classifier_logistic, classifier_grid_logistic, "Logistic Regression"),
 (classifier_svm, classifier_grid_svm, "SVM"),
 (classifier_rf, classifier_grid_rf, "Random Forest"), 
 (classifier_dt, classifier_grid_dt, "Decision Tree"), 
]


# --- Selectors ---
from sklearn.feature_selection import VarianceThreshold
selector_variance = VarianceThreshold(threshold=0.01)
selector_grid_variance = ('feature_selector__threshold', [0.001, 0.01, 0.05, 0.1])

from sklearn.feature_selection import SelectKBest, f_classif
selector_anova = SelectKBest(score_func=f_classif, k=10)
selector_grid_anova = ('feature_selector__k', [5, 10, 25, 50, 100]) 

from sklearn.decomposition import PCA
selector_pca = PCA(n_components=10)
selector_grid_pca = ('feature_selector__n_components', [5, 10, 25, 50, 100])

Selectors = [
    (selector_variance, selector_grid_variance, "Variance Threshold"),
    (selector_pca, selector_grid_pca, "PCA"),
    (selector_anova, selector_grid_anova, "ANOVA")
]

INNER_SPLITS = 5
OUTER_SPLITS = 3
REPEATS = 1

FEATURES_FOLDER_PATH = Path("FEATURES")

def classify_per_window_size():
    """ train a classifier to predict difficulty """
    if not FEATURES_FOLDER_PATH.exists():
        print(f"Error: Path {FEATURES_FOLDER_PATH} does not exist.")
        return

    for file in FEATURES_FOLDER_PATH.iterdir():
        if "mapping" in file.name or not file.name.endswith(".csv"):
            continue 
        print(f"\n==================================================")
        print(f"Loading features from {file.name}")
        df = pd.read_csv(file)

        X = df.drop(columns=["difficulty"])  
        Y = df["difficulty"]  

        # Clean constant features out upfront globally
        selector_vt = VarianceThreshold(threshold=0.0)
        try:
            X_transformed = selector_vt.fit_transform(X)
            X = pd.DataFrame(X_transformed, columns=selector_vt.get_feature_names_out(X.columns))
        except ValueError:
            print(f"Skipping {file.name}: No features have variance above 0.")
            continue
      
        run_all_combinations(title=f"CLASSIFIED_{file.stem}", X=X, Y=Y)


# --- Nested Cross-Validation Mechanics ---
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

def _safe_values(values, maximum):
    safe = [value for value in values if value <= maximum]
    if safe:
        return safe
    return [max(1, maximum)]

def _build_parameter_grid(X, Y, selector_tuple, classifier_tuple):
    params_grid = {}
    
    for idx in range(0, len(selector_tuple), 2):
        params_grid[selector_tuple[idx]] = selector_tuple[idx+1]
        
    for idx in range(0, len(classifier_tuple), 2):
        params_grid[classifier_tuple[idx]] = classifier_tuple[idx+1]

    outer_train_size = len(Y) - int(np.ceil(len(Y) / OUTER_SPLITS))
    inner_train_size = outer_train_size - int(np.ceil(outer_train_size / INNER_SPLITS))
    
    n_features = X.shape[1]
    absolute_maximum = min(inner_train_size, n_features)

    for key in list(params_grid.keys()):
        # Only cap structural parameters that rely on integer dimensions
        if key.endswith("__n_components") or key.endswith("__k"):
            params_grid[key] = _safe_values(params_grid[key], absolute_maximum)

        if key.endswith("__n_neighbors"):
            min_class_count = int(Y.value_counts().min())
            max_safe_neighbors = max(1, min(min_class_count, inner_train_size))
            params_grid[key] = _safe_values(params_grid[key], max_safe_neighbors)
            
        # Leave 'feature_selector__threshold' untouched since decimals are safe

    return params_grid

def run_classification(X, Y, classifier, classifier_grid, selector, selector_grid, text=""):
    print(f"\n ------ {text} ------")

    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('feature_selector', selector),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    params_grid = _build_parameter_grid(X, Y, selector_grid, classifier_grid)

    all_outer_accuracies = []
    all_results_raw = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        for i in range(REPEATS):
            inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=i)
            
            # CRITICAL FIX: n_jobs=None here prevents multiprocessing deadlocks
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=params_grid,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=None,
                return_train_score=True
            )

            outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=i + 100)
            
            # CRITICAL FIX: Parallelize splits safely at the outer loop layer instead
            nested_scores = cross_validate(
                grid_search,
                X, Y,
                cv=outer_cv,
                n_jobs=-1,
                scoring=("accuracy", "precision_macro", "recall_macro", "f1_macro")
            )

            print(f"Run {i+1}/{REPEATS} - Outer test accuracy scores: {nested_scores['test_accuracy']}")
            all_outer_accuracies.extend(nested_scores['test_accuracy'])
            
            # Format performance metrics safely to string for storage & up to 3 decimal places
            cleaned_scores = {k: [f"{v:.3f}" for v in vi] for k, vi in nested_scores.items()}
            all_results_raw.append(cleaned_scores)

        # Fit final estimator to obtain the best parameter profile
        grid_search.fit(X, Y)

    stats = {
        "Run details": text,
        "Average Performance (V)": np.mean(all_outer_accuracies),
        "Outer scores per repeat": all_results_raw,
        "Best parameters": grid_search.best_params_,
    }

    print(f"Result: {stats['Average Performance (V)']:.4f} using {stats['Best parameters']}")
    return stats

def run_all_combinations(X, Y, title=""):
    results_dir = Path("RESULTS")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL FIX: Removed illegal colon character (:) from timestamp string format
    timestamp = pd.Timestamp.now().strftime("%d_%H_%M")
    path = results_dir / f"{title}_({timestamp})_in{INNER_SPLITS}_out{OUTER_SPLITS}.tsv"

    for selector, selector_grid, selector_name in Selectors:
        for classifier, classifier_grid, classifier_name in Classifiers:
            stats = run_classification(
                X=X, Y=Y,
                classifier=classifier, classifier_grid=classifier_grid,
                selector=selector, selector_grid=selector_grid,
                text=f"{classifier_name} with {selector_name}"
            )

            data = {
                "classifier": classifier_name,
                "selector": selector_name,
                "avg_performance_v": f"{stats.get('Average Performance (V)'):.3f}",
                "outer_scores_per_repeat": str(stats.get("Outer scores per repeat")),
                "best_parameters": str(stats.get("Best parameters")),
            }

            df = pd.DataFrame([data])
            df.to_csv(path, sep="\t", index=False, mode='a', header=not path.exists())

    return path

if __name__ == "__main__":
    classify_per_window_size()