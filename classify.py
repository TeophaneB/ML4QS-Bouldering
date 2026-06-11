"""
5. Define an appropriate train/test setup and apply classical machine learning techniques
(cf. Chapter 7) on the resulting dataset from (4). Describe your rationale, the results,
and how you optimized the hyperparameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


### Classfiers

# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=3)
classifier_grid_knn = ('kneighborsclassifier__n_neighbors', [1, 3, 5, 10, 20, 50], 'kneighborsclassifier__weights', ['uniform', 'distance'])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(max_iter=5000)
classifier_grid_logistic = ('logisticregression__C', [0.001, 0.01, 0.1, 0.5, 1, 10])

# SVM
from sklearn.svm import SVC
classifier_svm = SVC()
classifier_grid_svm = ('svc__C', [0.001, 0.01, 0.1, 0.5, 1, 10], 'svc__kernel', ['linear', 'rbf'])

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier()
classifier_grid_rf = ('randomforestclassifier__n_estimators', [1, 3, 5, 10, 20, 50], 'randomforestclassifier__max_depth', [None, 5, 10, 20])

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_grid_dt = ('decisiontreeclassifier__max_depth', [None, 5, 10, 20])

Classifiers = [
 (classifier_knn, classifier_grid_knn, "KNN"),
 (classifier_logistic, classifier_grid_logistic, "Logistic Regression"),
 (classifier_svm, classifier_grid_svm, "SVM"),
 (classifier_rf, classifier_grid_rf, "Random Forest"), # main
 (classifier_dt, classifier_grid_dt, "Decision Tree"), # baseline
]

### Selectors

# # Forward Selection
# from sklearn.feature_selection import SequentialFeatureSelector
# selector_forward = SequentialFeatureSelector(estimator=classifier_knn, direction="forward", cv=2, n_features_to_select=10, scoring="accuracy")
# selector_grid_forward = ('sequentialfeatureselector__n_features_to_select', [2, 5, 10, 25])

# # Mutual Information
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
# selector_grid_mi = ('selectkbest__k', [5, 10, 25])

# Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector_variance = VarianceThreshold(threshold=0.01)
selector_grid_variance = ('variancethreshold__threshold', [0.001, 0.01, 0.05, 0.1])

# ANOVA 
from sklearn.feature_selection import SelectKBest, f_classif
selector_anova = SelectKBest(score_func=f_classif, k=10)
selector_grid_anova = ('selectkbest__k', [5, 10, 25, 50, 100]) 

# PCA 
from sklearn.decomposition import PCA
selector_pca = PCA(n_components=10)
selector_grid_pca = ('pca__n_components', [5, 10, 25, 50, 100])

Selectors = [
#(selector_anova, selector_grid_anova, "ANOVA"),
#(selector_variance, selector_grid_variance, "Variance Threshold"),
(selector_pca, selector_grid_pca, "PCA"),

# Skip these?
#(selector_mi, selector_grid_mi, "Mutual Information"),
#(selector_forward, selector_grid_forward, "Forward Selection"),
 ]


INNER_SPLITS = 5
OUTER_SPLITS = 3
REPEATS = 1

FEATURES_FOLDER_PATH = Path("C:\\Users\\teoph\\OneDrive\\Documents\\Master\\P6\\ML4QS\\FEATURES")

def classify_per_window_size():
    """
    train a classifier to predict difficulty
    """
    
    # classify each file/window size separately
    for file in FEATURES_FOLDER_PATH.iterdir():  # classify each file/window size separately
        if "mapping" in file.name:
            continue  # skip mapping files
        print(f"Loading features from {file.name}")
        df = pd.read_csv(file)

        # Prepare features and labels
        X = df.drop(columns=["difficulty"])  # Features
        Y = df["difficulty"]  # Labels

        # Remove constant features
        X = VarianceThreshold(threshold=0.0).fit_transform(X)
      
        run_all_combinations(title=f"Classifiers with Selectors for {file.stem}", X=X, Y=Y)



### Nested Cross-Validation for Robust Performance Estimation and Hyperparameter Tuning

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def _safe_values(values, maximum):
    safe = [value for value in values if value <= maximum]
    if safe:
        return safe
    return [max(1, maximum)]


def _build_parameter_grid(X, Y, selector_grid, classifier_grid):
    params_grid = {
        selector_grid[0]: selector_grid[1],
        classifier_grid[0]: classifier_grid[1],
    }

    n_features = X.shape[1]
    if selector_grid[0].endswith("__n_components") or selector_grid[0].endswith("__k"):
        params_grid[selector_grid[0]] = _safe_values(selector_grid[1], n_features)

    if classifier_grid[0].endswith("__n_neighbors"):
        min_class_count = int(Y.value_counts().min())
        outer_train_size = len(Y) - int(np.ceil(len(Y) / OUTER_SPLITS))
        inner_train_size = outer_train_size - int(np.ceil(outer_train_size / INNER_SPLITS))
        max_safe_neighbors = max(1, min(min_class_count, inner_train_size))
        params_grid[classifier_grid[0]] = _safe_values(classifier_grid[1], max_safe_neighbors)

    return params_grid


def run_classification(X, Y, classifier, classifier_grid, selector, selector_grid, text=""):
    print(f"\n\n ------ {text} ------")

    # -- steps of the model
    pipeline = make_pipeline(VarianceThreshold(threshold=0.0), selector, StandardScaler(), classifier)

    # -- hyperparameters to test
    params_grid = _build_parameter_grid(X, Y, selector_grid, classifier_grid)

    all_outer_scores = []

    # Repeat the nested CV
    for i in range(REPEATS):
        
        # -- Inner loop: train the model and evaluate the performance
        inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=i)
        grid_search = GridSearchCV(
            estimator=pipeline,  # selector then classifier
            param_grid=params_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True
        )

        # -- Outer loop: repeat the inner loop and report the average performance
        outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=i + 100)
        nested_scores = cross_val_score(
            grid_search,
            X, Y,
            cv=outer_cv,
            n_jobs=-1
        )

        print(f"Run {i+1}/{REPEATS} - Outer scores: {nested_scores}")
        all_outer_scores.append(nested_scores)

    grid_search.fit(X, Y)


    stats = {
        "Run details": text,
        "Average Performance (V)": np.mean(np.concatenate(all_outer_scores)),  # scalar over all repeats
        "Outer scores per repeat": all_outer_scores,
        "Best parameters": grid_search.best_params_,
    }

    print(stats)
    return stats


import pandas as pd
from pathlib import Path



def run_all_combinations(X, Y, title=""):

    # save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("(%d_%H:%M)")
    path = results_dir / f"{title} {timestamp} in{INNER_SPLITS} out{OUTER_SPLITS} rep{REPEATS}.tsv"

    for selector, selector_grid, selector_name in Selectors: # eaach combination
        for classifier, classifier_grid, classifier_name in Classifiers:
            stats = run_classification(
                X=X,
                Y=Y,
                classifier=classifier,
                classifier_grid=classifier_grid,
                selector=selector,
                selector_grid=selector_grid,
                text=f"{classifier_name} with {selector_name}"
            )

            data = {
                "classifier": classifier_name,
                "selector": selector_name,
                "avg_performance_v": stats.get("Average Performance (V)"),
                "outer_scores_per_repeat": str(stats.get("Outer scores per repeat")),
                "best_parameters": str(stats.get("Best parameters")),
                "final_predictions_unseen": str(stats.get("Final predictions on unseen data"))
            }

            # Save data
            df = pd.DataFrame([data])
            df.to_csv(path, sep="\t", index=False, mode='a', header=not path.exists())  # Append to file, write header only if file doesn't exist


    return path


classify_per_window_size()