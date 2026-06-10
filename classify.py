"""
5. Define an appropriate train/test setup and apply classical machine learning techniques
(cf. Chapter 7) on the resulting dataset from (4). Describe your rationale, the results,
and how you optimized the hyperparameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
# import KNN, RandomForest, LogisticRegression, DecisionTree, SVM, etc.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# cross validation and hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

features_folder_path = Path("C:\\Users\\teoph\\OneDrive\\Documents\\Master\\P6\\ML4QS\\FEATURES")

def classify_attempts(features_folder_path):
    """
    train a classifier to predict difficulty
    """
    

    # classify each file/window size separately
    for file in features_folder_path.iterdir():  # classify each file/window size separately
        try:
            if "mapping" in file.name:
                continue  # skip mapping files
            print(f"Loading features from {file.name}")
            df = pd.read_csv(file)

            # Prepare features and labels
            X = df.drop(columns=["difficulty"])  # Features
            y = df["difficulty"]  # Labels

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            final_results_clean = {}

            def evaluate_a_model(model, model_name, model_params):
                grid = GridSearchCV(model, model_params, cv=5, scoring='accuracy')
                grid.fit(X_train, y_train)
                cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5)
                test_score = grid.score(X_test, y_test)

                final_results_clean[model_name] = {
                    "best_params": grid.best_params_,
                    "cv_accuracy_mean": np.mean(cv_scores),
                    "cv_accuracy_std": np.std(cv_scores),
                    "test_accuracy": test_score
                }

            # Evaluate different classifiers
            evaluate_a_model(KNeighborsClassifier(), "KNN", {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']})
            evaluate_a_model(RandomForestClassifier(random_state=42), "Random Forest", {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]})
            evaluate_a_model(LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression", {'C': [0.01, 0.1, 1, 10]})
            evaluate_a_model(DecisionTreeClassifier(random_state=42), "Decision Tree", {'max_depth': [5, 10, 15]})
            evaluate_a_model(SVC(random_state=42), "SVM", {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})

            print(f"Results for {file.name}:")
            print(pd.DataFrame(final_results_clean).T)  # Transpose for better readability
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    classify_attempts(features_folder_path)