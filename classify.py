"""
5. Define an appropriate train/test setup and apply classical machine learning techniques
(cf. Chapter 7) on the resulting dataset from (4). Describe your rationale, the results,
and how you optimized the hyperparameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
# import KNN, RandomForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# cross validation and hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV

features_folder_path = Path("C:\\Users\\teoph\\OneDrive\\Documents\\Master\\P6\\ML4QS\\FEATURES")

def classify_attempts(features_folder_path):
    """
    train a classifier to predict difficulty
    """
    

    # classify each file/window size separately
    for file in features_folder_path.iterdir():  # classify each file/window size separately
        try:
            print(f"Loading features from {file.name}")
            df = pd.read_csv(file)

            # Prepare features and labels
            X = df.drop(columns=["difficulty"])  # Features
            y = df["difficulty"]  # Labels

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train KNN classifier
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            knn_score = knn.score(X_test, y_test)

            # Train Random Forest classifier
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            rf_score = rf.score(X_test, y_test)

            print(f"KNN Accuracy: {knn_score:.2f}")
            print(f"Random Forest Accuracy: {rf_score:.2f}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    classify_attempts(features_folder_path)