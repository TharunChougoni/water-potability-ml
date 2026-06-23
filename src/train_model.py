"""Train and evaluate a water-potability classifier.

This script mirrors the notebook workflow in a reproducible command-line form:
load data, impute missing values, scale features, train Random Forest, and print
classification metrics plus cross-validation accuracy.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "water_potability.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=["Potability"])
    y = df["Potability"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    print("Dataset shape:", df.shape)
    print("Target balance:")
    print(y.value_counts(normalize=True).rename("ratio"))
    print()
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("5-fold CV accuracy:", [round(float(v), 4) for v in cv_scores])
    print("Mean CV accuracy:", round(cv_scores.mean(), 4))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
