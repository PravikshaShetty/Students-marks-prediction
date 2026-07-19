"""
preprocessing.py
-----------------
Handles data loading, feature engineering, and preprocessing pipeline
(encoding + scaling) for the Student Marks Prediction project.

Dataset: "Students Performance in Exams" (Kaggle, spscientist)
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

Columns in the raw file:
  gender, race/ethnicity, parental level of education, lunch,
  test preparation course, math score, reading score, writing score

We predict `final_marks` (math score) from the student's reading and
writing scores (prior academic performance) plus demographic /
program features (gender, race/ethnicity, parental education, lunch
type, test preparation course).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


CATEGORICAL_COLS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]

NUMERIC_COLS = [
    "reading_score",
    "writing_score",
]

TARGET_COL = "final_marks"

RENAME_MAP = {
    "gender": "gender",
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parental_level_of_education",
    "lunch": "lunch",
    "test preparation course": "test_preparation_course",
    "math score": "final_marks",
    "reading score": "reading_score",
    "writing score": "writing_score",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME_MAP)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features that give the model more predictive signal."""
    df = df.copy()

    # Average prior performance across the two other subjects
    df["avg_prior_score"] = (df["reading_score"] + df["writing_score"]) / 2

    # Gap between reading and writing can flag subject-specific weakness
    df["reading_writing_gap"] = df["reading_score"] - df["writing_score"]

    return df


class Preprocessor:
    """
    Fits label encoders + a scaler on training data, and applies the
    same transformations consistently at inference time.
    """

    def __init__(self):
        self.encoders = {col: LabelEncoder() for col in CATEGORICAL_COLS}
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit_transform(self, df: pd.DataFrame):
        df = engineer_features(df)

        for col in CATEGORICAL_COLS:
            df[col] = self.encoders[col].fit_transform(df[col])

        feature_cols = (
            NUMERIC_COLS
            + CATEGORICAL_COLS
            + ["avg_prior_score", "reading_writing_gap"]
        )
        self.feature_names_ = feature_cols

        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        y = df[TARGET_COL].values if TARGET_COL in df.columns else None
        return X_scaled, y

    def transform(self, df: pd.DataFrame):
        df = engineer_features(df)

        for col in CATEGORICAL_COLS:
            # handle unseen categories gracefully by falling back to the most common class
            known_classes = set(self.encoders[col].classes_)
            df[col] = df[col].apply(lambda v: v if v in known_classes else self.encoders[col].classes_[0])
            df[col] = self.encoders[col].transform(df[col])

        X = df[self.feature_names_].values
        return self.scaler.transform(X)
