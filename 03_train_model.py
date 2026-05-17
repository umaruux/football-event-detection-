"""
03_train_model.py
------------------
1. Loads the feature matrix from data/features.csv.
2. Performs basic Exploratory Data Analysis (EDA) with plots.
3. Handles class imbalance using SMOTE.
4. Trains a Random Forest classifier.
5. Evaluates with precision, recall, F1 and a confusion matrix.
6. Saves the trained model to models/rf_model.joblib.

Requirements:
    pip install scikit-learn imbalanced-learn pandas matplotlib seaborn joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from imblearn.over_sampling import SMOTE

from utils import FEATURE_COLS

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FEATURES_CSV  = Path("data/features.csv")
MODEL_OUT     = Path("models/rf_model.joblib")
ENCODER_OUT   = Path("models/label_encoder.joblib")
PLOTS_DIR     = Path("plots")

TARGET_CLASSES = ["Penalty", "Free-kick", "Kick-off", "Corner", "Throw-in", "Background"]

RANDOM_STATE  = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15  

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":     12,
    "min_samples_leaf": 2,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}
# ───────────────────────────────────────────────────────────────────────────────

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["label"].isin(TARGET_CLASSES)].reset_index(drop=True)
    print(f"Loaded feature matrix: {df.shape}")
    print(df["label"].value_counts())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df):
    print("\n── EDA ───────────────────────────────────────────────────────────")

    # 2a. Class distribution bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["label"].value_counts()
    palette = ["#534AB7","#0F6E56","#993C1D","#BA7517","#A32D2D","#888888"]
    bars = ax.bar(counts.index, counts.values, color=palette[:len(counts)])
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Event class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=150)
    plt.close()
    print("Saved: plots/01_class_distribution.png")

    # 2b. Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_feature_correlation.png", dpi=150)
    plt.close()
    print("Saved: plots/02_feature_correlation.png")

    # 2c. Box plots of key features by class
    key_features = ["ball_dist_spot_mid", "wall_index_mid",
                    "ball_dist_corner_mid", "ball_dist_sideline_mid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flatten(), key_features):
        df.boxplot(column=feat, by="label", ax=ax, grid=False)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    plt.suptitle("Key feature distributions by class")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_feature_boxplots.png", dpi=150)
    plt.close()
    print("Saved: plots/03_feature_boxplots.png")

    # 2d. Missing values summary
    missing = df[FEATURE_COLS].isnull().sum()
    if missing.any():
        print("\nMissing values:\n", missing[missing > 0])
    else:
        print("No missing values in feature columns.")

    print("─────────────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df):
        # ── Remove classes with too few samples for SMOTE (needs >= k+1 = 6) ─
    MIN_SAMPLES = 6
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
    removed = class_counts[class_counts < MIN_SAMPLES]
    if not removed.empty:
        print(f"\nWarning: Removing classes with too few samples: {removed.to_dict()}")
    df = df[df["label"].isin(valid_classes)].reset_index(drop=True)
    # ───────────────────────────────────────────────────────────────────
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[FEATURE_COLS].values

    print(f"Classes: {list(le.classes_)}")

    # Train / val / test split (stratified to preserve class ratios)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac, random_state=RANDOM_STATE, stratify=y_trainval
    )

    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Test  : {len(X_test)} samples")

    # Handle class imbalance with SMOTE on training set only
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_train_res)} training samples")

    return X_train_res, y_train_res, X_val, y_val, X_test, y_test, le


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train, y_train):
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(clf, X_train, y_train):
    print("\nRunning 5-fold cross-validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_train,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"  CV F1 (weighted): {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(clf, X_val, y_val, X_test, y_test, le):
    class_names = list(le.classes_)

    # --- Validation set ---
    y_val_pred = clf.predict(X_val)
    print("\n── Validation Set Results ───────────────────────────────────────")
    print(classification_report(y_val, y_val_pred, target_names=class_names))

    # --- Test set ---
    y_test_pred = clf.predict(X_test)
    print("── Test Set Results ─────────────────────────────────────────────")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    print(f"Overall weighted F1 on test set: {test_f1:.3f}")
    print("─────────────────────────────────────────────────────────────────\n")

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix — test set")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: plots/04_confusion_matrix.png")

    return test_f1


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importance[::-1], color="#534AB7")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — feature importances")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: plots/05_feature_importance.png")

    print("\nFeature importances:")
    for feat, imp in zip(sorted_features, sorted_importance):
        print(f"  {feat:<25} {imp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(clf, le):
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(le, ENCODER_OUT)
    print(f"\nModel saved : {MODEL_OUT}")
    print(f"Encoder saved: {ENCODER_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()
    run_eda(df)

    X_train, y_train, X_val, y_val, X_test, y_test, le = preprocess(df)
    clf = train(X_train, y_train)
    cross_validate(clf, X_train, y_train)
    evaluate(clf, X_val, y_val, X_test, y_test, le)
    plot_feature_importance(clf)
    save_model(clf, le)

    print("\nDone. All plots saved to: plots/")


if __name__ == "__main__":
    main()
