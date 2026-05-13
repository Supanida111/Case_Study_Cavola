"""
Calvora — Supervised Learning
Target: awareness_tier (Low/Mid/High) from raw brand_correct count.
NO leakage: excludes brand_correct, brand_decoy_hit from features.
Compares 5 models with StratifiedKFold cross-validation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

BASE = Path(__file__).parent
CLUSTERED = BASE / "data" / "calvora_clustered.csv"
METRICS = BASE / "data" / "supervised_metrics.json"
MODELS = BASE / "models"

# Leakage-safe features (exclude brand_correct, brand_decoy_hit, awareness_tier itself)
SUPERVISED_FEATURES = [
    "tagline_known", "tagline_alignment", "tagline_rating",
    "products_recognized_count", "products_tried_count",
    "ebisen_familiarity", "age",
    "gender_ชาย", "gender_หญิง", "gender_LGBTQ+",
]

TIER_ORDER = ["Low", "Mid", "High"]


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42),
    }


def main():
    print("Loading clustered data...")
    df = pd.read_csv(CLUSTERED)

    # Ensure all expected gender columns exist
    for g in ["gender_ชาย", "gender_หญิง", "gender_LGBTQ+"]:
        if g not in df.columns:
            df[g] = 0

    available = [f for f in SUPERVISED_FEATURES if f in df.columns]
    print(f"  features used: {len(available)}/{len(SUPERVISED_FEATURES)}")
    X = df[available].values
    le = LabelEncoder().fit(TIER_ORDER)
    y = le.transform(df["awareness_tier"])

    print(f"  X shape: {X.shape}, y dist: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  train: {len(y_train)} | test: {len(y_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    results = {}

    print("\n=== Training & Cross-Validation ===")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="f1_weighted", n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(TIER_ORDER)))
        report = classification_report(y_test, y_pred,
                                        target_names=TIER_ORDER,
                                        output_dict=True, zero_division=0)
        results[name] = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
            "test_f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        print(f"  {name}:")
        print(f"    CV F1 = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"    Test Acc = {acc:.3f}, Test F1 = {f1:.3f}")

    # === GridSearchCV: tune KNN hyperparameters ===
    print("\n=== GridSearchCV: K-Nearest Neighbors ===")
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])
    param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }
    print(f"  Searching {7*2*2} combinations × 5-fold CV...")
    gs = GridSearchCV(knn_pipe, param_grid, cv=cv,
                       scoring="f1_weighted", n_jobs=-1,
                       return_train_score=False)
    gs.fit(X_train, y_train)
    best_idx = gs.best_index_
    knn_tuned = gs.best_estimator_
    knn_tuned_cv_std = float(gs.cv_results_["std_test_score"][best_idx])

    # Evaluate tuned KNN on test set
    y_pred_tuned = knn_tuned.predict(X_test)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned, average="weighted",
                          zero_division=0)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned,
                                  labels=range(len(TIER_ORDER)))
    rep_tuned = classification_report(y_test, y_pred_tuned,
                                        target_names=TIER_ORDER,
                                        output_dict=True, zero_division=0)
    results["KNN (Tuned)"] = {
        "cv_f1_mean": float(gs.best_score_),
        "cv_f1_std": knn_tuned_cv_std,
        "test_accuracy": float(acc_tuned),
        "test_f1": float(f1_tuned),
        "confusion_matrix": cm_tuned.tolist(),
        "classification_report": rep_tuned,
        "best_params": gs.best_params_,
    }
    models["KNN (Tuned)"] = knn_tuned

    # Top-5 candidates table
    cv_df = pd.DataFrame(gs.cv_results_)
    top5 = cv_df.nlargest(5, "mean_test_score")[
        ["params", "mean_test_score", "std_test_score"]
    ].to_dict(orient="records")
    baseline_knn = results["K-Nearest Neighbors"]["cv_f1_mean"]
    improvement = gs.best_score_ - baseline_knn

    print(f"  Best params:  {gs.best_params_}")
    print(f"  Tuned CV F1 = {gs.best_score_:.3f} ± "
          f"{knn_tuned_cv_std:.3f}")
    print(f"  Baseline KNN CV F1 = {baseline_knn:.3f}")
    print(f"  Improvement: {improvement:+.3f}")
    print(f"  Test Acc = {acc_tuned:.3f}, Test F1 = {f1_tuned:.3f}")

    # Best by CV F1 (will now include tuned KNN as candidate)
    best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
    print(f"\nBest model (by CV F1): {best_name}")

    # Feature importance from Random Forest
    rf = models["Random Forest"]
    feat_imp = dict(zip(available, rf.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

    summary = {
        "features": available,
        "tier_order": TIER_ORDER,
        "tier_distribution": {t: int(c) for t, c in
                              zip(TIER_ORDER, np.bincount(y))},
        "results": results,
        "best_model": best_name,
        "feature_importance_rf": feat_imp,
        "knn_tuning": {
            "search_space": {
                "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            "best_params": gs.best_params_,
            "baseline_cv_f1": float(baseline_knn),
            "tuned_cv_f1": float(gs.best_score_),
            "tuned_cv_f1_std": knn_tuned_cv_std,
            "improvement": float(improvement),
            "top5_candidates": [
                {"params": {k: (v if not isinstance(v, (np.integer, np.floating))
                                  else float(v))
                              for k, v in row["params"].items()},
                 "mean_test_score": float(row["mean_test_score"]),
                 "std_test_score": float(row["std_test_score"])}
                for row in top5
            ],
        },
    }
    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics → {METRICS}")

    # Save best model
    best_model = models[best_name]
    joblib.dump({"model": best_model, "features": available,
                 "tier_order": TIER_ORDER, "label_encoder": le},
                MODELS / "supervised_best.joblib")
    print(f"Saved best model → {MODELS / 'supervised_best.joblib'}")


if __name__ == "__main__":
    main()
