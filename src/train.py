"""Train the XGBoost customer-segment classifier and export it for the web app.

Usage:  python -m src.train
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src import dataset
from src.utils import export_trees

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_model.json"
WEB_MODEL = ROOT / "web" / "model.json"
SEED = 42

PARAM_GRID = [
    {"max_depth": d, "learning_rate": lr, "min_child_weight": mcw}
    for d in (3, 4, 5) for lr in (0.03, 0.06) for mcw in (1, 5)
]
BASE = dict(objective="multi:softprob", n_estimators=300, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            eval_metric="mlogloss", random_state=SEED, n_jobs=-1)


def segment_profiles(df: pd.DataFrame) -> dict:
    """Human-readable statistics for each segment, shown on the website."""
    out = {}
    for seg, g in df.groupby(dataset.TARGET):
        out[seg] = {
            "count": int(len(g)),
            "avg_age": round(float(g["Age"].mean()), 1),
            "married_pct": round(float((g["Ever_Married"] == "Yes").mean() * 100), 1),
            "graduated_pct": round(float((g["Graduated"] == "Yes").mean() * 100), 1),
            "avg_family": round(float(g["Family_Size"].mean()), 1),
            "avg_experience": round(float(g["Work_Experience"].mean()), 1),
            "top_professions": g["Profession"].value_counts(normalize=True).head(3).round(3).to_dict(),
            "spending": g["Spending_Score"].value_counts(normalize=True).round(3).to_dict(),
        }
    return out


def main() -> None:
    df = dataset.load()
    X = dataset.encode(df)
    y = df[dataset.TARGET].map({c: i for i, c in enumerate(dataset.CLASSES)})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Pick hyper-parameters with 5-fold CV on the training split.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, best_score = None, -1.0
    for p in PARAM_GRID:
        score = cross_val_score(xgb.XGBClassifier(**BASE, **p), X_train, y_train,
                                cv=cv, scoring="accuracy").mean()
        print(f"{p}  cv_acc={score:.4f}")
        if score > best_score:
            best, best_score = p, score
    print("best:", best, round(best_score, 4))

    # 2) Find the number of trees with early stopping on a validation slice.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
    probe = xgb.XGBClassifier(**{**BASE, "n_estimators": 1000}, **best, early_stopping_rounds=40)
    probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    n_trees = probe.best_iteration + 1

    # 3) Final model on the full training split, evaluated once on the held-out test set.
    model = xgb.XGBClassifier(**{**BASE, "n_estimators": n_trees}, **best)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"trees={n_trees}  test_acc={acc:.4f}  macro_f1={f1:.4f}")
    print(classification_report(y_test, pred, target_names=dataset.CLASSES))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)

    names = dataset.feature_names()
    exported = export_trees(model, X_test.to_numpy()[:300], names)
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values())
    report = classification_report(y_test, pred, target_names=dataset.CLASSES, output_dict=True)

    payload = {
        "classes": dataset.CLASSES,
        "features": names,
        "professions": dataset.PROFESSIONS,
        "categories": dataset.CATEGORIES,
        **exported,
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "cv_accuracy": round(float(best_score), 4),
            "baseline_accuracy": round(float(y_test.value_counts(normalize=True).max()), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_trees": n_trees,
            "params": best,
            "per_class": {c: {k: round(report[c][k], 3) for k in ("precision", "recall", "f1-score")}
                          for c in dataset.CLASSES},
            "confusion": confusion_matrix(y_test, pred).tolist(),
        },
        "importance": sorted(({"feature": k, "gain": round(v / total, 4)} for k, v in gain.items()),
                             key=lambda d: -d["gain"]),
        "profiles": segment_profiles(df),
    }
    WEB_MODEL.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {WEB_MODEL.relative_to(ROOT)} "
          f"({WEB_MODEL.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
