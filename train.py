import argparse, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, precision_recall_curve)
from features import build_feature_frame, build_matrix, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=3000)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--out", default="model.pkl")
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(a.data).dropna(subset=["question1", "question2", "is_duplicate"])
    df = df.drop_duplicates(subset=["question1", "question2"])
    if a.sample and a.sample < len(df):
        df = df.sample(a.sample, random_state=42)
    df = df.reset_index(drop=True)
    print(f"Rows: {len(df):,} | duplicate rate: {df.is_duplicate.mean():.3f}")

    q1 = df.question1.astype(str).tolist()
    q2 = df.question2.astype(str).tolist()
    y = df.is_duplicate.values
    print("Building features ...")
    feat_df, c1, c2 = build_feature_frame(q1, q2)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
    cv = CountVectorizer(max_features=a.max_features)
    cv.fit([c1[i] for i in idx_tr] + [c2[i] for i in idx_tr])

    X, feat_full = build_matrix(feat_df, c1, c2, cv)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    print("Training RandomForest ...")
    model = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=42,
                                   class_weight="balanced_subsample", min_samples_leaf=2)
    model.fit(Xtr, ytr)

    prob = model.predict_proba(Xte)[:, 1]
    p, r, th = precision_recall_curve(yte, prob)
    f1s = 2 * p * r / (p + r + 1e-9)
    best_th = float(th[np.argmax(f1s[:-1])])
    pred = (prob >= best_th).astype(int)

    imp = model.feature_importances_[-len(FEATURE_COLS):]
    metrics = {
        "rows": int(len(df)), "duplicate_rate": float(df.is_duplicate.mean()),
        "threshold": best_th,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred)),
        "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, prob)),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "feature_importance": dict(sorted(zip(FEATURE_COLS, map(float, imp)), key=lambda x: -x[1])),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump({"vectorizer": cv, "model": model, "threshold": best_th, "metrics": metrics},
                a.out, compress=3)
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importance"}, indent=2))
    print("Saved", a.out, "and metrics.json")


if __name__ == "__main__":
    main()