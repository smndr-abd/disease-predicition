"""
Rigorous evaluation + baseline comparison for the disease-prediction project.

Run:  python evaluate.py
Outputs: figures/*.png and results.csv

What it adds over train.py
  1. Repeated stratified CV (mean +/- std) instead of a single split.
     The decision tree is tuned INSIDE each fold (nested CV), so the
     tuning never sees the fold it is scored on.
  2. Top-3 accuracy: a ranked "differential" is more useful than one label.
  3. Robustness: hide a share of a patient's symptoms at test time.
  4. Learning curves: does more data help?
  5. Baselines: Naive Bayes, Logistic Regression, Random Forest, Gradient Boosting.
  6. Calibration: when a model says "80% sure", is it right 80% of the time?
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate, learning_curve)
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from make_dataset import generate

warnings.filterwarnings("ignore")
SEED = 42
TREE = "Decision Tree"


def make_models():
    tree = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 5, 10]},
        cv=3,
    )
    return {
        TREE: tree,
        "Naive Bayes": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(200, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def hide_symptoms(X, frac, rng):
    """Randomly turn a share of PRESENT symptoms off (patient forgot to mention them)."""
    arr = X.to_numpy().copy()
    arr[(rng.random(arr.shape) < frac) & (arr == 1)] = 0
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def expected_calibration_error(proba, y_true, classes, bins=10):
    """Gap between confidence and accuracy, averaged over confidence bins."""
    conf = proba.max(axis=1)
    pred = np.array(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            curve.append((conf[m].mean(), correct[m].mean()))
    return ece, np.array(curve)


def main():
    df = generate(n_per_disease=300, seed=SEED)
    X, y = df.drop(columns="prognosis"), df["prognosis"]
    classes = sorted(y.unique())
    models = make_models()
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = {name: {} for name in models}

    # 1. Repeated CV -------------------------------------------------
    print("1/5 repeated nested CV ...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cv_acc = {}
    for name, m in models.items():
        r = cross_validate(clone(m), X, y, cv=rskf,
                           scoring=["accuracy", "f1_macro"], n_jobs=-1)
        cv_acc[name] = r["test_accuracy"]
        rows[name]["acc_mean"] = r["test_accuracy"].mean()
        rows[name]["acc_std"] = r["test_accuracy"].std()
        rows[name]["f1_macro"] = r["test_f1_macro"].mean()

    # 2 + 6. Top-3 accuracy and calibration (out-of-fold probabilities)
    print("2/5 top-3 accuracy + calibration ...")
    curves = {}
    for name, m in models.items():
        proba = cross_val_predict(clone(m), X, y, cv=cv5,
                                  method="predict_proba", n_jobs=-1)
        rows[name]["top3"] = top_k_accuracy_score(y, proba, k=3, labels=classes)
        rows[name]["ECE"], curves[name] = expected_calibration_error(proba, y, classes)

    # 3. Robustness to missing symptoms ------------------------------
    print("3/5 robustness to hidden symptoms ...")
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rng = np.random.default_rng(SEED)
    rob = {name: np.zeros(len(levels)) for name in models}
    for tr, te in cv5.split(X, y):
        for name, m in models.items():
            fit = clone(m).fit(X.iloc[tr], y.iloc[tr])
            for i, p in enumerate(levels):
                reps = 1 if p == 0 else 3
                rob[name][i] += np.mean([
                    fit.score(hide_symptoms(X.iloc[te], p, rng), y.iloc[te])
                    for _ in range(reps)
                ]) / cv5.get_n_splits()
    for name in models:
        rows[name]["acc_30pct_hidden"] = rob[name][levels.index(0.3)]

    # 4. Learning curves ---------------------------------------------
    print("4/5 learning curves ...")
    lc = {}
    for name in [TREE, "Naive Bayes", "Logistic Regression", "Random Forest"]:
        sizes, _, te = learning_curve(clone(models[name]), X, y,
                                      train_sizes=np.linspace(0.1, 1.0, 6),
                                      cv=cv5, n_jobs=-1)
        lc[name] = (sizes, te.mean(axis=1), te.std(axis=1))

    # RESULTS TABLE --------------------------------------------------
    res = pd.DataFrame(rows).T
    res["accuracy (mean±std)"] = res.apply(
        lambda r: f"{r.acc_mean:.3f} ± {r.acc_std:.3f}", axis=1)
    out = res[["accuracy (mean±std)", "f1_macro", "top3", "ECE", "acc_30pct_hidden"]]
    print("\n", out.round(3).to_string())
    res.drop(columns="accuracy (mean±std)").round(4).to_csv("results.csv")

    # FIGURES --------------------------------------------------------
    print("5/5 saving figures ...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(res.index, res.acc_mean, yerr=res.acc_std, capsize=4)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Accuracy (5-fold x 3 repeats)")
    ax.set_title("Model comparison (decision tree tuned in nested CV)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig("figures/model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in models:
        ax.plot([p * 100 for p in levels], rob[name], marker="o", label=name)
    ax.set_xlabel("% of present symptoms hidden at test time")
    ax.set_ylabel("Accuracy"); ax.set_title("Robustness to missing symptoms")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/robustness.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (s, mu, sd) in lc.items():
        ax.plot(s, mu, marker="o", label=name)
        ax.fill_between(s, mu - sd, mu + sd, alpha=.15)
    ax.set_xlabel("Training examples"); ax.set_ylabel("CV accuracy")
    ax.set_title("Learning curves"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/learning_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name, c in curves.items():
        ax.plot(c[:, 0], c[:, 1], marker="o", label=f"{name} (ECE {res.loc[name, 'ECE']:.3f})")
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Actual accuracy")
    ax.set_title("Calibration (reliability diagram)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/calibration.png", dpi=150); plt.close(fig)

    # Is the tree really different from the others? (paired, same folds)
    best = res.acc_mean.idxmax()
    print(f"\nBest mean accuracy: {best}")
    for name in models:
        if name != best:
            d = cv_acc[best] - cv_acc[name]
            print(f"  {best} minus {name}: {d.mean():+.3f} (std of diff {d.std():.3f})")


if __name__ == "__main__":
    main()