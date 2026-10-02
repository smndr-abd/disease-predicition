"""
Disease Prediction with Decision Trees: full experiment pipeline.

Run:   python train.py
Real:  python train.py --real path/to/Training.csv   (optional cross-dataset test)
Outputs go to ./figures
"""
import argparse
import matplotlib
matplotlib.use("Agg")  # save figures to files; works in terminal and VS Code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from make_dataset import align_to_real, generate

SEED = 42


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main(real_path=None):
    # 1. DATA --------------------------------------------------------
    df = generate(n_per_disease=300, seed=SEED)
    X, y = df.drop(columns="prognosis"), df["prognosis"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    section("1. Data")
    print(f"rows={len(df)}  symptoms={X.shape[1]}  diseases={y.nunique()}")

    # 2. OVERFITTING DEMO: unrestricted tree -------------------------
    section("2. Unrestricted tree (overfitting demo)")
    deep = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    print(f"depth={deep.get_depth()}  leaves={deep.get_n_leaves()}")
    print(f"train acc={deep.score(Xtr, ytr):.3f}   test acc={deep.score(Xte, yte):.3f}")

    # 3. TUNING with cross-validation --------------------------------
    section("3. Tuning (5-fold CV grid search)")
    grid = {
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "ccp_alpha": [0.0, 0.001, 0.005],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    ).fit(Xtr, ytr)
    best = gs.best_estimator_
    print("best params:", gs.best_params_)
    print(f"CV acc={gs.best_score_:.3f}   test acc={best.score(Xte, yte):.3f}")
    print(f"depth={best.get_depth()}  leaves={best.get_n_leaves()}")

    # 4. EVALUATION --------------------------------------------------
    section("4. Evaluation on held-out synthetic test set")
    pred = best.predict(Xte)
    print(classification_report(yte, pred))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        yte, pred, ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title("Confusion matrix (tuned tree, synthetic test)")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 5. INTERPRETATION ----------------------------------------------
    section("5. Interpretation")
    print(export_text(best, feature_names=list(X.columns), max_depth=3))
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(best, feature_names=X.columns, class_names=best.classes_,
              filled=True, max_depth=3, fontsize=8, ax=ax)
    fig.savefig("figures/tree.png", dpi=120)
    plt.close(fig)

    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(12).plot.barh(ax=ax)
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig("figures/feature_importance.png", dpi=150)
    plt.close(fig)

    # 6. BONUS: Random Forest comparison -----------------------------
    section("6. Random Forest comparison")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    print(f"tuned tree test acc={best.score(Xte, yte):.3f}   forest test acc={rf.score(Xte, yte):.3f}")

    # 7. CROSS-DATASET TEST (optional) -------------------------------
    if real_path:
        section("7. Synthetic <-> Real alignment")
        real = align_to_real(pd.read_csv(real_path), df)
        Xr, yr = real.drop(columns="prognosis"), real["prognosis"]
        print(f"aligned real rows={len(real)}  shared diseases={yr.nunique()}")
        dead = [c for c in Xr.columns if Xr[c].sum() == 0]
        if dead:
            print("WARNING all-zero columns (name mismatch?):", dead)
        print(f"train synthetic -> test real : {best.score(Xr, yr):.3f}")
        real_tree = DecisionTreeClassifier(max_depth=best.get_params()["max_depth"],
                                           random_state=SEED).fit(Xr, yr)
        print(f"train real -> test synthetic : {real_tree.score(X, y):.3f}")
    else:
        print("\n(skip) pass --real path/to/Training.csv to run the cross-dataset test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="path to real Kaggle CSV")
    main(ap.parse_args().real)