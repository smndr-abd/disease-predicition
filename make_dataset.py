"""
Synthetic disease-symptom dataset generator (EDUCATIONAL ONLY, not medical data).

Idea: each disease has a "profile": symptoms with a probability of appearing.
  core      -> very likely   (0.80-0.95)
  secondary -> sometimes     (0.30-0.50)
  noise     -> any other symptom appears rarely (0.03), like real-world confusion

Column names follow the Kaggle "Disease Prediction Using Machine Learning"
convention (binary symptom columns + 'prognosis') so the two can be aligned.
Verify names against the real CSV's header after downloading it.
"""
import numpy as np
import pandas as pd

# Illustrative profiles. Edit freely; this is YOUR domain knowledge layer.
PROFILES = {
    "Common Cold": {
        "core": ["continuous_sneezing", "cough", "chills", "headache"],
        "secondary": ["fatigue", "high_fever", "muscle_pain"],
    },
    "Pneumonia": {
        "core": ["high_fever", "cough", "breathlessness", "chills"],
        "secondary": ["fatigue", "sweating", "chest_pain"],
    },
    "Malaria": {
        "core": ["chills", "high_fever", "sweating", "vomiting"],
        "secondary": ["headache", "nausea", "muscle_pain"],
    },
    "Dengue": {
        "core": ["high_fever", "headache", "joint_pain", "skin_rash"],
        "secondary": ["nausea", "vomiting", "fatigue", "muscle_pain"],
    },
    "Typhoid": {
        "core": ["high_fever", "headache", "abdominal_pain", "fatigue"],
        "secondary": ["vomiting", "nausea", "diarrhoea"],
    },
    "Migraine": {
        "core": ["headache", "nausea", "visual_disturbances"],
        "secondary": ["vomiting", "fatigue"],
    },
    "Jaundice": {
        "core": ["yellowish_skin", "dark_urine", "fatigue"],
        "secondary": ["vomiting", "abdominal_pain", "high_fever"],
    },
    "Allergy": {
        "core": ["continuous_sneezing", "shivering", "skin_rash"],
        "secondary": ["chills", "headache"],
    },
}

CORE_P, SECONDARY_P, NOISE_P = (0.80, 0.95), (0.30, 0.50), 0.03


def all_symptoms(profiles=PROFILES):
    s = set()
    for p in profiles.values():
        s.update(p["core"], p["secondary"])
    return sorted(s)


def generate(n_per_disease=300, seed=42, profiles=PROFILES):
    rng = np.random.default_rng(seed)
    symptoms = all_symptoms(profiles)
    rows = []
    for disease, p in profiles.items():
        for _ in range(n_per_disease):
            row = {s: int(rng.random() < NOISE_P) for s in symptoms}
            for s in p["core"]:
                row[s] = int(rng.random() < rng.uniform(*CORE_P))
            for s in p["secondary"]:
                row[s] = int(rng.random() < rng.uniform(*SECONDARY_P))
            row["prognosis"] = disease
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def align_to_real(real_df, synth_df, target="prognosis"):
    """Put a real dataset on the same schema as the synthetic one.

    - keep only diseases both datasets know
    - keep only symptoms synthetic knows; add missing ones as 0
    - same column order
    Returns the aligned real dataframe.
    """
    shared = set(real_df[target].str.strip()) & set(synth_df[target])
    real = real_df.copy()
    real[target] = real[target].str.strip()
    real = real[real[target].isin(shared)]
    feats = [c for c in synth_df.columns if c != target]
    for c in feats:
        if c not in real.columns:
            real[c] = 0
    return real[feats + [target]].reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_disease.csv", index=False)
    print(df.shape)
    print(df["prognosis"].value_counts().to_string())
    print("\nSymptoms:", len(all_symptoms()))

    # quick sanity baseline: tree on synthetic, tested on held-out synthetic
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = df.drop(columns="prognosis"), df["prognosis"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xtr, ytr)
    print(f"\nSynthetic->synthetic accuracy: {tree.score(Xte, yte):.3f}")