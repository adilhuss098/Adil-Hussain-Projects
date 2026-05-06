"""
analysis.py
-----------
ML pipeline for pipe failure prediction — City of Kitchener water main data.

Pipeline:
  1. Load & explore data
  2. Pre-processing (encoding, undersampling)
  3. Train four classifiers: Decision Tree, Random Forest, AdaBoost, RUSBoost
  4. Evaluate: confusion matrices, ROC curves, AUC
  5. Predictor importance
  6. Pipe network failure probability map (real GPS coordinates)
  7. Save all figures to figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

DATA_PATH    = Path("data/pipe_network.csv")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

COLORS = {
    "RUSBoost":      "#E63946",
    "AdaBoost":      "#2196F3",
    "Random Forest": "#4CAF50",
    "Decision Tree": "#FF9800",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PIPE FAILURE MODELLING — City of Kitchener Water Mains")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {len(df):,} pipes")
print(f"    Failure rate: {df['failure'].mean():.2%}  ({df['failure'].sum():,} failures)")

# ── Figure 1: Data distributions ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Pipe Network Data Distributions — City of Kitchener",
             fontsize=16, fontweight="bold", y=1.01)

axes[0,0].hist(df["age"].dropna(), bins=40, color="#2196F3",
               edgecolor="white", linewidth=0.5)
axes[0,0].set_xlabel("Age [years]"); axes[0,0].set_ylabel("Count")
axes[0,0].set_title("Age Distribution")

axes[0,1].hist(df["diameter"].dropna(), bins=30, color="#4CAF50",
               edgecolor="white", linewidth=0.5)
axes[0,1].set_xlabel("Diameter [mm]"); axes[0,1].set_ylabel("Count")
axes[0,1].set_title("Diameter Distribution")

axes[0,2].hist(np.log10(df["length"].clip(lower=0.1)), bins=30, color="#FF9800",
               edgecolor="white", linewidth=0.5)
axes[0,2].set_xlabel("Log₁₀ Length [m]"); axes[0,2].set_ylabel("Count")
axes[0,2].set_title("Pipe Length (log scale)")

axes[1,0].hist(df["Failure_Tot"], bins=range(0, 12), color="#E63946",
               edgecolor="white", linewidth=0.5, log=True)
axes[1,0].set_xlabel("Number of Breaks"); axes[1,0].set_ylabel("Count (log scale)")
axes[1,0].set_title("Total Breaks per Pipe")

mat_counts = df["material"].value_counts()
axes[1,1].bar(mat_counts.index, mat_counts.values, color="#9C27B0",
              edgecolor="white")
axes[1,1].set_xlabel("Material"); axes[1,1].set_ylabel("Count")
axes[1,1].set_title("Material Distribution")
axes[1,1].tick_params(axis="x", rotation=45)

cond_fail   = df[df["failure"] == 1]["condition_score"].dropna()
cond_nofail = df[df["failure"] == 0]["condition_score"].dropna()
axes[1,2].hist(cond_nofail, bins=30, alpha=0.6, color="#4CAF50",
               edgecolor="white", linewidth=0.3, label="No failure")
axes[1,2].hist(cond_fail,   bins=30, alpha=0.6, color="#E63946",
               edgecolor="white", linewidth=0.3, label="Failure")
axes[1,2].set_xlabel("Condition Score"); axes[1,2].set_ylabel("Count")
axes[1,2].set_title("Condition Score by Failure Label")
axes[1,2].legend()

plt.tight_layout()
plt.savefig(FIGURES_PATH / "01_data_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure 1 saved: data distributions")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Pre-processing...")

FEATURE_COLS = [
    "age", "diameter", "length",
    "condition_score", "criticality", "pressure_zone",
    "material",
]
TARGET_COL = "failure"

df_clean = df[FEATURE_COLS + [TARGET_COL]].dropna()

df_encoded = pd.get_dummies(df_clean[FEATURE_COLS], columns=["material"])
X = df_encoded.astype(float)
y = df_clean[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"    Train failure rate: {y_train.mean():.2%}")


def stratified_undersample(X, y, ratio=1.0, random_state=42):
    df_temp = X.copy()
    df_temp["__target__"] = y
    majority = df_temp[df_temp["__target__"] == 0]
    minority = df_temp[df_temp["__target__"] == 1]
    n_sample = int(len(minority) * ratio)
    majority_down = resample(majority, n_samples=n_sample,
                             random_state=random_state, replace=False)
    df_bal = pd.concat([majority_down, minority])
    y_bal = df_bal.pop("__target__").values
    return df_bal, y_bal


X_train_bal, y_train_bal = stratified_undersample(X_train, y_train, ratio=1.0)
print(f"    Balanced train: {len(X_train_bal):,} "
      f"(failure rate: {y_train_bal.mean():.2%})")

FEATURE_NAMES = list(X.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Training classifiers...")


class RUSBoostClassifier:
    """
    RUSBoost: AdaBoost with per-iteration Random UnderSampling.
    Seiffert et al. (2010), IEEE Trans. Systems, Man, and Cybernetics.
    """
    def __init__(self, n_estimators=100, learning_rate=1.0, random_state=42):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.estimators_   = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n   = len(y)
        w   = np.ones(n) / n
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y
        for m in range(self.n_estimators):
            pos_idx = np.where(y_arr == 1)[0]
            neg_idx = np.where(y_arr == 0)[0]
            n_samp  = min(len(pos_idx), len(neg_idx))
            chosen_pos = rng.choice(pos_idx, n_samp, replace=False)
            chosen_neg = rng.choice(neg_idx, n_samp, replace=False)
            idx = np.concatenate([chosen_pos, chosen_neg])
            clf = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            clf.fit(X_arr[idx], y_arr[idx],
                    sample_weight=w[idx] / w[idx].sum())
            pred  = clf.predict(X_arr)
            error = np.clip(np.sum(w * (pred != y_arr)), 1e-10, 1 - 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            w = w * np.exp(-alpha * (2 * y_arr - 1) * (2 * pred - 1))
            w /= w.sum()
            self.estimators_.append(clf)
            self.estimator_weights_.append(alpha)
        return self

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        scores = np.zeros(len(X_arr))
        for alpha, clf in zip(self.estimator_weights_, self.estimators_):
            pred    = clf.predict(X_arr)
            scores += alpha * (2 * pred - 1)
        prob1 = 1 / (1 + np.exp(-2 * scores))
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        importances = np.zeros(len(FEATURE_NAMES))
        total_alpha = sum(self.estimator_weights_)
        for alpha, clf in zip(self.estimator_weights_, self.estimators_):
            importances += (alpha / total_alpha) * clf.feature_importances_
        return importances


classifiers = {
    "Decision Tree":  DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                             n_jobs=-1),
    "AdaBoost":       AdaBoostClassifier(
                          estimator=DecisionTreeClassifier(max_depth=1),
                          n_estimators=100, random_state=RANDOM_STATE),
    "RUSBoost":       RUSBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

trained = {}
for name, clf in classifiers.items():
    if name == "RUSBoost":
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train_bal, y_train_bal)
    trained[name] = clf
    print(f"    [✓] {name} trained")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Evaluating classifiers...")

results = {}
for name, clf in trained.items():
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    acc     = accuracy_score(y_test, y_pred)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr_arr, tpr_arr)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) * 100
    fpr = fp / (fp + tn) * 100
    results[name] = {
        "y_pred": y_pred, "y_proba": y_proba,
        "cm": cm, "acc": acc,
        "fpr_arr": fpr_arr, "tpr_arr": tpr_arr, "auc": roc_auc,
        "TPR": tpr, "FPR": fpr,
    }
    print(f"    {name:15s} | Acc: {acc:.3f} | AUC: {roc_auc:.3f} | "
          f"TPR: {tpr:.1f}% | FPR: {fpr:.1f}%")

# ── Figure 2: Confusion Matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrices — Predicted vs Actual (%)",
             fontsize=15, fontweight="bold")

for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm_norm = res["cm"].astype(float)
    cm_norm[0] /= cm_norm[0].sum() / 100
    cm_norm[1] /= cm_norm[1].sum() / 100
    im = ax.imshow(cm_norm, interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=100)
    ax.set_title(f"{name}\nAcc={res['acc']:.3f}  AUC={res['auc']:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Failure", "Failure"])
    ax.set_yticklabels(["No Failure", "Failure"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.1f}%",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm_norm[i, j] > 60 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(FIGURES_PATH / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure 2 saved: confusion matrices")

# ── Figure 3: ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1], [0,1], "k--", linewidth=1.2, label="No Discrimination", alpha=0.6)

for (name, res), ls in zip(results.items(), ["-", "--", "-.", ":"]):
    ax.plot(res["fpr_arr"], res["tpr_arr"],
            color=COLORS[name], linestyle=ls, linewidth=2.2,
            label=f"{name} (AUC = {res['auc']:.2f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Pipe Failure Classification", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)
ax.axhline(0.9, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(0.02, 0.91, "AUC = 0.9 (Outstanding threshold)", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig(FIGURES_PATH / "03_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 3 saved: ROC curves")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTOR IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Computing predictor importance...")

base_features = ["age", "diameter", "length", "condition_score",
                 "criticality", "pressure_zone", "Material"]

def aggregate_importances(importances, feature_names):
    agg = {f: 0.0 for f in base_features}
    for imp, fname in zip(importances, feature_names):
        matched = False
        for base in base_features:
            if fname.lower().startswith(base.lower()) or \
               fname.startswith("material_"):
                key = "Material" if fname.startswith("material_") else base
                agg[key] += imp
                matched = True
                break
    return agg

# Plot DT/RF/AdaBoost together; RUSBoost separately since it concentrates
# entirely on condition_score and would crush the scale of the others
fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                         gridspec_kw={"width_ratios": [3, 1]})
fig.suptitle("Feature Importance by Classifier", fontsize=13, fontweight="bold")

main_clfs = ["Decision Tree", "Random Forest", "AdaBoost"]
x      = np.arange(len(base_features))
width  = 0.25
offset = [-1, 0, 1]

for i, name in enumerate(main_clfs):
    clf = trained[name]
    imp = clf.feature_importances_
    agg = aggregate_importances(imp, FEATURE_NAMES)
    if i == 0:
        sorted_keys = sorted(agg, key=lambda k: agg[k], reverse=True)
    vals = [agg[k] for k in sorted_keys]
    axes[0].bar(x + offset[i] * width, vals, width,
                label=name, color=COLORS[name], alpha=0.85, edgecolor="white")

axes[0].set_xticks(x)
axes[0].set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=10)
axes[0].set_ylabel("Predictor Importance", fontsize=12)
axes[0].set_title("Decision Tree, Random Forest, AdaBoost", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(axis="y", alpha=0.3)

# RUSBoost panel — shows it leans almost entirely on condition_score
rus_agg = aggregate_importances(trained["RUSBoost"].feature_importances_, FEATURE_NAMES)
rus_vals = [rus_agg[k] for k in sorted_keys]
axes[1].bar(x, rus_vals, 0.6, color=COLORS["RUSBoost"], alpha=0.85, edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=10)
axes[1].set_ylabel("Predictor Importance", fontsize=12)
axes[1].set_title("RUSBoost", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(FIGURES_PATH / "04_predictor_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 4 saved: predictor importance")


# ══════════════════════════════════════════════════════════════════════════════
# 6. NETWORK FAILURE MAP — real GPS coordinates (Kitchener, Ontario)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Generating network failure probability maps...")

map_clf = trained["AdaBoost"]

# Pipes with GPS coordinates (those that appear in the breaks dataset)
df_map = df.dropna(subset=["longitude", "latitude"]).copy()
df_map_clean = df_map[FEATURE_COLS].dropna()
df_map = df_map.loc[df_map_clean.index]

df_map_enc = pd.get_dummies(df_map_clean[FEATURE_COLS], columns=["material"])
df_map_enc = df_map_enc.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)

cmap = plt.cm.RdYlGn_r

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Predicted Failure Probability — Kitchener, Ontario Water Mains",
             fontsize=13, fontweight="bold", y=1.01)

for ax, delta_yr, title in zip(axes, [0, 5, 10],
                                ["Current state", "+5 years", "+10 years"]):
    df_fut = df_map_enc.copy()
    df_fut["age"] = df_fut["age"] + delta_yr

    proba = map_clf.predict_proba(df_fut)[:, 1]

    sc = ax.scatter(
        df_map["longitude"], df_map["latitude"],
        c=proba, cmap=cmap, s=8, alpha=0.8,
        vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_facecolor("#e8e8e8")
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate("N ↑", xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, va="top")
    plt.colorbar(sc, ax=ax, label="Failure probability",
                 fraction=0.046, pad=0.04)
    ax.grid(False)

plt.tight_layout()
plt.savefig(FIGURES_PATH / "05_network_failure_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 5 saved: network failure maps")

# ── Figure 6: Failure probability histograms ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Failure Probability Distribution — Full Network",
             fontsize=13, fontweight="bold")

X_full_enc = pd.get_dummies(df_clean[FEATURE_COLS], columns=["material"])
X_full_enc = X_full_enc.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)

for ax, delta_yr, title in zip(axes, [0, 10], ["0 years (current)", "10 years"]):
    X_fut = X_full_enc.copy()
    X_fut["age"] = X_fut["age"] + delta_yr
    proba = map_clf.predict_proba(X_fut)[:, 1]
    ax.hist(proba * 100, bins=25, color="#E63946", edgecolor="white",
            linewidth=0.5, range=(0, 100))
    ax.set_xlabel("Failure probability [%]")
    ax.set_ylabel("Number of pipes")
    ax.set_title(f"Failure probability histogram\n{title}")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / "06_failure_probability_histograms.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 6 saved: failure probability histograms")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Classifier':<18} {'Accuracy':>10} {'AUC':>8} {'TPR (%)':>10} {'FPR (%)':>10}")
print("-" * 58)
for name, res in results.items():
    print(f"{name:<18} {res['acc']:>10.3f} {res['auc']:>8.3f} "
          f"{res['TPR']:>10.1f} {res['FPR']:>10.1f}")
print("-" * 58)
print("\n[✓] All figures saved to figures/")
print("[✓] Analysis complete.")
