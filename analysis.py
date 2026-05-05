"""
analysis.py
-----------
ML pipeline for pipe failure prediction in water distribution networks.

Pipeline:
  1. Load & explore data
  2. Pre-processing (curation, encoding, subsampling)
  3. Train four classifiers: Decision Tree, Random Forest, AdaBoost, RUSBoost
  4. Evaluate: confusion matrices, ROC curves, AUC
  5. Predictor importance analysis
  6. Pipe network failure probability maps (current, +5yr, +10yr)
  7. Save all figures to figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/pipe_network.csv")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Colour palette ─────────────────────────────────────────────────────────────
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
print("PIPE FAILURE MODELLING — Water Distribution Networks")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {len(df):,} pipes, {df.columns.tolist()}")
print(f"    Failure rate: {df['failure'].mean():.2%}  ({df['failure'].sum():,} failures)")

# ── Figure 1: Distribution histograms (replicating paper Figure 2) ────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Pipe Network Data Distributions", fontsize=16, fontweight="bold", y=1.01)

# Age
axes[0,0].hist(df["age"], bins=40, color="#2196F3", edgecolor="white", linewidth=0.5)
axes[0,0].set_xlabel("Age [years]"); axes[0,0].set_ylabel("Count")
axes[0,0].set_title("Age Distribution")

# Diameter
axes[0,1].hist(df["diameter"], bins=30, color="#4CAF50", edgecolor="white", linewidth=0.5)
axes[0,1].set_xlabel("Diameter [mm]"); axes[0,1].set_ylabel("Count")
axes[0,1].set_title("Diameter Distribution")

# Log length
axes[0,2].hist(np.log10(df["length"]+0.01), bins=30, color="#FF9800", edgecolor="white", linewidth=0.5)
axes[0,2].set_xlabel("Log₁₀ Length [m]"); axes[0,2].set_ylabel("Count")
axes[0,2].set_title("Pipe Length (log scale)")

# Failures
axes[1,0].hist(df["Failure_Tot"], bins=range(0,10), color="#E63946", edgecolor="white",
               linewidth=0.5, log=True)
axes[1,0].set_xlabel("Number of Failures"); axes[1,0].set_ylabel("Count (log scale)")
axes[1,0].set_title("Total Failures per Pipe")

# Material
mat_counts = df["material"].value_counts()
axes[1,1].bar(mat_counts.index, mat_counts.values, color="#9C27B0", edgecolor="white")
axes[1,1].set_xlabel("Material"); axes[1,1].set_ylabel("Count")
axes[1,1].set_title("Material Distribution")
axes[1,1].tick_params(axis="x", rotation=45)

# Type
type_counts = df["type"].value_counts()
axes[1,2].bar(type_counts.index, type_counts.values,
              color=["#00BCD4","#FF5722","#607D8B"])
axes[1,2].set_xlabel("Pipe Type"); axes[1,2].set_ylabel("Count")
axes[1,2].set_title("Pipe Type\n(HC=House Connection, DP=Distribution, HY=Hydrant)")

plt.tight_layout()
plt.savefig(FIGURES_PATH / "01_data_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Figure 1 saved: data distributions")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Pre-processing...")

# Sub-classify materials by manufacturing era
def subclassify_material(row):
    mat, yr = row["material"], row["install_year"]
    if mat == "CI":
        if yr < 1930: return "CI_1G"
        elif yr < 1970: return "CI_2G"
        else: return "CI_3G"
    elif mat == "DI":
        if yr < 1980: return "DI_1G"
        elif yr < 2000: return "DI_2G"
        else: return "DI_3G"
    elif mat == "ST":
        if yr < 1940: return "ST_1G"
        elif yr < 1980: return "ST_2G"
        elif yr < 2000: return "ST_3G"
        else: return "ST_4G"
    elif mat == "PE":
        if yr < 1975: return "PE_1G"
        elif yr < 1995: return "PE_2G"
        else: return "PE_3G"
    return mat

df["material_sub"] = df.apply(subclassify_material, axis=1)

# Feature columns
FEATURE_COLS = [
    "age", "diameter", "length", "pressure",
    "HC_Str", "HY_Str", "Valves_Tot", "Valves_St",
    "Failure_Tot", "Failure_New",
    "material_sub", "type"
]
TARGET_COL = "failure"

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[FEATURE_COLS], columns=["material_sub", "type"])

X = df_encoded.astype(float)
y = df[TARGET_COL].values

# 50/50 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"    Train failure rate: {y_train.mean():.2%}")

# ── Stratified Random Undersampling (PSRS) for DT/RF/AdaBoost training ────────
def stratified_undersample(X, y, ratio=1.0, random_state=42):
    """Balance classes by undersampling the majority class."""
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
print(f"    Balanced train set: {len(X_train_bal):,} "
      f"(failure rate: {y_train_bal.mean():.2%})")

FEATURE_NAMES = list(X.columns)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Training classifiers...")

# ── RUSBoost: AdaBoost with per-iteration Random UnderSampling ─────────────────
class RUSBoostClassifier:
    """
    RUSBoost (Seiffert et al., 2010).
    AdaBoost where each weak learner is trained on a randomly
    undersampled version of the data — handles class imbalance well.
    """
    def __init__(self, n_estimators=100, learning_rate=1.0,
                 random_state=42):
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
            # Random undersample majority class for this iteration
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
            error = np.sum(w * (pred != y_arr))
            error = np.clip(error, 1e-10, 1 - 1e-10)
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
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "AdaBoost":       AdaBoostClassifier(
                          estimator=DecisionTreeClassifier(max_depth=1),
                          n_estimators=100, random_state=RANDOM_STATE),
    "RUSBoost":       RUSBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

trained = {}
for name, clf in classifiers.items():
    # RUSBoost trains on full (skewed) data; others on balanced
    if name == "RUSBoost":
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train_bal, y_train_bal)
    trained[name] = clf
    print(f"    [✓] {name} trained")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE — Confusion Matrices
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

    # Rates as %
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) * 100
    fpr = fp / (fp + tn) * 100
    tnr = tn / (tn + fp) * 100
    fnr = fn / (fn + tp) * 100

    results[name] = {
        "y_pred": y_pred, "y_proba": y_proba,
        "cm": cm, "acc": acc,
        "fpr_arr": fpr_arr, "tpr_arr": tpr_arr, "auc": roc_auc,
        "TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr,
    }
    print(f"    {name:15s} | Acc: {acc:.3f} | AUC: {roc_auc:.3f} | "
          f"TPR: {tpr:.1f}% | FPR: {fpr:.1f}%")


# ── Figure 2: Confusion Matrices (rate %) ──────────────────────────────────────
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


# ── Figure 3: ROC Curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1], [0,1], "k--", linewidth=1.2, label="No Discrimination", alpha=0.6)

line_styles = ["-", "--", "-.", ":"]
for (name, res), ls in zip(results.items(), line_styles):
    ax.plot(res["fpr_arr"], res["tpr_arr"],
            color=COLORS[name], linestyle=ls, linewidth=2.2,
            label=f"{name} (AUC = {res['auc']:.2f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Pipe Failure Classification",
             fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)

# Annotate AUC threshold lines from paper
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

# Map one-hot features back to original names for cleaner display
def aggregate_importances(importances, feature_names):
    """Sum importances of one-hot-encoded features back to original feature."""
    base_features = [
        "age", "diameter", "length", "pressure",
        "HC_Str", "HY_Str", "Valves_Tot", "Valves_St",
        "Failure_Tot", "Failure_New", "Material", "Type"
    ]
    agg = {f: 0.0 for f in base_features}
    for imp, fname in zip(importances, feature_names):
        matched = False
        for base in base_features:
            if fname.startswith(base.replace(" ", "").lower()) or \
               fname.lower().startswith(base.lower()):
                agg[base] += imp
                matched = True
                break
        if not matched:
            for base in ["material_sub", "type"]:
                if fname.startswith(base):
                    key = "Material" if base == "material_sub" else "Type"
                    agg[key] += imp
                    matched = True
                    break
    return agg

fig, ax = plt.subplots(figsize=(12, 6))

x      = np.arange(12)
width  = 0.2
names  = list(trained.keys())
offset = [-1.5, -0.5, 0.5, 1.5]

for i, (name, clf) in enumerate(trained.items()):
    imp = clf.feature_importances_
    agg = aggregate_importances(imp, FEATURE_NAMES)
    # Sort by RUSBoost order (first classifier, matching paper Figure 4)
    if i == 0:
        sorted_keys = sorted(agg, key=lambda k: agg[k], reverse=True)
    vals = [agg[k] for k in sorted_keys]
    ax.bar(x + offset[i] * width, vals, width,
           label=name, color=COLORS[name], alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Predictor Importance", fontsize=12)
ax.set_title("Feature Importance by Classifier\n(sorted by RUSBoost importance)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "04_predictor_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 4 saved: predictor importance")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPE NETWORK FAILURE MAP (current, +5yr, +10yr)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Generating network failure probability maps...")

import matplotlib.collections as mc

# Build a synthetic street network (grid of pipe segments)
np.random.seed(7)
GRID_W, GRID_H = 520, 420

# Irregular street spacing to look like a real city
x_streets = np.sort(np.concatenate([
    np.linspace(20, GRID_W - 20, 18),
    np.random.uniform(30, GRID_W - 30, 6)
]))
y_streets = np.sort(np.concatenate([
    np.linspace(20, GRID_H - 20, 14),
    np.random.uniform(30, GRID_H - 30, 5)
]))
x_streets = np.unique(np.round(x_streets, 1))
y_streets = np.unique(np.round(y_streets, 1))

# Each block edge between two intersections is one pipe segment
segs = []
for y in y_streets:
    for i in range(len(x_streets) - 1):
        segs.append(((x_streets[i], y), (x_streets[i+1], y)))
for x in x_streets:
    for i in range(len(y_streets) - 1):
        segs.append(((x, y_streets[i]), (x, y_streets[i+1])))

n_segs = len(segs)
idx = np.random.choice(len(df), n_segs, replace=True)
df_net = df.iloc[idx].copy().reset_index(drop=True)
df_net["material_sub"] = df_net.apply(
    lambda r: (lambda mat, yr:
        "CI_1G" if mat == "CI" and yr < 1930 else
        "CI_2G" if mat == "CI" and yr < 1970 else
        "CI_3G" if mat == "CI" else
        "DI_1G" if mat == "DI" and yr < 1980 else
        "DI_2G" if mat == "DI" and yr < 2000 else
        "DI_3G" if mat == "DI" else
        "ST_1G" if mat == "ST" and yr < 1940 else
        "ST_2G" if mat == "ST" and yr < 1980 else
        "ST_3G" if mat == "ST" and yr < 2000 else
        "ST_4G" if mat == "ST" else
        "PE_1G" if mat == "PE" and yr < 1975 else
        "PE_2G" if mat == "PE" and yr < 1995 else
        "PE_3G" if mat == "PE" else mat
    )(r["material"], r["install_year"]),
    axis=1
)
df_net_enc = pd.get_dummies(df_net[FEATURE_COLS], columns=["material_sub", "type"])
df_net_enc = df_net_enc.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)

map_clf = trained["AdaBoost"]  # best-calibrated probabilities for visualisation

# ── Figure 5: Network map — layout matches current + future views ─────────────
fig = plt.figure(figsize=(18, 8))
gs_fig = fig.add_gridspec(1, 3, wspace=0.35)

cmap = plt.cm.RdYlGn_r

for col, delta_yr, title in zip([0, 1, 2], [0, 5, 10],
                                 ["Current state", "+5 years", "+10 years"]):
    df_fut = df_net_enc.copy()
    df_fut["age"] = df_fut["age"] + delta_yr
    proba = map_clf.predict_proba(df_fut)[:, 1]

    colors = cmap(proba)
    lc = mc.LineCollection(segs, colors=colors, linewidths=1.4, alpha=0.9)

    ax = fig.add_subplot(gs_fig[col])
    ax.add_collection(lc)
    ax.set_xlim(0, GRID_W); ax.set_ylim(0, GRID_H)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_facecolor("#f5f5f5")
    ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Failure probability [0–1]",
                 fraction=0.046, pad=0.04)

fig.suptitle("Pipe Network Failure Probability — AdaBoost Predictions",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig(FIGURES_PATH / "05_network_failure_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Figure 5 saved: network failure maps")


# ── Figure 6: Failure probability histograms ──────────────────────────────────
X_full_enc = pd.get_dummies(df[FEATURE_COLS], columns=["material_sub", "type"])
X_full_enc = X_full_enc.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Failure Probability Distribution — Full Network",
             fontsize=13, fontweight="bold")

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
# 7. SUMMARY TABLE
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
