# Pipe Failure Modelling for Water Distribution Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## The Problem

Water utilities manage thousands of kilometres of buried pipes. When a pipe fails it causes road closures, property damage, and service outages — and emergency repairs cost significantly more than planned ones. The challenge is that most pipes look fine from the surface: they fail without warning.

This project builds a machine learning classifier to identify **which pipes are most likely to fail**, so utilities can prioritise inspections and schedule proactive replacements before a failure happens.

---

## Dataset

Real open data from the **City of Kitchener, Ontario** (GIS open data portal):

- **Water_Mains.csv** — 15,903 pipes covering the city's full distribution network, including material, diameter, length, installation date, condition score, and pressure zone
- **Water_Main_Breaks.csv** — 2,766 recorded break incidents from 1985 to 2023

Each pipe is labelled as **failed** (has had at least one recorded break) or **healthy** (no recorded breaks). This gives a dataset of 15,768 pipes with a 21.6% failure rate — higher than typical studies because Kitchener has a large proportion of cast iron and ductile iron mains installed in the 1950s–70s.

Features used for prediction: **age, diameter, length, material, condition score, criticality rating, pressure zone.**

---

## Approach

Four classifiers are trained and compared:

- **Decision Tree** — single tree, used as a baseline
- **Random Forest** — ensemble of 100 trees (bagging)
- **AdaBoost** — boosting with adaptive sample weighting
- **RUSBoost** — AdaBoost with random undersampling at each iteration, specifically designed for imbalanced datasets

The dataset has a ~4:1 healthy-to-failed ratio. To stop classifiers from just predicting everything as healthy, stratified undersampling is applied to the training set for DT, RF, and AdaBoost. RUSBoost handles this internally. Train/test split is 50/50.

---

## Results

| Classifier | Accuracy | AUC | TPR | FPR |
|---|---|---|---|---|
| **Random Forest** | 0.764 | 0.848 | 75.0% | 23.2% |
| AdaBoost | 0.750 | 0.821 | 72.8% | 24.4% |
| Decision Tree | 0.747 | 0.786 | 73.1% | 24.9% |
| RUSBoost | 0.791 | 0.677 | 49.7% | 12.8% |

**Random Forest is the best overall classifier** with AUC 0.848 — it correctly identifies 75% of pipes that will fail (TPR) while raising false alarms on 23% of healthy pipes (FPR). In this domain, missing a real failure (false negative) is more costly than a false alarm, so TPR matters more than raw accuracy.

RUSBoost had the highest accuracy but the lowest TPR — it plays it safe by predicting fewer failures, which reduces false alarms but also means more real failures get missed.

---

## Figures

### 1. Data Distributions
![Data Distributions](figures/01_data_distributions.png)

Overview of the dataset. The age distribution peaks around 10–25 years reflecting Kitchener's recent PVC rollout, with a secondary hump around 50–60 years from the older cast iron and ductile iron mains. The break count per pipe drops off exponentially — most pipes have 0–2 breaks, but some old cast iron mains have 8+. The condition score overlay (bottom right) shows that failed and healthy pipes both score highly on this metric, meaning condition score alone isn't a strong predictor but is still informative alongside other features.

---

### 2. Confusion Matrices
![Confusion Matrices](figures/02_confusion_matrices.png)

For each classifier, shows what percentage of actual failures were correctly caught (top-right cell = TPR) and what percentage of healthy pipes were wrongly flagged (bottom-right cell = FPR). A perfect classifier would have 100% top-right and 0% bottom-right. Random Forest and AdaBoost strike the best balance — they catch ~73–75% of real failures while keeping false alarms around 23–24%.

---

### 3. ROC Curves
![ROC Curves](figures/03_roc_curves.png)

The ROC curve shows the trade-off between catching failures (TPR) and generating false alarms (FPR) across all decision thresholds. A curve hugging the top-left corner is better. AUC (area under the curve) summarises this — higher is better, with 1.0 being perfect and 0.5 being no better than random. Random Forest (AUC 0.848) clearly outperforms the others, followed by AdaBoost (0.821). RUSBoost's lower AUC (0.677) reflects that its conservative predictions, while accurate, miss too many real failures.

---

### 4. Feature Importance
![Feature Importance](figures/04_predictor_importance.png)

Shows which features each classifier relies on most. For DT, RF, and AdaBoost (left panel): **age and pipe length** are consistently the most important features — older and longer pipes have accumulated more wear and are harder to access for maintenance. Condition score matters more for AdaBoost. The RUSBoost panel (right) is shown separately because it converges almost entirely on condition score — this is a characteristic of the RUS undersampling approach, where the balanced subsets at each iteration make condition score the single most discriminative split.

---

### 5. Network Failure Probability Map
![Failure Maps](figures/05_network_failure_map.png)

The AdaBoost model is applied to every pipe with a GPS coordinate (those appearing in the break records) and each pipe is coloured by its predicted failure probability — green = low risk, red = high risk. The three panels show predictions for the current pipe ages, and if every pipe were 5 or 10 years older. The city's older central areas and legacy cast iron zones show higher risk (more orange/red), while newer PVC infrastructure in outer areas stays greener. The shift towards higher probabilities over time is visible as the colour distribution warms.

---

### 6. Failure Probability Histograms
![Histograms](figures/06_failure_probability_histograms.png)

Across all 15,768 pipes in the network, this shows how predicted failure probabilities are distributed — now and in 10 years. The distribution shifts rightward over time as pipes age, meaning the model predicts a gradual increase in overall network risk. This kind of output is directly usable for rehabilitation planning: a utility could use the 10-year histogram to estimate how many pipes will cross a risk threshold and budget accordingly.

---

## Conclusion

Random Forest gave the best results (AUC 0.848, TPR 75%) on this real dataset. The key predictors were **age and pipe length** — consistent with what you'd expect from infrastructure degradation. Older pipes in Kitchener's city centre are at highest risk, and the model predicts a measurable increase in network-wide failure probability over the next decade as those pipes continue to age.

The approach is directly transferable to any water utility that maintains a GIS asset database with break history — no additional data collection required.

---

## How to run

```bash
git clone https://github.com/adilhuss098/Adil-Hussain-Projects.git
cd Adil-Hussain-Projects
pip install -r requirements.txt
python prepare_real_data.py   # builds data/pipe_network.csv from raw files
jupyter notebook pipe_failure_modelling.ipynb
```

or as a script:
```bash
python analysis.py
```

---

## References

- Seiffert, C. et al. (2010). *RUSBoost: A hybrid approach to alleviating class imbalance.* IEEE Trans. Systems, Man, and Cybernetics, 40(1), 185–197.
- Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
- City of Kitchener Open Data Portal — Water Mains & Water Main Breaks datasets.

---

MIT License
