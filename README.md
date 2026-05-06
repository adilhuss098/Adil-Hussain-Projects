# Pipe Failure Modelling for Water Distribution Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

ML project predicting water main failures using real municipal data from the City of Kitchener, Ontario (open data portal). Four classifiers benchmarked on 15,768 pipes with recorded break history going back to 1985.

---

## Results

| Classifier | Accuracy | AUC | TPR | FPR |
|---|---|---|---|---|
| **Random Forest** | 0.764 | 0.848 | 75.0% | 23.2% |
| AdaBoost | 0.750 | 0.821 | 72.8% | 24.4% |
| Decision Tree | 0.747 | 0.786 | 73.1% | 24.9% |
| RUSBoost | 0.791 | 0.677 | 49.7% | 12.8% |

Random Forest had the best AUC (0.848) and TPR — it catches the most at-risk pipes. RUSBoost had the best accuracy but lower TPR, which matters less here since missing a real failure is more costly than a false alarm.

---

## Visualisations

### Data Distributions
![Data Distributions](figures/01_data_distributions.png)

### Confusion Matrices
![Confusion Matrices](figures/02_confusion_matrices.png)

### ROC Curves
![ROC Curves](figures/03_roc_curves.png)

### Feature Importance
![Feature Importance](figures/04_predictor_importance.png)

### Network Failure Probability Map — Kitchener, Ontario (Present / +5yr / +10yr)
![Failure Maps](figures/05_network_failure_map.png)

### Failure Probability Histograms
![Histograms](figures/06_failure_probability_histograms.png)

---

## Dataset

Real open data from the City of Kitchener GIS portal:
- **Water_Mains.csv** — 15,903 pipes with material, diameter, length, install date, condition score
- **Water_Main_Breaks.csv** — 2,766 recorded break incidents (1985–2023)
- **Failure label** — binary: pipe has had at least one recorded break

Failure rate: 21.6% — higher than typical studies because Kitchener has a significant proportion of older cast iron and ductile iron mains installed in the 1950s–70s.

Features used: age, diameter, length, material, condition score, criticality, pressure zone.

---

## How to run

```bash
git clone https://github.com/adilhuss098/Adil-Hussain-Projects.git
cd Adil-Hussain-Projects
pip install -r requirements.txt
python prepare_real_data.py   # builds data/pipe_network.csv from raw files
jupyter notebook pipe_failure_modelling.ipynb
```

or run as a script:
```bash
python analysis.py
```

---

## Methods

**Classifiers:**
- Decision Tree (CART, Gini impurity) — baseline
- Random Forest — 100 trees, best overall AUC
- AdaBoost — adaptive sample weighting
- RUSBoost — AdaBoost + random undersampling per iteration for imbalanced data

**Class imbalance:** ~78:22 healthy-to-failed ratio. Used stratified undersampling (PSRS) for DT/RF/AdaBoost; RUS embedded per iteration in RUSBoost.

**Evaluation:** confusion matrix, ROC/AUC, feature importance.

**Network map:** AdaBoost model applied to all pipes with GPS coordinates — shows real Kitchener pipe locations coloured by predicted failure probability for current state and 5/10-year projections.

---

## Repo structure

```
├── pipe_failure_modelling.ipynb   <- main notebook
├── analysis.py                    <- script version
├── prepare_real_data.py           <- builds dataset from raw files
├── data/
│   ├── pipe_network.csv           <- processed dataset
│   └── raw/                       <- City of Kitchener open data
├── figures/
└── requirements.txt
```

---

## References

- Seiffert, C. et al. (2010). *RUSBoost: A hybrid approach to alleviating class imbalance.* IEEE Trans. Systems, Man, and Cybernetics, 40(1), 185–197.
- Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
- City of Kitchener Open Data Portal — Water Mains & Water Main Breaks datasets.

---

MIT License
