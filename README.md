# Pipe Failure Modelling for Water Distribution Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

ML project predicting pipe failures in water distribution networks. Built and tested four classifiers on a dataset of ~40k pipes (8.6% failure rate) to see which best identifies pipes at risk.

---

## Results

| Classifier | Accuracy | AUC | TPR | FPR |
|---|---|---|---|---|
| **RUSBoost** | 0.724 | 0.617 | 48.7% | 25.4% |
| AdaBoost | 0.665 | 0.671 | 60.2% | 32.9% |
| Random Forest | 0.621 | 0.648 | 59.8% | 37.7% |
| Decision Tree | 0.658 | 0.599 | 54.3% | 33.1% |

RUSBoost came out on top for accuracy — it handles the class imbalance well since it undersamples the majority class at each boosting iteration.

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

### Network Failure Probability Maps (Present / +5yr / +10yr)
![Failure Maps](figures/05_network_failure_map.png)

### Failure Probability Histograms
![Histograms](figures/06_failure_probability_histograms.png)

---

## How to run

```bash
git clone https://github.com/adilhuss098/Adil-Hussain-Projects.git
cd Adil-Hussain-Projects
pip install -r requirements.txt
python generate_data.py      # creates data/pipe_network.csv
jupyter notebook pipe_failure_modelling.ipynb
```

or just run everything as a script:
```bash
python analysis.py
```

---

## Methods

**Dataset:** 39,637 pipes, 9 material types (AC, CI, DI, GRP, PP, PE, PVC, Pb, ST), 12 features including age, diameter, length, pressure, valve counts, and failure history. 50/50 train/test split.

**Classifiers:**
- Decision Tree (CART, Gini impurity) — baseline
- Random Forest — 100 trees
- AdaBoost — adaptive sample weighting
- RUSBoost — AdaBoost + random undersampling per iteration for imbalanced data

**Class imbalance:** ~9:1 healthy-to-failed ratio. Used PSRS (stratified undersampling) for DT/RF/AdaBoost, and RUS embedded in each RUSBoost iteration.

**Evaluation:** confusion matrix, ROC/AUC, predictor importance.

---

## Repo structure

```
├── pipe_failure_modelling.ipynb   <- main notebook
├── analysis.py                    <- script version
├── generate_data.py               <- synthetic data generator
├── data/pipe_network.csv
├── figures/
└── requirements.txt
```

---

MIT License
