# ⏱️ Survival Analysis -> Breast Cancer Recurrence Modelling

Standard classification asks whether an event will happen. Survival analysis asks when, and crucially, it handles the fundamental challenge of patients who haven't experienced the event yet. This project builds a complete survival analysis pipeline on a real breast cancer dataset, from Kaplan-Meier curves through to a multi-variable Cox regression model with clinical interpretation.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Dataset** | GBSG Breast Cancer Survival Dataset |
| **Records** | 686 patients |
| **Follow-up** | Up to 7.3 years |
| **Event** | Cancer recurrence |
| **Censored patients** | 387 (56.4%) — still alive or lost to follow-up |
| **Libraries** | `lifelines` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn` |

---

## 🗂️ The Dataset

The German Breast Cancer Study Group (GBSG) dataset follows 686 breast cancer patients over time, recording whether and when their cancer recurred after initial treatment. The central challenge is that 387 patients (56.4%) did not experience recurrence during the study period, they are censored. You cannot delete them (that would bias results toward faster-recurring cases) and you cannot treat them as non-events. Survival analysis handles this mathematically using the partial information each censored patient provides.

**Features:** age · menopausal status · tumour size · tumour grade · positive lymph nodes · progesterone receptor · estrogen receptor · hormonal treatment

---

## 📐 Methods

### Kaplan-Meier Estimator
Non-parametric estimation of the survival function S(t), the probability of remaining recurrence-free beyond time t. Updates at each event time using only patients still under observation, handling censored patients automatically. Produces the characteristic step-function survival curve.

**Results:**
- Median recurrence-free survival: **4.95 years**
- 1-year survival: **91.6%**
- 3-year survival: **64.3%**
- 5-year survival: **49.2%**

### Log-Rank Test
Statistical test comparing survival curves between groups. Produces a p-value indicating whether the observed survival difference is statistically significant or could be explained by chance.

| Comparison | p-value | Significant? |
|------------|---------|-------------|
| Grade 1 vs Grade 3 | 0.000009 | ✓ Yes -> extremely strong |
| With vs Without Hormonal Treatment | 0.003427 | ✓ Yes |
| 1–3 nodes vs 7+ nodes | < 0.000001 | ✓ Yes -> strongest effect |

### Cox Proportional Hazards Model
Multivariable regression modelling the simultaneous effect of all features on recurrence risk. Output is a Hazard Ratio (HR) for each feature, the multiplicative change in instantaneous recurrence risk per unit increase.

| Feature | HR | Direction | Significant? |
|---------|-----|-----------|-------------|
| Lymph nodes | 1.287 | ↑ risk | ✓ p < 0.001 |
| Tumour grade | 1.303 | ↑ risk | ✓ p < 0.01 |
| Progesterone receptor | 0.767 | Protective | ✓ p < 0.001 |
| Hormonal treatment | 0.762 | Protective | ✓ p < 0.05 |

**C-index: 0.688** -> the model correctly ranks 68.8% of patient pairs by recurrence risk.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_dataset_overview.png` | Follow-up times, event distribution, tumour grade breakdown |
| `plot2_km_overall.png` | Overall Kaplan-Meier curve with median and confidence interval |
| `plot3_km_by_grade.png` | KM curves by tumour grade with log-rank test |
| `plot4_km_treatment.png` | KM curves comparing hormonal treatment groups |
| `plot5_hazard_ratios.png` | Forest plot of Cox model hazard ratios |
| `plot6_predicted_survival.png` | Predicted survival for low-risk vs high-risk patient profiles |
| `plot7_km_nodes.png` | KM curves by lymph node involvement |

---

## 🔍 Key Findings

Lymph node involvement is the strongest prognostic factor -> patients with 7+ positive nodes have dramatically worse survival than those with 1–3 nodes, and this difference is statistically extremely significant. Grade 3 tumours recur far earlier than grade 1, with curves separating sharply within the first two years. Hormonal treatment shows a meaningful and statistically significant protective effect, patients on treatment have consistently higher survival probability at every time point.

The C-index of 0.688, while below 0.70, is clinically reasonable for a dataset of this size. Progesterone receptor status being protective is biologically meaningful, hormone receptor-positive tumours are generally more responsive to targeted therapies.

---

## 📂 Repository Structure

```
survival-analysis/
├── gbsg2.csv
├── survival_analysis.py
├── plot1_dataset_overview.png
├── plot2_km_overall.png
├── plot3_km_by_grade.png
├── plot4_km_treatment.png
├── plot5_hazard_ratios.png
├── plot6_predicted_survival.png
├── plot7_km_nodes.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/survival-analysis.git
cd survival-analysis
pip3 install lifelines scikit-learn pandas matplotlib seaborn
python3 survival_analysis.py
```

**Note:** The dataset is loaded from `gbsg2.csv`. Download it from the lifelines GitHub repository or use the provided file.

---

## 📚 Skills Developed

- Understanding censoring -> what it is, why it cannot be ignored, and how survival analysis handles it
- Fitting and plotting Kaplan-Meier survival curves with confidence intervals
- Interpreting median survival time and survival probability at fixed time points
- Log-rank test -> null hypothesis, p-value interpretation, and clinical significance vs statistical significance
- Cox Proportional Hazards -> hazard ratios, confidence intervals, and the proportional hazards assumption
- Reading and building forest plots for hazard ratio visualisation
- C-index as the survival equivalent of AUC-ROC

---

## 🗺️ Learning Roadmap

_**Project 4 of 10**_ -> a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | **Survival Analysis** ← | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
