# ⏱️ Survival Analysis — Cancer Patient Prognosis

A Machine Learning project applying survival analysis techniques to predict cancer recurrence over time using the GBSG Breast Cancer dataset. This is **Project 4 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Dataset | GBSG Breast Cancer Survival Dataset (via lifelines) |
| Patients | 686 breast cancer patients |
| Features | Age, tumour size, grade, lymph nodes, hormone receptors, treatment |
| Target | Time to recurrence + event indicator (1=recurrence, 0=censored) |
| Techniques | Kaplan-Meier, Log-rank test, Cox Proportional Hazards, C-index |
| Libraries | `lifelines`, `scikit-learn`, `pandas`, `matplotlib` |

---

## 🧠 Key Concepts

### Censoring
Not all patients experienced recurrence during follow-up — some were still alive, others dropped out. These **censored** patients cannot be ignored or removed. Survival analysis handles them by using the information "this patient survived at least X months."

### Kaplan-Meier Estimator
Non-parametric method for estimating the survival function S(t) — the probability of surviving beyond time t. Handles censored data and produces the characteristic step-function survival curves.

### Log-Rank Test
Statistical test comparing survival curves between groups. Produces a p-value — if p < 0.05, the survival difference is statistically significant.

### Cox Proportional Hazards Model
Models the simultaneous effect of multiple features on survival. Outputs **Hazard Ratios (HR)**:
- HR > 1 = feature increases recurrence risk
- HR < 1 = feature is protective (decreases risk)

### C-Index (Concordance Index)
The survival equivalent of AUC-ROC. Measures how well the model ranks patients by risk. C-index = 0.5 is random, 1.0 is perfect, ≥ 0.70 is considered good.

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| Dataset Overview | Follow-up times, event distribution, grade distribution |
| Overall KM Curve | Population-level survival with median and confidence interval |
| KM by Tumour Grade | Survival comparison across grades 1, 2, 3 with log-rank test |
| KM by Treatment | Effect of hormonal treatment on survival |
| Hazard Ratio Forest Plot | Cox model HR + 95% CI for all features |
| Predicted Survival | Low-risk vs high-risk patient profile comparison |
| KM by Lymph Nodes | Survival by 0 / 1-3 / 4+ positive nodes |

---

## 📂 Project Structure

```
survival-analysis/
├── survival_analysis.py          # Main script
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

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/survival-analysis.git
cd survival-analysis
```

**2. Install dependencies**
```bash
pip3 install lifelines scikit-learn pandas matplotlib seaborn
```

**3. Run the script**
```bash
python3 survival_analysis.py
```

---

## 🔬 Connection to Research Proposal

This project directly implements tools described in **Aim 3** of a computational biology research proposal on colorectal cancer risk prediction in the Emirati population:

> *"Cox proportional hazards for survival to inform prognosis"*
> *"C-index ≥ 0.70 for survival prediction"*

The same Kaplan-Meier and Cox techniques will be applied to Emirati CRC patient cohorts, comparing survival across genomic risk groups defined by population-specific polygenic risk scores.

---

## 📚 What I Learned

- What **censoring** is and why it makes survival analysis fundamentally different from classification
- How the **Kaplan-Meier estimator** handles censored data to estimate survival probabilities
- How to use the **log-rank test** to determine if survival differences between groups are statistically significant
- How **Cox Proportional Hazards** models multiple features simultaneously and produces hazard ratios
- How to interpret **Hazard Ratios** — direction, magnitude, and confidence intervals
- What the **C-index** measures and why it is the correct evaluation metric for survival models
- How to build a **forest plot** to visualise feature effects in a Cox model

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | ✅ Complete |
| 5 | Customer Segmentation | 🔜 Next |
| 6 | Gene Expression Clustering | ⏳ Upcoming |
| 7 | Explainable AI with SHAP | ⏳ Upcoming |
| 8 | Counterfactual Explanations | ⏳ Upcoming |
| 9 | Multi-Modal Data Fusion | ⏳ Upcoming |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
