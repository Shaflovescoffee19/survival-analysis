# ============================================================
# PROJECT 4: Survival Analysis — Cancer Patient Survival
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the GBSG breast cancer survival dataset
#   2. Explores survival times and censoring
#   3. Builds Kaplan-Meier survival curves
#   4. Compares survival across patient groups (log-rank test)
#   5. Builds a Cox Proportional Hazards model
#   6. Computes the C-index (survival AUC equivalent)
#   7. Visualises hazard ratios and feature effects
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# ===========================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ===========================================================
# GBSG = German Breast Cancer Study Group
# 686 breast cancer patients followed over time.
#
# Columns:
#   pid        - Patient ID
#   htreat     - Hormonal treatment (0=No, 1=Yes)
#   age        - Age at diagnosis
#   meno       - Menopausal status (0=Pre, 1=Post)
#   size       - Tumour size (mm)
#   grade      - Tumour grade (1=low, 2=medium, 3=high)
#   nodes      - Number of positive lymph nodes
#   pgr        - Progesterone receptor (fmol/l)
#   er         - Estrogen receptor (fmol/l)
#   duration   - Follow-up time in days ← TIME
#   event      - 1=recurrence occurred, 0=censored ← EVENT

df = pd.read_csv("gbsg2.csv")

# Rename for clarity
df = df.rename(columns={
    "time": "duration", "cens": "event",
    "horTh": "htreat", "menostat": "meno",
    "tsize": "size", "tgrade": "grade",
    "pnodes": "nodes", "progrec": "pgr", "estrec": "er"
})
df["grade"] = df["grade"].map({"I": 1, "II": 2, "III": 3})
df["htreat"] = df["htreat"].map({"no": 0, "yes": 1})
df["meno"] = df["meno"].map({"Pre": 0, "Post": 1})

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"  Total patients      : {len(df)}")
print(f"  Events (recurrence) : {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
print(f"  Censored patients   : {(df['event']==0).sum()} ({(df['event']==0).mean()*100:.1f}%)")
print(f"  Median follow-up    : {df['duration'].median():.0f} days ({df['duration'].median()/365:.1f} years)")
print(f"  Max follow-up       : {df['duration'].max():.0f} days ({df['duration'].max()/365:.1f} years)")
print()
print("  Statistical summary:")
print(df[["age", "size", "grade", "nodes", "pgr", "er", "duration"]].describe().round(1).to_string())
print()

# ===========================================================
# STEP 2: VISUALISE SURVIVAL TIMES AND CENSORING
# ===========================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Follow-up duration distribution
axes[0].hist(df["duration"] / 365, bins=30,
             color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Follow-up Time (years)")
axes[0].set_ylabel("Number of Patients")
axes[0].set_title("Distribution of Follow-up Times", fontweight="bold")

# Event vs censored
event_counts = df["event"].value_counts()
axes[1].bar(["Censored (0)", "Recurrence (1)"],
            [event_counts[0], event_counts[1]],
            color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[1].set_ylabel("Count")
axes[1].set_title("Event Distribution", fontweight="bold")
for i, v in enumerate([event_counts[0], event_counts[1]]):
    axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")

# Tumour grade distribution
grade_counts = df["grade"].value_counts().sort_index()
axes[2].bar([f"Grade {g}" for g in grade_counts.index],
            grade_counts.values,
            color=["#55A868", "#DD8452", "#C44E52"], edgecolor="white")
axes[2].set_ylabel("Count")
axes[2].set_title("Tumour Grade Distribution", fontweight="bold")

fig.suptitle("Dataset Overview — GBSG Breast Cancer Survival",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_dataset_overview.png")
plt.close()
print("Saved: plot1_dataset_overview.png")

# ===========================================================
# STEP 3: OVERALL KAPLAN-MEIER SURVIVAL CURVE
# ===========================================================
# The KM curve estimates the probability of surviving
# (no recurrence) beyond each time point.
# Each drop in the curve = a recurrence event.
# Flat sections = periods where only censored patients were observed.

kmf = KaplanMeierFitter()
kmf.fit(df["duration"] / 365,         # Convert days to years
        event_observed=df["event"],
        label="All Patients")

fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot_survival_function(
    ax=ax, ci_show=True,
    color="#4C72B0", linewidth=2.5
)

# Add median survival line
median_surv = kmf.median_survival_time_
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
ax.axvline(x=median_surv, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
ax.annotate(f"Median survival: {median_surv:.1f} years",
            xy=(median_surv, 0.5),
            xytext=(median_surv + 0.3, 0.55),
            fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Probability of Recurrence-Free Survival", fontsize=12)
ax.set_title("Kaplan-Meier Overall Survival Curve\n(Shaded area = 95% Confidence Interval)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_km_overall.png")
plt.close()
print("Saved: plot2_km_overall.png")

print()
print("=" * 60)
print("STEP 3: KAPLAN-MEIER RESULTS")
print("=" * 60)
print(f"  Median survival time : {median_surv:.2f} years")
print(f"  1-year survival      : {kmf.survival_function_at_times([1.0]).values[0]:.3f} ({kmf.survival_function_at_times([1.0]).values[0]*100:.1f}%)")
print(f"  3-year survival      : {kmf.survival_function_at_times([3.0]).values[0]:.3f} ({kmf.survival_function_at_times([3.0]).values[0]*100:.1f}%)")
print(f"  5-year survival      : {kmf.survival_function_at_times([5.0]).values[0]:.3f} ({kmf.survival_function_at_times([5.0]).values[0]*100:.1f}%)")
print()

# ===========================================================
# STEP 4: KM CURVES BY TUMOUR GRADE (with log-rank test)
# ===========================================================
# Compare survival between grade 1, 2, and 3 tumours.
# The log-rank test checks if differences are statistically
# significant (p < 0.05 = significant difference).

fig, ax = plt.subplots(figsize=(10, 7))
colors_grade = ["#55A868", "#DD8452", "#C44E52"]
grade_labels = {1: "Grade 1 (Low)", 2: "Grade 2 (Medium)", 3: "Grade 3 (High)"}

kmf_list = {}
for grade, color in zip([1, 2, 3], colors_grade):
    mask = df["grade"] == grade
    kmf_g = KaplanMeierFitter()
    kmf_g.fit(df.loc[mask, "duration"] / 365,
              event_observed=df.loc[mask, "event"],
              label=f"{grade_labels[grade]} (n={mask.sum()})")
    kmf_g.plot_survival_function(ax=ax, ci_show=False,
                                  color=color, linewidth=2.5)
    kmf_list[grade] = kmf_g

# Log-rank test: Grade 1 vs Grade 3
result = logrank_test(
    df[df["grade"] == 1]["duration"],
    df[df["grade"] == 3]["duration"],
    event_observed_A=df[df["grade"] == 1]["event"],
    event_observed_B=df[df["grade"] == 3]["event"]
)

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Probability of Recurrence-Free Survival", fontsize=12)
ax.set_title(f"Kaplan-Meier Curves by Tumour Grade\nLog-rank test (Grade 1 vs 3): p = {result.p_value:.4f}",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Significance annotation
sig_text = "***Highly Significant" if result.p_value < 0.001 else "**Significant" if result.p_value < 0.01 else "*Significant" if result.p_value < 0.05 else "Not Significant"
ax.annotate(sig_text, xy=(0.02, 0.05), xycoords="axes fraction", fontsize=11,
            color="darkred", fontweight="bold")

plt.tight_layout()
plt.savefig("plot3_km_by_grade.png")
plt.close()
print("Saved: plot3_km_by_grade.png")

print("=" * 60)
print("STEP 4: LOG-RANK TEST (Grade 1 vs Grade 3)")
print("=" * 60)
print(f"  Test statistic : {result.test_statistic:.4f}")
print(f"  p-value        : {result.p_value:.6f}")
print(f"  Significant    : {'Yes (p < 0.05)' if result.p_value < 0.05 else 'No'}")
print()

# ===========================================================
# STEP 5: KM CURVES BY HORMONAL TREATMENT
# ===========================================================

fig, ax = plt.subplots(figsize=(10, 6))
for treat, color, label in [(0, "#C44E52", "No Hormonal Treatment (n={})"),
                              (1, "#55A868", "Hormonal Treatment (n={})")]:
    mask = df["htreat"] == treat
    kmf_t = KaplanMeierFitter()
    kmf_t.fit(df.loc[mask, "duration"] / 365,
              event_observed=df.loc[mask, "event"],
              label=label.format(mask.sum()))
    kmf_t.plot_survival_function(ax=ax, ci_show=True,
                                  color=color, linewidth=2.5)

result_treat = logrank_test(
    df[df["htreat"] == 0]["duration"],
    df[df["htreat"] == 1]["duration"],
    event_observed_A=df[df["htreat"] == 0]["event"],
    event_observed_B=df[df["htreat"] == 1]["event"]
)

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Probability of Recurrence-Free Survival", fontsize=12)
ax.set_title(f"Effect of Hormonal Treatment on Survival\nLog-rank test p = {result_treat.p_value:.4f}",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_km_treatment.png")
plt.close()
print("Saved: plot4_km_treatment.png")

print("=" * 60)
print("STEP 5: LOG-RANK TEST (Treatment Effect)")
print("=" * 60)
print(f"  p-value     : {result_treat.p_value:.6f}")
print(f"  Significant : {'Yes (p < 0.05)' if result_treat.p_value < 0.05 else 'No'}")
print()

# ===========================================================
# STEP 6: COX PROPORTIONAL HAZARDS MODEL
# ===========================================================
# Cox regression models multiple features simultaneously.
# Output: hazard ratio (HR) for each feature.
# HR > 1 = increases risk, HR < 1 = protective.

print("=" * 60)
print("STEP 6: COX PROPORTIONAL HAZARDS MODEL")
print("=" * 60)

# Prepare features for Cox model
cox_df = df[["duration", "event", "age", "size", "grade",
             "nodes", "pgr", "er", "htreat", "meno"]].copy()

# Normalise continuous features for better coefficient interpretation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cont_cols = ["age", "size", "nodes", "pgr", "er"]
cox_df[cont_cols] = scaler.fit_transform(cox_df[cont_cols])

# Fit Cox model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col="duration", event_col="event")

print(cph.summary[["coef", "exp(coef)", "p", "coef lower 95%", "coef upper 95%"]].round(4))
print()

# C-index
c_index = cph.concordance_index_
print(f"  C-index (concordance): {c_index:.4f}")
print(f"  Interpretation       : Model correctly ranks {c_index*100:.1f}% of patient pairs by risk")
print()

# ===========================================================
# STEP 7: HAZARD RATIO FOREST PLOT
# ===========================================================
# A forest plot shows the HR and 95% CI for each feature.
# If the confidence interval crosses 1.0, the effect is NOT
# statistically significant.

summary = cph.summary.copy()
summary = summary.sort_values("exp(coef)", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = range(len(summary))
feature_labels = {
    "age": "Age",
    "size": "Tumour Size",
    "grade": "Tumour Grade",
    "nodes": "Lymph Nodes",
    "pgr": "Progesterone Receptor",
    "er": "Estrogen Receptor",
    "htreat": "Hormonal Treatment",
    "meno": "Menopausal Status"
}

labels = [feature_labels.get(f, f) for f in summary.index]
hrs = summary["exp(coef)"].values
lower = summary["exp(coef) lower 95%"].values
upper = summary["exp(coef) upper 95%"].values
pvals = summary["p"].values

colors_hr = ["#55A868" if hr < 1 else "#C44E52" for hr in hrs]

for i, (hr, lo, hi, pv, color) in enumerate(zip(hrs, lower, upper, pvals, colors_hr)):
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, alpha=0.8)
    marker = "D" if pv < 0.05 else "o"
    ax.plot(hr, i, marker=marker, color=color, markersize=10,
            markeredgecolor="white", markeredgewidth=1)
    sig_label = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
    ax.text(hi + 0.02, i, f" HR={hr:.2f}{sig_label}", va="center", fontsize=9)

ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Hazard Ratio (HR) with 95% CI", fontsize=12)
ax.set_title("Cox Model — Hazard Ratios\n(Green = protective, Red = harmful, Diamond = significant p<0.05)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0.3, max(upper) + 0.5)

protective = mpatches.Patch(color="#55A868", label="HR < 1 (Protective)")
harmful = mpatches.Patch(color="#C44E52", label="HR > 1 (Increases risk)")
ax.legend(handles=[protective, harmful], loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("plot5_hazard_ratios.png")
plt.close()
print("Saved: plot5_hazard_ratios.png")

# ===========================================================
# STEP 8: PREDICTED SURVIVAL CURVES FOR PATIENT PROFILES
# ===========================================================
# Cox model can predict survival curves for hypothetical
# patient profiles. Compare a low-risk vs high-risk patient.

fig, ax = plt.subplots(figsize=(10, 6))

# Low-risk profile: young, small tumour, low grade, few nodes,
#                   high pgr/er, with treatment
low_risk = pd.DataFrame({
    "age": [-1.0], "size": [-1.0], "grade": [1],
    "nodes": [-0.5], "pgr": [1.0], "er": [1.0],
    "htreat": [1], "meno": [0]
})

# High-risk profile: older, large tumour, high grade, many nodes,
#                    low pgr/er, no treatment
high_risk = pd.DataFrame({
    "age": [1.0], "size": [1.5], "grade": [3],
    "nodes": [2.0], "pgr": [-0.5], "er": [-0.5],
    "htreat": [0], "meno": [1]
})

cph.predict_survival_function(low_risk).T.squeeze().plot(
    ax=ax, color="#55A868", linewidth=2.5,
    label="Low-risk profile (young, small tumour, treated)")
cph.predict_survival_function(high_risk).T.squeeze().plot(
    ax=ax, color="#C44E52", linewidth=2.5,
    label="High-risk profile (older, large tumour, grade 3, untreated)")

ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Predicted Survival Probability", fontsize=12)
ax.set_title("Cox Model: Predicted Survival for Low vs High Risk Patients",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot6_predicted_survival.png")
plt.close()
print("Saved: plot6_predicted_survival.png")

# ===========================================================
# STEP 9: NODES IMPACT ON SURVIVAL (KM)
# ===========================================================
# Number of positive lymph nodes is a critical prognostic
# factor. Split into 0, 1-3, and 4+ nodes groups.

df["node_group"] = pd.cut(df["nodes"],
                           bins=[0, 3, 6, 100],
                           labels=["1-3 nodes", "4-6 nodes", "7+ nodes"])
df["node_group"] = df["node_group"].astype(str)

fig, ax = plt.subplots(figsize=(10, 6))
colors_nodes = ["#55A868", "#DD8452", "#C44E52"]

for group, color in zip(["1-3 nodes", "4-6 nodes", "7+ nodes"], colors_nodes):
    mask = df["node_group"] == group
    kmf_n = KaplanMeierFitter()
    kmf_n.fit(df.loc[mask, "duration"] / 365,
              event_observed=df.loc[mask, "event"],
              label=f"{group} (n={mask.sum()})")
    kmf_n.plot_survival_function(ax=ax, ci_show=False,
                                  color=color, linewidth=2.5)

result_nodes = logrank_test(
    df[df["node_group"] == "1-3 nodes"]["duration"],
df[df["node_group"] == "7+ nodes"]["duration"],
event_observed_A=df[df["node_group"] == "1-3 nodes"]["event"],
event_observed_B=df[df["node_group"] == "7+ nodes"]["event"]
)

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Probability of Recurrence-Free Survival", fontsize=12)
ax.set_title(f"Survival by Lymph Node Involvement\nLog-rank (0 vs 4+ nodes): p = {result_nodes.p_value:.6f}",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot7_km_nodes.png")
plt.close()
print("Saved: plot7_km_nodes.png")

# ===========================================================
# FINAL SUMMARY
# ===========================================================

print()
print("=" * 60)
print("PROJECT 4 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset             : GBSG Breast Cancer ({len(df)} patients)")
print(f"  Events (recurrence) : {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
print(f"  Censored            : {(df['event']==0).sum()} ({(df['event']==0).mean()*100:.1f}%)")
print()
print(f"  Median survival     : {median_surv:.2f} years")
print(f"  5-year survival     : {kmf.survival_function_at_times([5.0]).values[0]*100:.1f}%")
print()
print(f"  Cox C-index         : {c_index:.4f}")
print(f"  Grade effect (p)    : {result.p_value:.6f} ({'Significant' if result.p_value < 0.05 else 'Not significant'})")
print(f"  Treatment effect(p) : {result_treat.p_value:.6f} ({'Significant' if result_treat.p_value < 0.05 else 'Not significant'})")
print()
print("  Most significant Cox features:")
sig_features = cph.summary[cph.summary["p"] < 0.05].sort_values("p")
for feat, row in sig_features.iterrows():
    direction = "increases" if row["exp(coef)"] > 1 else "decreases"
    print(f"    {feat:8s}: HR={row['exp(coef)']:.3f} ({direction} risk), p={row['p']:.4f}")
print()
print("  7 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
