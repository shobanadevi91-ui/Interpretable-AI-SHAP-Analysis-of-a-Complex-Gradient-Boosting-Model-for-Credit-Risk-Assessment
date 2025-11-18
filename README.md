# Interpretable-AI-SHAP-Analysis-of-a-Complex-Gradient-Boosting-Model-for-Credit-Risk-Assessment
Creates a realistic LendingClub-like synthetic dataset (so you get a dataset included and the code runs offline),  Trains and hyperparameter-tunes a LightGBM gradient boosting classifier for credit-default (binary),  Handles class imbalance sensibly,  Computes tree-based feature importance and SHAP explanations (global &amp; local).
Interpretable AI for Credit Risk Assessment
SHAP Analysis of a Gradient Boosting Model (LightGBM)

This project demonstrates how to build a high-performance, fully interpretable, and regulator-ready credit default prediction system using:

Gradient Boosting Machine (LightGBM)

SHAP (SHapley Additive exPlanations) for global & local explainability

Non-linear SHAP interaction analysis

SMOTE + hyperparameter tuning

A synthetic LendingClub-style dataset that runs offline

The project is intended for advanced machine learning students and practitioners aiming to build explainable, deployable, high-stakes credit risk models compatible with regulatory expectations (e.g., adverse-action reasoning, model transparency, and risk governance).

📌 Project Goals

Train & tune a Gradient Boosting Machine

Uses LightGBM with extensive hyperparameter search (RandomizedSearchCV + StratifiedKFold).

Handles class imbalance using SMOTE inside a pipeline to avoid leakage.

Global interpretability

LightGBM tree-based feature importance (gain).

SHAP global summary (beeswarm), mean |SHAP| bar charts.

Comparison between tree importance and SHAP importance.

Local interpretability

Select three representative applicants:

High-risk

Low-risk

Borderline case

Generate SHAP waterfall plots and list top contributing features.

Interaction effects

Compute SHAP interaction values to identify top three non-linear interactions driving model predictions.

Complete reproducibility

Self-contained synthetic dataset (no external downloads needed).

All plots, tables, model artifacts, and analysis saved automatically.

📂 Repository Structure
project-root/
│
├── credit_shap_lgbm.py            # Main runnable script
├── README.md                      # Project documentation
│
└── outputs_credit_shap/           # Auto-generated results folder
    ├── analysis_report.txt        # Full text analysis & explanation
    ├── shap_summary_beeswarm.png
    ├── shap_importance_top15.png
    ├── tree_importance_top15.png
    ├── compare_tree_shap.csv
    ├── shap_interactions.csv
    ├── shap_local_high_risk_*.png
    ├── shap_local_low_risk_*.png
    ├── shap_local_borderline_*.png
    ├── raw_row_*.csv
    ├── lgb_model.txt              # Saved LightGBM model
    └── ... (additional plots)


All contents inside outputs_credit_shap/ are created automatically when you run the script.

⚙️ Installation
1. Clone or download the repository
git clone <your-repo-url>
cd project-folder

2. Install required packages
pip install numpy pandas scikit-learn lightgbm shap matplotlib seaborn imbalanced-learn


Note: SHAP may require installing wheels appropriate to your Python version.

▶️ Running the Project

Execute the main script:

python credit_shap_lgbm.py


Upon completion, the script outputs:

Model performance metrics (AUC, F1)

SHAP plots (global & local)

Interaction analysis

An automatically generated analysis report summarizing the full results

📊 What the Analysis Includes
1. Model Performance

ROC AUC

F1 score

Confusion matrix

Classification report

2. Global Explainability

SHAP summary plot (beeswarm)

SHAP mean |value| importance

LightGBM tree gain importance

CSV comparison of the two methods

3. Local Explainability

For each selected applicant (high, low, borderline risk), the analysis includes:

SHAP waterfall plot showing top positive & negative feature contributions

Raw applicant record

Explanation of how model arrived at the decision

4. SHAP Interaction Analysis

Computes full SHAP interaction matrix

Identifies top three non-linear interactions

Saves a CSV ranking all feature pairs

🧠 Key Methodological Notes

Synthetic Dataset:
A LendingClub-style dataset is auto-generated using realistic distributions and non-linear default dynamics.

Imbalance Handling:
SMOTE is applied inside the cross-validation pipeline to prevent leakage.

Hyperparameter Tuning:
Uses RandomizedSearchCV with 40 sampled configurations for strong performance without excessive compute cost.

Regulatory Suitability:
SHAP-based explanations meet interpretability criteria for high-stakes credit risk models.

📈 Example Outputs

The project automatically generates:

Global feature importance charts

Local explanation waterfall plots

SHAP dependence plots

SHAP interaction scatter plots

Model performance report

Comparative global feature importance table
