# Body-Performance-Analysis-ML-Project
End-to-end ML pipeline on 13,393 real fitness assessments. Tackles 4-class performance tier classification &amp; broad jump regression using KNN, Decision Tree, SVM (Linear/RBF), and MLP. Includes domain-informed preprocessing, stratified K-Fold CV, and full model comparison across 3 split ratios.
📌 Overview
A complete, end-to-end machine learning pipeline applied to the Body Performance Dataset — a collection of real fitness assessments spanning age groups and genders. The project tackles two complementary predictive tasks:
TaskTypeTargetMetricPerformance Tier PredictionClassificationClass A / B / C / DMacro F1Broad Jump Distance PredictionRegressionbroad jump_cmR² Score

📁 Project Structure
body-performance-ml/
│
├── Machine_Learning_Project.ipynb   # Main notebook (full pipeline)
├── bodyPerformance.csv              # Dataset (add manually)
│
├── INSIGHTS.md                      # Key findings & analytical insights
├── MODEL_COMPARISON.md              # Full model comparison tables & results
│
└── README.md

📊 Dataset
PropertyValueSourceKaggle — Body Performance DataRecords13,393Features11 (age, gender, height, weight, body fat %, blood pressure, grip force, sit-ups, broad jump)Target (Classification)class — A, B, C, DTarget (Regression)broad jump_cm

⚙️ Pipeline
1 — Data Preprocessing

Type verification — stripped column whitespace, validated dtypes
Duplicate removal — prevented sample memorisation
Domain-informed filtering — removed physiologically impossible blood pressure values (diastolic ≥ systolic, diastolic ≤ 40, systolic ≤ 70)
Rule-based outlier removal — medically anchored ranges for age, weight, height, body fat %
IQR Winsorization — clipped remaining extremes to fence values (floor at 0 for non-negative columns) — preserves dataset size
Encoding & Scaling inside Pipeline — zero data leakage guaranteed

2 — Modelling Strategy
Classification Models:
ModelVariantsK-Nearest Neighboursk = 3, 5, 7Decision Treemax_depth = 3, 5, NoneSupport Vector MachineLinear kernel, RBF kernelNeural Network (MLP)64 → 32, ReLU, 500 iterations
Regression Models:
ModelVariantsLinear RegressionOLS baselineKNN Regressork = 3, 5Decision Tree Regressormax_depth = 3, 5Support Vector RegressorLinear, RBFNeural Network Regressor64 → 32, ReLU
3 — Evaluation Protocol

Three train/test splits: 80:20, 70:30, 50:50
Stratified K-Fold (5-fold) for classification
K-Fold (5-fold) for regression
Final ranking: Macro F1 (classification) · R² (regression)


🏆 Results Summary
Classification
RankModelCV F1 Macro🥇SVM (RBF)Best🥈Neural Network (MLP)~Equal🥉KNN (k=5)CompetitiveDecision Tree (unconstrained)OverfitSVM (Linear)Below RBF
Regression
RankModelCV R²🥇Neural Network RegressorBest🥈SVR (RBF)~Equal🥉Decision Tree (depth=5)ModerateLinear RegressionBelow non-linear

Full numeric results → MODEL_COMPARISON.md


💡 Key Insights

Full analysis → INSIGHTS.md


Non-linear models consistently win — both tasks confirm that performance data contains complex interaction effects between age, grip force, and flexibility that linear models cannot capture
Grip force ↔ broad jump correlation ≈ 0.73 (strongest predictor)
Age ↔ broad jump correlation ≈ −0.50 (performance declines with age)
SVM Linear vs RBF gap proves class boundaries are not linearly separable in the original 11-dimensional feature space


🚀 Getting Started
Prerequisites
bashpip install pandas numpy scikit-learn matplotlib seaborn jupyter
Run the Notebook
bashgit clone https://github.com/YOUR_USERNAME/body-performance-ml.git
cd body-performance-ml
jupyter notebook Machine_Learning_Project.ipynb

⚠️ Place bodyPerformance.csv in the root directory before running, or update DATA_PATH in cell 1.


⚠️ Known Limitations

Best-model final evaluation is done on the full training set (optimistically biased — no dedicated hold-out test set)
Hyperparameters were fixed at sensible defaults, not tuned via grid/random search
MLP convergence was not monitored (no validation loss curve or early stopping)


🔮 Future Work

 Hyperparameter tuning with RandomizedSearchCV (nested CV)
 Add Random Forest & Gradient Boosting (XGBoost, LightGBM)
 Feature engineering: BMI, strength-to-weight ratio, age-normalised scores
 Proper held-out test set
 SHAP explainability layer for individual predictions
 Class-specific threshold optimisation via Precision-Recall curves


📄 Report
A full written report (ML_Report_Body_Performance.pdf) is included, covering methodology, results, critical analysis, and future improvements.

👤 Author
Ahmed Mahmoud Khalifa · March 2026
