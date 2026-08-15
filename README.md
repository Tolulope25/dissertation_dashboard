 Gender Bias Audit and Mitigation Dashboard

This repository has the analysis code and dashboard I built for my dissertation, which looks at how data reweighting and fairness constraints work together to reduce gender bias in credit scoring models.

Full pipeline:  the full analysis pipeline. This includes data preprocessing, the bias audit framework, Meta's Balance reweighting (LASSO-IPW and CBPS), Microsoft's Fairlearn fairness constraints, all 18 experimental conditions, and the post-mitigation audit.
app.py: the code for the Streamlit dashboard.
requirements.txt: the Python packages needed to run the dashboarbaseline_results_v3.csv, group1_results_v3.csv, group2_results_v3.csv, group3_results_v3.csv : the saved results used by the dashboard.

## Dashboard

The dashboard is live here: https://gender-bias-audit-2026.streamlit.app

It has five tabs: Pre-Mitigation Audit, Baseline Results, the 18-Condition Experiment, Post-Mitigation Audit, and a Recommendation Engine. There is also an Upload Mode where users can test the bias audit framework on their own credit scoring data.

## Dataset

This study uses the German Credit dataset (Hofmann, 1994), which is publicly available and widely used in credit scoring fairness research.

## Tools used

- Python 3.12
- scikit-learn 1.6.1
- Meta's Balance 0.23.0
- Microsoft's Fairlearn 0.14.0
- Streamlit 1.28
