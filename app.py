import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Gender Bias Audit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTabs [data-baseweb="tab"] { color: #6B2D8B; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #6B2D8B; }
    .stTabs [aria-selected="true"] { color: #6B2D8B; font-weight: bold; }
    .stMetric { background-color: #f3e8ff; border-radius: 8px; padding: 10px; }
    h1, h2, h3 { color: #6B2D8B; }
    .stButton > button { background-color: #6B2D8B; color: white; border-radius: 8px; }
    .stButton > button:hover { background-color: #4a1f63; color: white; }
    .stSidebar { background-color: #f3e8ff; }
    </style>
""", unsafe_allow_html=True)

st.title("Gender Bias Audit and Mitigation Dashboard")
st.markdown("**Evaluating the Interaction Between Data Reweighting and Fairness Constraints in Reducing Gender Bias in Credit Scoring**")
st.markdown("*Tolulope Animashaun | c7546560 | Leeds Beckett University | MSc Data Science*")
st.divider()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    ["Demo Mode (Dissertation Results)", "Upload Mode (New Dataset)"],
    help="Demo Mode loads official dissertation results. Upload Mode allows you to test a new dataset."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This dashboard presents findings from a dissertation study evaluating
gender bias mitigation in credit scoring models using:
- **Meta's Balance** (LASSO-IPW and CBPS)
- **Microsoft's Fairlearn** (DP, EO, EOP constraints)
- **18-condition factorial experiment**
""")
st.sidebar.markdown("---")
st.sidebar.markdown("*Leeds Beckett University | MSc Data Science | 2026*")

DATA_PATH = "data/"

@st.cache_data
def load_demo_data():
    baseline = pd.read_csv(DATA_PATH + "baseline_results_v3.csv", index_col=0)
    group1 = pd.read_csv(DATA_PATH + "group1_results_v3.csv")
    group2 = pd.read_csv(DATA_PATH + "group2_results_v3.csv")
    group3 = pd.read_csv(DATA_PATH + "group3_results_v3.csv")
    all_conditions = pd.concat([group1, group2, group3], ignore_index=True)
    all_conditions['Balance'] = all_conditions['Balance'].fillna('No Balance')
    return baseline, group1, group2, group3, all_conditions

data_loaded = False

if "Demo" in mode:
    try:
        baseline, group1, group2, group3, all_conditions = load_demo_data()
        st.success("Official dissertation results loaded successfully")
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure all v3 files are in the data/ folder")
        data_loaded = False

else:
    st.info("Upload Mode — upload any credit scoring dataset to run the bias audit")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"File uploaded — {df_uploaded.shape[0]} rows, {df_uploaded.shape[1]} columns")

        st.subheader("Step 1 — Preview your data")
        st.dataframe(df_uploaded.head(), use_container_width=True)

        st.subheader("Step 2 — Specify your columns")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            target_col = st.selectbox("Target column (loan outcome):", df_uploaded.columns)
        with col_b:
            gender_col = st.selectbox("Gender column:", df_uploaded.columns)
        with col_c:
            female_value = st.text_input("Value that means female:", "female")

        st.subheader("Step 3 — Specify target encoding")
        approved_value = st.text_input("Value that means approved/good credit:", "1")

        if st.button("Run Pre-Mitigation Bias Audit", key="upload_audit_btn"):
            try:
                df_work = df_uploaded.copy()
                df_work['gender_binary'] = df_work[gender_col].astype(str).str.contains(str(female_value), case=False, na=False).astype(int)
                df_work['target_binary'] = (df_work[target_col].astype(str) == str(approved_value)).astype(int)

                female_approval = df_work[df_work['gender_binary']==1]['target_binary'].mean()
                male_approval = df_work[df_work['gender_binary']==0]['target_binary'].mean()
                approval_gap = abs(male_approval - female_approval)

                female_proportion = df_work['gender_binary'].mean()
                representation_gap = abs(0.5 - female_proportion)

                feature_cols = [c for c in df_work.columns if c not in [target_col, gender_col, 'gender_binary', 'target_binary']]
                X_proxy = pd.get_dummies(df_work[feature_cols])
                X_proxy = X_proxy.fillna(0)
                lr_proxy = LogisticRegression(solver='liblinear', max_iter=5000, random_state=42)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                sauc_raw = cross_val_score(lr_proxy, X_proxy, df_work['gender_binary'], cv=cv, scoring='roc_auc').mean()
                sauc_normalised = (sauc_raw - 0.5) * 2

                audit_score = (approval_gap * 0.5) + (representation_gap * 0.3) + (sauc_normalised * 0.2)

                if audit_score <= 0.05:
                    classification = "LOW"
                elif audit_score <= 0.15:
                    classification = "MODERATE"
                else:
                    classification = "HIGH"

                st.markdown("---")
                st.subheader("Pre-Mitigation Bias Audit Results")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Audit Score", f"{audit_score:.4f}")
                with col2:
                    st.metric("Classification", classification)
                with col3:
                    st.metric("Female Approval Rate", f"{female_approval*100:.2f}%")
                with col4:
                    st.metric("Female Proportion", f"{female_proportion*100:.2f}%")

                st.markdown("---")
                audit_breakdown = pd.DataFrame({
                    'Component': ['Approval Rate Gap', 'Representation Gap', 'Proxy Feature Strength (sAUC)'],
                    'Raw Value': [round(approval_gap, 4), round(representation_gap, 4), round(sauc_normalised, 4)],
                    'Weight': [0.5, 0.3, 0.2],
                    'Contribution': [round(approval_gap*0.5, 4), round(representation_gap*0.3, 4), round(sauc_normalised*0.2, 4)]
                })
                st.dataframe(audit_breakdown, use_container_width=True)

                if classification == "HIGH":
                    st.error(f"Audit score of {audit_score:.4f} is classified as HIGH — serious gender bias detected.")
                elif classification == "MODERATE":
                    st.warning(f"Audit score of {audit_score:.4f} is classified as MODERATE — some gender bias detected.")
                else:
                    st.success(f"Audit score of {audit_score:.4f} is classified as LOW — minimal gender bias detected.")

                st.info(f"Female approval rate: {female_approval*100:.2f}% vs Male approval rate: {male_approval*100:.2f}%")
                data_loaded = True

            except Exception as e:
                st.error(f"Error running audit: {e}")
                data_loaded = False
    else:
        data_loaded = False

if data_loaded and "Demo" in mode:

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Pre-Mitigation Audit",
        "Baseline Results",
        "18-Condition Experiment",
        "Post-Mitigation Audit",
        "Recommendation Engine"
    ])

    with tab1:
        st.header("Pre-Mitigation Bias Audit")
        st.markdown("Measures the level of gender bias in the German Credit dataset **before** any model training or fairness intervention.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Audit Score", "0.1714", help="Composite bias score")
        with col2:
            st.metric("Classification", "HIGH", help="Above 0.15 threshold")
        with col3:
            st.metric("Female Approval Rate", "64.98%", help="vs 72.26% for males")
        with col4:
            st.metric("Female Proportion", "31%", help="Gap of 0.19 from ideal 50%")

        st.markdown("---")
        st.subheader("Audit Score Breakdown")
        audit_breakdown = pd.DataFrame({
            'Component': ['Approval Rate Gap', 'Representation Gap', 'Proxy Feature Strength (sAUC)'],
            'Raw Value': [0.0728, 0.1900, 0.3901],
            'Weight': [0.5, 0.3, 0.2],
            'Contribution': [0.0364, 0.0570, 0.0780]
        })
        st.dataframe(audit_breakdown, use_container_width=True)

        st.markdown("---")
        st.subheader("Key Finding")
        st.warning("The pre-mitigation audit score of 0.1714 is classified as HIGH — indicating serious gender bias in the dataset before any intervention.")
        st.info("Age was identified as the strongest individual proxy for gender (Cramers V = 0.2588)")

    with tab2:
        st.header("Baseline Model Performance")
        st.markdown("Performance of both classifiers **before** any fairness intervention was applied.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression")
            lr_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Balanced Accuracy', 'Sensitivity', 'Specificity', 'DPD', 'EOD', 'FNR Women'],
                'Value': [0.7633, 0.6849, 0.8810, 0.4889, 0.0407, 0.0415, 0.1167]
            })
            st.dataframe(lr_metrics, use_container_width=True)

        with col2:
            st.subheader("Random Forest")
            rf_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Balanced Accuracy', 'Sensitivity', 'Specificity', 'DPD', 'EOD', 'FNR Women'],
                'Value': [0.7640, 0.6575, 0.9238, 0.3911, 0.0674, 0.1101, 0.0833]
            })
            st.dataframe(rf_metrics, use_container_width=True)

        st.markdown("---")
        st.info("Logistic Regression achieved lower DPD and EOD. Random Forest had higher sensitivity and slightly higher accuracy, with lower FNR Women.")

    with tab3:
        st.header("18-Condition Factorial Experiment")
        st.markdown("Results across all 18 combinations of Balance method, Fairlearn constraint and classifier type.")

        st.subheader("All 18 Conditions")
        display_cols = ['Condition', 'Balance', 'Constraint', 'Classifier', 'Accuracy', 'DPD', 'EOD', 'FNR_Women']
        st.dataframe(all_conditions[display_cols].round(4), use_container_width=True)

        st.markdown("---")
        st.subheader("DPD Comparison Chart")
        fig, ax = plt.subplots(figsize=(14, 5))
        no_balance = all_conditions[all_conditions['Balance'] == 'No Balance']
        lasso = all_conditions[all_conditions['Balance'] == 'LASSO']
        cbps = all_conditions[all_conditions['Balance'] == 'CBPS']
        x = np.arange(6)
        width = 0.25
        ax.bar(x - width, no_balance['DPD'].values, width, label='No Balance', color='#6B2D8B', alpha=0.85)
        ax.bar(x, lasso['DPD'].values, width, label='LASSO Balance', color='#9B59B6', alpha=0.85)
        ax.bar(x + width, cbps['DPD'].values, width, label='CBPS Balance', color='#D7BDE2', alpha=0.85)
        ax.axhline(y=0.0407, color='black', linestyle='--', linewidth=1.5, label='Baseline LR DPD (0.0407)')
        labels = ['DP+LR', 'DP+RF', 'EO+LR', 'EO+RF', 'EOP+LR', 'EOP+RF']
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('DPD', fontsize=12)
        ax.set_title('DPD Across All 18 Conditions', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Compatibility Heatmap")
        all_conditions['Group'] = all_conditions['Balance'] + '\n' + all_conditions['Classifier']
        all_conditions['Constraint_short'] = all_conditions['Constraint']
        pivot = all_conditions.pivot_table(values='DPD', index='Group', columns='Constraint_short', aggfunc='mean')
        row_order = [
            'No Balance\nLR', 'No Balance\nRF',
            'LASSO\nLR', 'LASSO\nRF',
            'CBPS\nLR', 'CBPS\nRF'
        ]
        pivot = pivot.reindex(row_order)
        pivot = pivot[['DP', 'EO', 'EOP']]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='Purples', ax=ax2,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'DPD (lower = fairer)'},
                    vmin=0, vmax=0.12)
        ax2.set_title('DPD Heatmap by Balance Method, Constraint and Classifier',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Fairlearn Constraint', fontsize=11)
        ax2.set_ylabel('Balance Method + Classifier', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("FNR Women Comparison Chart")
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        ax3.bar(x - width, no_balance['FNR_Women'].values, width, label='No Balance', color='#6B2D8B', alpha=0.85)
        ax3.bar(x, lasso['FNR_Women'].values, width, label='LASSO Balance', color='#9B59B6', alpha=0.85)
        ax3.bar(x + width, cbps['FNR_Women'].values, width, label='CBPS Balance', color='#D7BDE2', alpha=0.85)
        ax3.axhline(y=0.1167, color='black', linestyle='--', linewidth=1.5, label='Baseline LR FNR Women (0.1167)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylabel('FNR Women', fontsize=12)
        ax3.set_title('FNR Women Across All 18 Conditions', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.set_ylim(0, 0.18)
        ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax3.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("Download Results")
        csv = all_conditions[display_cols].round(4).to_csv(index=False)
        st.download_button(
            label="Download All 18 Conditions (CSV)",
            data=csv,
            file_name="dissertation_results_18_conditions.csv",
            mime="text/csv",
            key="download_btn"
        )

    with tab4:
        st.header("Post-Mitigation Bias Audit")
        st.markdown("Comparing bias levels before and after mitigation using the same composite formula.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pre-Mitigation", "0.1714", "HIGH")
        with col2:
            st.metric("Baseline LR", "0.1049", "+38.8%")
        with col3:
            st.metric("C15 CBPS+Fairlearn", "0.0720", "+58.0%")
        with col4:
            st.metric("C1 Fairlearn Only (Best)", "0.0619", "+63.9%")

        st.markdown("---")
        st.subheader("Audit Comparison Table")
        audit_table = pd.DataFrame({
            'Condition': ['Pre-mitigation dataset', 'Baseline LR (no mitigation)', 'C1: No Balance + DP + LR', 'C15: CBPS + EO + LR', 'C17: CBPS + EOP + LR'],
            'Approval Rate Gap': [0.0748, 0.0407, 0.0061, 0.0281, 0.0249],
            'Representation Gap': [0.1900, 0.2013, 0.1883, 0.1872, 0.1910],
            'Proxy Strength (sAUC)': [0.3901, 0.1210, 0.0119, 0.0090, 0.0150],
            'Audit Score': [0.1714, 0.1049, 0.0619, 0.0720, 0.0728],
            'Classification': ['HIGH', 'MODERATE', 'MODERATE', 'MODERATE', 'MODERATE'],
            'Improvement': ['—', '+38.8%', '+63.9%', '+58.0%', '+57.6%']
        })
        st.dataframe(audit_table, use_container_width=True)

        st.markdown("---")
        st.subheader("Key Findings")
        st.success("Fairlearn's Demographic Parity constraint with Logistic Regression alone (Condition 1) achieved the best overall improvement of 63.9%")
        st.warning("No condition reached the LOW classification threshold of 0.05")
        st.info("The Representation Gap was the most resistant component to reduce")

    with tab5:
        st.header("Recommendation Engine")
        st.markdown("Select your priority fairness metric and the engine will recommend the best mitigation combination.")

        priority = st.selectbox(
            "What is your primary fairness priority?",
            [
                "Best overall bias reduction (Audit Score)",
                "Best approval rate equality (DPD)",
                "Best error rate equality (EOD)",
                "Fewest creditworthy women wrongly rejected (FNR Women)"
            ]
        )

        if st.button("Get Recommendation", key="rec_btn"):
            if priority == "Best overall bias reduction (Audit Score)":
                st.success("Recommended: Condition 1 — Fairlearn Demographic Parity + Logistic Regression (No Balance)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Audit Improvement", "+63.9%")
                with col2:
                    st.metric("DPD", "0.0227")
                with col3:
                    st.metric("Accuracy", "0.7580")
                st.info("This combination achieved the best overall audit score of 0.0619 — a 63.9% improvement from the pre-mitigation baseline of 0.1714.")

            elif priority == "Best approval rate equality (DPD)":
                st.success("Recommended: Condition 17 — CBPS Balance + Equal Opportunity + Logistic Regression")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("DPD", "0.0209")
                with col2:
                    st.metric("EOD", "0.0600")
                with col3:
                    st.metric("Accuracy", "0.7540")
                st.info("This combination achieved the lowest DPD of 0.0209 across all 18 conditions.")

            elif priority == "Best error rate equality (EOD)":
                st.success("Recommended: Condition 1 — Fairlearn Demographic Parity + Logistic Regression (No Balance)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("EOD", "0.0209")
                with col2:
                    st.metric("DPD", "0.0227")
                with col3:
                    st.metric("Accuracy", "0.7580")
                st.info("This combination achieved the lowest EOD of 0.0209 across all 18 conditions.")

            elif priority == "Fewest creditworthy women wrongly rejected (FNR Women)":
                st.success("Recommended: Condition 2 — Fairlearn Demographic Parity + Random Forest (No Balance)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("FNR Women", "0.0700")
                with col2:
                    st.metric("DPD", "0.0707")
                with col3:
                    st.metric("Accuracy", "0.7553")
                st.warning("Note: While this condition achieved the lowest FNR Women, it has a higher DPD than other conditions — illustrating fairness metric incompatibility.")

        st.markdown("---")
        st.subheader("All Conditions Ranked by Your Priority")

        if priority == "Best overall bias reduction (Audit Score)":
            ranked = pd.DataFrame({
                'Rank': [1, 2, 3, 4, 5],
                'Condition': ['C1: No Balance+DP+LR', 'C15: CBPS+EO+LR', 'C17: CBPS+EOP+LR', 'C8: LASSO+DP+RF', 'Baseline LR'],
                'Audit Score': [0.0619, 0.0720, 0.0728, 0.0878, 0.1049],
                'Improvement': ['+63.9%', '+58.0%', '+57.6%', '+48.8%', '+38.8%']
            })
            st.dataframe(ranked, use_container_width=True)

        elif priority in ["Best approval rate equality (DPD)", "Best error rate equality (EOD)"]:
            metric_col = 'DPD' if 'DPD' in priority else 'EOD'
            display_cols_rec = ['Condition', 'Balance', 'Constraint', 'Classifier', metric_col, 'Accuracy']
            ranked = all_conditions[display_cols_rec].sort_values(metric_col).reset_index(drop=True)
            ranked.index += 1
            st.dataframe(ranked.round(4), use_container_width=True)

        elif priority == "Fewest creditworthy women wrongly rejected (FNR Women)":
            display_cols_rec = ['Condition', 'Balance', 'Constraint', 'Classifier', 'FNR_Women', 'DPD', 'Accuracy']
            ranked = all_conditions[display_cols_rec].sort_values('FNR_Women').reset_index(drop=True)
            ranked.index += 1
            st.dataframe(ranked.round(4), use_container_width=True)
