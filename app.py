import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def predict_probability(patient_dict):
    """
    Takes a single patient's dictionary of features
    and returns the predicted readmission probability.
    """
    # Build aligned dataframe
    aligned_df = build_aligned_df(patient_dict, TRAIN_COLS)

    # Predict probability from model pipeline
    proba = xgb_model.predict_proba(aligned_df)[0][1]

    return float(proba)

def categorize_risk(prob):
    """
    Map a probability (0–1) to a risk category label.
    Use the same thresholds you’re using in the main app.
    """
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Moderate Risk"
    else:
        return "Low Risk"



def clean_feature_name(f):
    """
    Convert raw feature names like:
    'cat__gender_Male' → 'Gender: Male'
    'num__number_inpatient' → 'Inpatient Visits'
    """

    # Handle categorical feature names
    if f.startswith("cat__"):
        base, category = f.replace("cat__", "").split("_", 1)
        base = base.replace("_", " ").title()   # gender → Gender
        category = category.replace("_", " ").title()
        return f"{base}: {category}"

    # Handle numerical feature names
    if f.startswith("num__"):
        base = f.replace("num__", "")
        base = base.replace("_", " ").title()
        return base

    # Fallback
    return f


# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🩺",
    layout="centered"
)

# ============ LOAD MODEL & COLUMNS ============

@st.cache_resource
def load_model_and_columns():
    model = joblib.load("xgboost_readmission_pipeline.pkl")
    cols = joblib.load("train_columns.pkl")
    return model, cols

xgb_model, TRAIN_COLS = load_model_and_columns()

def get_global_feature_importance(model, top_n=15):
    """
    Returns a dataframe with the top_n most important features
    according to the XGBoost model inside the pipeline.
    """

    # Steps from pipeline
    preprocessor = model.named_steps["preprocessor"]
    xgb_clf = model.named_steps["model"]  # XGBClassifier

    # Raw feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # XGBoost feature importances
    importances = xgb_clf.feature_importances_

    # Build compact dataframe: only clean names + importance
    imp_df = pd.DataFrame({
        "Clean Feature": [clean_feature_name(f) for f in feature_names],
        "Relative Importance": importances,
    })

    # Sort and keep top_n
    imp_df = (
        imp_df
        .sort_values("Relative Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return imp_df

# ============ BACKEND HELPERS (NO UI) ============

def build_aligned_df(patient_dict, train_cols):
    """
    Build a 1-row DataFrame containing all training columns.
    Missing columns are filled with NaN and order matches training.
    """
    df = pd.DataFrame([patient_dict])
    for col in train_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[train_cols]
    return df


def predict_readmission_streamlit(patient_dict, threshold=0.5):
    """
    Use the saved XGBoost pipeline to predict readmission.
    Returns: (pred_class, probability, risk_label)
    """
    df_input = build_aligned_df(patient_dict, TRAIN_COLS)
    proba = float(xgb_model.predict_proba(df_input)[0][1])
    pred_class = int(proba >= threshold)

    if proba >= 0.7:
        risk = "High Risk"
    elif proba >= 0.4:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    return pred_class, proba, risk


def simulate_scenarios_streamlit(base_patient, scenarios):
    results = []

    for sc in scenarios:
        modified_patient = base_patient.copy()

        # apply scenario changes
        for feature, new_value in sc["changes"].items():
            modified_patient[feature] = new_value

        # predict probability
        prob = predict_probability(modified_patient)

        # add record
        results.append({
            "Scenario": sc["name"],
            "Changes": sc["changes"],
            "Probability": round(prob, 3),
            "Risk": categorize_risk(prob)
        })

    return pd.DataFrame(results)

def risk_color_badge(risk: str) -> str:
    """
    Return a markdown string with an emoji badge for the given risk label.
    """
    if risk == "High Risk":
        return "🟥 **HIGH RISK**"
    elif risk == "Moderate Risk":
        return "🟧 **MODERATE RISK**"
    else:
        return "🟩 **LOW RISK**"

# ============ STREAMLIT UI ============

def main():
    # Page config
    st.set_page_config(
        page_title="Diabetes Readmission Digital Twin",
        layout="wide",
    )

    # ---- HEADER ----
    st.title("Diabetes Readmission Prediction & Digital Twin")
    st.write(
        "This tool uses a machine learning model (XGBoost) trained on a large hospital dataset "
        "to estimate the probability that a diabetic patient will be readmitted. "
        "It also provides a **digital twin** to simulate treatment changes and see how risk may change."
    )

    # ---- STEP 1: INPUT FORM ----
    st.markdown("## Step 1: Enter Patient Information")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            race = st.selectbox(
                "Race",
                ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                index=0,
            )
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            age = st.selectbox(
                "Age group",
                ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)"],
                index=4,  # [40-50)
            )
            time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 4)

        with col2:
            number_inpatient = st.slider("Number of prior inpatient visits (2 years)", 0, 10, 1)
            insulin = st.selectbox("Insulin status", ["No", "Steady", "Up", "Down"], index=0)
            metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"], index=0)
            change = st.selectbox("Any medication change this encounter?", ["No", "Ch", "Up", "Down"], index=0)
            diabetesMed = st.selectbox("On any diabetes medication?", ["Yes", "No"], index=0)

        submitted = st.form_submit_button("Predict Readmission Risk")

    # ---- AFTER SUBMIT ----
    # Remember that we have run at least one prediction
    if submitted:
        st.session_state["has_prediction"] = True

    # Use this flag to decide whether to show results
    if st.session_state.get("has_prediction", False):
        # Build patient dictionary
        base_patient = {
            "race": race,
            "gender": gender,
            "age": age,
            "time_in_hospital": time_in_hospital,
            "number_inpatient": number_inpatient,
            "insulin": insulin,
            "metformin": metformin,
            "change": change,
            "diabetesMed": diabetesMed,
        }

        # ---- PATIENT SUMMARY ----
        st.markdown("## Step 2: Patient Summary")
        st.json(base_patient)
        
        st.markdown(
    """
    **What this means:**  
    This section shows the key medical and demographic details of the patient that our model uses to estimate the risk of being readmitted to the hospital.  
    These factors include age, medications, number of hospital stays, and other relevant information.
    """
)


        # ---- BASELINE PREDICTION ----
        pred_class, proba, risk = predict_readmission_streamlit(base_patient)

        st.markdown("### Readmission Risk")
        st.markdown(f"**Estimated probability of readmission:** {proba * 100:.1f}%")
        st.markdown(
            f"**Risk category:** {risk_color_badge(risk)}",
            unsafe_allow_html=True,
        )

        # ===== RISK INTERPRETATION BLOCK =====
        st.markdown("#### Interpretation")

        if risk == "High Risk":
            st.write(
                "The patient has a **relatively high likelihood of being readmitted** based on their current profile. "
                "Key contributors may include prior inpatient visits, duration of hospital stay, and medication "
                "adjustments. Digital twin scenarios below suggest that stabilizing insulin therapy and reducing "
                "inpatient recurrence patterns may help reduce this probability."
            )

        elif risk == "Moderate Risk":
            st.write(
                "The patient shows a **moderate likelihood of readmission**. Factors such as inpatient recurrence, "
                "hospital stay duration, and diabetic medication behavior are contributing to this profile. "
                "Consider reviewing medication adherence, stability of insulin dosing, and routine follow-up planning. "
                "Digital twin scenarios below provide simulated strategies that may help lower the risk."
            )

        else:
            st.write(
                "The patient displays a **relatively low likelihood of readmission** under current conditions. "
                "Maintaining stable medication patterns and minimizing avoidable inpatient occurrences should help "
                "sustain this reduced risk level."
            )

        st.markdown("---")
        st.markdown(
    """
    **What this means:**  
    The number above represents the chance that this patient may be readmitted to the hospital after discharge.  
    It is based on patterns learned from thousands of past real-world patient records.  
    - A higher percentage means the patient is more likely to return to the hospital soon.
    - A lower percentage means the risk is smaller.
    
    The colored risk tag (Low/Moderate/High) gives a quick interpretation for easier decision-making.
    """
)


            # ------------- STEP 3: DIGITAL TWIN SCENARIOS -------------
        st.markdown("---")
        st.markdown("### Step 3: Digital Twin – What-If Treatment Scenarios")
        st.write(
            "We create a digital twin of this patient and apply hypothetical treatment changes "
            "to see how the predicted readmission risk changes."
        )

        st.markdown("**Choose which treatment changes to simulate:**")

        col_left, col_right = st.columns(2)

        with col_left:
            sc_insulin = st.checkbox(
                "Start steady insulin",
                value=True,
                key="sc_insulin",
            )
            sc_reduce_inp = st.checkbox(
                "Reduce inpatient visits to 0",
                value=True,
                key="sc_reduce_inp",
            )
            sc_younger = st.checkbox(
                "Move to younger age group",
                value=False,
                key="sc_younger",
            )

        with col_right:
            sc_metformin = st.checkbox(
                "Add metformin (Steady)",
                value=True,
                key="sc_metformin",
            )
            sc_combo = st.checkbox(
                "Insulin steady + 0 inpatient visits",
                value=True,
                key="sc_combo",
            )

        # --------- Custom Scenario Builder (new) ----------
        st.markdown("#### Custom scenario (optional)")

        with st.expander("Configure a custom what-if scenario", expanded=False):
            custom_label = st.text_input("Scenario name", value="Custom scenario")

            colA, colB, colC = st.columns(3)

            # Age options – include 'No change'
            with colA:
                age_options = ["No change", "[40-50]", "[50-60]", "[60-70]", "[70-80]", "[80-90]"]
                base_age = base_patient.get("age", "No change")
                default_age_idx = age_options.index(base_age) if base_age in age_options else 0
                custom_age = st.selectbox(
                    "Age group",
                    age_options,
                    index=default_age_idx,
                    key="custom_age",
                )

            # Insulin status options
            with colB:
                insulin_options = ["No change", "No", "Steady"]
                base_ins = base_patient.get("insulin", "No change")
                if base_ins == "No":
                    default_ins_idx = 1
                elif base_ins == "Steady":
                    default_ins_idx = 2
                else:
                    default_ins_idx = 0

                custom_insulin = st.selectbox(
                    "Insulin status",
                    insulin_options,
                    index=default_ins_idx,
                    key="custom_ins",
                )

            # Number of inpatient visits
            with colC:
                custom_inp = st.number_input(
                    "Number of inpatient visits",
                    min_value=0,
                    max_value=20,
                    step=1,
                    value=int(base_patient.get("number_inpatient", 0)),
                    key="custom_inp",
                )

            include_custom = st.checkbox(
                "Include this custom scenario",
                value=False,
                key="include_custom",
            )

        # -------- Build scenarios list from toggles + custom --------
        scenarios = []

        if sc_insulin:
            scenarios.append({
                "name": "Start steady insulin",
                "changes": {"insulin": "Steady"},
            })

        if sc_reduce_inp:
            scenarios.append({
                "name": "Reduce inpatient visits to 0",
                "changes": {"number_inpatient": 0},
            })

        if sc_younger:
            # Simple “younger age” example – you can tune this mapping if you like
            younger_age = "[50-60]"
            scenarios.append({
                "name": "Move to younger age group",
                "changes": {"age": younger_age},
            })

        if sc_metformin:
            scenarios.append({
                "name": "Add metformin (Steady)",
                "changes": {"metformin": "Steady"},
            })

        if sc_combo:
            scenarios.append({
                "name": "Insulin steady + 0 inpatient visits",
                "changes": {"insulin": "Steady", "number_inpatient": 0},
            })

        # Custom scenario – only add if user opted in and there is at least one actual change
        if include_custom:
            custom_changes = {}

            if custom_age != "No change" and custom_age != base_patient.get("age"):
                custom_changes["age"] = custom_age

            if custom_insulin != "No change" and custom_insulin != base_patient.get("insulin"):
                custom_changes["insulin"] = custom_insulin

            if int(custom_inp) != int(base_patient.get("number_inpatient", 0)):
                custom_changes["number_inpatient"] = int(custom_inp)

            if custom_changes:
                scenarios.append({
                    "name": custom_label,
                    "changes": custom_changes,
                })

        # -------- Run simulations & display results --------
        if scenarios:
            twin_df = simulate_scenarios_streamlit(base_patient, scenarios)

            st.markdown("#### Scenario Table")
            st.dataframe(
                twin_df.style.format({
                    "Probability": "{:.3f}",
                }),
                use_container_width=True,
            )

            st.markdown("#### Scenario Comparison (Risk Probability)")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(twin_df["Scenario"], twin_df["Probability"])
            ax.set_ylabel("Readmission Probability")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(twin_df["Scenario"], rotation=30, ha="right")
            st.pyplot(fig)
            
                        # ---- Explanation for Scenario Comparison Chart ----
            st.markdown(
                """
                **How to interpret this chart:**  
                Each bar shows how the likelihood of readmission changes under different treatment scenarios.  
                - Taller bars mean **higher risk**
                - Shorter bars mean **lower risk**

                By comparing bar heights, you can quickly see which interventions may help lower risk more effectively.
                """
            )

            # Short textual summary of best (lowest-risk) scenario
            try:
                best_idx = twin_df["Probability"].idxmin()
                best_row = twin_df.loc[best_idx]

                st.markdown(
                    f"**Best scenario so far:** `{best_row['Scenario']}` "
                    f"with estimated readmission risk **{best_row['Probability']*100:.1f}%** "
                    f"({best_row['Risk']})."
                )
            except Exception:
                st.caption(
                    "Could not compute a summary for scenarios. "
                    "Please check that scenario probabilities are available."
                )

        else:
            st.info("Select at least one scenario or include a custom scenario to run the digital twin.")


    
                # Short textual summary of the best scenario
        try:
            best_idx = twin_df["Probability"].idxmin()
            best_row = twin_df.loc[best_idx]

            st.markdown(
                f"💡 **Best scenario so far:** `{best_row['Scenario']}` "
                f"with estimated readmission risk **{best_row['Probability']*100:.1f}%** "
                f"({best_row['Risk']})."
            )
        except Exception:
            st.caption(
                "Could not compute a summary for scenarios. "
                "Please check that scenario probabilities are available."
            )
            
        st.markdown(
    """
    **How to read this table:**  
    Each row represents a hypothetical “what-if” situation.  
    We apply specific treatment changes (like starting steady insulin or reducing inpatient visits) to see how much they affect the chances of readmission.

    - **Scenario** → Description of the change  
    - **Changes** → Which patient details were modified  
    - **Probability** → New predicted readmission risk  
    - **Risk** → Interpreted level (Low, Moderate, High)

    This helps identify which medical actions may reduce the chances of readmission the most.
    """
)

            
        st.markdown(
            "> **Note:** This app is a research prototype for academic purposes only and must not be used "
            "for real clinical decisions."
        )
        

            # -------------- MODEL INSIGHTS SECTION --------------
    st.markdown("---")
    st.subheader("Model Insights: Top Predictive Features")

    with st.expander("Show top features that drive readmission predictions", expanded=False):
        try:
            imp_df = get_global_feature_importance(xgb_model, top_n=15)

            # 👉 OPTIONAL: debug, you can remove after checking once
            # st.write("DEBUG columns:", imp_df.columns)
            # st.write(imp_df.head())

            # ===== Table of top features =====
            st.dataframe(
                imp_df[["Clean Feature", "Relative Importance"]]
                    .style.format({"Relative Importance": "{:.3f}"})
            )

            # ===== Bar chart of feature importance =====
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(imp_df["Clean Feature"], imp_df["Relative Importance"])
            ax.set_ylabel("Relative Importance Score")
            ax.set_xticklabels(imp_df["Clean Feature"], rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(
    """
    **What this chart tells you:**  
    This graph shows which patient characteristics have the strongest influence on readmission risk according to our machine learning model.  
    Higher bars mean that the feature plays a bigger role in predicting risk.

    Example:  
    If “Insulin: Steady” is high on the chart, it means that stabilizing insulin levels strongly affects whether a patient may be readmitted.
    """
)


            # ===== Short textual summary of top features =====
            top3 = imp_df.head(3)

            st.markdown("**Most important drivers according to the model:**")
            bullets = []
            for _, row in top3.iterrows():
                bullets.append(
                    f"- **{row['Clean Feature']}** "
                    f"(relative importance: {row['Relative Importance']:.3f})"
                )
            st.markdown("\n".join(bullets))

        except Exception as e:
            st.warning(
                "Could not compute model feature importances inside the app. "
                "This does not affect predictions or digital twin functionality."
            )
            st.caption(f"Internal error: {e}")




if __name__ == "__main__":
    main()
