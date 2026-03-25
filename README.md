# 🏥 Diabetes Readmission Prediction

A machine learning project that predicts the **probability of hospital readmission within 30 days** for diabetic patients. The project includes an end-to-end ML pipeline — from exploratory data analysis and model training to an interactive **Streamlit web app with Digital Twin simulation** for exploring treatment interventions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Data Preprocessing](#data-preprocessing)
- [Models & Results](#models--results)
- [Feature Importance](#feature-importance)
- [Streamlit App](#streamlit-app)
- [Tech Stack](#tech-stack)

---

## Overview

Hospital readmissions within 30 days of discharge are costly and often preventable. This project builds a predictive model using clinical data to identify diabetic patients at high risk of readmission, enabling early interventions.

**Key highlights:**
- Trained on **101,766** real-world patient records
- Compared **3 ML models**: Logistic Regression, Random Forest, and XGBoost
- XGBoost achieved the best recall (**59.2%**) and ROC-AUC (**0.684**)
- Interactive **Streamlit app** to predict risk and simulate treatment scenarios
- **Digital Twin** feature lets clinicians explore "what-if" interventions to lower patient risk

---

## Dataset

**Source:** [UCI Diabetes 130-US hospitals dataset (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

| Property | Value |
|---|---|
| Records | 101,766 patient encounters |
| Features | 50 original features |
| Target | `readmitted` → binarised to `1` if `<30` days, else `0` |
| Class distribution | ~10% readmitted within 30 days (imbalanced) |

**Key feature groups:**

| Category | Features |
|---|---|
| Demographics | `race`, `gender`, `age` |
| Hospital metrics | `time_in_hospital`, `admission_type_id`, `discharge_disposition_id` |
| Visit history | `number_outpatient`, `number_emergency`, `number_inpatient` |
| Clinical tests | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` |
| Glucose/A1C | `max_glu_serum`, `A1Cresult` |
| Diagnoses | `diag_1`, `diag_2`, `diag_3` (ICD-9 codes) |
| Medications | 23 diabetes drugs (insulin, metformin, glipizide, etc.) |

---

## Project Structure

```
diabetes-readmission-prediction/
├── app.py                              # Streamlit web application
├── diabetes_project.ipynb              # Jupyter notebook (EDA → training → evaluation)
├── diabetic_data.csv                   # Raw dataset
├── xgboost_readmission_pipeline.pkl    # Saved XGBoost pipeline (generated after training)
├── train_columns.pkl                   # Saved training column order (generated after training)
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/RajatThakral01/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit matplotlib seaborn joblib
```

---

## How to Use

### 1. Train the Model (Jupyter Notebook)

Open and run `diabetes_project.ipynb` end-to-end. This will:
- Perform exploratory data analysis
- Preprocess the data
- Train and compare Logistic Regression, Random Forest, and XGBoost
- Save the trained pipeline to `xgboost_readmission_pipeline.pkl`

```bash
jupyter notebook diabetes_project.ipynb
```

### 2. Launch the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Data Preprocessing

The full preprocessing pipeline is built using scikit-learn `Pipeline` and `ColumnTransformer`:

1. **Data Cleaning**
   - Replace `'?'` with `NaN`
   - Drop low-value columns: `encounter_id`, `patient_nbr`, `weight`, `payer_code`, `medical_specialty`, `examide`, `citoglipton`

2. **Target Engineering**
   - Convert multi-class `readmitted` → binary: `1` if `<30` days, `0` otherwise

3. **ICD-9 Diagnosis Categorisation** — maps raw codes to clinical groups:
   | Code Range | Category |
   |---|---|
   | 390–459 | Circulatory |
   | 460–519 | Respiratory |
   | 250–250 | Diabetes |
   | 520–579 | Digestive |
   | 280–289 | Blood |
   | Others | Other |

4. **Imputation**
   - Numerical columns → median imputation
   - Categorical columns → most-frequent imputation

5. **Encoding** — One-hot encoding for all categorical features

6. **Class Imbalance** — [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) applied inside the training pipeline to oversample the minority class

---

## Models & Results

Three models were trained and evaluated. The primary metric is **Recall** (minimising missed readmissions) alongside **ROC-AUC**.

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 88.83% | 0.018 | 0.466 | 0.035 | 0.641 |
| Random Forest | 88.88% | 0.006 | 0.667 | 0.012 | 0.647 |
| **XGBoost ✅** | **66.55%** | **0.592** | **0.186** | **0.283** | **0.684** |

**Why XGBoost was chosen:**
- **Recall of 59.2%** — catches the majority of actual readmission cases, which is critical in a clinical setting where a missed readmission is more costly than a false alarm
- **Best ROC-AUC (0.684)** — strongest overall discrimination between classes
- Better calibrated for imbalanced clinical data with `scale_pos_weight=8`

### XGBoost Configuration
```python
XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=8,
    random_state=42,
    n_jobs=-1
)
```

---

## Feature Importance

Top predictors identified by XGBoost:

| Rank | Feature | Description |
|---|---|---|
| 1 | `number_inpatient` | Prior inpatient visits — strongest predictor |
| 2 | `time_in_hospital` | Longer stays indicate higher severity |
| 3 | `insulin_Steady` | Stable insulin therapy reduces risk |
| 4 | `change` | Medication changes during encounter |
| 5 | `age` | Older patients face higher risk |
| 6 | `num_medications` | Polypharmacy indicator |
| 7 | `A1Cresult` | Glycaemic control |
| 8 | `max_glu_serum` | Blood glucose levels |

**Clinical insight:** Patients with a history of inpatient admissions, unstable insulin regimens, or poor glycaemic control are at the highest risk of readmission.

---

## Streamlit App

The web app provides three interactive sections:

### 🔮 Risk Prediction
Enter patient information (demographics, medication status, visit history) to get:
- **Readmission probability (%)**
- **Risk category**: 🟩 Low (`< 40%`) · 🟧 Moderate (`40–69%`) · 🟥 High (`≥ 70%`)

### 🧬 Digital Twin Simulation
Simulate clinical interventions and see how they affect risk:

| Scenario | Description |
|---|---|
| Start steady insulin | Switch insulin to Steady regimen |
| Reduce inpatient visits | Set prior inpatient visits to 0 |
| Add metformin (Steady) | Add stable metformin therapy |
| Insulin + 0 inpatient | Combine both above |
| Move to younger age group | Assess age-based risk reduction |
| Custom scenario | Define your own combination of changes |

Results show a comparison table and bar chart highlighting the best intervention.

### 📊 Model Insights
- Top 15 feature importances from XGBoost
- Bar chart visualisation
- Explanation of what drives readmission predictions

---

## Tech Stack

| Category | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost` |
| Imbalanced learning | `imbalanced-learn` (SMOTE) |
| Visualisation | `matplotlib`, `seaborn` |
| Web app | `streamlit` |
| Model serialisation | `joblib` |
