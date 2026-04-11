# 📡 Customer Churn Prediction — End-to-End ML Pipeline

> **Telco Customer Churn** | IBM Watson Dataset | Random Forest Classifier | Deployed on Streamlit

---

## 🗂️ Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Step-by-Step Pipeline (Google Colab)](#-step-by-step-pipeline-google-colab)
   - [Step 1 — Import Libraries](#step-1--import-libraries)
   - [Step 2 — Load Dataset & Initial Exploration](#step-2--load-dataset--initial-exploration)
   - [Step 3 — Data Cleaning](#step-3--data-cleaning)
   - [Step 4 — Exploratory Data Analysis (EDA)](#step-4--exploratory-data-analysis-eda)
   - [Step 5 — Data Preprocessing & Label Encoding](#step-5--data-preprocessing--label-encoding)
   - [Step 6 — Train-Test Split](#step-6--train-test-split)
   - [Step 7 — Handling Class Imbalance with SMOTE](#step-7--handling-class-imbalance-with-smote)
   - [Step 8 — Model Training & Cross-Validation](#step-8--model-training--cross-validation)
   - [Step 9 — Final Model Training & Evaluation](#step-9--final-model-training--evaluation)
   - [Step 10 — Saving Model & Encoders](#step-10--saving-model--encoders)
   - [Step 11 — Building a Predictive System](#step-11--building-a-predictive-system)
6. [Key Metrics & Results](#-key-metrics--results)
7. [Insights & Observations](#-insights--observations)
8. [Streamlit Deployment](#-streamlit-deployment)
9. [How to Run Locally](#-how-to-run-locally)
10. [Known Limitations & Future Improvements](#-known-limitations--future-improvements)

---

## 🎯 Project Overview

Customer churn refers to the phenomenon where customers stop using a company's service. In the telecom industry, retaining existing customers is significantly cheaper than acquiring new ones. This project builds a **binary classification model** to predict whether a customer will churn (`Yes`) or not (`No`) based on their demographics, account details, and service subscriptions.

**Problem Type:** Binary Classification  
**Target Variable:** `Churn` (Yes / No)  
**Best Model:** Random Forest Classifier  
**CV Accuracy:** ~84%  
**Test Accuracy:** ~77.9%

---

## 📊 Dataset

- **Name:** Telco Customer Churn Dataset  
- **Source:** IBM Watson Analytics Sample Data  
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Shape:** 7,043 rows × 21 columns  
- **Target:** `Churn` (26.5% churned, 73.5% retained — imbalanced)

### Feature Categories

| Category | Features |
|---|---|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Account Info** | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **Google Colab** | Notebook environment for model development |
| **Pandas** | Data loading, manipulation, cleaning |
| **NumPy** | Numerical operations |
| **Matplotlib + Seaborn** | Visualization (histograms, box plots, heatmaps, bar charts) |
| **Scikit-learn** | Preprocessing, model training, cross-validation, evaluation |
| **imbalanced-learn (SMOTE)** | Handling class imbalance in training data |
| **XGBoost** | Gradient boosted trees for comparison |
| **Pickle** | Serializing and saving trained model & encoders |
| **Streamlit** | Web application for live predictions |

---

## 📁 Project Structure

```
Customer-Churn/
├── Customer_Churn.ipynb      # Main Colab notebook — full ML pipeline
├── Dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── rfc_model.pkl             # Saved Random Forest model + feature names
├── encoders.pkl              # Saved LabelEncoders for all categorical columns
├── app.py                    # Streamlit web app for live predictions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔬 Step-by-Step Pipeline (Google Colab)

### Step 1 — Import Libraries

All required libraries were imported at the top of the notebook:

```python
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

### Step 2 — Load Dataset & Initial Exploration

The dataset was loaded from Google Drive / Colab local:

```python
data = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

**Initial Checks Performed:**
- `data.shape` → **(7043, 21)** — 7,043 rows, 21 columns
- `data.columns` → Listed all 21 feature names
- `data.info()` → Identified data types: `TotalCharges` incorrectly typed as `object` (should be `float64`)
- `data.head()` → Viewed sample rows
- `data.describe()` → Statistical summary (mean, std, min/max) for numeric columns: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- `data[col].unique()` for each column → Explored distinct values and detected issues

**Key Observation:** `customerID` is a unique identifier with no predictive value — dropped immediately:

```python
data.drop(["customerID"], axis=1, inplace=True)
```

---

### Step 3 — Data Cleaning

#### 3.1 — Missing Value Detection

```python
data.isnull().sum()   # Output: 0 null values across all columns
```

Though `isnull().sum()` returned 0, `TotalCharges` was stored as `object` dtype and contained **11 whitespace strings** (`" "`) masquerading as missing values:

```python
len(data[data['TotalCharges'] == " "])   # Output: 11
```

#### 3.2 — Handling Hidden Missing Values

Since only 11 rows (~0.15%) had blank `TotalCharges`, **dropping them was not done** (to preserve data). They were replaced with `"0.0"` instead:

```python
data['TotalCharges'] = data['TotalCharges'].replace(" ", "0.0")
data['TotalCharges'] = data['TotalCharges'].astype(float)
```

#### 3.3 — Class Imbalance Check

```python
data['Churn'].value_counts()
# No     5174  (73.5%)
# Yes    1869  (26.5%)
```

⚠️ **Class imbalance identified** — the dataset has roughly a 73:27 ratio between non-churners and churners. This noted as a critical issue to handle during training.

**Key Insights after cleaning:**
1. `customerID` removed (no predictive value)
2. No structural null values found
3. `TotalCharges` dtype corrected from `object` → `float64`
4. 11 whitespace blanks in `TotalCharges` replaced with `0.0`
5. Class imbalance identified in the target column

---

### Step 4 — Exploratory Data Analysis (EDA)

#### 4.1 — Numerical Feature Distribution (Histograms)

A reusable `histogram()` function was defined to plot `KDE + histplot` with mean and median reference lines:

```python
def histogram(data, col):
    plt.figure(figsize=(5, 3))
    sns.histplot(data[col], kde=True)
    plt.axvline(data[col].mean(), color='red', linestyle='dashed')    # Mean line
    plt.axvline(data[col].median(), color='green', linestyle='dashed') # Median line
    plt.legend(['Mean', 'Median'])
    plt.show()

numeric_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_col:
    histogram(data, col)
```

> 💡 **Insight:** If Mean > Median → indicates **right skew / high outliers**

#### 4.2 — Box Plots (Outlier Detection)

```python
def box_plt(data, col):
    plt.figure(figsize=(5, 3))
    sns.boxplot(data[col])
    plt.title(f"BOX-PLOT of {col}")
    plt.show()

for col in numeric_col:
    box_plt(data, col)
```

> ✅ **Result:** No outliers detected (no dots beyond whiskers) for all three numerical features.

#### 4.3 — Correlation Heatmap

```python
plt.figure(figsize=(5, 3))
sns.heatmap(data[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

> ⚠️ **Insight:** `tenure` and `TotalCharges` are **highly correlated** (customers who stay longer accumulate higher charges). This flags a potential **multicollinearity** concern — noted for future feature selection.

#### 4.4 — Categorical Features Analysis (Count Plots)

All categorical + boolean columns were identified and visualized:

```python
object_col = data.select_dtypes(include='object').columns.to_list()
object_col = ["SeniorCitizen"] + object_col  # SeniorCitizen is int (0/1) but categorical

for col in object_col:
    plt.figure(figsize=(5, 3))
    sns.countplot(data=data, x=col)
    plt.title(f"Count Plot of {col}")
    plt.show()
```

This generated **17 count plots** covering all categorical columns including the target `Churn`.

---

### Step 5 — Data Preprocessing & Label Encoding

#### 5.1 — Encode Target Column

The `Churn` column (`Yes`/`No`) was encoded to binary (1/0):

```python
data['Churn'] = LabelEncoder().fit_transform(data['Churn'])
# 0 = No Churn (5174 rows)
# 1 = Churn    (1869 rows)
```

#### 5.2 — Encode All Categorical Features

All object-type columns were label encoded. **Crucially, each encoder was stored** so it can be reused during inference without training again:

```python
object_columns = data.select_dtypes(include='object').columns.to_list()
# ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
#  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#  'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
#  'PaperlessBilling', 'PaymentMethod']

encoders = {}
for column in object_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder  # ← Save each fitted encoder
```

#### 5.3 — Save Encoders to Pickle

```python
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
```

This allows the Streamlit app to **apply the exact same encoding** on new user inputs.

---

### Step 6 — Train-Test Split

Features (`X`) and target (`Y`) were separated. The dataset was split 80-20:

```python
X = data.drop("Churn", axis=1)   # Shape: (7043, 19)
Y = data["Churn"]                  # Shape: (7043,)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
# x_train: (5634, 19) | y_train: (5634,)
# x_test:  (1409, 19) | y_test:  (1409,)
```

**Train set class distribution:**
```
Churn
0    4138   (73.5%)
1    1496   (26.5%)
```

> ⚠️ Class imbalance persists in training data — addressed in the next step with SMOTE.

---

### Step 7 — Handling Class Imbalance with SMOTE

**SMOTE (Synthetic Minority Oversampling TEchnique)** was applied **only on training data** to prevent data leakage:

```python
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# x_train_smote: (8276, 19)
# y_train_smote: (8276,)

y_train_smote.value_counts()
# Churn
# 0    4138
# 1    4138   ← perfectly balanced now
```

> ✅ SMOTE created **2,642 synthetic churn samples** to balance the minority class.  
> ℹ️ SMOTE was applied **after** the train-test split to avoid contaminating test data.

---

### Step 8 — Model Training & Cross-Validation

Three models were trained using 5-fold cross-validation on the SMOTE-balanced training set:

```python
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}
for model_name, model in models.items():
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} → CV Accuracy: {scores.mean():.4f}")
```

**Cross-Validation Results (5-Fold):**

| Model | Fold Scores | Mean CV Accuracy |
|---|---|---|
| Decision Tree | [0.683, 0.713, 0.822, 0.836, 0.836] | **77.78%** |
| **Random Forest** | [0.725, 0.778, 0.905, 0.894, 0.901] | **84.08% ✅** |
| XGBoost | [0.700, 0.756, 0.903, 0.895, 0.900] | **83.10%** |

A bar chart was plotted to compare mean CV accuracies:

```python
sns.barplot(x=model_names, y=mean_cv_accuracies, palette='viridis')
plt.title('Cross-Validation Accuracy Comparison')
```

> 🏆 **Random Forest** achieved the highest cross-validation accuracy and was selected as the final model.

---

### Step 9 — Final Model Training & Evaluation

The Random Forest Classifier was retrained on the full SMOTE-balanced training set:

```python
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train_smote, y_train_smote)
```

Predictions were made on the **original (unbalanced) test set**:

```python
y_pred = rfc.predict(x_test)
```

**Test Set Evaluation:**

```
Test set distribution:   Class 0: 1036 | Class 1: 373

Accuracy Score : 0.7786  (77.86%)

Confusion Matrix:
          Pred 0   Pred 1
Actual 0    878      158    (85% precision, 85% recall)
Actual 1    154      219    (58% precision, 59% recall)

Classification Report:
              precision  recall  f1-score  support
Class 0           0.85    0.85      0.85     1036
Class 1           0.58    0.59      0.58      373
accuracy                            0.78     1409
macro avg         0.72    0.72      0.72     1409
```

A confusion matrix heatmap was plotted:

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
```

> ⚠️ **Key Insight:** Test accuracy (78%) is lower than CV accuracy (84%). This is because the test set is **imbalanced** (26% churn), while the model was trained on a **balanced** dataset via SMOTE. Accuracy alone is misleading here — **F1-score, Precision, and Recall** are the reliable metrics for imbalanced classification.

---

### Step 10 — Saving Model & Encoders

The trained model and feature names were serialized for use in the Streamlit app:

```python
model_data = {
    "model": rfc,
    "feature_names": X.columns.tolist()   # 19 features in exact training order
}

with open('rfc_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
```

> 💡 Saving `feature_names` alongside the model ensures prediction inputs always match the exact column order used during training.

---

### Step 11 — Building a Predictive System

A full predictive system was built and tested within the notebook itself:

```python
# Load saved model and encoders
with open("rfc_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model  = model_data["model"]
feature_names = model_data["feature_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
```

**Sample prediction with a real customer profile:**

```python
input_data = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 1, 'PhoneService': 'No',
    'MultipleLines': 'No phone service', 'InternetService': 'DSL',
    'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85, 'TotalCharges': 29.85
}

input_df = pd.DataFrame([input_data])

# Apply saved encoders (same transformation as training)
for column, encoder in encoders.items():
    if column in input_df.columns:
        input_df[column] = encoder.transform(input_df[column])

# Align feature order
input_df = input_df[feature_names]

# Make prediction
prediction = loaded_model.predict(input_df)
pred_prob  = loaded_model.predict_proba(input_df)

# Output:
# prediction : No Churn
# Probability of Churn : [[0.78, 0.22]]
```

---

## 📈 Key Metrics & Results

| Metric | Value |
|---|---|
| **Dataset Size** | 7,043 customers, 19 features |
| **Class Distribution (original)** | 73.5% No Churn / 26.5% Churn |
| **SMOTE Balanced Training Set** | 4,138 each class (8,276 total) |
| **Best Model** | Random Forest Classifier |
| **CV Accuracy (5-fold)** | **84.08%** |
| **Test Accuracy** | **77.86%** |
| **Churn Precision (Class 1)** | 0.58 |
| **Churn Recall (Class 1)** | 0.59 |
| **Churn F1-Score (Class 1)** | 0.58 |
| **Non-Churn F1-Score (Class 0)** | 0.85 |

---

## 💡 Insights & Observations

1. **Class Imbalance is Critical:** The raw dataset has a 73:27 class split. Without SMOTE, models would learn to predict "No Churn" for almost everything and still hit ~73% accuracy — while being useless for identifying actual churners.

2. **SMOTE on Training Data Only:** SMOTE was applied exclusively on `x_train` / `y_train` (never on test data), correctly preventing **data leakage**.

3. **CV vs Test Accuracy Gap:** The ~6% gap between cross-validation (84%) and test accuracy (78%) is expected and partly explained by the test set's natural imbalance.

4. **Accuracy is Misleading:** In imbalanced problems, looking at accuracy alone is dangerous. F1-score, Precision, and Recall for the minority class (Churn = 1) are the true indicators of model quality.

5. **TotalCharges ↔ Tenure Correlation:** These two features are highly correlated. In future iterations, dropping one or applying PCA could reduce multicollinearity.

6. **Encoders Must Be Saved:** Using `fit_transform()` on training and `transform()` only during inference is a critical deployment practice — saving encoders ensures consistent label mappings.

---

## 🚀 Streamlit Deployment

A **Streamlit web application** (`app.py`) was built to serve live churn predictions:

- **Input:** 19 customer attributes via interactive dropdowns and number inputs
- **Processing:** Applies the saved `LabelEncoder` for categorical fields, aligns features to training order
- **Output:** Churn prediction (✅ No / ⚠️ Yes) with confidence probability

**Live Demo (if deployed):** [Streamlit Cloud link here]

---

## 🔧 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/Customer-Churn.git
cd Customer-Churn

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost streamlit

# Run Streamlit app
streamlit run app.py
```

---

## ⚠️ Known Limitations & Future Improvements

| Limitation | Planned Fix |
|---|---|
| Default hyperparameters used | Apply **GridSearchCV / RandomizedSearchCV** for hyperparameter tuning |
| High correlation between `tenure` and `TotalCharges` | Feature selection / PCA |
| SMOTE may not always be the best technique | Try **ADASYN**, **class_weight='balanced'** in RFC, or **undersampling** |
| Model not re-evaluated after SMOTE tuning | Run final evaluation after hyperparameter tuning |
| No feature importance plot | Add `feature_importances_` visualization |

---

## 👤 Author

**Nithin Kumar**  
Data Science & Machine Learning | Python · Scikit-learn · Streamlit  

---

*Built with ❤️ in Google Colab | Deployed on Streamlit Cloud*