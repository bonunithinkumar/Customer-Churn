# Customer-Churn
Customer churn prediction using a telecom dataset with a complete ML pipeline. Includes EDA, class imbalance handling using SMOTE, model training and cross-validation. Highlights accuracy drop on test data and the need for precision, recall, and F1-score over accuracy alone.

🔍 Project Overview

This project implements a machine learning pipeline to predict customer churn using a telecom dataset. It includes data preprocessing, exploratory data analysis, handling class imbalance using SMOTE, and training multiple classification models.

Models were evaluated using cross-validation and test data. Although Random Forest performed best during cross-validation, a drop in test accuracy highlighted the impact of class imbalance and the limitations of relying on accuracy alone. Evaluation was therefore extended to precision, recall, F1-score, and confusion matrix.

The project also saves trained models and label encoders to support future predictions and deployment.
