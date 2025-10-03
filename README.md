# Breast Cancer Prediction with Machine Learning

This project demonstrates an end-to-end **machine learning pipeline** for predicting breast cancer using the **Breast Cancer Wisconsin Diagnostic Dataset**.
The Logistic Regression model classifies tumors as **malignant (cancerous)** or **benign (non-cancerous)**, showcasing the full ML workflow from data cleaning to model evaluation.

**Note:** This project is for educational and demonstration purposes only.
It must **not** be used for clinical decision-making.

---

## Project Workflow

The pipeline follows these steps:

1. **Data loading & cleaning**

2. **Exploratory Data Analysis (EDA)**
   - Class distribution of malignant vs benign
   - Feature distributions and correlations

3. **Preprocessing**
   - Scaling
   - Train/test split with stratification

4. **Model Training**
   - Logistic Regression with **GridSearchCV** for hyperparameter tuning
   - Pipelines with `ColumnTransformer`, `StandardScaler`

5. **Threshold Selection for Recall**
   - Precision-recall analysis
   - Cross-validation to pick a threshold ensuring ~99% recall
   - Prioritizes recall to minimize false negatives in diagnosis context

6. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Confusion matrix, precision-recall curves

7. **Saving Model & Threshold**
   - Best model saved with `joblib`
   - Optimal threshold stored in JSON for reproducibility

---

## Results

- Achieved **95.6% accuracy** and **97.6% recall** on the test set
- High recall ensures the model rarely misses malignant cases
- Shows potential of ML for assisting medical diagnostics (but not replacing doctors)

---

## Jupyter Notebook

For a full walkthrough with code, outputs, and visualizations, see the
Jupyter Notebook [BreastCancerPredictAI.ipynb](BreastCancerPredictAI.ipynb)
