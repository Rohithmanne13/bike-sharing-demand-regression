# Bike Sharing Demand Regression Analysis

## Overview
This project implements and evaluates multiple regression models to predict hourly bike-sharing demand using the Kaggle Bike Sharing Demand dataset. All models are implemented from scratch using NumPy, with careful preprocessing to ensure a leakage-free evaluation.

The objective is to compare linear and nonlinear regression models and identify the best-performing model based solely on test-set performance.

---

## Dataset
- **Source:** Kaggle Bike Sharing Demand Dataset  
- **Target variable:** `count` (total hourly rentals)

---

## Preprocessing (Leakage-Free Workflow)
To ensure correct evaluation and avoid train–test leakage, the following preprocessing steps are applied:

- The `datetime` feature is decomposed into:
  - year
  - month
  - day
  - hour
  - dayofweek
- Leakage-prone variables (`casual`, `registered`) are removed because the target variable is defined as  
count = casual + registered
which would otherwise introduce direct target leakage.
- Data is sorted chronologically and split into:
- 80% training data
- 20% testing data
- Categorical variables (`season`, `holiday`, `workingday`, `weather`) are:
- One-hot encoded using training data categories only
- Numeric features are standardized using training-set mean and standard deviation only

This workflow ensures the model is trained strictly on past data and evaluated on future data.

---

## Models Implemented
All regression coefficients are computed using the Normal Equation.

- Linear Regression (baseline)
- Polynomial Regression (no interaction terms)
- Degree 2
- Degree 3
- Degree 4
- Quadratic Regression with interaction terms (degree 2 only)

---

## Evaluation Metrics
Models are evaluated on the test set using:
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)

Predicted vs. Actual plots are generated for each model.

---

## Results Summary
- Degree-3 polynomial regression achieves the best generalization performance
- Linear regression underfits due to high bias
- Degree-4 polynomial regression shows mild overfitting
- Quadratic interaction model suffers from high variance and multicollinearity

These results align with bias–variance tradeoff theory.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib

---

## Files
- `regression.py` – full implementation and evaluation
- `Optimization_Regression_Analysis_Report.pdf` – mathematical formulation and analysis

---

## Execution
```bash
python3 regression.py
