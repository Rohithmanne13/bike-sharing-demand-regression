import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset and extract features 
df = pd.read_csv("train.csv")

#Extract structured time components from the datetime column
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

#Remove target leakage columns
df = df.drop(columns=["casual", "registered"])

#Train–test split (sorted by timestamp, 80/20)
df = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(0.80 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
y_train = train["count"].to_numpy()
y_test = test["count"].to_numpy()

#We do not pass datetime itself to the model
X_train_raw = train.drop(columns=["count", "datetime"])
X_test_raw = test.drop(columns=["count", "datetime"])

#2. One-hot encoding (train–test safe)
categorical_cols = ["season", "holiday", "workingday", "weather"]

def fit_one_hot_categories(df_train, cat_cols):
    #Identify unique categories in the training data.
    categories = {}
    for col in cat_cols:
        categories[col] = np.unique(df_train[col].to_numpy())
    return categories

def transform_one_hot(df_part, categories):
    #encode categorical columns using only training categories
    encoded_list = []
    for col in categorical_cols:
        values = df_part[col].to_numpy()
        cats = categories[col]
        encoded = (values[:, None] == cats[None, :]).astype(int)
        encoded_list.append(encoded)
    return np.hstack(encoded_list)

cat_categories = fit_one_hot_categories(train, categorical_cols)
X_train_cat = transform_one_hot(train, cat_categories)
X_test_cat = transform_one_hot(test, cat_categories)

#3. Numeric preprocessing (standardization)
numeric_cols = ["temp", "atemp", "humidity", "windspeed",
                "year", "month", "day", "hour", "dayofweek"]

X_train_num = train[numeric_cols].to_numpy()
X_test_num = test[numeric_cols].to_numpy()

#Standardization parameters (training data only)
num_mean = X_train_num.mean(axis=0)
num_std = X_train_num.std(axis=0)
num_std[num_std == 0] = 1.0

#Apply standardization
X_train_num_std = (X_train_num - num_mean) / num_std
X_test_num_std = (X_test_num - num_mean) / num_std

#Build base feature matrices
def build_base_features(X_num_std, X_cat):
    #Combine standardized numeric and one-hot categorical data
    return np.hstack([X_num_std, X_cat])

X_train_base = build_base_features(X_train_num_std, X_train_cat)
X_test_base = build_base_features(X_test_num_std, X_test_cat)

#4. Polynomial features without interactions
def polynomial_no_interactions(X_num_std, degree):
    #Generate polynomial features for numeric variables only
    polys = [X_num_std]
    for d in range(2, degree + 1):
        polys.append(X_num_std ** d)
    return np.hstack(polys)

#5. Quadratic features with interactions
def quadratic_with_interactions_numeric(X_num_std):
    #Generate squared and interaction terms for numeric variables
    n_samples, n_features = X_num_std.shape
    features = [X_num_std, X_num_std ** 2]

    interactions = []
    for i in range(n_features):
        for j in range(i, n_features):
            interactions.append((X_num_std[:, i] * X_num_std[:, j]).reshape(-1, 1))

    return np.hstack(features + interactions)

#6. Normal equation regression
def normal_equation_fit_predict(X_train, y_train, X_test):
    #Fit using the normal equation and return test predictions
    X_train_d = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_d = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    XtX = X_train_d.T @ X_train_d
    Xty = X_train_d.T @ y_train

    beta = np.linalg.pinv(XtX) @ Xty
    return X_test_d @ beta

#7. Performance metrics (MSE and R²)
def regression_metrics(y_true, y_pred):
    #Computing test MSE and R²
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, r2

#8. Train and evaluate all models
results = []
#Linear model
y_pred_lin = normal_equation_fit_predict(X_train_base, y_train, X_test_base)
results.append(("Linear Regression (baseline)", *regression_metrics(y_test, y_pred_lin)))

#Polynomial models (no interactions)
for d in [2, 3, 4]:
    X_train_poly_num = polynomial_no_interactions(X_train_num_std, d)
    X_test_poly_num = polynomial_no_interactions(X_test_num_std, d)

    X_train_poly = np.hstack([X_train_poly_num, X_train_cat])
    X_test_poly = np.hstack([X_test_poly_num, X_test_cat])

    y_pred_poly = normal_equation_fit_predict(X_train_poly, y_train, X_test_poly)
    results.append((f"Polynomial Regression degree {d} (no interactions)",
                    *regression_metrics(y_test, y_pred_poly)))

#Quadratic model with interactions
X_train_quad = np.hstack([quadratic_with_interactions_numeric(X_train_num_std), X_train_cat])
X_test_quad = np.hstack([quadratic_with_interactions_numeric(X_test_num_std), X_test_cat])

y_pred_quad = normal_equation_fit_predict(X_train_quad, y_train, X_test_quad)
results.append(("Quadratic with interactions (degree 2)",
                *regression_metrics(y_test, y_pred_quad)))

#9. Output results
print("=== MODEL PERFORMANCE ===")
for name, mse, r2 in results:
    print(f"{name:55s} | MSE = {mse:10.2f} | R2 = {r2:7.4f}")

best_model = max(results, key=lambda x: x[2])
print("\n=== BEST MODEL ===")
print(f"Best Model: {best_model[0]}")
print(f"MSE: {best_model[1]:.2f}")
print(f"R2:  {best_model[2]:.4f}")

#10. Predicted vs Actual Plots for Each Model 
def plot_pred_vs_actual(filename_prefix, model_name, y_test, y_pred, mse, r2):
    #Generating Predicted vs Actual plot for a model 
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    # Perfect prediction diagonal
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label="Perfect prediction")

    # Titles and labels
    plt.title(f"{model_name} (test)\nMSE={mse:.2f}, R2={r2:.3f}")
    plt.xlabel("Actual count (test)")
    plt.ylabel("Predicted count (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_pred_vs_actual.png")
    plt.close()

#Generating plots for each model
for (name, mse, r2) in results:
    #Linear model
    if name.startswith("Linear Regression"):
        prefix = "linear"
        y_pred = y_pred_lin
    #Polynomial degree 2 without interactions 
    elif name.startswith("Polynomial Regression degree 2 "):
        prefix = "poly2"
        X_train_poly_num = polynomial_no_interactions(X_train_num_std, 2)
        X_test_poly_num = polynomial_no_interactions(X_test_num_std, 2)
        y_pred = normal_equation_fit_predict(
            np.hstack([X_train_poly_num, X_train_cat]),
            y_train,
            np.hstack([X_test_poly_num, X_test_cat])
           )
    #Polynomial degree 3 without interactions
    elif name.startswith("Polynomial Regression degree 3 "):
        prefix = "poly3"
        X_train_poly_num = polynomial_no_interactions(X_train_num_std, 3)
        X_test_poly_num = polynomial_no_interactions(X_test_num_std, 3)
        y_pred = normal_equation_fit_predict(
            np.hstack([X_train_poly_num, X_train_cat]),
            y_train,
            np.hstack([X_test_poly_num, X_test_cat])
        )
    #Polynomial degree 4 without interactions 
    elif name.startswith("Polynomial Regression degree 4 "):
        prefix = "poly4"
        X_train_poly_num = polynomial_no_interactions(X_train_num_std, 4)
        X_test_poly_num = polynomial_no_interactions(X_test_num_std, 4)
        y_pred = normal_equation_fit_predict(
            np.hstack([X_train_poly_num, X_train_cat]),
            y_train,
            np.hstack([X_test_poly_num, X_test_cat])
        )
    #Quadratic model with interactions  
    elif name.startswith("Quadratic with interactions"):
        prefix = "quad"
        y_pred = normal_equation_fit_predict(
            X_train_quad,
            y_train,
            X_test_quad
        )
    plot_pred_vs_actual(prefix, name, y_test, y_pred, mse, r2)
print("\nSaved Predicted vs Actual plots for all models.")
