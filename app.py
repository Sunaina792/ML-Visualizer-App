import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from Regression import *  # Your model functions here

st.title("🧠 Regression Visualizer")

# Load dataset
df = pd.read_csv("iris_dataset.csv")
st.write("### 📄 Sample Data", df.head())

# Feature and target selection
feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist(), default=df.columns[0])
target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)

X = df[feature_cols]
y = df[target_col]

# Model selection
model_choice = st.selectbox(
    "Choose Regression Model",
    ["Linear", "Multiple Linear", "Polynomial", "Ridge", "Lasso", "Logistic"]
)

# Hyperparameters
if model_choice == "Polynomial":
    degree = st.slider("Polynomial Degree", 2, 5, 2)
if model_choice in ["Ridge", "Lasso"]:
    alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0)

# Run model
if st.button("Run Model"):
    if model_choice == "Linear":
        model, pred = linear_regression(X, y)
    elif model_choice == "Multiple Linear":
        model, pred = multiple_linear_regression(X, y)
    elif model_choice == "Polynomial":
        model, pred = polynomial_regression(X, y, degree)
    elif model_choice == "Ridge":
        model, pred = ridge_regression(X, y, alpha)
    elif model_choice == "Lasso":
        model, pred = lasso_regression(X, y, alpha)
    elif model_choice == "Logistic":
        model, pred = logistic_regression(X, y)

    # Coefficients
    st.write("### 📊 Model Coefficients")
    try:
        st.write("Coefficients:", model.coef_)
        st.write("Intercept:", model.intercept_)
    except:
        st.warning("Model does not expose coefficients directly.")

    # Metrics
    st.write("### 📈 Model Evaluation")
    try:
        mse = mean_squared_error(y, pred)
        r2 = r2_score(y, pred)
        st.metric("📉 Mean Squared Error (MSE)", f"{mse:.4f}")
        st.metric("📊 R² Score", f"{r2:.4f}")
    except:
        st.warning("Unable to calculate metrics — check data shape or model type.")

    # Plot
    st.write("### 📉 Prediction Plot")
    if X.shape[1] == 1:
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Actual')
        ax.plot(X, pred, color='red', label='Regression Line')
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        ax.scatter(y, pred, color='green')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
