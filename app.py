# Complete app.py with all functions included
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def linear_regression(X, y):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = LinearRegression()
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def multiple_linear_regression(X, y):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = LinearRegression()
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def polynomial_regression(X, y, degree=2):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def ridge_regression(X, y, alpha=1.0):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = Ridge(alpha=alpha)
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def lasso_regression(X, y, alpha=1.0):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = Lasso(alpha=alpha)
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def svr_regression(X, y, kernel='rbf'):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = SVR(kernel=kernel)
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def random_forest_regression(X, y, n_estimators=100):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def decision_tree_regression(X, y):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = DecisionTreeRegressor()
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

def knn_regression(X, y, n_neighbors=5):
    y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y_array)
    pred = model.predict(X)
    mse = mean_squared_error(y_array, pred)
    r2 = r2_score(y_array, pred)
    return model, pred, mse, r2

# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, acc, cm, report

def random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, acc, cm, report

def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, acc, cm, report

def svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf'):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, acc, cm, report

def logistic_classifier(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, y_pred, acc, cm, report

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="ML Visualizer", layout="wide")
st.title("🔍 ML Visualizer – Supervised Learning Explorer")


# Sidebar navigation
visualizer = st.sidebar.selectbox("Choose Visualizer", ["Regression", "Classification"])

# ========== REGRESSION ==========
if visualizer == "Regression":
    st.header("🧠 Regression Visualizer")

    model_choice = st.selectbox(
        "Choose Regression Model",
        ["Linear", "Multiple Linear", "Polynomial", "Ridge", "Lasso", "SVR", "Random Forest", "Decision Tree", "KNN"]
    )

    # Dataset based on model
    if model_choice in ["Linear", "Multiple Linear", "Polynomial", "Ridge", "Lasso"]:
        try:
            df = pd.read_csv("iris_dataset.csv")
        except FileNotFoundError:
            st.error("❌ iris_dataset.csv not found. Please make sure the file exists.")
            st.stop()
    else:
        try:
            df = pd.read_csv("marketing_spend.csv")
        except FileNotFoundError:
            st.error("❌ marketing_spend.csv not found. Please make sure the file exists.")
            st.stop()

    st.write("### 📄 Sample Data", df.head())

    # Feature and target selection
    feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist(), default=[df.columns[0]])
    target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)

    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column.")
        st.stop()

    X = df[feature_cols]
    y = df[[target_col]]

    # Hyperparameters
    if model_choice == "Polynomial":
        degree = st.slider("Polynomial Degree", 2, 5, 2)
    if model_choice in ["Ridge", "Lasso"]:
        alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0)
    if model_choice == "SVR":
        kernel = st.selectbox("SVR Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
    if model_choice == "KNN":
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5)

    if st.button("Run Model"):
        try:
            if model_choice == "Linear":
                model, pred, mse, r2 = linear_regression(X, y)
            elif model_choice == "Multiple Linear":
                model, pred, mse, r2 = multiple_linear_regression(X, y)
            elif model_choice == "Polynomial":
                model, pred, mse, r2 = polynomial_regression(X, y, degree)
            elif model_choice == "Ridge":
                model, pred, mse, r2 = ridge_regression(X, y, alpha)
            elif model_choice == "Lasso":
                model, pred, mse, r2 = lasso_regression(X, y, alpha)
            elif model_choice == "SVR":
                model, pred, mse, r2 = svr_regression(X, y, kernel)
            elif model_choice == "Random Forest":
                model, pred, mse, r2 = random_forest_regression(X, y, n_estimators)
            elif model_choice == "Decision Tree":
                model, pred, mse, r2 = decision_tree_regression(X, y)
            elif model_choice == "KNN":
                model, pred, mse, r2 = knn_regression(X, y, n_neighbors)

            st.write("### 📊 Model Coefficients")
            try:
                # For pipeline models (polynomial), access the final step
                if hasattr(model, 'steps'):
                    final_model = model.steps[-1][1]
                    st.write("Coefficients:", final_model.coef_)
                    st.write("Intercept:", final_model.intercept_)
                else:
                    st.write("Coefficients:", model.coef_)
                    st.write("Intercept:", model.intercept_)
            except:
                st.warning("Model does not expose coefficients directly.")

            st.write("### 📈 Model Evaluation")
            st.metric("📉 Mean Squared Error (MSE)", f"{mse:.4f}")
            st.metric("📊 R² Score", f"{r2:.4f}")
            
            st.write("### 📉 Prediction Plot")
            if X.shape[1] == 1:
                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue', label='Actual')
                ax.plot(X, pred, color='red', label='Prediction')
                ax.set_xlabel(X.columns[0])
                ax.set_ylabel(target_col)
                ax.legend()
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                ax.scatter(y, pred, color='green')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Error running model: {e}")

# ========== CLASSIFICATION ==========
elif visualizer == "Classification":
    st.header("🧠 Classification Visualizer")

    try:
        df1 = pd.read_csv("employee_shortlisting.csv")
    except FileNotFoundError:
        st.error("❌ employee_shortlisting.csv not found. Please make sure the file exists.")
        st.stop()
    
    st.write("### 📄 Sample Data", df1.head())

    # Check if default columns exist
    default_features = []
    if "Interview_Score" in df1.columns:
        default_features.append("Interview_Score")
    if "Skills_Matched" in df1.columns:
        default_features.append("Skills_Matched")
    
    if not default_features:
        default_features = [df1.columns[0]] if len(df1.columns) > 1 else []

    features = st.multiselect("Select Feature Columns", df1.columns.tolist(), default=default_features)
    target = st.selectbox("Select Target Column", df1.columns.tolist(), index=len(df1.columns) - 1)

    if not features:
        st.warning("⚠️ Please select at least one feature column.")
        st.stop()

    X1 = df1[features]
    y1 = df1[target]

    test_size = st.slider("Test Size (%)", 10, 50, 20)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size/100, random_state=42)

    scaler = StandardScaler()
    X1_train_scaled = scaler.fit_transform(X1_train)
    X1_test_scaled = scaler.transform(X1_test)

    model_choice = st.selectbox("Choose Classification Model", ["Decision Tree", "Random Forest", "KNN", "SVM", "Logistic Regression"])

    if model_choice == "KNN":
        k = st.slider("Number of Neighbors (K)", 1, 20, 5)
    if model_choice == "SVM":
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)

    if st.button("Run Model"):
        try:
            if model_choice == "Decision Tree":
                model, y_pred, acc, cm, report = decision_tree_classifier(X1_train_scaled, X1_test_scaled, y1_train, y1_test)
            elif model_choice == "Random Forest":
                model, y_pred, acc, cm, report = random_forest_classifier(X1_train_scaled, X1_test_scaled, y1_train, y1_test)
            elif model_choice == "KNN":
                model, y_pred, acc, cm, report = knn_classifier(X1_train_scaled, X1_test_scaled, y1_train, y1_test, k)
            elif model_choice == "SVM":
                model, y_pred, acc, cm, report = svm_classifier(X1_train_scaled, X1_test_scaled, y1_train, y1_test, kernel)
            elif model_choice == "Logistic Regression":
                model, y_pred, acc, cm, report = logistic_classifier(X1_train_scaled, X1_test_scaled, y1_train, y1_test)

            st.write("### ✅ Accuracy Score", f"{acc:.4f}")
            
            st.write("### 📋 Classification Report")
            st.text(report)

            st.write("### 📊 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            if len(features) == 2:
                st.write("### 🧩 Feature Space Visualization")
                fig2, ax2 = plt.subplots()
                for label in df1[target].unique():
                    subset = df1[df1[target] == label]
                    ax2.scatter(subset[features[0]], subset[features[1]], label=label, alpha=0.6, edgecolors='k')
                ax2.set_xlabel(features[0])
                ax2.set_ylabel(features[1])
                ax2.set_title("Feature Space")
                ax2.legend()
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"❌ Error running classification model: {e}")
            import traceback
            st.error(traceback.format_exc())