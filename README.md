# 🤖 ML Visualizer – Supervised Learning Playground

An interactive **Streamlit web app** designed to visualize and evaluate popular **supervised machine learning algorithms**. This tool empowers users to experiment with different **regression** and **classification** models, inspect dataset behavior, view evaluation metrics, and visualize prediction results in real-time.

---

## 🧠 Supported ML Algorithms

### 📈 Regression Models  
*(Uses Iris and Marketing Spend datasets)*  
- Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- K-Nearest Neighbors (KNN) Regressor  
- Support Vector Regressor (SVR)

#### ✅ Regression Evaluation Metrics:
- Mean Squared Error (MSE)  
- R² Score  
- Actual vs Predicted Scatter Plot

---

### 🎯 Classification Models  
*(Uses Employee Shortlisting dataset)*  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN) Classifier  
- Support Vector Machine (SVM) Classifier  

#### ✅ Classification Evaluation Metrics:
- Accuracy Score  
- Classification Report (Precision, Recall, F1-Score)  
- Confusion Matrix

---

## 📂 Datasets Used

| Dataset               | Type         | Description                                  |
|-----------------------|--------------|----------------------------------------------|
| `iris.csv`            | Regression   | Predicting petal/sepal measurements          |
| `marketing_spend.csv` | Regression   | Predicting profit based on ad spend          |
| `employee_data.csv`   | Classification | Employee shortlisting based on skills & attributes |

---

## 🚀 Features

- Interactive model selection and configuration  
- Feature/target column selector  
- Clean UI with real-time updates using Streamlit  
- Evaluation metrics and prediction plots  
- Extensible codebase (easily add more models)

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- Matplotlib & Seaborn

---



## 🏃‍♀️ Getting Started

```bash
# Clone the repo
git clone https://github.com/Sunaina792/ML-Visualizer-App.git
cd ML-Visualizer-App
```
# Create virtual environment 
```
python -m venv .venv
source .venv/Scripts/activate  # or use .venv\Scripts\activate for Windows
```
# Install dependencies
```
pip install -r requirements.txt
```
# Run the app
```
streamlit run app.py
```
### 🙋‍♀️ Made With ❤️ by Sunaina

