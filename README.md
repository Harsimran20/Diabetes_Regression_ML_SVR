# Diabetes_Regression_ML_SVR
## 🚀 Project Overview

This project implements an end-to-end regression pipeline:

✅ Data Loading  
✅ Train-Test Split  
✅ Feature Scaling  
✅ Target Variable Scaling  
✅ Support Vector Regression (SVR)  
✅ Hyperparameter Tuning using GridSearchCV  
✅ Model Evaluation using R² Score  

---

## 📂 Dataset

The dataset used is the **Diabetes Dataset** provided by Scikit-Learn.

```python
from sklearn import datasets

df = datasets.load_diabetes(as_frame=True).frame
```

### Dataset Information

| Attribute | Value |
|-----------|-------|
| Samples | 442 |
| Features | 10 |
| Target | Disease Progression |

---

## 🛠️ Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-Learn
- 📈 Matplotlib
- 📉 Seaborn

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Preprocessing

- Separate Features and Target
- Train-Test Split
- Standard Scaling

### 2️⃣ Model Training

```python
model = SVR()
model.fit(X_train, y_train_scaled)
```

---

### 3️⃣ Hyperparameter Tuning

```python
param_grid = {
    "C": [1,2,5,10,50,100],
    "kernel": ["rbf", "linear"],
    "epsilon": [0.01,0.1,0.2,0.3,0.5]
}
```

Grid Search with 5-Fold Cross Validation:

```python
grid_search = GridSearchCV(
    SVR(),
    param_grid,
    scoring="r2",
    cv=5
)
```

---

## 📊 Evaluation Metric

### R² Score

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- ✅ R² = 1 → Perfect Model
- ⚠️ R² = 0 → Average Prediction
- ❌ R² < 0 → Poor Model

---

## 🏆 Best Hyperparameters

Example:

```python
print(grid_search.best_params_)
```

Output:

```python
{
    'C': 10,
    'kernel': 'linear',
    'epsilon': 0.1
}
```

---

## 📁 Project Structure

```text
svr-diabetes-regression/
│
├── data/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── svm_regressor.py
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/username/svr-diabetes-regression.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python svm_regressor.py
```

---

## 📌 Future Improvements

- 🔥 RandomizedSearchCV
- 🔥 Bayesian Optimization
- 🔥 Feature Engineering
- 🔥 Model Deployment with Flask/FastAPI
- 🔥 Streamlit Dashboard

---
